import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ..storage.store import FileStore
from ..storage.mongo_store import MongoDBStore
from ..models import Transcript, Segment
from ..config import settings
from .steps import (
    normalize_audio,
    detect_silence_segments,
    cut_segment,
    transcribe_openai,
    transcribe_faster_whisper,
    diarize_pyannote,
)
from .diarization_light import diarize_over_asr_segments  # lightweight diarization over ASR segments
from ..jobs import job_store

# Choose storage backend based on environment
USE_MONGODB = os.getenv('USE_MONGODB_STORAGE', '').lower() in ('true', '1', 'yes')
Store = MongoDBStore if USE_MONGODB else FileStore

# REMOVED: filter_hallucinated_segments function
# Hallucination filtering is now handled by LLM correction for better context-aware cleaning
# This avoids removing valid Portuguese speech that may look "repetitive" to rule-based filters


def normalize_speaker_labels(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove all speaker labels from segments.
    
    Diarization is currently not reliable enough, so we remove speaker labels entirely
    to avoid cluttering transcripts with incorrect single-speaker labels.
    
    Args:
        segments: List of segment dicts with 'speaker' field
    
    Returns:
        Segments with speaker labels removed
    """
    if not segments:
        return segments
    
    try:
        print(f"[TA][FILTER] Removing all speaker labels (diarization disabled for quality)")
    except Exception:
        pass
    
    # Remove speaker labels from all segments
    for seg in segments:
        seg.pop("speaker", None)
        # Also remove from words if present
        if seg.get("words"):
            for w in seg["words"]:
                if isinstance(w, dict):
                    w.pop("speaker", None)
    
    return segments


def calculate_optimal_chunk_size(total_duration_seconds: float, target_processing_minutes: float = 5.0) -> int:
    """
    Calculate optimal chunk size to target a specific total processing time.
    
    Based on empirical testing:
    - Local STT (faster-whisper medium): ~0.15-0.25x realtime (4-6 minutes per hour of audio)
    - OpenAI Whisper API: ~0.05-0.1x realtime (3-6 minutes per hour, depending on network)
    - Diarization adds ~10-20% overhead
    
    Strategy: Divide audio into N chunks such that processing time ≈ target_processing_minutes
    
    Args:
        total_duration_seconds: Total audio duration in seconds
        target_processing_minutes: Desired total processing time in minutes (default: 5)
    
    Returns:
        Optimal chunk size in seconds (minimum 30, maximum 300)
    """
    # Estimate processing speed multiplier based on STT provider
    if settings.stt_provider == "openai":
        # OpenAI is faster but has network overhead
        speed_multiplier = 0.08  # ~5 minutes per hour
    else:
        # Local STT speed depends on model size
        model = settings.stt_model.lower()
        if "tiny" in model or "base" in model:
            speed_multiplier = 0.05  # Very fast
        elif "small" in model:
            speed_multiplier = 0.10
        elif "medium" in model:
            speed_multiplier = 0.20
        elif "large" in model:
            speed_multiplier = 0.35
        else:
            speed_multiplier = 0.20  # Default to medium
    
    # Add diarization overhead if enabled
    if settings.enable_diarization:
        speed_multiplier *= 1.15  # 15% overhead for speaker detection
    
    # Calculate how many chunks we need to hit target processing time
    # processing_time = duration * speed_multiplier
    # If we split into N chunks: processing_time_per_chunk = (duration/N) * speed_multiplier
    # With concurrency C: total_time = (duration/N) * speed_multiplier / C
    # Solve for N: N = (duration * speed_multiplier) / (target_minutes * 60 * C)
    
    target_seconds = target_processing_minutes * 60
    concurrency = max(1, settings.transcribe_concurrency)
    
    # Expected processing time without chunking
    expected_processing_seconds = total_duration_seconds * speed_multiplier
    
    # If already fast enough, use larger chunks
    if expected_processing_seconds <= target_seconds:
        # Use moderate chunks for better parallelization
        return min(180, int(total_duration_seconds / 2)) if total_duration_seconds > 120 else int(total_duration_seconds)
    
    # Calculate required number of chunks for target time
    # total_time = (total_duration / num_chunks) * speed_multiplier / concurrency = target_seconds
    # num_chunks = (total_duration * speed_multiplier) / (target_seconds * concurrency)
    num_chunks_needed = (total_duration_seconds * speed_multiplier) / (target_seconds * concurrency)
    num_chunks_needed = max(2, num_chunks_needed)  # At least 2 chunks for parallelization
    
    # Calculate chunk size
    optimal_chunk_seconds = total_duration_seconds / num_chunks_needed
    
    # Clamp to reasonable bounds
    # Minimum 30 seconds (too small = too much overhead)
    # Maximum 300 seconds (5 minutes - good balance)
    optimal_chunk_seconds = max(30, min(300, optimal_chunk_seconds))
    
    return int(optimal_chunk_seconds)

class PipelineState(dict):
    pass

async def run_pipeline(job_id: str, input_path: str, org_id: str | None, meeting_ref: str | None, mode: str | None, profile: str | None) -> Dict[str, Any]:
    work = Path("artifacts") / "work" / job_id
    work.mkdir(parents=True, exist_ok=True)
    # Shortcut: if input is a .txt, persist as transcript without audio processing
    if str(input_path).lower().endswith('.txt'):
        text = Path(input_path).read_text(encoding='utf-8', errors='ignore')
        transcript_id = Store.new_transcript_id()
        payload = Transcript(
            transcript_id=transcript_id,
            org_id=org_id,
            meeting_id=meeting_ref,
            mode=mode,
            profile=profile,
            source="upload_text",
            segments=[],
            text=text.strip()
        ).model_dump()
        
        # Save to storage (MongoDB or FileStore based on config)
        if USE_MONGODB:
            await MongoDBStore.save_transcript(transcript_id, payload, org_id, meeting_ref)
        else:
            FileStore.save_transcript_json(transcript_id, payload)
        
        return {"transcript_id": transcript_id, "text": text}

    def _sanitize_segments(items):
        """Sanitize a list of STT segments.
        - Keep only dicts
        - Ensure keys: start, end, text, words, speaker
        - Coerce types and defaults
        """
        if not items:
            return []
        def _sanitize_words(words):
            if not words:
                return None
            out = []
            for w in words:
                if not isinstance(w, dict):
                    continue
                ws = w.get("start")
                we = w.get("end")
                wt = w.get("text")
                wspk = w.get("speaker") if isinstance(w, dict) else None
                try:
                    ws_f = float(ws) if ws is not None else 0.0
                except Exception:
                    ws_f = 0.0
                try:
                    we_f = float(we) if we is not None else 0.0
                except Exception:
                    we_f = 0.0
                out.append({
                    "start": ws_f,
                    "end": we_f,
                    "text": "" if wt is None else str(wt),
                    "speaker": wspk
                })
            return out or None
        clean = []
        for s in items:
            if not isinstance(s, dict):
                continue
            start = s.get("start")
            end = s.get("end")
            text = s.get("text")
            words = s.get("words")
            speaker = s.get("speaker")
            try:
                start_f = float(start) if start is not None else 0.0
            except Exception:
                start_f = 0.0
            try:
                end_f = float(end) if end is not None else 0.0
            except Exception:
                end_f = 0.0
            clean.append({
                "start": start_f,
                "end": end_f,
                "text": "" if text is None else str(text),
                "words": _sanitize_words(words),
                "speaker": speaker
            })
        return clean

    async def node_normalize(state: PipelineState):
        try:
            job_store.update(job_id, phase="normalize", status="processing", progress=0.0)
        except Exception:
            pass
        try:
            norm = await normalize_audio(input_path, work)
            state["normalized"] = norm
            state["skip_normalize"] = False
            return state
        except Exception as e:
            # Fallback: continue with original input file (may still work for some STT backends)
            state["normalized"] = input_path
            state["skip_normalize"] = True
            return state

    async def node_segment(state: PipelineState):
        try:
            job_store.update(job_id, phase="segment", status="processing")
        except Exception:
            pass
        # Be defensive: fall back to the original input if normalization state is missing
        norm = state.get("normalized", input_path)
        # Probe total duration for clamping
        try:
            from .steps import probe_duration_seconds
            total_dur = probe_duration_seconds(norm)
        except Exception:
            total_dur = None
        # Fallback: estimate duration from normalized PCM WAV size (16kHz mono 16-bit => 32000 B/s)
        if (not total_dur) and (not state.get("skip_normalize")):
            try:
                p = Path(norm)
                if p.exists():
                    sz = p.stat().st_size
                    # WAV header ~44 bytes; guard non-negative
                    if sz > 44:
                        total_dur = max(0.0, (sz - 44) / 32000.0)
            except Exception:
                pass
        if total_dur and total_dur > 0:
            state["duration"] = float(total_dur)
        if state.get("skip_normalize"):
            # If we skipped normalization, still try to chunk by duration if possible
            try:
                from .steps import probe_duration_seconds
                dur = probe_duration_seconds(norm)
            except Exception:
                dur = None
            if dur and dur > 0:
                # Build fixed-size chunks across the duration with overlap
                segs = []
                start = 0.0
                while start < dur:
                    end = min(dur, start + settings.chunk_seconds)
                    segs.append((max(0.0, start - settings.overlap_seconds), end))
                    start = end
            else:
                # Fall back to single full-file segment
                segs = [(0.0, 0.0)]
        elif settings.use_silence_segmentation:
            try:
                segs = detect_silence_segments(norm)
            except Exception as e:
                # Fall back to single segment
                segs = [(0.0, 0.0)]
        else:
            # No silence segmentation; use dynamic chunking optimized for target processing time
            try:
                from .steps import probe_duration_seconds
                dur = probe_duration_seconds(norm)
            except Exception:
                dur = None
            if dur and dur > 0:
                # Calculate optimal chunk size to target 5-minute processing time
                target_minutes = float(os.environ.get("TARGET_PROCESSING_MINUTES", "5.0"))
                optimal_chunk = calculate_optimal_chunk_size(dur, target_minutes)
                try:
                    print(f"[TA][DEBUG] segment: duration={dur:.1f}s, optimal_chunk={optimal_chunk}s (target={target_minutes}min)")
                except Exception:
                    pass
                segs = []
                start = 0.0
                while start < dur:
                    end = min(dur, start + optimal_chunk)
                    segs.append((max(0.0, start - settings.overlap_seconds), end))
                    start = end
            else:
                segs = [(0.0, 0.0)]  # unbounded marker = full file downstream
        # Cap segments to dynamic chunk size and add overlap; clamp to duration if known
        final: List[Tuple[float, float]] = []
        if segs == [(0.0, 0.0)]:
            final = [(0.0, 0.0)]
        else:
            # When using silence segmentation, respect those boundaries but still apply dynamic chunking
            # if segments are too large
            try:
                from .steps import probe_duration_seconds
                dur_check = total_dur or probe_duration_seconds(norm)
            except Exception:
                dur_check = None
            
            # Calculate optimal chunk size once for capping
            if dur_check and dur_check > 0:
                target_minutes = float(os.environ.get("TARGET_PROCESSING_MINUTES", "5.0"))
                max_chunk = calculate_optimal_chunk_size(dur_check, target_minutes)
            else:
                max_chunk = settings.chunk_seconds
            
            for (s, e) in segs:
                dur = e - s
                if dur <= max_chunk:
                    s2 = max(0.0, s - settings.overlap_seconds)
                    e2 = e
                    if total_dur and total_dur > 0:
                        e2 = min(e2, total_dur)
                    final.append((s2, e2))
                else:
                    # slice into multiple chunks with overlap using optimal size
                    start = s
                    while start < e:
                        end = min(e, start + max_chunk)
                        s2 = max(0.0, start - settings.overlap_seconds)
                        e2 = end
                        if total_dur and total_dur > 0:
                            e2 = min(e2, total_dur)
                        final.append((s2, e2))
                        start = end
        state["segments"] = final or [(0.0, 0.0)]
        return state

    async def node_transcribe(state: PipelineState):
        try:
            job_store.update(job_id, phase="transcribe", status="processing", progress=0.0)
        except Exception:
            pass
        norm = state.get("normalized")
        if not norm:
            norm = input_path
            state["normalized"] = norm
            state["skip_normalize"] = True

        segs: List[Tuple[float, float]] = state.get("segments", [(0.0, 0.0)])
        try:
            print(f"[TA][DEBUG] transcribe: planned segments count={len(segs)} sample={segs[:3]}")
        except Exception:
            pass
        # If single unbounded, try to split by duration into fixed chunks for performance
        results: List[Dict[str, Any]] = []
        if segs == [(0.0, 0.0)]:
            try:
                from .steps import probe_duration_seconds
                dur = probe_duration_seconds(norm)
            except Exception:
                dur = None
            # Guard: if using OpenAI and the file is near/over the 25MB limit, force chunking
            forced_chunk = False
            if settings.stt_provider == "openai":
                try:
                    import os as _os
                    size_bytes = _os.path.getsize(norm) if _os.path.exists(norm) else None
                except Exception:
                    size_bytes = None
                # Use ~24.5MB as safety threshold
                threshold = int(24.5 * 1024 * 1024)
                if size_bytes and size_bytes >= threshold:
                    # If we have normalized PCM WAV, estimate duration from bytes (16kHz mono 16-bit => 32000 B/s)
                    if not state.get("skip_normalize"):
                        try:
                            dur = (size_bytes / 32000.0)
                        except Exception:
                            pass
                    forced_chunk = True

            if (dur and dur > 0 and dur > settings.chunk_seconds) or forced_chunk:
                # Create dynamic chunks optimized for target processing time
                fixed: List[Tuple[float, float]] = []
                start = 0.0
                # If duration is unknown but forced_chunk is True, approximate a minimal split into two parts
                if (not dur or dur <= 0) and forced_chunk:
                    dur = float(settings.chunk_seconds * 2)
                
                # Calculate optimal chunk size for target processing time
                target_minutes = float(os.environ.get("TARGET_PROCESSING_MINUTES", "5.0"))
                optimal_chunk = calculate_optimal_chunk_size(dur, target_minutes)
                try:
                    print(f"[TA][DEBUG] transcribe: using dynamic chunks of {optimal_chunk}s for {dur:.1f}s audio (target={target_minutes}min)")
                except Exception:
                    pass
                
                while start < (dur or 0.0):
                    end = min(dur, start + optimal_chunk)
                    fixed.append((max(0.0, start - settings.overlap_seconds), end))
                    start = end
                segs = fixed or [(0.0, 0.0)]
                state["segments"] = segs
            else:
                # Single-shot path: still apply a timeout for local STT to avoid indefinite stalls
                import time
                t0 = time.perf_counter()
                if settings.stt_provider == "local":
                    try:
                        fw = await asyncio.wait_for(transcribe_faster_whisper(norm), timeout=settings.chunk_timeout_seconds)
                    except Exception as e:
                        raise RuntimeError(f"local_stt_decode_failed: {e}. Please install ffmpeg or switch STT_PROVIDER=openai with OPENAI_API_KEY.")
                    results = fw["segments"]
                else:
                    try:
                        oai = await transcribe_openai(norm)
                    except Exception as e:
                        raise RuntimeError(f"openai_stt_failed: {e}")
                    # Prefer fine-grained segments when available
                    if oai.get("segments"):
                        results = [{"start": s["start"], "end": s["end"], "text": s["text"], "words": None} for s in oai["segments"]]
                    else:
                        dur_v = state.get("duration")
                        end_ts = float(dur_v) if dur_v else 0.0
                        results = [{"start": 0.0, "end": end_ts, "text": oai.get("text", ""), "words": None}]
                t1 = time.perf_counter()
                try:
                    print(f"[TA] transcribe single-shot done in {(t1 - t0):.1f}s")
                except Exception:
                    pass
                state["stt_segments"] = _sanitize_segments(results)
                # Mark progress complete for single-shot
                try:
                    job_store.update(job_id, phase="transcribe", status="processing", progress=1.0)
                except Exception:
                    pass
                return state

        # Else, cut and process concurrently
        # Safety: if OpenAI and the resulting list still indicates a single large chunk, try to derive duration and split
        if settings.stt_provider == "openai" and segs == [(0.0, 0.0)]:
            try:
                from .steps import probe_duration_seconds
                dur2 = probe_duration_seconds(norm)
            except Exception:
                dur2 = None
            if dur2 and dur2 > settings.chunk_seconds:
                fixed2: List[Tuple[float, float]] = []
                start2 = 0.0
                while start2 < dur2:
                    end2 = min(dur2, start2 + settings.chunk_seconds)
                    fixed2.append((max(0.0, start2 - settings.overlap_seconds), end2))
                    start2 = end2
                segs = fixed2
                state["segments"] = segs
        sem = asyncio.Semaphore(settings.transcribe_concurrency)
        out_dir = work / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)

        async def worker(idx: int, start: float, end: float):
            async with sem:
                chunk_path = out_dir / f"chunk_{idx:04d}.wav"
                try:
                    # Guard against zero/negative durations due to rounding
                    dur_total = state.get("duration")
                    if dur_total and end > dur_total:
                        end = dur_total
                    if (end - start) <= 0.005:
                        # Skip generating this chunk; return empty output
                        return idx, []
                    await cut_segment(norm, start, end, chunk_path)
                except Exception as e:
                    try:
                        print(f"[TA][ERROR] transcribe: cut_segment_failed idx={idx} start={start:.3f} end={end:.3f} err={e}")
                    except Exception:
                        pass
                    # Skip this chunk but continue the pipeline
                    return idx, []
                if settings.stt_provider == "local":
                    try:
                        # Apply a per-chunk timeout to avoid stalls
                        fw = await asyncio.wait_for(transcribe_faster_whisper(str(chunk_path)), timeout=settings.chunk_timeout_seconds)
                    except Exception as e:
                        if settings.hybrid_fallback and settings.openai_api_key:
                            # Fallback this chunk to OpenAI to keep pipeline moving
                            try:
                                txt = await transcribe_openai(str(chunk_path))
                                return idx, [{"start": start, "end": end, "text": txt, "words": None}]
                            except Exception as e2:
                                try:
                                    print(f"[TA][ERROR] transcribe: faster_whisper_failed_then_openai_failed idx={idx} err1={e} err2={e2}")
                                except Exception:
                                    pass
                                # Skip this chunk
                                return idx, []
                        # No fallback available: log and skip this chunk instead of crashing the job
                        import traceback
                        try:
                            print(f"[TA][ERROR] transcribe: faster_whisper_failed idx={idx} err_type={type(e).__name__} err_str={str(e)} err_repr={repr(e)}")
                            print(f"[TA][ERROR] transcribe: traceback:\n{traceback.format_exc()}")
                        except Exception:
                            pass
                        return idx, []
                    # Offset words/segments by start
                    seg_out = []
                    for seg in fw["segments"]:
                        seg_out.append({
                            "start": (seg["start"] or 0.0) + start,
                            "end": (seg["end"] or 0.0) + start,
                            "text": seg.get("text", ""),
                            "words": [{"start": (w.get("start") or 0.0) + start, "end": (w.get("end") or 0.0) + start, "text": w.get("text", "")} for w in (seg.get("words") or [])] or None
                        })
                    if not seg_out and settings.hybrid_fallback and settings.openai_api_key:
                        try:
                            txt = await transcribe_openai(str(chunk_path))
                            return idx, [{"start": start, "end": end, "text": txt, "words": None}]
                        except Exception:
                            pass
                    return idx, seg_out
                else:
                    try:
                        oai = await transcribe_openai(str(chunk_path))
                    except Exception as e:
                        try:
                            print(f"[TA][ERROR] transcribe: openai_stt_failed idx={idx} err={e}")
                        except Exception:
                            pass
                        return idx, []
                    if oai.get("segments"):
                        segs_o = []
                        for s in oai["segments"]:
                            segs_o.append({"start": start + (s.get("start") or 0.0), "end": start + (s.get("end") or 0.0), "text": s.get("text", ""), "words": None})
                        return idx, segs_o
                    return idx, [{"start": start, "end": end, "text": oai.get("text", ""), "words": None}]

        tasks = [worker(i, s, e) for i, (s, e) in enumerate(segs)]
        done = []
        total = len(tasks)
        for idx, coro in enumerate(asyncio.as_completed(tasks), start=1):
            try:
                out = await coro
                done.append(out)
            except Exception as e:
                # Catch any unexpected task failure and continue processing remaining chunks
                try:
                    print(f"[TA][ERROR] transcribe: task failed at idx={idx} err={e}")
                except Exception:
                    pass
            finally:
                # update progress
                try:
                    job_store.update(job_id, phase="transcribe", status="processing", progress=idx/total)
                    print(f"[TA] transcribe progress {idx}/{total} ({(idx/total)*100:.1f}%)")
                except Exception:
                    pass
        done.sort(key=lambda x: x[0])
        merged: List[Dict[str, Any]] = []
        for _, seglist in done:
            merged.extend(seglist)
        state["stt_segments"] = _sanitize_segments(merged)
        try:
            print(f"[TA][DEBUG] transcribe: merged segments count={len(state['stt_segments'])}")
        except Exception:
            pass
        # If local STT produced no segments, try a single-shot local decode on the full normalized audio,
        # then (optionally) an OpenAI fallback.
        if settings.stt_provider == "local" and not state["stt_segments"]:
            try:
                fw_full = await asyncio.wait_for(transcribe_faster_whisper(norm), timeout=max(60, settings.chunk_timeout_seconds))
                segs_full = _sanitize_segments(fw_full.get("segments") or [])
                if segs_full:
                    try:
                        print("[TA][DEBUG] transcribe: recovered text via single-shot local decode")
                    except Exception:
                        pass
                    state["stt_segments"] = segs_full
            except Exception:
                pass
            # If still empty and hybrid fallback is available, try OpenAI one-shot on full file
            if not state["stt_segments"] and settings.hybrid_fallback and settings.openai_api_key:
                try:
                    txt_fb = await transcribe_openai(norm)
                    if txt_fb and txt_fb.strip():
                        state["stt_segments"] = _sanitize_segments([{"start": 0.0, "end": state.get("duration", 0.0) or 0.0, "text": txt_fb, "words": None}])
                        try:
                            print("[TA][DEBUG] transcribe: applied OpenAI fallback for empty local result")
                        except Exception:
                            pass
                except Exception:
                    # Ignore fallback failure; leave empty
                    pass
        # mark transcribe step complete
        try:
            job_store.update(job_id, phase="transcribe", status="processing", progress=1.0)
        except Exception:
            pass
        return state

    async def node_diarize(state: PipelineState):
        # Fast path: skip diarization entirely when disabled
        if not settings.enable_diarization:
            try:
                job_store.update(job_id, phase="diarize", status="processing")
            except Exception:
                pass
            stt_segs = _sanitize_segments(state.get("stt_segments", []) or [])
            # Ensure each segment has a default speaker label for downstream formatting
            # Don't add default speaker labels - we don't use them
            state["stt_segments"] = stt_segs
            # Optional regrouping remains available
            try:
                if settings.use_sentence_segmentation:
                    from .steps import regroup_segments_by_sentence as _regroup
                    reg = _regroup(state.get("stt_segments", []) or [])
                    if reg:
                        state["stt_segments"] = reg
            except Exception:
                pass
            
            # Filter hallucinations and normalize speaker labels
            # NOTE: Hallucination filtering DISABLED - was too aggressive and removing valid Portuguese speech
            # Whisper's built-in thresholds + LLM correction are sufficient
            try:
                segs = state.get("stt_segments", []) or []
                # segs = filter_hallucinated_segments(segs)  # DISABLED - removing too much valid content
                segs = normalize_speaker_labels(segs)
                state["stt_segments"] = segs
            except Exception as e:
                try:
                    print(f"[TA][WARN] Segment filtering failed: {e}")
                except Exception:
                    pass
            
            return state
        try:
            job_store.update(job_id, phase="diarize", status="processing")
        except Exception:
            pass
        # Ensure segments are sanitized before diarization
        stt_segs = _sanitize_segments(state.get("stt_segments", []))
        state["stt_segments"] = stt_segs
        # Be defensive if previous steps failed to populate 'normalized'
        norm = state.get("normalized", input_path)
        diar = None
        # Mode selection
        mode = (getattr(settings, "diarization_mode", "auto") or "auto").lower()
        if mode == "light":
            # Fast: cluster ASR segments with speaker embeddings
            try:
                diarized = diarize_over_asr_segments(
                    audio_path=norm,
                    asr_segments=stt_segs,
                    n_speakers=getattr(settings, "diarization_speakers", None) or "auto",
                )
                if diarized:
                    # Count unique speakers
                    unique_speakers = set(seg.get("speaker") for seg in diarized if seg.get("speaker"))
                    print(f"[TA][DIAR] ✓ Success using light mode: {len(unique_speakers)} speaker(s) detected: {sorted(unique_speakers)}")
                    state["stt_segments"] = _sanitize_segments(diarized)
                    # Optional regrouping into sentence-level segments
                    try:
                        if settings.use_sentence_segmentation:
                            from .steps import regroup_segments_by_sentence as _regroup
                            reg = _regroup(state.get("stt_segments", []) or [])
                            if reg:
                                state["stt_segments"] = reg
                    except Exception:
                        pass
                    return state
                else:
                    print(f"[TA][DIAR] ✗ Light mode failed - falling back to full diarization")
            except Exception as e:
                print(f"[TA][DIAR] ✗ Light mode error: {e} - falling back to full diarization")
                diar = None
        # full/auto modes: try external diarizers producing (start,end,speaker)
        diarization_method = None
        try:
            diar = await diarize_pyannote(norm)
            if diar:
                diarization_method = "pyannote"
        except Exception as e:
            print(f"[TA][DIAR] PyAnnote failed: {e}")
            diar = None
        if not diar:
            try:
                from .steps import diarize_resemblyzer
                diar = await diarize_resemblyzer(norm)
                if diar:
                    diarization_method = "resemblyzer"
            except Exception as e:
                print(f"[TA][DIAR] Resemblyzer failed: {e}")
                diar = None
        if not diar:
            try:
                from .steps import diarize_simple
                diar = await diarize_simple(norm)
                if diar:
                    diarization_method = "simple"
            except Exception as e:
                print(f"[TA][DIAR] Simple diarizer failed: {e}")
                diar = None
        
        # Report diarization results
        if diar:
            unique_speakers = set(spk for _, _, spk in diar)
            print(f"[TA][DIAR] ✓ Success using {diarization_method}: {len(unique_speakers)} speaker(s) detected: {sorted(unique_speakers)}")
            print(f"[TA][DIAR]   Total diarization segments: {len(diar)}")
        else:
            print(f"[TA][DIAR] ✗ FAILED - All diarization methods failed. Segments will have NO speaker labels.")
        
        # Attach speaker labels if available by simple overlap majority; otherwise leave None
        if stt_segs:
            segments_with_speakers = 0
            segments_without_speakers = 0
            for seg in stt_segs:
                s = seg.get("start", 0.0) or 0.0
                e = seg.get("end", 0.0) or s
                label = None
                if diar:
                    best_overlap = 0.0
                    for (ds, de, spk) in diar:
                        ov = max(0.0, min(e, de) - max(s, ds))
                        if ov > best_overlap:
                            best_overlap = ov
                            label = spk
                # NO DEFAULT - if diarization failed or no overlap, speaker stays None
                # This makes failures obvious instead of silently masking them
                if label:
                    segments_with_speakers += 1
                    seg["speaker"] = label
                else:
                    segments_without_speakers += 1
                    # Don't set speaker at all - let it be None/missing
                
                # Ensure words inherit or get mapped by overlap
                if seg.get("words"):
                    new_words = []
                    for w in seg["words"]:
                        ws = (w.get("start") if isinstance(w, dict) else None) or s
                        we = (w.get("end") if isinstance(w, dict) else None) or ws
                        wlabel = label  # Inherit segment's label (could be None)
                        if diar:
                            wbest = 0.0
                            for (ds, de, spk) in diar:
                                ov = max(0.0, min(we, de) - max(ws, ds))
                                if ov > wbest:
                                    wbest = ov
                                    wlabel = spk
                        # Only set speaker on word if we have a label
                        if wlabel:
                            nw = {**w, "speaker": wlabel} if isinstance(w, dict) else {"start": ws, "end": we, "text": str(w), "speaker": wlabel}
                        else:
                            nw = w if isinstance(w, dict) else {"start": ws, "end": we, "text": str(w)}
                        new_words.append(nw)
                    seg["words"] = new_words
            
            print(f"[TA][DIAR] Speaker assignment: {segments_with_speakers} segments with speakers, {segments_without_speakers} without")

            # Refinement step: split segments for more precise speaker attribution
            # - If words exist with speaker labels: split segments into contiguous word-speaker runs
            # - Else, if diarization exists: split segment by diarization overlaps and distribute text proportionally
            if diar:
                refined: List[Dict[str, Any]] = []
                for seg in stt_segs:
                    s = float(seg.get("start", 0.0) or 0.0)
                    e = float(seg.get("end", 0.0) or s)
                    txt = seg.get("text", "")
                    words = seg.get("words")
                    if words:
                        cur_group: List[Dict[str, Any]] = []
                        cur_spk = None
                        for w in words:
                            spk_w = w.get("speaker") or seg.get("speaker") or "SPEAKER_0"
                            if cur_spk is None:
                                cur_spk = spk_w
                            if spk_w != cur_spk:
                                # flush
                                if cur_group:
                                    start_g = float(cur_group[0].get("start", s) or s)
                                    end_g = float(cur_group[-1].get("end", start_g) or start_g)
                                    text_g = " ".join([(ww.get("text") or "").strip() for ww in cur_group]).strip()
                                    refined.append({"start": start_g, "end": end_g, "text": text_g, "speaker": cur_spk, "words": cur_group})
                                cur_group = []
                                cur_spk = spk_w
                            cur_group.append(w)
                        if cur_group:
                            start_g = float(cur_group[0].get("start", s) or s)
                            end_g = float(cur_group[-1].get("end", start_g) or start_g)
                            text_g = " ".join([(ww.get("text") or "").strip() for ww in cur_group]).strip()
                            refined.append({"start": start_g, "end": end_g, "text": text_g, "speaker": cur_spk, "words": cur_group})
                    else:
                        # No words: split by diarization overlaps
                        overlaps: List[Tuple[float, float, str]] = []
                        for (ds, de, spk) in diar:
                            os_ = max(s, float(ds))
                            oe_ = min(e, float(de))
                            if (oe_ - os_) > 0.05:
                                overlaps.append((os_, oe_, spk))
                        if not overlaps:
                            refined.append(seg)
                        else:
                            overlaps.sort(key=lambda t: t[0])
                            total_dur = sum(oe_ - os_ for (os_, oe_, _) in overlaps)
                            total_chars = len(txt or "")
                            if total_dur <= 0 or total_chars <= 0:
                                first = True
                                for (os_, oe_, spk_) in overlaps:
                                    refined.append({"start": os_, "end": oe_, "text": (txt if first else ""), "speaker": spk_, "words": None})
                                    first = False
                            else:
                                # Allocate characters proportionally to duration
                                lens: List[int] = []
                                accum = 0
                                for (os_, oe_, _) in overlaps:
                                    l = int(round(total_chars * ((oe_ - os_) / total_dur)))
                                    lens.append(l)
                                diff = total_chars - sum(lens)
                                if diff != 0:
                                    lens[-1] += diff
                                cur = 0
                                for (os_, oe_, spk_), ln in zip(overlaps, lens):
                                    slice_text = (txt or "")[cur:cur+ln]
                                    refined.append({"start": os_, "end": oe_, "text": slice_text.strip(), "speaker": spk_, "words": None})
                                    cur += ln
                if refined:
                    state["stt_segments"] = _sanitize_segments(refined)
        # Optional regrouping into sentence-level segments to avoid tiny fragments
        try:
            if settings.use_sentence_segmentation:
                from .steps import regroup_segments_by_sentence as _regroup
                reg = _regroup(state.get("stt_segments", []) or [])
                if reg:
                    state["stt_segments"] = reg
        except Exception:
            pass
        
        # Filter hallucinations and normalize speaker labels
        # NOTE: Hallucination filtering DISABLED - was too aggressive and removing valid Portuguese speech
        # Whisper's built-in thresholds + LLM correction are sufficient
        try:
            segs = state.get("stt_segments", []) or []
            # segs = filter_hallucinated_segments(segs)  # DISABLED - removing too much valid content
            segs = normalize_speaker_labels(segs)
            state["stt_segments"] = segs
        except Exception as e:
            try:
                print(f"[TA][WARN] Segment filtering failed: {e}")
            except Exception:
                pass
        
        return state

    async def node_merge_persist(state: PipelineState):
        try:
            job_store.update(job_id, phase="persist", status="processing")
        except Exception:
            pass
        
        # ===== LLM CORRECTION STEP (if enabled) =====
        # Apply intelligent error correction BEFORE persisting
        if settings.enable_llm_correction:
            if not settings.openai_api_key:
                print("[TA][LLM_CORRECTION] ⚠️ WARNING: LLM correction is enabled but OPENAI_API_KEY is not set!")
                print("[TA][LLM_CORRECTION] ⚠️ Hallucinations will NOT be removed. Please set OPENAI_API_KEY in .env")
            else:
                try:
                    from .transcription_correction import correct_transcription
                    
                    segs_in = state.get("stt_segments", []) or []
                    if segs_in:
                        # Extract text from segments
                        pieces: List[str] = []
                        for s in segs_in:
                            if isinstance(s, dict):
                                pieces.append(s.get("text", ""))
                            else:
                                pieces.append(getattr(s, "text", ""))
                        
                        raw_text = " ".join(pieces).strip()
                        
                        if raw_text:
                            print(f"[TA][LLM_CORRECTION] Starting correction (mode={settings.llm_correction_mode}, passes={settings.llm_correction_passes})")
                            
                            # Apply correction
                            corrected_text = correct_transcription(
                                raw_text,
                                api_key=settings.openai_api_key,
                                base_url=settings.openai_base_url,
                                model=settings.llm_correction_model,
                                mode=settings.llm_correction_mode,
                                num_passes=settings.llm_correction_passes,
                            )
                            
                            # Update segment texts with corrected version
                            # Simple approach: split corrected text and redistribute to segments proportionally
                            if corrected_text and corrected_text != raw_text:
                                # For now, just update the first segment with full corrected text
                                # (More sophisticated: split by sentence boundaries and redistribute)
                                print(f"[TA][LLM_CORRECTION] ✓ Correction applied (chars: {len(raw_text)} → {len(corrected_text)})")
                                
                                # Replace segment texts with corrected version
                                # Strategy: Keep timing info, update text
                                corrected_segs = []
                                for s in segs_in:
                                    if isinstance(s, dict):
                                        corrected_segs.append({**s, "text": corrected_text if s == segs_in[0] else ""})
                                    else:
                                        # Handle object-like segments
                                        corrected_segs.append(s)
                                
                                # Better: distribute corrected text across segments by word count
                                # This preserves timing while fixing errors
                                raw_words = raw_text.split()
                                corrected_words = corrected_text.split()
                                
                                if len(raw_words) > 0 and len(corrected_words) > 0:
                                    # Redistribute corrected words proportionally across all segments
                                    # This handles cases where LLM removed hallucinations (fewer words)
                                    corrected_segs = []
                                    total_raw_words = len(raw_words)
                                    total_corrected_words = len(corrected_words)
                                    word_idx = 0
                                    
                                    for s in segs_in:
                                        if isinstance(s, dict):
                                            seg_text = s.get("text", "")
                                        else:
                                            seg_text = getattr(s, "text", "")
                                        
                                        seg_word_count = len(seg_text.split())
                                        if seg_word_count == 0:
                                            continue
                                        
                                        # Calculate proportional word count for this segment
                                        # If LLM removed words, scale down proportionally
                                        proportion = seg_word_count / total_raw_words
                                        target_word_count = max(1, int(proportion * total_corrected_words))
                                        
                                        # Take words from corrected text (or remaining words if near end)
                                        remaining_words = len(corrected_words) - word_idx
                                        actual_word_count = min(target_word_count, remaining_words)
                                        
                                        if actual_word_count > 0:
                                            seg_corrected_words = corrected_words[word_idx:word_idx + actual_word_count]
                                            seg_corrected_text = " ".join(seg_corrected_words)
                                            
                                            if seg_corrected_text.strip():
                                                if isinstance(s, dict):
                                                    corrected_segs.append({**s, "text": seg_corrected_text})
                                                else:
                                                    corrected_segs.append(s)
                                            
                                            word_idx += actual_word_count
                                    
                                    state["stt_segments"] = corrected_segs
                            else:
                                print(f"[TA][LLM_CORRECTION] No changes needed")
                except Exception as e:
                    print(f"[TA][LLM_CORRECTION] ✗ Correction failed: {e}")
                    # Continue with uncorrected text
        
        # ===== FINAL SANITIZE AND PERSIST =====
        # Final sanitize before persisting
        segs_in = state.get("stt_segments", []) or []
        try:
            print("[TA][DEBUG] persist: segs_in type/len", type(segs_in).__name__, len(segs_in) if hasattr(segs_in, "__len__") else "n/a")
            if isinstance(segs_in, list) and segs_in:
                print("[TA][DEBUG] persist: segs_in sample", repr(segs_in[:2]))
        except Exception:
            pass
        segs = _sanitize_segments(segs_in)
        try:
            print("[TA][DEBUG] persist: segs sanitized len", len(segs))
            if segs:
                print("[TA][DEBUG] persist: segs sanitized sample", repr(segs[:2]))
        except Exception:
            pass
        # Safe text join (supports dict-like and object-like)
        pieces: List[str] = []
        for s in segs:
            if isinstance(s, dict):
                pieces.append(s.get("text", ""))
            else:
                pieces.append(getattr(s, "text", ""))
        text = " ".join(pieces).strip()
        transcript_id = FileStore.new_transcript_id()
        seg_models = []
        for s in segs:
            if isinstance(s, dict):
                words = s.get("words") or None
                start_v = s.get("start")
                end_v = s.get("end")
                text_v = s.get("text", "")
                speaker_v = s.get("speaker")
            else:
                words = getattr(s, "words", None)
                start_v = getattr(s, "start", 0.0)
                end_v = getattr(s, "end", 0.0)
                text_v = getattr(s, "text", "")
                speaker_v = getattr(s, "speaker", None)
            try:
                start_f = float(start_v or 0.0)
            except Exception:
                start_f = 0.0
            try:
                end_f = float(end_v or 0.0)
            except Exception:
                end_f = 0.0
            seg_models.append(Segment(
                start=start_f,
                end=end_f,
                text=text_v or "",
                speaker=speaker_v,
                words=None if words is None else words
            ))

        payload = Transcript(
            transcript_id=transcript_id,
            org_id=org_id,
            meeting_id=meeting_ref,
            mode=mode,
            profile=profile,
            source="upload_audio",
            segments=seg_models,
            text=text
        ).model_dump()
        try:
            print("[TA][DEBUG] persist: built payload type", type(payload).__name__)
        except Exception:
            pass
        if not isinstance(payload, dict):
            # Fallback for unexpected pydantic versioning edge cases
            try:
                payload = payload.dict()
            except Exception:
                payload = {
                    "transcript_id": transcript_id,
                    "org_id": org_id,
                    "meeting_id": meeting_ref,
                    "mode": mode,
                    "profile": profile,
                    "source": "upload_audio",
                    "segments": [s.model_dump() if hasattr(s, "model_dump") else (s.dict() if hasattr(s, "dict") else s) for s in seg_models],
                    "text": text,
                }
        try:
            print("[TA][DEBUG] persist: saving transcript", {"id": transcript_id, "segments": len(seg_models), "text_len": len(text)})
        except Exception:
            pass
        
        # Save to storage (MongoDB or FileStore based on config)
        # CRITICAL: Save BEFORE exposing transcript_id to avoid race condition
        try:
            if USE_MONGODB:
                await MongoDBStore.save_transcript(transcript_id, payload, org_id, meeting_ref)
            else:
                FileStore.save_transcript_json(transcript_id, payload)
            
            # Only set state values after successful save
            state["transcript_id"] = transcript_id
            state["text"] = text
            
            # ONLY expose transcript_id AFTER successful save to prevent race condition
            # where API callers see the ID before the transcript is persisted
            try:
                job_store.update(job_id, transcript_id=transcript_id)
                print(f"[TA][DEBUG] persist: ✓ Transcript {transcript_id} saved and exposed to job store")
            except Exception as e:
                print(f"[TA][WARN] persist: Failed to update job store: {e}")
        except Exception as save_error:
            # If save fails, don't expose transcript_id and propagate error
            print(f"[TA][ERROR] persist: Failed to save transcript {transcript_id}: {save_error}")
            raise RuntimeError(f"transcript_save_failed: {save_error}")
        
        return state

    # Sequential orchestration (no LangGraph). Each node mutates state and returns it.
    state: PipelineState = {}
    # Normalize
    state = await node_normalize(state)
    # Segment
    state = await node_segment(state)
    # Transcribe
    state = await node_transcribe(state)
    # Diarize
    state = await node_diarize(state)
    # Optional selective word timestamp refinement (only for long segments)
    try:
        level = (getattr(settings, "timestamp_level", "segment") or "segment").lower()
        min_secs = float(getattr(settings, "word_ts_min_seconds", 12.0) or 12.0)
        if level == "word" or (level == "auto" and any((s.get("end", 0)-s.get("start", 0)) >= min_secs for s in (state.get("stt_segments") or []))):
            from .steps import refine_word_timestamps_by_cut as _refine
            refined = await _refine(state.get("normalized", input_path), state.get("stt_segments", []) or [], min_seconds=min_secs)
            if refined:
                state["stt_segments"] = refined
    except Exception:
        pass
    # Persist
    state = await node_merge_persist(state)
    try:
        print("[TA][DEBUG] run_pipeline: sequential final keys", list(state.keys()))
    except Exception:
        pass
    return {"transcript_id": state.get("transcript_id"), "text": state.get("text", "")}
