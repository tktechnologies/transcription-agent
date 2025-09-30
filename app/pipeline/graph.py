import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ..storage.store import FileStore
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

class PipelineState(dict):
    pass

async def run_pipeline(job_id: str, input_path: str, org_id: str | None, meeting_ref: str | None, mode: str | None, profile: str | None) -> Dict[str, Any]:
    work = Path("artifacts") / "work" / job_id
    work.mkdir(parents=True, exist_ok=True)
    # Shortcut: if input is a .txt, persist as transcript without audio processing
    if str(input_path).lower().endswith('.txt'):
        text = Path(input_path).read_text(encoding='utf-8', errors='ignore')
        transcript_id = FileStore.new_transcript_id()
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
            # No silence segmentation; use fixed-size chunking over full duration
            try:
                from .steps import probe_duration_seconds
                dur = probe_duration_seconds(norm)
            except Exception:
                dur = None
            if dur and dur > 0:
                segs = []
                start = 0.0
                while start < dur:
                    end = min(dur, start + settings.chunk_seconds)
                    segs.append((max(0.0, start - settings.overlap_seconds), end))
                    start = end
            else:
                segs = [(0.0, 0.0)]  # unbounded marker = full file downstream
        # Cap segments to chunk_seconds and add overlap; clamp to duration if known
        final: List[Tuple[float, float]] = []
        if segs == [(0.0, 0.0)]:
            final = [(0.0, 0.0)]
        else:
            for (s, e) in segs:
                dur = e - s
                if dur <= settings.chunk_seconds:
                    s2 = max(0.0, s - settings.overlap_seconds)
                    e2 = e
                    if total_dur and total_dur > 0:
                        e2 = min(e2, total_dur)
                    final.append((s2, e2))
                else:
                    # slice into multiple chunks with overlap
                    start = s
                    while start < e:
                        end = min(e, start + settings.chunk_seconds)
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
                # create fixed chunks now
                fixed: List[Tuple[float, float]] = []
                start = 0.0
                # If duration is unknown but forced_chunk is True, approximate a minimal split into two parts
                if (not dur or dur <= 0) and forced_chunk:
                    dur = float(settings.chunk_seconds * 2)
                while start < (dur or 0.0):
                    end = min(dur, start + settings.chunk_seconds)
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
                        try:
                            print(f"[TA][ERROR] transcribe: faster_whisper_failed idx={idx} err={e}")
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
            for seg in stt_segs:
                if not seg.get("speaker"):
                    seg["speaker"] = "SPEAKER_0"
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
            except Exception:
                diar = None
        # full/auto modes: try external diarizers producing (start,end,speaker)
        try:
            diar = await diarize_pyannote(norm)
        except Exception:
            diar = None
        if not diar:
            try:
                from .steps import diarize_resemblyzer
                diar = await diarize_resemblyzer(norm)
            except Exception:
                diar = None
        if not diar:
            try:
                from .steps import diarize_simple
                diar = await diarize_simple(norm)
            except Exception:
                diar = None
        # Attach speaker labels if available by simple overlap majority; otherwise default
        if stt_segs:
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
                # Default if diarization unavailable/no overlap
                if not label:
                    label = "SPEAKER_0"
                seg["speaker"] = label
                # Ensure words inherit or get mapped by overlap
                if seg.get("words"):
                    new_words = []
                    for w in seg["words"]:
                        ws = (w.get("start") if isinstance(w, dict) else None) or s
                        we = (w.get("end") if isinstance(w, dict) else None) or ws
                        wlabel = label
                        if diar:
                            wbest = 0.0
                            for (ds, de, spk) in diar:
                                ov = max(0.0, min(we, de) - max(ws, ds))
                                if ov > wbest:
                                    wbest = ov
                                    wlabel = spk
                        # set/inherit speaker on word
                        nw = {**w, "speaker": wlabel} if isinstance(w, dict) else {"start": ws, "end": we, "text": str(w), "speaker": wlabel}
                        new_words.append(nw)
                    seg["words"] = new_words

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
        return state

    async def node_merge_persist(state: PipelineState):
        try:
            job_store.update(job_id, phase="persist", status="processing")
        except Exception:
            pass
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
        FileStore.save_transcript_json(transcript_id, payload)
        state["transcript_id"] = transcript_id
        state["text"] = text
        # Expose transcript_id immediately for job pollers
        try:
            job_store.update(job_id, transcript_id=transcript_id)
        except Exception:
            pass
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
