import asyncio
import os
import sys
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from ..config import settings

FFMPEG = settings.ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = settings.ffprobe_path or shutil.which("ffprobe") or "ffprobe"

# Initialize threading lock at module level to avoid race conditions
_fw_model_lock = threading.Lock()

# Run a subprocess in a background thread (Windows-friendly for asyncio)
async def _run_proc(args: List[str]) -> tuple[int, bytes, bytes]:
    def _call():
        try:
            res = subprocess.run(args, capture_output=True)
            return res.returncode, res.stdout, res.stderr
        except Exception as e:
            # Simulate a process error with rc=1 and error text
            return 1, b"", str(e).encode()
    return await asyncio.to_thread(_call)

async def normalize_audio(input_path: str, work_dir: Path) -> str:
    """Transcode to 16kHz mono wav for predictable STT behavior."""
    out = work_dir / "normalized.wav"
    if not Path(input_path).exists():
        raise RuntimeError(f"input_not_found: {input_path}")
    ff = settings.ffmpeg_path or shutil.which("ffmpeg") or FFMPEG
    try:
        print("[TA][DEBUG] normalize: ffmpeg=", ff, "input=", str(input_path), "out=", str(out))
    except Exception:
        pass
    rc, stdout, stderr = await _run_proc([str(ff), "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out)])
    if rc != 0:
        err = (stderr or b"").decode(errors="ignore").strip()
        raise RuntimeError(f"ffmpeg_normalize_failed: bin={ff} err={err[:500]}")
    try:
        sz = Path(out).stat().st_size if Path(out).exists() else 0
        print("[TA][DEBUG] normalize: out size bytes=", sz)
    except Exception:
        pass
    return str(out)

# -------- Segmentation ---------
def detect_silence_segments(input_wav: str, min_silence_len_ms: int = 700, silence_thresh_db: int = -38, keep_silence_ms: int = 200) -> List[Tuple[float, float]]:
    """Use pydub to split on silence. Returns list of (start_sec, end_sec)."""
    try:
        from pydub import AudioSegment, silence
    except Exception as e:
        # Fallback: single segment
        return [(0.0, 0.0)]  # (0.0, 0.0) means unbounded; handle upstream

    audio = AudioSegment.from_wav(input_wav)
    chunks = silence.split_on_silence(audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_db, keep_silence=keep_silence_ms)
    segments: List[Tuple[float, float]] = []
    cursor = 0
    for ch in chunks:
        start = cursor / 1000.0
        end = (cursor + len(ch)) / 1000.0
        segments.append((start, end))
        cursor += len(ch)
    if not segments:
        # Whole file
        duration = len(audio) / 1000.0
        return [(0.0, duration)]
    return segments

def probe_duration_seconds(input_media: str) -> Optional[float]:
    """Return media duration in seconds using ffprobe, or None if unavailable."""
    ffprobe = settings.ffprobe_path or shutil.which("ffprobe") or FFPROBE
    try:
        res = subprocess.run([str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_media)],
                             shell=False, capture_output=True, text=True)
        if res.returncode != 0:
            return None
        out = (res.stdout or "").strip()
        return float(out)
    except Exception:
        return None

async def cut_segment(input_wav: str, start: float, end: float, out_path: Path) -> str:
    start_s = max(0.0, start)
    dur_s = max(0.0, end - start)
    if not Path(input_wav).exists():
        raise RuntimeError(f"input_not_found: {input_wav}")
    ff = settings.ffmpeg_path or shutil.which("ffmpeg") or FFMPEG
    try:
        print(f"[TA][DEBUG] cut: ffmpeg={ff} in={input_wav} start={start_s:.3f} dur={dur_s:.3f} out={out_path}")
    except Exception:
        pass
    rc, stdout, stderr = await _run_proc([str(ff), "-y", "-i", str(input_wav), "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out_path)])
    if rc != 0:
        err = (stderr or b"").decode(errors="ignore").strip()
        out = (stdout or b"").decode(errors="ignore").strip()
        raise RuntimeError(
            f"ffmpeg_cut_failed: bin={ff} rc={rc} start={start_s:.3f} end={end:.3f} dur={dur_s:.3f} | stderr={err[:300]} | stdout={out[:120]}"
        )
    try:
        sz = Path(out_path).stat().st_size if Path(out_path).exists() else 0
        from .steps import probe_duration_seconds as _probe
        dur = _probe(str(out_path))
        print("[TA][DEBUG] cut: out size=", sz, "bytes, probed dur=", dur)
    except Exception:
        pass
    return str(out_path)

# -------- STT Providers ---------
async def transcribe_openai(file_path: str) -> Dict[str, Any]:
    """OpenAI Whisper API call (async) with retry/backoff, honoring base URL and org.
    Returns dict with keys: text (str), segments (optional list of {start,end,text}).
    """
    from tenacity import retry, wait_exponential_jitter, stop_after_attempt
    @retry(wait=wait_exponential_jitter(initial=0.5, max=8), stop=stop_after_attempt(5), reraise=True)
    async def _call() -> Dict[str, Any]:
        import httpx
        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError("missing_openai_key")
        base = settings.openai_base_url.rstrip("/")
        # If using Azure OpenAI (deployment provided), construct the Azure-style URL
        if settings.openai_deployment:
            # Example base: https://<resource>.openai.azure.com/openai
            # Full: {base}/deployments/{deployment}/audio/transcriptions?api-version=...
            url = f"{base}/deployments/{settings.openai_deployment}/audio/transcriptions?api-version={settings.openai_api_version}"
        else:
            url = f"{base}/audio/transcriptions"
        # Build headers depending on provider
        headers = {}
        if settings.openai_deployment:
            # Azure uses 'api-key' header
            headers["api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
            if settings.openai_organization:
                headers["OpenAI-Organization"] = settings.openai_organization
        files = {"file": (Path(file_path).name, open(file_path, "rb"), "audio/wav")}
        # Payload differs between providers
        if settings.openai_deployment:
            data = {}  # Azure: deployment implies model, response is JSON by default (text only)
        else:
            # Use verbose_json to get segments with timestamps when using OpenAI
            data = {"model": settings.openai_transcribe_model, "response_format": "verbose_json"}
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(url, headers=headers, data=data, files=files)
            if r.status_code >= 400:
                body = (r.text or "").strip()
                raise httpx.HTTPStatusError(f"OpenAI transcribe failed {r.status_code}: {body[:300]}", request=r.request, response=r)
            j = r.json()
            # Normalize output
            if settings.openai_deployment:
                # Azure returns { text: "..." }
                return {"text": j.get("text", ""), "segments": None}
            else:
                # OpenAI returns verbose { text: "...", segments: [{start,end,text,...}, ...] }
                segs = []
                for s in (j.get("segments") or []):
                    try:
                        segs.append({"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": s.get("text", "").strip()})
                    except Exception:
                        continue
                return {"text": j.get("text", ""), "segments": segs or None}
    return await _call()

_fw_model = None
_fw_model_lock = threading.Lock()

def get_faster_whisper_model():
    global _fw_model
    if _fw_model is None:
        with _fw_model_lock:  # Lock ONLY for initialization
            # Double-check pattern: another thread might have loaded it while we waited
            if _fw_model is None:
                try:
                    # Avoid Windows privilege issues with hardlinks/symlinks in HF cache
                    if not os.getenv("HF_HUB_DISABLE_SYMLINKS"):
                        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
                    if not os.getenv("HF_HUB_DISABLE_HARD_LINKS"):
                        os.environ["HF_HUB_DISABLE_HARD_LINKS"] = "1"
                    # Route Hugging Face cache into our storage directory for clearer permissions
                    cache_dir = Path(settings.storage_dir) / "hf-cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    if not os.getenv("HUGGINGFACE_HUB_CACHE"):
                        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)

                    from faster_whisper import WhisperModel
                    device = None if settings.stt_device == "auto" else settings.stt_device
                    try:
                        print("[TA] faster-whisper cache", {
                            "HF_HOME": os.getenv("HF_HOME"),
                            "HUGGINGFACE_HUB_CACHE": os.getenv("HUGGINGFACE_HUB_CACHE"),
                            "HF_HUB_DISABLE_SYMLINKS": os.getenv("HF_HUB_DISABLE_SYMLINKS"),
                            "HF_HUB_DISABLE_HARD_LINKS": os.getenv("HF_HUB_DISABLE_HARD_LINKS"),
                            "download_root": str(cache_dir),
                            "device": device or "auto",
                            "model": settings.stt_model,
                        })
                    except Exception:
                        pass
                    # Map auto/invalids to library defaults
                    compute_type = settings.stt_compute_type
                    if not compute_type or compute_type == "auto":
                        compute_type = "default"
                    cpu_threads = settings.stt_cpu_threads if settings.stt_cpu_threads and settings.stt_cpu_threads > 0 else 0

                    _fw_model = WhisperModel(
                        settings.stt_model,
                        device=device or "auto",
                        compute_type=compute_type,
                        cpu_threads=cpu_threads,
                        num_workers=max(1, settings.stt_num_workers),
                        download_root=str(cache_dir),
                    )
                except Exception as e:
                    raise RuntimeError(f"faster_whisper_unavailable: {e}")
    return _fw_model

async def transcribe_faster_whisper(file_path: str, *, word_timestamps: Optional[bool] = None, beam_size: Optional[int] = None) -> Dict[str, Any]:
    """Return segments with word timestamps when available."""
    def _run():
        # Get model instance (shared, but transcribe() call is thread-safe in faster-whisper >= 0.9.0)
        # The tqdm issue is avoided by setting log_progress=False
        model = get_faster_whisper_model()
        wt = settings.stt_word_timestamps if word_timestamps is None else bool(word_timestamps)
        bs = max(1, settings.stt_beam_size if beam_size is None else int(beam_size))
        # Pass optional language and vad_filter when available
        kwargs = {
            "word_timestamps": wt,
            "beam_size": bs,
            "temperature": getattr(settings, "stt_temperature", 0.0),
            "condition_on_previous_text": getattr(settings, "stt_condition_prev_text", False),
            "no_speech_threshold": getattr(settings, "stt_no_speech_threshold", 0.65),
            "compression_ratio_threshold": getattr(settings, "stt_compression_ratio_threshold", 2.6),
            "log_prob_threshold": getattr(settings, "stt_logprob_threshold", -1.0),
            "log_progress": False,  # Disable tqdm progress bar to avoid threading conflicts
        }
        if settings.stt_language:
            kwargs["language"] = settings.stt_language
        try:
            # Some versions accept vad_filter
            if settings.stt_vad_filter:
                kwargs["vad_filter"] = True
        except Exception:
            pass
        segments, info = model.transcribe(file_path, **kwargs)
        out = []
        for seg in segments:
            words = []
            if wt and seg.words:
                for w in seg.words:
                    words.append({"start": w.start, "end": w.end, "text": w.word})
            out.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": words or None
            })
        try:
            print("[TA][DEBUG] faster-whisper: lang=", getattr(info, 'language', None), "segments=", len(out), "file=", file_path)
        except Exception:
            pass
        return {"language": info.language, "segments": out}
    
    # Run in thread pool WITHOUT lock - faster-whisper model.transcribe() is thread-safe
    # The log_progress=False prevents tqdm threading issues
    return await asyncio.to_thread(_run)

async def refine_word_timestamps_by_cut(
    normalized_wav: str,
    segments: List[Dict[str, Any]],
    *,
    min_seconds: float = 12.0,
) -> List[Dict[str, Any]]:
    """Re-run faster-whisper with word_timestamps=True for long segments only.
    - Cuts the audio for each long segment to a temp wav
    - Transcribes with word timestamps and offsets times back
    - Preserves speaker labels
    """
    if not segments:
        return []
    out: List[Dict[str, Any]] = []
    work_dir = Path(normalized_wav).parent / "refine"
    work_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(segments):
        try:
            st = float(s.get("start") or 0.0)
            en = float(s.get("end") or st)
        except Exception:
            out.append(s)
            continue
        dur = max(0.0, en - st)
        if dur < max(0.0, float(min_seconds)):
            out.append(s)
            continue
        cut_path = work_dir / f"seg_{i:04d}.wav"
        try:
            await cut_segment(normalized_wav, st, en, cut_path)
            fw = await transcribe_faster_whisper(str(cut_path), word_timestamps=True, beam_size=1)
        except Exception as e:
            print(f"[TA][WARN] refine word-ts failed for seg {i}: {e}", file=sys.stderr)
            out.append(s)
            continue
        # Convert and offset
        segs = fw.get("segments") or []
        if not segs:
            out.append(s)
            continue
        for r in segs:
            r2 = {
                "start": float(r.get("start") or 0.0) + st,
                "end": float(r.get("end") or 0.0) + st,
                "text": r.get("text", ""),
                "words": None,
                "speaker": s.get("speaker"),
            }
            wlist = []
            for w in (r.get("words") or []):
                wlist.append({
                    "start": float(w.get("start") or 0.0) + st,
                    "end": float(w.get("end") or 0.0) + st,
                    "text": w.get("text", ""),
                    "speaker": s.get("speaker"),
                })
            r2["words"] = wlist or None
            out.append(r2)
    return out

# -------- Optional Alignment / Diarization ---------
async def align_with_whisperx(normalized_wav: str, language: Optional[str], segments_text: str) -> Optional[Dict[str, Any]]:
    """Optional: try to align words using whisperx if installed. Returns None if unavailable."""
    try:
        import whisperx  # type: ignore
    except Exception:
        return None
    # Placeholder: a full whisperx integration would run ASR + alignment. Here we skip for MVP.
    return None

async def diarize_pyannote(normalized_wav: str) -> Optional[List[Tuple[float, float, str]]]:
    """Optional: returns list of (start, end, speaker_label). Returns None if unavailable."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return None
    try:
        from pyannote.audio import Pipeline  # type: ignore
        try:
            print("[TA][DEBUG] diarize: initializing pyannote pipeline")
        except Exception:
            pass
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=hf_token)
        diar = await asyncio.to_thread(pipeline, normalized_wav)
        segments = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        try:
            print("[TA][DEBUG] diarize: segments=", len(segments))
        except Exception:
            pass
        return segments
    except Exception:
        return None

# -------- Token-free Diarization Fallback (Resemblyzer + Clustering) ---------
async def diarize_resemblyzer(normalized_wav: str) -> Optional[List[Tuple[float, float, str]]]:
    """Diarize using speaker embeddings clustered over sliding windows.
    Returns list of (start, end, speaker_label). No HF token needed.
    """
    try:
        import numpy as np  # type: ignore
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
    except Exception:
        return None

    def _run() -> Optional[List[Tuple[float, float, str]]]:
        try:
            wav = preprocess_wav(normalized_wav)
            if wav is None or len(wav) < 16000:
                return None
            encoder = VoiceEncoder()
            # Get partial embeddings with their time slices
            _, partial_embeds, partial_slices = encoder.embed_utterance(wav, return_partials=True)
            if partial_embeds is None or len(partial_embeds) < 4:
                return None
            embeds = np.stack(partial_embeds)  # [N, D]
            # Try K=2 using a tiny k-means fallback to avoid heavy deps
            try:
                import numpy as _np  # type: ignore
                # Init with two farthest points
                D = embeds @ embeds.T
                i0 = 0
                i1 = int(_np.argmax(D[i0]))
                c0 = embeds[i0]
                c1 = embeds[i1]
                for _ in range(20):
                    d0 = _np.sum((embeds - c0) ** 2, axis=1)
                    d1 = _np.sum((embeds - c1) ** 2, axis=1)
                    lab = (d1 < d0).astype(_np.int32)
                    if _np.all(lab == 0) or _np.all(lab == 1):
                        break
                    if _np.any(lab == 0):
                        c0 = embeds[lab == 0].mean(axis=0)
                    if _np.any(lab == 1):
                        c1 = embeds[lab == 1].mean(axis=0)
                best_labels = lab
                best_k = 2
            except Exception:
                return None
            # Merge contiguous windows with same label into turns
            turns: List[Tuple[float, float, str]] = []
            sr = 16000.0  # preprocess_wav resamples to 16 kHz by default
            cur_lab = None
            cur_start = None
            cur_end = None
            def slice_to_time(slc) -> Tuple[float, float]:
                # partial_slices are objects with .start and .stop in samples
                st = getattr(slc, 'start', 0) / sr
                en = getattr(slc, 'stop', st) / sr
                return float(st), float(en)
            for lab, slc in zip(best_labels, partial_slices):
                st, en = slice_to_time(slc)
                if cur_lab is None:
                    cur_lab, cur_start, cur_end = int(lab), st, en
                elif int(lab) == cur_lab and st <= (cur_end + 0.2):
                    # extend current turn (allow small gaps)
                    cur_end = max(cur_end, en)
                else:
                    turns.append((cur_start, cur_end, f"SPEAKER_{cur_lab}"))
                    cur_lab, cur_start, cur_end = int(lab), st, en
            if cur_lab is not None and cur_start is not None and cur_end is not None:
                turns.append((cur_start, cur_end, f"SPEAKER_{cur_lab}"))
            # Merge very short adjacent turns with same label
            merged: List[Tuple[float, float, str]] = []
            for t in turns:
                if merged and merged[-1][2] == t[2] and (t[0] - merged[-1][1]) <= 0.3:
                    a = merged[-1]
                    merged[-1] = (a[0], t[1], a[2])
                else:
                    merged.append(t)
            return merged or None
        except Exception:
            return None

    return await asyncio.to_thread(_run)

# -------- Lightweight token-free Diarization (NumPy + optional scikit-learn) ---------
async def diarize_simple(normalized_wav: str) -> Optional[List[Tuple[float, float, str]]]:
    """Heuristic diarization using basic spectral features and clustering.
    - No internet/token required
    - Pure Python/NumPy; uses scikit-learn if available, otherwise a tiny 2-means fallback
    Returns list of (start, end, speaker_label)
    """
    try:
        import wave
        import numpy as np  # type: ignore
    except Exception:
        return None

    def _run() -> Optional[List[Tuple[float, float, str]]]:
        try:
            with wave.open(normalized_wav, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                fr = wf.getframerate()
                nframes = wf.getnframes()
                if fr <= 0 or nframes <= 0:
                    return None
                raw = wf.readframes(nframes)
            # Expect 16kHz mono s16le from normalize step; be defensive
            if n_channels != 1 or sampwidth not in (2, 3, 4):
                # Unsupported format
                return None
            if sampwidth == 2:
                sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                sig /= 32768.0
            elif sampwidth == 3:
                b = np.frombuffer(raw, dtype=np.uint8).astype(np.uint32)
                # Convert 24-bit little endian to int32
                b = b.reshape(-1, 3)
                v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)).astype(np.int32)
                v = (v << 8) >> 8  # sign extend 24->32
                sig = v.astype(np.float32) / (2**23)
            else:  # 32-bit
                sig = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / (2**31)

            sr = float(fr)
            win_s = int(sr * 1.0)  # 1.0s windows
            hop_s = int(sr * 0.5)  # 0.5s hop
            if win_s <= 0 or hop_s <= 0 or len(sig) < win_s:
                return None
            # Precompute frequency bins for spectral centroid
            n_fft = 1024
            win = np.hanning(n_fft).astype(np.float32)
            freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
            feats = []
            times = []
            for start in range(0, len(sig) - win_s + 1, hop_s):
                end = start + win_s
                x = sig[start:end]
                if len(x) < win_s:
                    break
                # Energy (log)
                eng = np.log10(1e-8 + float(np.mean(x * x)))
                # Zero-crossing rate
                zc = float(np.mean(np.abs(np.diff(np.sign(np.clip(x, -1, 1)))) > 0))
                # Spectral centroid over short frames and average across window
                # Frame the window into smaller overlapping slices for centroid stability
                step = n_fft // 2
                cents = []
                for i in range(0, len(x) - n_fft + 1, step):
                    xx = x[i:i+n_fft]
                    X = np.fft.rfft(xx * win)
                    mag = np.abs(X) + 1e-10
                    cen = float(np.sum(freqs * mag) / np.sum(mag))
                    cents.append(cen)
                cen = float(np.mean(cents)) if cents else 0.0
                # Normalize centroid to [0,1] by Nyquist
                cen_n = cen / (sr / 2.0)
                feats.append([eng, cen_n, zc])
                times.append((start / sr, end / sr))

            if not feats:
                return None
            F = np.asarray(feats, dtype=np.float32)
            # Standardize
            mu = F.mean(axis=0, keepdims=True)
            sigma = F.std(axis=0, keepdims=True) + 1e-6
            Z = (F - mu) / sigma

            labels = None
            k_chosen = None
            # Try scikit first
            try:
                from sklearn.cluster import AgglomerativeClustering  # type: ignore
                from sklearn.metrics import silhouette_score  # type: ignore
                best_sc = -1.0
                best_lab = None
                best_k = None
                for k in (2, 3):
                    cl = AgglomerativeClustering(n_clusters=k)
                    lab = cl.fit_predict(Z)
                    if len(set(lab)) < 2:
                        continue
                    sc = silhouette_score(Z, lab)
                    if sc > best_sc:
                        best_sc, best_lab, best_k = sc, lab, k
                if best_lab is not None:
                    labels, k_chosen = best_lab, best_k
            except Exception:
                labels = None
            # Fallback tiny 2-means
            if labels is None:
                # Init with two farthest points
                D = Z @ Z.T
                i0 = 0
                i1 = int(np.argmax(D[i0]))
                c0 = Z[i0]
                c1 = Z[i1]
                for _ in range(30):
                    d0 = np.sum((Z - c0) ** 2, axis=1)
                    d1 = np.sum((Z - c1) ** 2, axis=1)
                    lab = (d1 < d0).astype(np.int32)
                    if np.all(lab == 0) or np.all(lab == 1):
                        break
                    c0 = Z[lab == 0].mean(axis=0)
                    c1 = Z[lab == 1].mean(axis=0)
                labels = lab
                k_chosen = 2

            # Merge contiguous windows with same label into turns
            turns: List[Tuple[float, float, str]] = []
            cur_lab = None
            cur_s = None
            cur_e = None
            for (s, e), lab in zip(times, labels):
                lab = int(lab)
                if cur_lab is None:
                    cur_lab, cur_s, cur_e = lab, s, e
                    continue
                if lab == cur_lab and s <= (cur_e + 0.2):
                    cur_e = max(cur_e, e)
                else:
                    turns.append((float(cur_s), float(cur_e), f"SPEAKER_{cur_lab}"))
                    cur_lab, cur_s, cur_e = lab, s, e
            if cur_lab is not None and cur_s is not None and cur_e is not None:
                turns.append((float(cur_s), float(cur_e), f"SPEAKER_{cur_lab}"))

            # Optional: compact adjacent same-speaker segments separated by tiny gaps
            merged: List[Tuple[float, float, str]] = []
            for t in turns:
                if merged and merged[-1][2] == t[2] and (t[0] - merged[-1][1]) <= 0.3:
                    a = merged[-1]
                    merged[-1] = (a[0], t[1], a[2])
                else:
                    merged.append(t)
            try:
                print("[TA][DEBUG] diarize_simple:", {"windows": len(times), "k": k_chosen, "segments": len(merged)})
            except Exception:
                pass
            return merged or None
        except Exception:
            return None

    return await asyncio.to_thread(_run)

# -------- Sentence regrouping ---------
def split_text_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter: splits on ., !, ?, using simple rules and preserving abbreviations heuristically."""
    if not text:
        return []
    import re
    # Protect common abbreviations (approx) by replacing the dot temporarily
    protected = {
        r"\bSr\.": "Sr<dot>",
        r"\bSra\.": "Sra<dot>",
        r"\bDr\.": "Dr<dot>",
        r"\bDra\.": "Dra<dot>",
        r"\betc\.": "etc<dot>",
        r"\be\.g\.": "e<dot>g<dot>",
        r"\bi\.e\.": "i<dot>e<dot>",
        r"\b\w\.(\w\.)+": None,  # initials pattern like A.B. or U.S.A. (handled by split regex)
    }
    t = text
    for pat, rep in protected.items():
        try:
            if rep is not None:
                t = re.sub(pat, rep, t)
        except Exception:
            pass
    # Split on terminal punctuation followed by space/cap or line end
    parts = re.split(r"([\.\!\?]+)\s+(?=[A-ZÀ-Ú])|([\.\!\?]+)$", t)
    # Re-stitch keeping punctuation, then unprotect
    out = []
    buf = ''
    for p in parts:
        if p is None:
            continue
        if p == '':
            continue
        buf += p
        if any(ch in p for ch in '.!?'):
            s = buf.replace('<dot>', '.')
            s = s.strip()
            if s:
                out.append(s)
            buf = ''
    if buf.strip():
        out.append(buf.replace('<dot>', '.').strip())
    return out

def regroup_segments_by_sentence(segments: List[dict]) -> List[dict]:
    """Regroup small fragments into sentence-level segments.
    Strategy:
      - If words present: accumulate words until encountering end punctuation or a reasonable pause gap (>0.6s)
      - If no words: split text via sentence splitter and allocate time proportionally within the segment
      - Speakers: majority vote by word in window; fallback to segment speaker
    """
    if not segments:
        return []
    out: List[dict] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        words = seg.get('words') or []
        if words:
            # accumulate contiguous words across subsequent segments if they are tiny and share speaker
            buf_words = []
            spk_votes = {}
            start_t = None
            last_end = None
            j = i
            pause_gap = getattr(settings, 'sentence_pause_gap', 0.6)
            min_chars = getattr(settings, 'sentence_min_chars', 12)
            while j < len(segments):
                s = segments[j]
                ws = s.get('words') or []
                # Merge if next is very short (<= 0.6s gap) or until punctuation reached
                for w in ws:
                    if start_t is None:
                        start_t = float(w.get('start') or s.get('start') or 0.0)
                    buf_words.append(w)
                    sp = w.get('speaker') or s.get('speaker')
                    if sp:
                        spk_votes[sp] = spk_votes.get(sp, 0) + 1
                    last_end = float(w.get('end') or s.get('end') or start_t)
                    # Prefer punctuation as sentence boundary
                    if isinstance(w.get('text'), str) and any(w.get('text').strip().endswith(ch) for ch in ['.', '!', '?']):
                        j += 1
                        break
                else:
                    # no punctuation hit; look at gap to next segment start
                    if j + 1 < len(segments):
                        next_s = segments[j+1]
                        gap = float((next_s.get('start') or 0.0) - (s.get('end') or 0.0))
                        if gap > pause_gap:
                            j += 1
                            break
                    j += 1
                    continue
                # punctuation branch lands here
                break
            text_join = ' '.join((w.get('text') or '').strip() for w in buf_words).strip()
            # If too short, try to pull one more segment forward (no words path or small words)
            if len(text_join) < min_chars and j < len(segments):
                s2 = segments[j]
                ws2 = s2.get('words') or []
                for w2 in ws2:
                    buf_words.append(w2)
                    sp = w2.get('speaker') or s2.get('speaker')
                    if sp:
                        spk_votes[sp] = spk_votes.get(sp, 0) + 1
                    last_end = float(w2.get('end') or s2.get('end') or last_end or 0.0)
                    if isinstance(w2.get('text'), str) and any(w2.get('text').strip().endswith(ch) for ch in ['.', '!', '?']):
                        j += 1
                        break
                text_join = ' '.join((w.get('text') or '').strip() for w in buf_words).strip()
            speaker = None
            if spk_votes:
                speaker = max(spk_votes.items(), key=lambda kv: kv[1])[0]
            speaker = speaker or seg.get('speaker') or 'SPEAKER_0'
            out.append({
                'start': float(start_t or seg.get('start') or 0.0),
                'end': float(last_end or seg.get('end') or start_t or 0.0),
                'text': text_join,
                'speaker': speaker,
                'words': buf_words,
            })
            i = max(i+1, j)
        else:
            # No words; split current segment text by sentences proportionally
            stext = seg.get('text') or ''
            parts = split_text_into_sentences(stext)
            if len(parts) <= 1:
                out.append(seg)
                i += 1
                continue
            # Merge tiny leading/trailing fragments with neighbors based on min_chars
            min_chars = getattr(settings, 'sentence_min_chars', 12)
            merged_parts: List[str] = []
            for p in parts:
                if not merged_parts:
                    merged_parts.append(p)
                elif len(merged_parts[-1]) < min_chars:
                    merged_parts[-1] = (merged_parts[-1] + ' ' + p).strip()
                else:
                    merged_parts.append(p)
            if merged_parts and len(merged_parts[-1]) < min_chars and len(merged_parts) > 1:
                merged_parts[-2] = (merged_parts[-2] + ' ' + merged_parts[-1]).strip()
                merged_parts.pop()
            parts = merged_parts
            total_chars = sum(len(p) for p in parts)
            if total_chars <= 0:
                out.append(seg)
                i += 1
                continue
            s = float(seg.get('start') or 0.0)
            e = float(seg.get('end') or s)
            dur = max(0.0, e - s)
            cursor = s
            for k, p in enumerate(parts):
                share = len(p) / total_chars
                pend = e if k == len(parts)-1 else (s + dur * sum(len(x) for x in parts[:k+1]) / total_chars)
                out.append({
                    'start': float(cursor),
                    'end': float(pend),
                    'text': p.strip(),
                    'speaker': seg.get('speaker') or 'SPEAKER_0',
                    'words': None
                })
                cursor = pend
            i += 1
    return out
