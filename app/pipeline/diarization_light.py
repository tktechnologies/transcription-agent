import os
import subprocess
from typing import List, Dict, Optional, Tuple

import numpy as np

def _resolve_ffmpeg() -> str:
    # Try explicit env, then settings (if importable), then PATH
    env = os.environ.get("FFMPEG_PATH") or os.environ.get("FFMPEG_BIN")
    if env:
        return env
    try:
        from ..config import settings  # type: ignore
        if getattr(settings, "ffmpeg_path", None):
            return settings.ffmpeg_path  # type: ignore
    except Exception:
        pass
    import shutil as _sh
    return _sh.which("ffmpeg") or "ffmpeg"

FFMPEG_BIN = _resolve_ffmpeg()


def _load_audio_ffmpeg(path: str, sr: int = 16000) -> np.ndarray:
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-nostdin",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pcm = proc.communicate()[0]
    if proc.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg failed decoding {path}")
    wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return wav


def _choose_n_speakers(embeds: np.ndarray, max_speakers: int = 5) -> int:
    if embeds.shape[0] < 2:
        return 1
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    except Exception:
        # Fallback: assume 2 speakers if any
        return min(2, embeds.shape[0])

    best_n, best_score = 2, -1.0
    upper = min(max_speakers, embeds.shape[0])
    for n in range(2, upper + 1):
        labels = AgglomerativeClustering(n_clusters=n, linkage="average").fit_predict(embeds)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeds, labels, metric="cosine")
        if score > best_score:
            best_score, best_n = score, n
    return best_n


def diarize_over_asr_segments(
    audio_path: str,
    asr_segments: List[Dict],
    n_speakers: Optional[str | int] = None,
    max_speakers_auto: int = 5,
) -> List[Dict]:
    """
    Light diarization over ASR segments:
    - Embed each segment with a speaker encoder
    - Cluster into speakers (auto-select K)
    - Assign labels and merge adjacent same-speaker segments

    Returns list of segments with 'speaker' and preserves 'words' where present.
    """
    if not asr_segments:
        return []

    # Compute MFCC-based embeddings with librosa to avoid heavy deps (resemblyzer/webrtcvad)
    try:
        import librosa  # type: ignore
    except Exception:
        # If librosa isn't available, just set a default speaker
        out = []
        last_spk = "SPK_1"
        for s in asr_segments:
            ss = dict(s)
            if "speaker" not in ss:
                ss["speaker"] = last_spk
            last_spk = ss["speaker"]
            out.append(ss)
        return out

    wav = _load_audio_ffmpeg(audio_path, sr=16000)
    sr = 16000
    embeds: List[np.ndarray] = []
    valid_idxs: List[int] = []
    for i, seg in enumerate(asr_segments):
        try:
            s = max(0, int(float(seg["start"]) * sr))
            e = max(s + 1, int(float(seg["end"]) * sr))
        except Exception:
            continue
        if e - s < int(0.3 * sr):  # skip ultra short
            continue
        clip = wav[s:e]
        if clip.size <= 0:
            continue
        # Extract MFCCs (e.g., 30 coeffs) and average + std over time -> 60-D embedding
        try:
            mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=30)
            if mfcc is None or mfcc.size == 0:
                continue
            mu = mfcc.mean(axis=1)
            sd = mfcc.std(axis=1)
            emb = np.concatenate([mu, sd], axis=0).astype(np.float32)
            # L2 normalize
            nrm = np.linalg.norm(emb) + 1e-8
            emb = emb / nrm
            embeds.append(emb)
            valid_idxs.append(i)
        except Exception:
            continue

    if not embeds:
        for seg in asr_segments:
            seg.setdefault("speaker", "SPK_1")
        return asr_segments

    embeds = np.vstack(embeds)

    if n_speakers in (None, "auto", "AUTO", ""):
        k = _choose_n_speakers(embeds, max_speakers_auto)
    else:
        try:
            k = max(1, int(n_speakers))
        except Exception:
            k = _choose_n_speakers(embeds, max_speakers_auto)

    try:
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(embeds)
    except Exception:
        # Simple 2-means fallback with NumPy
        D = embeds @ embeds.T
        i0 = 0
        i1 = int(np.argmax(D[i0]))
        c0 = embeds[i0]
        c1 = embeds[i1]
        for _ in range(20):
            d0 = np.sum((embeds - c0) ** 2, axis=1)
            d1 = np.sum((embeds - c1) ** 2, axis=1)
            lab = (d1 < d0).astype(np.int32)
            if np.all(lab == 0) or np.all(lab == 1):
                break
            if np.any(lab == 0):
                c0 = embeds[lab == 0].mean(axis=0)
            if np.any(lab == 1):
                c1 = embeds[lab == 1].mean(axis=0)
        labels = lab
        k = 2

    # Assign labels back
    for idx, lab in zip(valid_idxs, labels):
        asr_segments[idx]["speaker"] = f"SPK_{int(lab) + 1}"
    last_spk = "SPK_1"
    for seg in asr_segments:
        if "speaker" not in seg or not seg.get("speaker"):
            seg["speaker"] = last_spk
        last_spk = seg["speaker"]

    # Merge adjacent same-speaker segments
    merged: List[Dict] = []
    for seg in asr_segments:
        if merged and merged[-1].get("speaker") == seg.get("speaker") and (seg.get("start", 0) - merged[-1].get("end", 0)) <= 0.25:
            # merge
            a = merged[-1]
            a["end"] = max(float(a.get("end", 0.0)), float(seg.get("end", a.get("end", 0.0))))
            # concatenate text
            a["text"] = (a.get("text") or "").strip() + " " + (seg.get("text") or "").strip()
            # merge words if present
            if a.get("words") and seg.get("words"):
                a["words"] = (a.get("words") or []) + (seg.get("words") or [])
        else:
            merged.append(dict(seg))

    return merged
