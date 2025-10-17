from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv
import zipfile

# Load a .env file early so environment variables are available to Settings
# Priority: transcription-agent/.env (sibling to app/), then CWD/.env
try:
    _env_candidates = [
        Path(__file__).resolve().parents[1] / ".env",
        Path.cwd() / ".env",
    ]
    for _p in _env_candidates:
        if _p.exists():
            load_dotenv(dotenv_path=str(_p), override=False)
            break
except Exception:
    # Non-fatal; continue with process environment
    pass

def _default_stt_provider():
    # Prefer explicit env; otherwise choose 'local' when no OPENAI_API_KEY is set
    v = os.getenv("STT_PROVIDER")
    if v:
        return v
    return "local" if not os.getenv("OPENAI_API_KEY") else "openai"


def _detect_ffmpeg_path() -> str | None:
    """Try to find a bundled ffmpeg binary if FFMPEG_PATH is not set.
    Looks in transcription-agent/bin/ffmpeg/ under the repo root.
    """
    # If explicitly provided, honor it
    env = os.getenv("FFMPEG_PATH")
    if env:
        return env
    try:
        root = Path(__file__).resolve().parents[1]
        # Prefer exact binaries placed directly under bin/ (user's current layout)
        direct_bin = root / "bin"
        direct_ffmpeg_exe = direct_bin / "ffmpeg.exe"
        direct_ffmpeg_unix = direct_bin / "ffmpeg"
        if direct_ffmpeg_exe.exists():
            return str(direct_ffmpeg_exe)
        if direct_ffmpeg_unix.exists():
            return str(direct_ffmpeg_unix)

        # Legacy/bundled layout: bin/ffmpeg folder (possibly with ffmpeg.zip)
        ff_dir = root / "bin" / "ffmpeg"
        # If executables missing but ffmpeg.zip exists, try to extract
        ff_exe = ff_dir / "ffmpeg.exe"
        ff_unix = ff_dir / "ffmpeg"
        if not ff_exe.exists() and not ff_unix.exists():
            z = ff_dir / "ffmpeg.zip"
            if z.exists():
                try:
                    with zipfile.ZipFile(str(z), 'r') as zf:
                        zf.extractall(str(ff_dir))
                except Exception:
                    pass
        # After potential extract, look for executables (search recursively as a fallback)
        if ff_exe.exists():
            return str(ff_exe)
        if ff_unix.exists():
            return str(ff_unix)
        try:
            # recursive search for ffmpeg(.exe)
            for p in ff_dir.rglob('ffmpeg.exe'):
                return str(p)
            for p in ff_dir.rglob('ffmpeg'):
                if p.is_file() and os.access(p, os.X_OK):
                    return str(p)
        except Exception:
            pass
        # Also consider a top-level FFmpeg folder with bin/ (user-provided)
        try:
            ff_alt = root / "FFmpeg"
            for sub in (ff_alt / "bin", ff_alt / "build" / "bin"):
                f1 = sub / "ffmpeg.exe"
                f2 = sub / "ffmpeg"
                if f1.exists():
                    return str(f1)
                if f2.exists():
                    return str(f2)
        except Exception:
            pass
    except Exception:
        pass
    return None


def _detect_ffprobe_path() -> str | None:
    env = os.getenv("FFPROBE_PATH")
    if env:
        return env
    try:
        root = Path(__file__).resolve().parents[1]
        # Prefer exact binaries placed directly under bin/
        direct_bin = root / "bin"
        direct_fp_exe = direct_bin / "ffprobe.exe"
        direct_fp_unix = direct_bin / "ffprobe"
        if direct_fp_exe.exists():
            return str(direct_fp_exe)
        if direct_fp_unix.exists():
            return str(direct_fp_unix)

        # Legacy/bundled layout: bin/ffmpeg folder
        ff_dir = root / "bin" / "ffmpeg"
        # Attempt zip extract if necessary (shared with ffmpeg)
        fp_exe = ff_dir / "ffprobe.exe"
        fp_unix = ff_dir / "ffprobe"
        if not fp_exe.exists() and not fp_unix.exists():
            z = ff_dir / "ffmpeg.zip"
            if z.exists():
                try:
                    with zipfile.ZipFile(str(z), 'r') as zf:
                        zf.extractall(str(ff_dir))
                except Exception:
                    pass
        if fp_exe.exists():
            return str(fp_exe)
        if fp_unix.exists():
            return str(fp_unix)
        try:
            for p in ff_dir.rglob('ffprobe.exe'):
                return str(p)
            for p in ff_dir.rglob('ffprobe'):
                if p.is_file() and os.access(p, os.X_OK):
                    return str(p)
        except Exception:
            pass
        # Also check a top-level FFmpeg folder with bin/
        try:
            ff_alt = root / "FFmpeg"
            for sub in (ff_alt / "bin", ff_alt / "build" / "bin"):
                f1 = sub / "ffprobe.exe"
                f2 = sub / "ffprobe"
                if f1.exists():
                    return str(f1)
                if f2.exists():
                    return str(f2)
        except Exception:
            pass
    except Exception:
        pass
    return None


class Settings(BaseModel):
    # External binaries
    ffmpeg_path: str | None = _detect_ffmpeg_path()
    ffprobe_path: str | None = _detect_ffprobe_path()

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_organization: str | None = os.getenv("OPENAI_ORG")
    openai_api_version: str = os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
    openai_deployment: str | None = os.getenv("OPENAI_DEPLOYMENT")
    openai_transcribe_model: str = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
    stt_provider: str = _default_stt_provider()  # openai|local
    stt_model: str = os.getenv("STT_MODEL", "medium")  # for faster-whisper
    stt_device: str = os.getenv("STT_DEVICE", "auto")  # auto|cpu|cuda
    stt_compute_type: str = os.getenv("STT_COMPUTE_TYPE", "auto")  # auto|int8|int8_float16|float16|float32
    stt_beam_size: int = int(os.getenv("STT_BEAM_SIZE", "5"))
    stt_word_timestamps: bool = os.getenv("STT_WORD_TIMESTAMPS", "1") not in ("0", "false", "False")
    stt_cpu_threads: int = int(os.getenv("STT_CPU_THREADS", "0"))  # 0 lets library decide
    stt_num_workers: int = int(os.getenv("STT_NUM_WORKERS", "1"))
    stt_language: str | None = os.getenv("STT_LANGUAGE")  # optional language hint like 'pt', 'en'
    # VAD filter enabled by default to prevent hallucinations on silence/noise/music
    # This skips transcribing segments with no speech, solving the root cause of repetitive hallucinations
    stt_vad_filter: bool = os.getenv("STT_VAD_FILTER", "1") not in ("0", "false", "False")
    # Additional faster-whisper decoding knobs
    stt_condition_prev_text: bool = os.getenv("STT_CONDITION_PREV_TEXT", "0") not in ("0", "false", "False")
    stt_temperature: float = float(os.getenv("STT_TEMPERATURE", "0"))
    # Increased threshold to be more aggressive at skipping non-speech (helps reduce hallucinations)
    stt_no_speech_threshold: float = float(os.getenv("STT_NO_SPEECH_THRESHOLD", "0.8"))
    stt_compression_ratio_threshold: float = float(os.getenv("STT_COMPRESSION_RATIO_THRESHOLD", "2.6"))
    stt_logprob_threshold: float = float(os.getenv("STT_LOGPROB_THRESHOLD", "-1.0"))
    chunk_seconds: int = int(os.getenv("CHUNK_SECONDS", "60"))
    overlap_seconds: float = float(os.getenv("OVERLAP_SECONDS", "0.8"))
    use_silence_segmentation: bool = os.getenv("USE_SILENCE_SEGMENTATION", "1") not in ("0", "false", "False")
    # Post-processing: regroup final segments by sentence boundaries to avoid tiny fragments
    use_sentence_segmentation: bool = os.getenv("USE_SENTENCE_SEGMENTATION", "1") not in ("0", "false", "False")
    sentence_min_chars: int = int(os.getenv("SENTENCE_MIN_CHARS", "12"))
    sentence_pause_gap: float = float(os.getenv("SENTENCE_PAUSE_GAP", "0.6"))
    # Diarization enabled by default to test effectiveness
    # Now with proper failure visibility to assess quality
    enable_diarization: bool = os.getenv("ENABLE_DIARIZATION", "1") not in ("0", "false", "False")
    diarization_mode: str = os.getenv("DIARIZATION_MODE", "auto")  # auto|light|full
    diarization_speakers: str | None = os.getenv("DIARIZATION_SPEAKERS")  # 'auto' or integer as string
    # Timestamp refinement level
    timestamp_level: str = os.getenv("TIMESTAMP_LEVEL", "segment")  # segment|word|auto
    word_ts_min_seconds: float = float(os.getenv("WORD_TS_MIN_SECONDS", "12"))
    transcribe_concurrency: int = int(os.getenv("TRANSCRIBE_CONCURRENCY", "2"))
    chunk_timeout_seconds: int = int(os.getenv("STT_CHUNK_TIMEOUT_SECONDS", "300"))
    hybrid_fallback: bool = os.getenv("HYBRID_FALLBACK", "1") not in ("0", "false", "False")
    meeting_agent_url: str | None = os.getenv("MEETING_AGENT_URL")
    parsing_agent_url: str | None = os.getenv("PARSING_AGENT_URL")
    service_token: str | None = os.getenv("SERVICE_TOKEN")
    storage_dir: str = os.getenv("STORAGE_DIR", "artifacts")
    
    # LLM-based transcription correction (post-processing to fix phonetic/domain errors)
    enable_llm_correction: bool = os.getenv("ENABLE_LLM_CORRECTION", "0") not in ("0", "false", "False")
    llm_correction_mode: str = os.getenv("LLM_CORRECTION_MODE", "llm")  # llm|hybrid|quick
    llm_correction_passes: int = int(os.getenv("LLM_CORRECTION_PASSES", "2"))
    llm_correction_model: str = os.getenv("LLM_CORRECTION_MODEL", "gpt-5-mini")

settings = Settings()
