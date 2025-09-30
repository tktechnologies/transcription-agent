from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import shutil

from .config import settings
from .jobs import job_store
from .storage.store import FileStore
from .models import JobStatus
from .pipeline.graph import run_pipeline

app = FastAPI(title="Transcription Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"]
)

@app.get("/health")
async def health():
    import shutil as _shutil
    from .config import settings as _settings
    ffmpeg_cmd = _settings.ffmpeg_path or "ffmpeg"
    ffprobe_cmd = _settings.ffprobe_path or "ffprobe"
    ffmpeg_ok = bool(_shutil.which(ffmpeg_cmd))
    ffprobe_ok = bool(_shutil.which(ffprobe_cmd))
    hf_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    return {
        "status": "ok",
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "ffmpeg_cmd": ffmpeg_cmd,
        "ffprobe_cmd": ffprobe_cmd,
        "stt_provider": settings.stt_provider,
        "openai": bool(settings.openai_api_key),
        "openai_base_url": settings.openai_base_url,
        "openai_org": bool(settings.openai_organization),
        "storage": settings.storage_dir,
        "hf_cache": hf_cache,
    }


def require_auth(req: Request):
    token = settings.service_token
    if not token:
        return
    auth = req.headers.get("authorization") or req.headers.get("Authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="unauthorized")
    provided = auth.split(" ", 1)[1].strip()
    if provided != token:
        raise HTTPException(status_code=401, detail="unauthorized")

@app.post("/transcriptions/upload")
async def upload(
    req: Request,
    file: UploadFile = File(...),
    mode: str = Form("transcribe_and_parse"),
    org_id: str | None = Form(None),
    meeting_ref: str | None = Form(None),
    profile: str | None = Form("balanced"),
):
    require_auth(req)
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing_file")
    # Log upload metadata
    try:
        print("[TA] Upload received", {"filename": file.filename, "mode": mode, "org_id": org_id, "meeting_ref": meeting_ref, "profile": profile})
    except Exception:
        pass
    # Save upload payload
    data = await file.read()
    input_path = FileStore.save_upload(file.filename, data)
    job = job_store.new(phase="queued")
    job_store.update(job.job_id, status="processing", phase="normalize")
    try:
        print("[TA] Job created", {"job_id": job.job_id, "input_path": input_path})
    except Exception:
        pass

    async def worker():
        try:
            job_store.update(job.job_id, phase="transcribe")
            result = await run_pipeline(job.job_id, input_path, org_id, meeting_ref, mode, profile)
            # Preserve existing transcript_id if present (persist step may have already set it)
            current = job_store.get(job.job_id)
            existing_tid = getattr(current, "transcript_id", None) if current else None
            tid = None
            try:
                tid = (result.get("transcript_id") if isinstance(result, dict) else None) or existing_tid
            except Exception:
                tid = existing_tid
            job_store.update(job.job_id, status="done", transcript_id=tid)
            try:
                print("[TA] Job done", {"job_id": job.job_id, "transcript_id": tid})
            except Exception:
                pass
        except Exception as e:
            try:
                print("[TA][ERROR] Job failed", {"job_id": job.job_id, "error": str(e)})
            except Exception:
                pass
            job_store.update(job.job_id, status="failed", error=str(e) or "unknown_error")

    asyncio.create_task(worker())
    return {"job_id": job.job_id, "status": job.status}

@app.get("/transcriptions/jobs/{job_id}")
async def get_job(req: Request, job_id: str):
    require_auth(req)
    j = job_store.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="not_found")
    try:
        print("[TA] Job status", {"job_id": job_id, "status": j.status, "phase": j.phase, "error": j.error})
    except Exception:
        pass
    return j

@app.get("/transcripts/{transcript_id}")
async def get_transcript(req: Request, transcript_id: str):
    require_auth(req)
    j = FileStore.load_transcript_json(transcript_id)
    if not j:
        raise HTTPException(status_code=404, detail="not_found")
    return JSONResponse(j)

def _to_srt(segments):
    lines = []
    def fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    for i, seg in enumerate(segments, start=1):
        start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        end = seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0)
        txt = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        spk = seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", None)
        header = f"{fmt(start)} --> {fmt(end)}"
        if spk:
            txt = f"[{spk}] {txt}"
        lines.append(f"{i}\n{header}\n{txt}\n")
    return "\n".join(lines)

def _to_vtt(segments):
    lines = ["WEBVTT\n"]
    def fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    for seg in segments:
        start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        end = seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0)
        txt = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        spk = seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", None)
        header = f"{fmt(start)} --> {fmt(end)}"
        if spk:
            txt = f"[{spk}] {txt}"
        lines.append(f"{header}\n{txt}\n")
    return "\n".join(lines)

def _to_txt(segments):
    # Plain text with timestamps and speakers per line: [HH:MM:SS.mmm - HH:MM:SS.mmm] [SPEAKER] text
    def fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    lines = []
    for seg in segments:
        start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        end = seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0)
        txt = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        spk = seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", None)
        if not spk:
            spk = "SPEAKER_0"
        lines.append(f"[{fmt(start)} - {fmt(end)}] [{spk}] {txt}".rstrip())
    return "\n".join(lines)

@app.get("/transcripts/{transcript_id}/download")
async def download_transcript(req: Request, transcript_id: str, format: str = Query("txt", pattern="^(txt|srt|vtt)$")):
    require_auth(req)
    j = FileStore.load_transcript_json(transcript_id)
    if not j:
        raise HTTPException(status_code=404, detail="not_found")
    if format == "txt":
        segments = j.get("segments") or []
        if segments:
            # Standard TXT now includes timestamps and speakers per segment
            out = _to_txt(segments)
            return PlainTextResponse(out)
        # Fallback: raw text when segments missing
        text = j.get("text", "")
        return PlainTextResponse(text)
    segments = j.get("segments") or []
    if format == "srt":
        srt = _to_srt(segments)
        return Response(content=srt, media_type="application/x-subrip")
    else:
        vtt = _to_vtt(segments)
        return Response(content=vtt, media_type="text/vtt")

@app.post("/transcripts/{transcript_id}/parse")
async def parse_later(req: Request, transcript_id: str, org_id: str | None = Form(None), meeting_ref: str | None = Form(None)):
    require_auth(req)
    # Placeholder – call parsing-agent here in future
    j = FileStore.load_transcript_json(transcript_id)
    if not j:
        raise HTTPException(status_code=404, detail="not_found")
    # Return a stub for now
    return {"transcript_id": transcript_id, "facts_inserted": 0}
