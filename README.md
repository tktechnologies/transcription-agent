# Transcription Agent (FastAPI)

Containerized service for fast, high-quality audio transcription with optional parsing.

## Features (MVP)
- FastAPI endpoints for upload and job status
- Sequential pipeline: normalize -> segment -> transcribe -> diarize -> persist
- Simple OpenAI Whisper integration (optional)
- Filesystem storage for artifacts (dev)

## Endpoints
- `GET /health` – readiness & config surface
- `POST /transcriptions/upload` – multipart file upload; form fields: `mode`, `org_id`, `meeting_ref`, `profile`
- `GET /transcriptions/jobs/{job_id}` – job state
- `GET /transcripts/{id}` – JSON transcript (meta + text)
- `GET /transcripts/{id}/download` – raw text
- `POST /transcripts/{id}/parse` – forward transcript to parsing-agent (optional)

## Quick start
1. Create and activate a venv
2. Install dependencies
3. Run the API

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

## Environment
- `OPENAI_API_KEY` (optional for Whisper API)
- `CHUNK_SECONDS` (default 180)
- `TRANSCRIBE_CONCURRENCY` (default 2)
- `MEETING_AGENT_URL` (optional integration)
- `PARSING_AGENT_URL` (optional integration)
- `SERVICE_TOKEN` (optional s2s auth)
