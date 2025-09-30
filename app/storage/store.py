import os
import json
import uuid
from pathlib import Path
from typing import Optional
from ..config import settings

ART_DIR = Path(settings.storage_dir)
ART_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR = ART_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = ART_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

class FileStore:
    @staticmethod
    def save_upload(filename: str, data: bytes) -> str:
        uid = str(uuid.uuid4())
        path = UPLOADS_DIR / f"{uid}_{filename}"
        with open(path, "wb") as f:
            f.write(data)
        return str(path)

    @staticmethod
    def new_transcript_id() -> str:
        return "t_" + uuid.uuid4().hex[:16]

    @staticmethod
    def save_transcript_json(transcript_id: str, payload: dict) -> str:
        path = TRANSCRIPTS_DIR / f"{transcript_id}.json"
        try:
            print("[TA][DEBUG] store.save: payload type before coerce:", type(payload).__name__)
        except Exception:
            pass
        # Be defensive: coerce payload to dict
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            try:
                payload = payload.model_dump()  # pydantic v2
            except Exception:
                try:
                    payload = payload.dict()  # pydantic v1
                except Exception:
                    payload = {"text": str(payload)}
        try:
            print("[TA][DEBUG] store.save: payload keys:", list(payload.keys()))
        except Exception:
            pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        # write raw text convenience file
        text = ""
        try:
            text = payload.get("text") or ""
        except Exception:
            try:
                text = str(payload)
            except Exception:
                text = ""
        with open(TRANSCRIPTS_DIR / f"{transcript_id}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return str(path)

    @staticmethod
    def load_transcript_json(transcript_id: str) -> Optional[dict]:
        path = TRANSCRIPTS_DIR / f"{transcript_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_transcript_text(transcript_id: str) -> Optional[str]:
        path = TRANSCRIPTS_DIR / f"{transcript_id}.txt"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
