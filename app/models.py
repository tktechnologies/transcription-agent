from pydantic import BaseModel
from typing import List, Optional

class Word(BaseModel):
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    speaker: Optional[str] = None

class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    words: Optional[List[Word]] = None

class Transcript(BaseModel):
    transcript_id: str
    org_id: Optional[str] = None
    meeting_id: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    mode: Optional[str] = None
    profile: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    segments: Optional[List[Segment]] = None
    text: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    phase: Optional[str] = None
    progress: Optional[float] = None
    transcript_id: Optional[str] = None
    facts_inserted: Optional[int] = None
    error: Optional[str] = None
