import time
import uuid
from typing import Dict, Optional
from .models import JobStatus

class JobStore:
    def __init__(self):
        self._jobs: Dict[str, JobStatus] = {}

    def new(self, phase: str = "queued") -> JobStatus:
        jid = "job_" + uuid.uuid4().hex
        js = JobStatus(job_id=jid, status="queued", phase=phase)
        self._jobs[jid] = js
        return js

    def update(self, job_id: str, **patch) -> Optional[JobStatus]:
        j = self._jobs.get(job_id)
        if not j:
            return None
        for k, v in patch.items():
            setattr(j, k, v)
        return j

    def get(self, job_id: str) -> Optional[JobStatus]:
        return self._jobs.get(job_id)

job_store = JobStore()
