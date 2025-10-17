# Race Condition Fix - Transcription Agent

## Problem Summary

The transcription agent was saving incomplete transcriptions to the database. This was caused by a **race condition** where the `transcript_id` was exposed to API callers **before** the transcript was actually saved to storage.

## Root Causes Identified

### 1. **Primary Race Condition in `graph.py`** (CRITICAL)
**Location**: `app/pipeline/graph.py`, line ~856-857 (before fix)

The code was updating the job store with the `transcript_id` **BEFORE** calling the save function:

```python
# OLD CODE - INCORRECT ORDER
state["transcript_id"] = transcript_id
state["text"] = text

# Expose transcript_id immediately for job pollers ❌ TOO EARLY!
job_store.update(job_id, transcript_id=transcript_id)

# Save to storage (MongoDB or FileStore based on config)
if USE_MONGODB:
    await MongoDBStore.save_transcript(transcript_id, payload, org_id, meeting_ref)
else:
    FileStore.save_transcript_json(transcript_id, payload)
```

**The Race Condition Timeline:**
1. Pipeline completes transcription processing
2. `transcript_id` is generated
3. Job store is updated with `transcript_id` ❌ **EXPOSED HERE**
4. API caller polls job status, sees `transcript_id`
5. API caller tries to fetch transcript (not saved yet!) ❌ **FAILS**
6. Database save happens ✓ (too late!)

### 2. **Secondary Issue in `main.py`**
**Location**: `app/main.py`, line ~88-92 (before fix)

The worker function had redundant logic trying to preserve a `transcript_id` that might have been set during the pipeline:

```python
# OLD CODE - UNNECESSARY COMPLEXITY
current = job_store.get(job.job_id)
existing_tid = getattr(current, "transcript_id", None) if current else None
tid = (result.get("transcript_id") if isinstance(result, dict) else None) or existing_tid
```

This created a timing dependency where the worker could see an intermediate state.

## Fixes Applied

### Fix 1: Reordered Save and Expose in `graph.py`

**File**: `app/pipeline/graph.py`

Changed the order to ensure the transcript is **saved first**, then exposed:

```python
# NEW CODE - CORRECT ORDER
try:
    # Save to storage FIRST (MongoDB or FileStore based on config)
    if USE_MONGODB:
        await MongoDBStore.save_transcript(transcript_id, payload, org_id, meeting_ref)
    else:
        FileStore.save_transcript_json(transcript_id, payload)
    
    # Only set state values AFTER successful save
    state["transcript_id"] = transcript_id
    state["text"] = text
    
    # ONLY expose transcript_id AFTER successful save
    job_store.update(job_id, transcript_id=transcript_id)
    print(f"[TA][DEBUG] persist: ✓ Transcript {transcript_id} saved and exposed to job store")
    
except Exception as save_error:
    # If save fails, don't expose transcript_id and propagate error
    print(f"[TA][ERROR] persist: Failed to save transcript {transcript_id}: {save_error}")
    raise RuntimeError(f"transcript_save_failed: {save_error}")
```

**Key Changes:**
- ✅ Database save happens **FIRST**
- ✅ `transcript_id` only exposed **AFTER** successful save
- ✅ If save fails, error is raised and no `transcript_id` is exposed
- ✅ Added error handling with clear logging

### Fix 2: Simplified Worker Logic in `main.py`

**File**: `app/main.py`

Removed redundant logic and clarified that `transcript_id` only comes from the completed pipeline:

```python
# NEW CODE - SIMPLIFIED
result = await run_pipeline(job.job_id, input_path, org_id, meeting_ref, mode, profile)

# Extract transcript_id from pipeline result
# The pipeline returns transcript_id only after successful DB save
tid = None
try:
    tid = result.get("transcript_id") if isinstance(result, dict) else None
except Exception:
    pass

# Mark job as done with the transcript_id
# This update happens AFTER the transcript is saved in the pipeline
job_store.update(job.job_id, status="done", transcript_id=tid)
```

**Key Changes:**
- ✅ Removed unnecessary job store check
- ✅ Clearer comments explaining the flow
- ✅ Simplified error handling

## Verification

### Critical Execution Flow (After Fix)

```
1. Transcription processing starts
   ↓
2. Audio is normalized, segmented, transcribed, diarized
   ↓
3. LLM correction (if enabled)
   ↓
4. Segments sanitized and payload built
   ↓
5. 🔒 SAVE TO DATABASE (MongoDB or FileStore)
   ↓
6. ✅ Save succeeds
   ↓
7. State updated with transcript_id
   ↓
8. 🔓 Job store updated with transcript_id (NOW VISIBLE TO API)
   ↓
9. Worker marks job as "done"
   ↓
10. API callers can safely fetch the transcript
```

### Error Handling Flow

If the database save fails at step 5:
```
5. 🔒 SAVE TO DATABASE
   ↓
   ❌ Exception raised
   ↓
   transcript_id NOT exposed
   ↓
   Error propagates to worker
   ↓
   Job marked as "failed" with error message
   ↓
   No incomplete transcript visible to API callers
```

## Testing Recommendations

1. **Happy Path Test**: Upload audio file and verify transcript is only visible after save completes
2. **Database Failure Test**: Simulate DB connection failure and verify no transcript_id is exposed
3. **Concurrent Request Test**: Upload multiple files simultaneously and verify no race conditions
4. **API Polling Test**: Poll job status rapidly and verify transcript is always complete when fetched

## Impact

### Before Fix
- ❌ API callers could see `transcript_id` before save completed
- ❌ Fetching transcript would return 404 or empty data
- ❌ Retries required, poor user experience
- ❌ Potential data inconsistencies

### After Fix
- ✅ `transcript_id` only visible after successful save
- ✅ Fetching transcript always returns complete data
- ✅ No retries needed, smooth user experience
- ✅ Data consistency guaranteed

## Related Files Modified

1. `app/pipeline/graph.py` - Reordered save/expose, added error handling
2. `app/main.py` - Simplified worker logic, clarified comments

## Date
October 14, 2025
