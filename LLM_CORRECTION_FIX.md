# LLM Correction Not Running - Missing API Key

## Problem Found ✅

LLM correction was **enabled** but **silently failing** because the OpenAI API key was not set in the `.env` file.

**Evidence:**
```python
LLM Correction Enabled: True
LLM Correction Mode: llm
LLM Correction Passes: 2
OpenAI API Key: NOT SET  ❌
```

**Result**: Hallucinations like this were NOT being removed:
```
[00:11:29.080 - 00:11:58.340] É, é, é, é, é, é, é, é, é, é, é, é, é, é...  ❌ HALLUCINATION
[00:11:59.400 - 00:12:01.400] Tchau, tchau, tchau.  ❌ HALLUCINATION
```

## Root Cause

The code checked:
```python
if settings.enable_llm_correction and settings.openai_api_key:
```

When `openai_api_key` is `None`, the entire LLM correction block is **silently skipped** with no warning to the user.

## Fixes Applied

### Fix 1: Added Missing API Key Configuration
**File**: `.env`

Added:
```env
# === OPENAI API CONFIGURATION ===
# REQUIRED for LLM correction to work
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_ORGANIZATION=
```

**Action Required**: Replace `your-openai-api-key-here` with your actual OpenAI API key.

### Fix 2: Added Warning When API Key Missing
**File**: `app/pipeline/graph.py`

Changed from:
```python
if settings.enable_llm_correction and settings.openai_api_key:
    try:
        # LLM correction code...
```

To:
```python
if settings.enable_llm_correction:
    if not settings.openai_api_key:
        print("[TA][LLM_CORRECTION] ⚠️ WARNING: LLM correction is enabled but OPENAI_API_KEY is not set!")
        print("[TA][LLM_CORRECTION] ⚠️ Hallucinations will NOT be removed. Please set OPENAI_API_KEY in .env")
    else:
        try:
            # LLM correction code...
```

Now you'll see a clear warning in the logs when the API key is missing!

## How to Fix

### Step 1: Get OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy it

### Step 2: Update `.env`
Open `.env` and replace:
```env
OPENAI_API_KEY=your-openai-api-key-here
```

With:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx  # Your actual key
```

### Step 3: Restart the Agent
```bash
# Stop the current agent
# Restart it - it will pick up the new API key
```

### Step 4: Test with Same Audio
Upload the same audio file again and check:
1. ✅ Transcript is complete
2. ✅ LLM correction runs (you'll see log messages)
3. ✅ Hallucinations are removed

## Expected Logs (After Fix)

### Before (No API Key):
```
[TA][LLM_CORRECTION] ⚠️ WARNING: LLM correction is enabled but OPENAI_API_KEY is not set!
[TA][LLM_CORRECTION] ⚠️ Hallucinations will NOT be removed. Please set OPENAI_API_KEY in .env
```

### After (With API Key):
```
[TA][LLM_CORRECTION] Starting correction (mode=llm, passes=2)
[TA][LLM_CORRECTION] ✓ Correction applied (chars: 15420 → 14280)
```

The character reduction (15420 → 14280) indicates hallucinations were removed!

## Why This Happened

The `.env.example` template includes `OPENAI_API_KEY`, but when you copied it to `.env`, this line was missing. The code didn't warn you, so LLM correction silently failed.

## Good News ✅

1. **Filtering is working**: Transcript is now complete (no aggressive filtering)
2. **Whisper is working**: Capturing all speech with permissive thresholds
3. **Pipeline is working**: No race conditions, proper save order

**Only missing**: The LLM post-processing step to remove hallucinations.

Once you add the API key, the full pipeline will work:
```
Whisper (permissive) → Raw text + hallucinations
         ↓
LLM Correction (2 passes) → Clean text
         ↓
Save to DB → Complete, clean transcript
```

## Testing After Fix

Upload the same `estrategia armazem (9).txt` audio again and verify:

**Should NOT see**:
```
❌ [00:11:29.080 - 00:11:58.340] É, é, é, é, é, é, é, é...
❌ [00:11:59.400 - 00:12:01.400] Tchau, tchau, tchau.
```

**Should see**:
```
✅ Clean transcript with hallucinations removed
✅ Natural repetition preserved (2-3 times for emphasis)
✅ All valid speech intact
```

---

**Date**: October 14, 2025  
**Status**: Fixed, awaiting API key configuration
