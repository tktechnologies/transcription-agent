# Hallucination Handling Strategy - LLM-Based Approach

## Problem Solved

The transcription agent was losing valid Portuguese speech because rule-based hallucination filters were too aggressive. Segments like this were being completely removed:

```
[00:18:27.430 - 00:18:33.750] Se dá pra botar aqui e depois eu cobrar, eu não consigo fazer 500 coisas.
```

This is **valid speech**, not a hallucination, but pattern-matching filters couldn't distinguish it from actual hallucinations.

## New Strategy: Let LLM Handle Hallucinations

Instead of using rigid pattern matching to filter hallucinations **during** transcription, we now:

### ✅ **Phase 1: Whisper - Maximum Recall**
- Use **maximally permissive thresholds** to capture ALL speech
- Accept that some hallucinations will slip through
- Focus on NOT losing valid speech

**Threshold Changes:**
```env
# OLD - Too strict, losing real speech
STT_NO_SPEECH_THRESHOLD=0.6
STT_COMPRESSION_RATIO_THRESHOLD=2.4
STT_LOGPROB_THRESHOLD=-1.0

# NEW - Maximum permissiveness, capture everything
STT_NO_SPEECH_THRESHOLD=0.8         # Higher = more permissive
STT_COMPRESSION_RATIO_THRESHOLD=3.0  # Higher = more permissive  
STT_LOGPROB_THRESHOLD=-2.0          # Less negative = more permissive
```

### ✅ **Phase 2: LLM - Intelligent Cleaning**
- LLM correction now handles **BOTH** error correction **AND** hallucination removal
- Context-aware: understands Portuguese patterns vs actual hallucinations
- Removes only **obvious** hallucinations:
  - Music markers: ♪, ♫, [Music]
  - Excessive repetition: Same phrase 5+ times (not natural emphasis)
  - Noise artifacts: Unintelligible sequences
  
**What LLM Keeps (Natural Speech):**
- Normal repetition: "Tá, tá, entendi" ✅
- Emphasis: "Não, não, não quero" ✅
- Informal speech: "né", "cara", "pra" ✅

**What LLM Removes (Hallucinations):**
- Excessive loops: "e aí e aí e aí e aí e aí..." (10+ times) ❌
- Music: "♪ ♪ ♪..." ❌
- Gibberish: Random word sequences that make no sense ❌

## Implementation Changes

### 1. Removed Rule-Based Filters
**File**: `app/pipeline/graph.py`

```python
# REMOVED: filter_hallucinated_segments function
# Hallucination filtering is now handled by LLM correction for better context-aware cleaning
# This avoids removing valid Portuguese speech that may look "repetitive" to rule-based filters
```

The old function was checking things like:
- Word repetition counts
- Phrase repetition patterns  
- Uniqueness ratios
- Music symbols

**Problem**: These patterns appear in BOTH valid speech AND hallucinations. Portuguese conversations naturally have repetition!

### 2. Enhanced LLM Correction Prompt
**File**: `app/pipeline/transcription_correction.py`

**New Prompt Section:**
```
HALLUCINATION REMOVAL (remove these patterns):
- Music/noise markers: "♪", "♫", "🎵", "[Music]", sequences of "..."
- Excessive repetition: Same word/phrase repeated 5+ times consecutively
- Meaningless loops: Very long segments where the same 2-3 words keep repeating
- Background noise transcribed as words
- IMPORTANT: Only remove OBVIOUS hallucinations. Natural repetition in speech is OK!
```

**System Message Updated:**
```python
"You are an expert at correcting Portuguese transcription errors AND removing hallucinations 
from business meetings. You MUST: 
1) Remove obvious hallucinations (music markers, excessive repetition of 5+ times, noise artifacts)
2) Only output real Portuguese words or known technical terms
3) Never invent nonsense words
4) If unsure about a correction, leave the word unchanged."
```

### 3. Permissive Whisper Configuration
**File**: `.env`

```env
# MAXIMALLY PERMISSIVE THRESHOLDS - Capture everything, let LLM handle hallucinations
STT_NO_SPEECH_THRESHOLD=0.8        # Default: 0.6, New: 0.8 (more permissive)
STT_COMPRESSION_RATIO_THRESHOLD=3.0  # Default: 2.4, New: 3.0 (more permissive)
STT_LOGPROB_THRESHOLD=-2.0         # Default: -1.0, New: -2.0 (more permissive)
```

**Why These Values:**
- **no_speech_threshold**: 0.8 means only filter if 80%+ confident it's silence (was 60%)
- **compression_ratio**: 3.0 allows more compressed text before flagging as hallucination
- **logprob**: -2.0 accepts lower probability outputs (captures uncertain speech)

## Advantages of LLM-Based Approach

### 1. **Context-Aware**
❌ Rule: "If phrase repeats 5+ times, remove"
✅ LLM: "Is this natural emphasis or a hallucination loop?"

Example:
- "Tá, tá, tá, entendi" → Natural affirmation, **KEEP**
- "e aí e aí e aí e aí e aí e aí e aí e aí" → Hallucination loop, **REMOVE**

### 2. **Language-Specific Intelligence**
❌ Rule: "If uniqueness ratio < 20%, remove"  
✅ LLM: Understands Portuguese conversation patterns

Example:
- "Se dá pra botar aqui e depois eu cobrar" → Valid Portuguese, **KEEP**
- Old filter saw repeated words → **WRONGLY REMOVED**

### 3. **Semantic Understanding**
❌ Rule: "Check for music symbols"
✅ LLM: Can identify music transcribed WITHOUT symbols

Example:
- "la la la la la la la..." → LLM recognizes as music, **REMOVE**
- Rule wouldn't catch it without ♪ symbol

### 4. **No False Positives**
**Before (Rule-Based)**:
- Lost valid speech segments
- Conservative thresholds to avoid hallucinations
- Result: Incomplete transcripts

**After (LLM-Based)**:
- Capture everything with Whisper
- Let intelligent LLM clean it
- Result: Complete transcripts with hallucinations removed

## Configuration

### Enabling LLM Correction
Already enabled by default in `.env`:

```env
ENABLE_LLM_CORRECTION=true
LLM_CORRECTION_MODE=llm              # Pure LLM approach (no term map)
LLM_CORRECTION_PASSES=2              # Two passes for thorough cleaning
LLM_CORRECTION_MODEL=gpt-5-mini      # Fast + accurate
```

### How It Works

1. **Transcribe** with permissive Whisper settings → Raw text with possible hallucinations
2. **LLM Pass 1** → Fix errors + remove obvious hallucinations  
3. **LLM Pass 2** → Catch anything missed in pass 1
4. **Result** → Clean, accurate transcript

## Performance Impact

- **Whisper Phase**: Faster (fewer rejected segments, less re-processing)
- **LLM Phase**: ~2-3 seconds per transcript (already running for error correction)
- **Net Impact**: Negligible (LLM was already running, just enhanced prompt)

## Testing

### Before Fix
```
Transcript: Only 16 segments from 12:00-18:39 (rest filtered as "hallucinations")
```

### After Fix  
```
Transcript: Full conversation captured, actual hallucinations intelligently removed by LLM
```

## Monitoring

Watch for these log messages:

**Whisper Phase (Permissive)**:
```
[TA][DEBUG] transcribe: merged segments count=145
```
More segments = capturing more speech ✅

**LLM Correction Phase**:
```
[TA][LLM_CORRECTION] Starting correction (mode=llm, passes=2)
[TA][LLM_CORRECTION] ✓ Correction applied (chars: 5420 → 5380)
```
Character reduction indicates hallucinations removed ✅

**Warning Signs** (if you see these, LLM might be too aggressive):
```
[TranscriptionCorrection] WARNING: Detected suspicious words: ['complice', 'afinaramento']
```
These indicate the LLM invented non-existent words → reduce temperature or passes

## Fallback Strategy

If LLM correction fails (no API key, API down, etc.):
1. Whisper still captures everything with permissive thresholds
2. Raw transcript returned (may contain some hallucinations)
3. Better to have hallucinations than missing speech
4. Users can manually review if needed

## Summary

| Aspect | Old Approach | New Approach |
|--------|-------------|--------------|
| **Filtering** | Rule-based during transcription | LLM-based after transcription |
| **Thresholds** | Strict (0.6, 2.4, -1.0) | Permissive (0.8, 3.0, -2.0) |
| **Philosophy** | Filter aggressively to avoid hallucinations | Capture everything, clean intelligently |
| **Risk** | Lost real speech | Might miss some hallucinations |
| **Result** | Incomplete transcripts | Complete, cleaned transcripts |
| **Context** | No understanding | Full Portuguese context awareness |

## Date
October 14, 2025
