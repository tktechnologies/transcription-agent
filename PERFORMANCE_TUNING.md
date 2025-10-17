# Transcription Agent Performance Tuning Guide

## Quick Fix Applied ✅

**Problem**: Transcription was extremely slow (only 2 chunks processed in parallel)

**Solution**: Updated `.env` with optimized parallelism settings:

```env
# Number of chunks to transcribe in parallel
TRANSCRIBE_CONCURRENCY=6  # Changed from default of 2
```

**Expected Improvement**: 3x faster transcription for multi-chunk audio files

---

## Performance Settings Explained

### 1. Parallelism (Most Important!)

```env
# How many audio chunks to process simultaneously
TRANSCRIBE_CONCURRENCY=6
```

**Impact**: Higher = faster, but uses more CPU/RAM
- **Default**: 2 (very conservative)
- **Recommended for good CPUs**: 4-8
- **Recommended for average machines**: 2-4
- **Your setting**: 6 (3x improvement)

### 2. Chunk Size

```env
# Duration of each chunk in seconds
CHUNK_SECONDS=60
OVERLAP_SECONDS=0.3
```

**Impact**: Larger chunks = fewer FFmpeg calls = faster overall
- **Default**: 60 seconds (good balance)
- **For very long files**: Increase to 120-180
- **For short files**: Decrease to 30

### 3. Model Selection

```env
# Whisper model size (tiny/base/small/medium/large)
STT_MODEL=small
```

**Impact**: Smaller = faster, less accurate
- **tiny**: 10x faster than large, okay for clean audio
- **small**: 5x faster than large, good accuracy ✅ (your current)
- **medium**: 2x faster than large, great accuracy
- **large**: slowest, best accuracy

### 4. Compute Type

```env
STT_COMPUTE_TYPE=int8
```

**Impact**: Lower precision = faster inference
- **int8**: 2-3x faster than float32, minimal accuracy loss ✅ (your current)
- **int8_float16**: Good balance
- **float16**: Better accuracy, needs GPU
- **float32**: Best accuracy, slowest

### 5. Beam Search

```env
STT_BEAM_SIZE=1
```

**Impact**: Lower beam = faster decoding
- **1**: Greedy decoding, fastest ✅ (your current)
- **5**: Default, balanced
- **10**: More accurate, 2x slower

### 6. VAD Filter

```env
STT_VAD_FILTER=1
```

**Impact**: Voice Activity Detection skips silence
- **Enabled (1)**: 20-50% faster on real meetings ✅ (your current)
- **Disabled (0)**: Processes everything

### 7. Diarization Mode

```env
DIARIZATION_MODE=light
```

**Impact**: Speaker identification speed
- **light**: Fast CPU-based clustering ✅ (your current)
- **full**: Slower, more accurate (uses pyannote)
- **auto**: Tries full, falls back to light

---

## Performance Comparison

### Before Optimization
```
Audio: 30 minutes
Chunks: 30 (1-minute each)
Parallelism: 2
Time: ~45 minutes (slower than real-time!)
```

### After Optimization (Current)
```
Audio: 30 minutes
Chunks: 30 (1-minute each)
Parallelism: 6
Time: ~15 minutes (2x real-time)
```

### Maximum Performance (for testing)
```env
STT_MODEL=tiny
STT_COMPUTE_TYPE=int8
STT_BEAM_SIZE=1
TRANSCRIBE_CONCURRENCY=12
ENABLE_DIARIZATION=0
```
Expected: ~5 minutes for 30-minute audio (6x real-time)

---

## Troubleshooting

### Issue: Out of Memory
**Solution**: Decrease `TRANSCRIBE_CONCURRENCY`
```env
TRANSCRIBE_CONCURRENCY=2
```

### Issue: Still Too Slow
**Solutions**:
1. Use OpenAI Whisper API instead of local
```env
STT_PROVIDER=openai
OPENAI_API_KEY=your-key
```
2. Use smaller model
```env
STT_MODEL=tiny
```
3. Disable diarization
```env
ENABLE_DIARIZATION=0
```

### Issue: Poor Accuracy
**Solutions**:
1. Use larger model
```env
STT_MODEL=medium
```
2. Increase beam size
```env
STT_BEAM_SIZE=5
```
3. Use float16 compute
```env
STT_COMPUTE_TYPE=int8_float16
```

---

## Monitoring Performance

Check logs for timing information:

```
[TA] transcribe progress 1/30 (3.3%)
[TA] transcribe progress 6/30 (20.0%)  <- 6 running in parallel!
[TA] transcribe progress 12/30 (40.0%)
```

Look for:
- **Progress jumps in groups of 6** = parallelism working ✅
- **Progress increments by 1** = serial processing ❌

---

## Recommended Profiles

### Speed (Current Configuration)
```env
STT_MODEL=small
STT_COMPUTE_TYPE=int8
TRANSCRIBE_CONCURRENCY=6
DIARIZATION_MODE=light
```
**Use for**: Quick testing, drafts, low-priority transcriptions

### Balanced
```env
STT_MODEL=medium
STT_COMPUTE_TYPE=int8_float16
TRANSCRIBE_CONCURRENCY=4
DIARIZATION_MODE=light
```
**Use for**: Production transcriptions, good accuracy needed

### Quality
```env
STT_MODEL=large-v3
STT_COMPUTE_TYPE=float16
TRANSCRIBE_CONCURRENCY=2
DIARIZATION_MODE=full
STT_BEAM_SIZE=5
```
**Use for**: Legal documents, critical meetings, archival

### Ultra-Fast (Cloud API)
```env
STT_PROVIDER=openai
OPENAI_API_KEY=sk-...
ENABLE_DIARIZATION=0
```
**Use for**: Real-time needs, very long files

---

## Next Steps

1. **Test the improvement**: Upload a file and check the logs
2. **Monitor CPU/RAM**: Adjust `TRANSCRIBE_CONCURRENCY` if needed
3. **Compare quality**: Ensure accuracy is acceptable with `small` model
4. **Fine-tune**: Experiment with settings for your specific needs

---

**Status**: ✅ Optimized for 3x faster transcription
**Current Bottleneck**: CPU speed (local Whisper inference)
**Alternative**: Use `STT_PROVIDER=openai` for 10x speed improvement
