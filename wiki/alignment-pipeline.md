# Alignment Pipeline

`src/alignment/` — converts text input to a timed viseme sequence using Piper TTS + Wav2TextGrid forced alignment + phoneme-to-viseme reduction.

## AcousticPipeline

`src/alignment/pipeline.py` — top-level orchestrator:

```
text → PiperProvider.synthesize() → audio + raw phonemes
     → Wav2TextGridAligner.align() → timestamped phoneme events
     → VisemeMapper.map_phonemes() → merged viseme events
     → return (audio, sample_rate, visemes)
```

## Wav2TextGrid Aligner

`src/alignment/mfa_aligner.py` — deep learning forced alignment using Wav2Vec2.

- Resamples audio to 16kHz (Wav2TextGrid requirement)
- Extracts x-vector speaker embedding
- CTC trellis + Viterbi decoding for phoneme boundary alignment
- Returns `List[PhonemeEvent]` with millisecond-accurate timestamps

```python
aligner = Wav2TextGridAligner()
phonemes = aligner.align(audio_data, sample_rate, text)
# phonemes[0] = PhonemeEvent(phoneme="HH", start_time=0.12, end_time=0.25)
```

## Viseme Mapper

`src/alignment/viseme_mapper.py` — reduces ~40 English phonemes to 13 visual viseme categories.

**Phoneme-to-viseme mapping:**

| Viseme | Phonemes |
|---|---|
| `IDLE` | silence, SP, SIL |
| `PP` | P, B, M |
| `FF` | F, V |
| `TH` | TH, DH |
| `DD` | T, D, N, L |
| `CH` | SH, ZH, CH, JH |
| `kk` | K, G, NG, HH |
| `SS` | S, Z |
| `RR` | R, ER |
| `aa` | AA, AE, AH, AO |
| `EE` | EH, IH, IY, AY, EY |
| `OO` | UW, UH, OW, OY, AW |
| `schwa` | AX, AH0, AXR |

**Merging:** consecutive identical visemes are collapsed into a single event with extended duration.

## File reference

| File | Role |
|---|---|
| `src/alignment/__init__.py` | Re-exports Wav2TextGridAligner |
| `src/alignment/pipeline.py` | Top-level acoustic pipeline orchestrator |
| `src/alignment/mfa_aligner.py` | Wav2TextGrid forced alignment wrapper |
| `src/alignment/viseme_mapper.py` | Phoneme-to-viseme reduction + merging |
