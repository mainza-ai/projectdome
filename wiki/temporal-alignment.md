# Temporal Alignment (Phonetic Segmentation)

Converts audio + text transcript into millisecond-accurate phoneme timestamps. This is the control signal that drives lip animation.

## Primary tool: Wav2TextGrid

**Wav2TextGrid** (MIT license) is a Python/PyTorch forced aligner built on Wav2Vec2.

- Self-supervised speech representations (no legacy MFCCs)
- Frame-wise phoneme predictions at 10ms granularity
- CTC trellis matrix + Viterbi decoding for optimal alignment
- Outputs Praat-compatible .TextGrid files or Python objects
- Native PyTorch stack (no legacy Kaldi dependencies)

## How it works

1. **G2P conversion** — orthographic transcript → phonetic sequence (via espeak-ng or CMU Pronouncing Dictionary)
2. **Audio encoding** — 24kHz audio → Wav2Vec2 per-frame feature vectors
3. **CTC trellis** — cross-references time axis (10ms frames) with label axis (expected phonemes)
4. **Viterbi decoding** — finds optimal monotonic path through the trellis, determining exact millisecond boundaries between phonemes

## Legacy alternative: Montreal Forced Aligner (MFA)

- Kaldi-based (GMM-HMM with triphone acoustic models)
- Proven accuracy
- Complex legacy dependency chain
- Falls outside the modern PyTorch stack

## Integration

```
TTS audio + transcript → Wav2TextGrid → phoneme_timestamps[]
```

Each timestamp: `{phoneme: "AA", start_ms: 1420, end_ms: 1650}`
