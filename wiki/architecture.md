# Architecture

Project Dome's pipeline has four layers:

```
Text Input
    │
    ▼
┌─────────────────┐
│  Cognitive Layer │  Local LLM (Qwen3) → dialogue + emotion tags
│  (Mind)          │
└────────┬────────┘
         │ text stream + affect tags
         ▼
┌─────────────────┐
│  Acoustic Layer  │  Kokoro TTS → audio + phoneme sequence
│  (Voice)         │
└────────┬────────┘
         │ audio + phonemes
         ▼
┌─────────────────┐
│  Alignment Layer │  Wav2TextGrid → millisecond-accurate phoneme timestamps
│  (Temporal)      │
└────────┬────────┘
         │ timestamped phonemes + emotion tags
         ▼
┌─────────────────┐
│  Mapping Layer   │  Path A (lookup) or Path B (neural) → 383-dim coefficient vector
│  (Expression)    │
└────────┬────────┘
         │ GNM coefficients
         ▼
┌─────────────────┐
│  Rendering Layer │  WebGPU / Three.js → 17,821-vertex mesh at 60fps
│  (GPU)           │
└─────────────────┘
```

## Layer details

- [[cognitive-layer|Cognitive Layer]] — LLM orchestration
- [[acoustic-layer|Acoustic Layer]] — TTS synthesis
- [[temporal-alignment|Temporal Alignment]] — phonetic forced alignment
- [[mapping-path-a|Path A — Deterministic Mapping]]
- [[mapping-path-b|Path B — Neural Regression]]
- [[rendering|Rendering Layer]]
