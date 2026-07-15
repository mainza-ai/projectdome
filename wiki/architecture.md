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

- [[cognitive-layer|Cognitive Layer]] — design overview (Qwen3, Ollama)
- [[mind-llm|Mind LLM Engine]] — implementation (provider protocol, LocalMindProvider)
- [[acoustic-layer|Acoustic Layer]] — TTS design comparison
- [[voice-tts|Voice TTS Provider]] — implementation (Piper)
- [[temporal-alignment|Temporal Alignment]] — phonetic forced alignment (concept)
- [[alignment-pipeline|Alignment Pipeline]] — implementation (Wav2TextGrid, viseme mapper)
- [[mapping-path-a|Path A — Deterministic Mapping]]
- [[mapping-path-b|Path B — Neural Regression]]
- [[animation-engine|Animation Engine]] — viseme table, interpolator, emotion blender, runtime loop
- [[rendering|Rendering Layer]] — design (WebGPU/TSL)
- [[web-renderer|Web Renderer]] — implementation (Three.js, LBS)
- [[web-ui|Web UI]] — HTML, CSS, audio sync
