# Acoustic Layer (Voice)

Produces high-quality speech audio from text. Must be fully local, CPU-runnable, and permissively licensed.

## Primary engine: Kokoro

**Kokoro** (Hexgrad, Apache 2.0) is an 82-million parameter TTS model.

| Property | Value |
|---|---|
| Parameters | 82M |
| License | Apache 2.0 |
| Architecture | StyleTTS 2 + ISTFTNet |
| Quality | MOS scores rivaling commercial APIs |
| Hardware | CPU (5-15x real-time) |
| Storage | <350MB |
| Output | 24kHz raw audio + phoneme sequence |
| Streaming | Yes — chunked output for low latency |

Kokoro's streaming capability is critical: the avatar can begin speaking the first synthesized sentence while the LLM generates subsequent tokens.

## Fallback: Piper TTS

- MIT licensed
- 10M–50M parameters
- ONNX/VITS-based
- CPU-runnable
- Functional quality (robotic, but reliable)

## Comparison

| Feature | Piper TTS | Kokoro | XTTS v2 |
|---|---|---|---|
| Quality | Baseline | High-fidelity | High-fidelity |
| License | MIT | Apache 2.0 | CPML (non-commercial) |
| Hardware | CPU | CPU | GPU (4-6GB) |
| Role | Legacy fallback | Primary | Disqualified |

## Integration

```python
# pip install kokoro
from kokoro import KPipeline
pipeline = KPipeline(lang_code='a')
audio, phonemes = pipeline(text, voice='af_heart')
```
