# Tool Stack Reference

## LLM / Cognitive

| Tool | Version | License | Role |
|---|---|---|---|
| [[cognitive-layer#primary-recommendation-qwen3-8b-or-14b|Qwen3 8B/14B]] | 2026 | Apache 2.0 | Primary cognitive engine |
| [[cognitive-layer#alternative-models|gpt-oss-20b]] | 2026 | Apache 2.0 | High-end alternative |
| [[cognitive-layer#deployment-ollama|Ollama]] | latest | MIT | Local LLM deployment daemon |

## TTS / Acoustic

| Tool | Version | License | Role |
|---|---|---|---|
| [[acoustic-layer#primary-engine-kokoro|Kokoro]] | v1.0 | Apache 2.0 | Primary TTS engine |
| [[acoustic-layer#fallback-piper-tts|Piper TTS]] | stable | MIT | Legacy fallback |

## Alignment

| Tool | Version | License | Role |
|---|---|---|---|
| [[temporal-alignment#primary-tool-wav2textgrid|Wav2TextGrid]] | latest | MIT | Forced alignment |
| [[temporal-alignment#legacy-alternative-montreal-forced-aligner-mfa|MFA]] | v3.3+ | Apache 2.0 | Legacy fallback |

## Rendering

| Tool | Version | License | Role |
|---|---|---|---|
| [[rendering|Three.js WebGPURenderer]] | r160+ | MIT | Web 3D rendering |
| [[gnm-head|GNM Head]] | v3_0 | Apache 2.0 | Parametric head model |
| Blender GNM Importer | add-on | ? | Viseme authoring |

## Audio preprocessing

| Tool | Version | License | Role |
|---|---|---|---|
| espeak-ng | latest | GPL | Grapheme-to-phoneme |
| CMU Pronouncing Dict | — | Public domain | Phoneme reference |
