# Cognitive Layer (Mind)

The avatar's "mind" — processes user input, generates dialogue, and emits synchronized affect tags.

## Requirements

- Real-time or near-real-time inference
- Structured JSON output (dialogue + emotion tags)
- Runs concurrently with rendering and TTS on consumer hardware
- Zero per-token cost

## Primary recommendation: Qwen3 (8B or 14B)

- Apache 2.0 licensed
- Native structured JSON output
- 100+ language support
- Strong instruction adherence
- Runs on consumer GPUs or Apple Silicon unified memory

## Alternative models

| Model | Params | Notes |
|---|---|---|
| Qwen3 8B/14B | 8–14B | Best balance of capability and resource use |
| gpt-oss-20b | 20B | Apache 2.0, 128K context, needs 24GB+ VRAM |
| DeepSeek-V4-Flash | MoE | Very high capability, needs aggressive quantization |
| GLM-5.2 | 744B MoE (40B active) | State-of-the-art open-weight, impractical for concurrent use |

## Deployment: Ollama

[[tools#ollama|Ollama]] runs as a daemon exposing a local REST API with GPU/CPU memory offloading. The LLM interface is kept modular behind an abstraction layer so the model is swappable.

## Output format

The cognitive layer emits two parallel outputs:
1. **Dialogue text** → sent to TTS
2. **Affect tags** (e.g. `[HAPPY]`, `[SURPRISE]`) → fed to GNM's ExpressionSampler for non-speech facial expression

Both streams are synchronized so expressions align with spoken words.

## Implementation

See [[mind-llm|Mind LLM Engine]] for the actual implementation: provider protocol, LocalMindProvider (OpenAI-compatible API client), and ConversationContext (dialogue history).
