# License Analysis

Critical for Project Dome's $0 budget and potential commercial future.

## Permissive (Apache 2.0 / MIT) — safe for any use

| Component | License |
|---|---|
| GNM Head | Apache 2.0 |
| Kokoro TTS | Apache 2.0 |
| Qwen3 | Apache 2.0 |
| gpt-oss-20b | Apache 2.0 |
| Ollama | MIT |
| Piper TTS | MIT |
| Wav2TextGrid | MIT |
| Three.js | MIT |

## Research-only — cannot use in commercial product

| Dataset | License |
|---|---|
| VOCASET | Non-commercial research |
| BIWI | Research use |
| Multiface | Meta research license |
| MEAD | Research use |

## Mitigation strategy

For commercial use, avoid training on research-only datasets. Instead:

1. **Path A** — no training needed at all (hand-authored viseme table)
2. **Synthetic dataset** — generate `(audio, GNM_coefficient)` pairs using Kokoro + Path A procedural pipeline → infinitely scalable, legally unencumbered
3. **Self-captured data** — record your own 4D facial data (you own the rights)
4. **Fine-tune** — start from synthetic foundation model, fine-tune on small proprietary dataset
