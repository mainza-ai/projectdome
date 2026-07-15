# HTTP Server & API

`src/server.py` — the local HTTP API server and static asset server. Serves the web client and exposes REST endpoints for speech synthesis, emotion sampling, identity generation, and blink pre-caching.

## Endpoints

### `GET /` — Static file server
Serves files from `web/`, `data/`, and `assets/` directories. Routes `/` to `index.html`.

### `POST /api/speak` — Speech synthesis
**Input:** `{ text, emotion?, intensity?, style_id? }`
**Output:** `{ audio_base64, visemes: [{name, start_time, end_time}] }`
**Pipeline:** AcousticPipeline → Piper TTS → Wav2TextGrid forced alignment → viseme mapping

### `POST /api/emotion` — Emotion coefficients
**Input:** `{ name, intensity? }`
**Output:** `{ coefficients: [383 floats] }`
**Source:** GNM ExpressionSampler CVAE

### `POST /api/identity` — Identity generation
**Input:** `{ gender: 0|1, ethnicity: 0-3 }`
**Output:** `{ coefficients: [253 floats] }`
**Source:** GNM IdentitySampler

### `POST /api/blink` — Blink coefficients
**Output:** `{ coefficients: [383 floats] }`
**Source:** Blended WINK_LEFT + WINK_RIGHT from ExpressionSampler

## Engine initialization

On startup, the server initializes:
- [[alignment-pipeline|AcousticPipeline]] (Piper + Wav2TextGrid)
- [[animation-engine#emotionblender|EmotionBlender]] (ExpressionSampler CVAE)
- [[gnm-head|IdentitySampler]]
- [[training-pipeline|SpeechToCoefficientsModel]] (if checkpoint exists at `voca/model/checkpoints/best_model.pt`)
