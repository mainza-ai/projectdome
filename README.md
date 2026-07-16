<p align="center">
  <img src="assets/logo.png" alt="Project Dome Logo" width="120">
</p>

<p align="center">
  <img src="assets/screenshots/project-dome-0.png" alt="Project Dome Screenshot" width="720">
</p>

# Project Dome

**3D conversational avatar engine.** Takes text input, synthesises speech, and drives a parametric 3D head model with synchronised lip movements and affective expressions — all in a browser, all on local/ open-source infrastructure.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://python.org)
[![GNM](https://img.shields.io/badge/GNM-Head_Apache_2.0-green)](https://github.com/google/GNM)
[![Piper](https://img.shields.io/badge/TTS-Piper_MIT-blue)](https://github.com/rhasspy/piper)

---

## Table of Contents

- [How it works](#how-it-works)
- [Key Features](#key-features)
- [Status](#status)
- [Getting Started](#getting-started)
- [API Quick Reference](#api-quick-reference)
- [Authors & Attributes](#authors--attributes)
- [Project Structure](#project-structure)

---

## How it works

```
Text input
    │
    ▼
Piper TTS ────────────► 16 kHz audio
    │
    ▼
Wav2TextGrid ─────────► millisecond-accurate phoneme boundaries
    │
    ▼
Phoneme→Viseme map ──► 13-category mouth-shape timeline
    │
    ▼
GNM coefficient blend ─► 383-dim expression vector (speech + optional emotion CVAE)
    │
    ▼
Three.js (WebGL) ─────► 17,821-vertex mesh with Linear Blend Skinning, 60 fps
```

---

## Key Features

- **Piper TTS** — CPU-based neural speech synthesis, MIT license, single American English voice
- **Wav2TextGrid forced alignment** — Wav2Vec2-based phoneme segmentation with millisecond precision
- **13-viseme lookup** — rule-based phoneme-to-viseme reduction with linear interpolation between keyframes
- **GNM Head v3.0** — 17,821 vertices, 383 expression PCA components, 253 identity components, 4-joint skeleton with Linear Blend Skinning
- **ExpressionSampler CVAE** — 20 semantic emotion labels (HAPPY, SURPRISE, DISGUST, …) blended per-channel against speech coefficients
- **IdentitySampler CVAE** — gender (2) × ethnicity (4) conditioned identity generation, plus 10 PCA sliders for fine-tuning
- **Speaking style selector** — 12 VOCASET subject styles (requires trained model checkpoint)
- **Browser renderer** — Three.js WebGL with sparse CPU deformation, per-vertex body-part colours (skin, eyes, teeth, tongue), automatic eyelid blinking, and camera-tracked eye gaze
- **HTTP API** — `/api/speak`, `/api/speak/stream`, `/api/emotion`, `/api/identity`, `/api/blink`, `/api/chat`, `/api/health`
- **Local LLM mind layer** — `/api/chat` calls an OpenAI-compatible endpoint (e.g. OMLX, Ollama) to generate dialogue + emotion tags

---

## Status

| Component | Maturity | Notes |
|---|---|---|
| TTS → alignment → viseme | **Working (MVP)** | Single Piper voice; Wav2TextGrid model downloads on first run |
| Browser renderer | **Working (MVP)** | CPU deformation; WebGL (not WebGPU); vertex colours, not textures |
| Emotion CVAE blending | **Working** | Per-channel max blend; upper face = emotion, lower face = speech-dominant |
| Identity CVAE sampling | **Working** | Gender/ethnicity-conditioned; applied to identity basis on every frame |
| Speaking styles (0–11) | **Stub** | Selector exists; requires training a multi-style model checkpoint |
| Neural regressor (Path B) | **Trainable** | `SpeechToCoefficientsModel` + VOCASET reprojection pipeline; no pretrained weights shipped |
| Mind layer (/api/chat) | **Functional** | Requires a running OpenAI-compatible local LLM (OMLX, Ollama, etc.) |
| Viseme mapping | **Rule-based** | 13 categories, hand-tuned coefficients for GNM expression basis; no coarticulation model |
| Multi-voice TTS | **Not implemented** | Single `en_US-lessac-medium` voice only |
| Streaming audio | **Working** | `/api/speak/stream` splits on sentence boundaries |

---

## Getting Started

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/), ~4 GB disk for model downloads.

```bash
# Install uv (macOS)
brew install uv

# Clone + setup
git clone https://github.com/mainza-ai/projectdome.git
cd projectdome
chmod +x setup.sh
./setup.sh

# Run
./venv/bin/python src/server.py

# Open
open http://localhost:8080/
```

The first startup downloads Piper TTS (~50 MB) and Wav2TextGrid (~1 GB) models. Subsequent starts use the cached files.

### VOCA Model & Training Data (Optional)

For **speaking style selection** (12 VOCASET styles) and **Path B neural training**, download from [https://voca.is.tue.mpg.de/download.php](https://voca.is.tue.mpg.de/download.php):

| File | Size | Place in | Purpose |
|------|------|----------|---------|
| `trained_model.zip` | ~12 MB | `voca/model/checkpoints/best_model.pt` | Pretrained VOCA speech-to-coefficients checkpoint. Enables the speaking-style dropdown (0–11) and serves as a baseline for Path B fine-tuning. |
| `training_data.zip` | ~8 GB | `voca/trainingdata/` | VOCASET: 12 subjects × 40 English sentences each, with FLAME-topology meshes, audio, and phoneme annotations. Required for Path B training (`src/training/run_pipeline.py`). |

After downloading:

```bash
# Trained model
unzip trained_model.zip -d voca/model/checkpoints/
mv voca/model/checkpoints/best_model.pt voca/model/checkpoints/  # adjust path if nested

# Training data
unzip training_data.zip -d voca/trainingdata/
```

The server detects `voca/model/checkpoints/best_model.pt` on startup and enables style-aware synthesis automatically. Without it, speech synthesis uses Path A (hand-tuned viseme lookup) with neutral style.

---

## API Quick Reference

| Endpoint | Method | Input | Returns |
|---|---|---|---|
| `/api/speak` | POST | `{text, emotion?, intensity?, style_id?}` | WAV audio (base64) + viseme timeline |
| `/api/speak/stream` | POST | `{text, …}` | Same, with per-sentence chunks |
| `/api/emotion` | POST | `{name, intensity?}` | 383-dim GNM expression coefficients |
| `/api/identity` | POST | `{gender, ethnicity}` | 253-dim identity coefficients |
| `/api/identity/info` | POST | `{n?}` | Identity dimension metadata |
| `/api/blink` | POST | `{}` | 383-dim blink coefficients (pre-cached) |
| `/api/chat` | POST | `{text}` | Response text + emotion + audio + visemes |
| `/api/chat/reset` | POST | `{}` | Clears conversation history |
| `/api/health` | POST | `{}` | Pipeline status |

---

## Authors & Attributes

- **Author:** Mainza Kangombe ([LinkedIn](https://www.linkedin.com/in/mainza-kangombe-6214295))
- **3D Head Model:** Google [GNM Head](https://github.com/google/GNM) — Apache 2.0
- **Speech-Driven Animation Reference:** [VOCA](https://github.com/TimoBolkart/voca) (Timo Bolkart et al., CVPR 2019) — research-only license for VOCASET data
- **TTS:** [Piper](https://github.com/rhasspy/piper) — MIT license
- **Forced Alignment:** [Wav2TextGrid](https://github.com/pkadambi/Wav2TextGrid) — MIT license
- **Browser 3D:** [Three.js](https://threejs.org) — MIT license

---

## Project Structure

<details>
<summary>Click to expand directory tree</summary>

```
├── setup.sh                 # Environment bootstrap (venv, submodules, dependencies, web buffers)
├── src/
│   ├── server.py            # HTTP server: static files + 9 API endpoints
│   ├── gnm_sanity_check.py  # GNM model loading and displacement verification
│   ├── mind/                # LLM cognitive layer (provider protocol, LocalMindProvider, ConversationContext)
│   ├── voice/               # Piper TTS wrapper (provider protocol, auto-download, LRU cache)
│   ├── alignment/           # Wav2TextGrid forced alignment, 13-viseme phoneme mapper, acoustic pipeline
│   ├── animation/           # Viseme table, interpolator, EmotionBlender, GNMDriver, runtime loop
│   └── training/            # VOCASET reprojection, SpeechToCoefficientsModel, dataset, training, evaluation, config, feature extraction
├── web/
│   ├── index.html           # Application shell with control panel
│   ├── style.css            # Apple-style light theme with Google colour accents
│   ├── renderer.js          # Three.js scene setup, buffer loading, sparse deformation, LBS, render loop
│   ├── animation_controller.js  # Client-side viseme interpolation + emotion blending
│   └── audio_sync.js        # Web Audio API playback with timeline sync
├── tools/
│   ├── export_basis.py      # Exports GNM model data → binary buffers for web client
│   └── tune_visemes.py      # Interactive CLI viseme coefficient editor
├── tests/
│   ├── run_all.py           # Test runner
│   ├── test_viseme_mapper.py    # Phoneme-to-viseme mapping logic
│   ├── test_viseme_table.py     # Viseme coefficient table
│   ├── test_emotion_blender.py  # Emotion coefficient blending
│   ├── test_interpolator.py     # Timeline interpolation
│   ├── test_dataset_split.py    # VOCA-compatible train/val/test split
│   ├── test_gnm_forward.py      # GNM export regression tests
│   └── test_server_endpoints.py # HTTP API integration tests
├── data/
│   ├── viseme_table.json    # GNM expression coefficients per viseme (182-dim)
│   └── web/                 # Exported binary buffers (generated by export_basis.py)
├── voca/                    # VOCASET reprojected data + model checkpoints (user-generated)
├── vendor/
│   ├── GNM/                 # Google GNM Head submodule (git)
│   └── voca/                # TimoBolkart VOCA submodule (git)
├── wiki/                    # Project knowledge base (markdown)
└── configs/                 # Training configuration YAML files
```
</details>
