# Setup & Environment

`setup.sh` and project configuration.

## Prerequisites

- Python 3.12
- `uv` package manager: `brew install uv`
- Git submodules initialized

## Setup Script

`setup.sh` runs the full installation:

```bash
chmod +x setup.sh
./setup.sh
```

**Steps:**
1. Create Python 3.12 virtual environment via `uv venv venv`
2. Install GNM shape package in editable mode: `uv pip install -e vendor/GNM/gnm/shape`
3. Install dependencies: `trimesh`, `numpy`, `matplotlib`, `onnxruntime`
4. Verify GNM import

## Running the server

```bash
./venv/bin/python src/server.py
```

Available at `http://localhost:8080/`

## Git submodules

- `vendor/GNM/` — Google GNM Head (github.com/google/GNM)
- `vendor/voca/` — TimoBolkart VOCA (github.com/TimoBolkart/voca)

Initialize with: `git submodule update --init --recursive`

## Project structure

```
├── setup.sh              # Environment bootstrap
├── README.md             # Project documentation
├── AGENTS.md             # LLM agent configuration
├── .gitignore            # Python, data, and OS ignores
├── .gitmodules           # Submodule references
├── src/                  # Python backend
│   ├── server.py         # HTTP API server
│   ├── gnm_sanity_check.py
│   ├── mind/             # LLM cognitive layer
│   ├── voice/            # TTS voice layer
│   ├── alignment/        # Forced alignment + viseme mapping
│   ├── animation/        # GNM animation engine
│   └── training/         # Neural regressor training
├── web/                  # Browser frontend
│   ├── index.html        # UI entry point
│   ├── style.css         # Styling
│   ├── renderer.js       # Three.js renderer
│   ├── audio_sync.js     # Audio playback
│   └── animation_controller.js  # Client anim logic
├── tools/                # Utility scripts
│   ├── export_basis.py   # Web buffer exporter
│   └── tune_visemes.py   # Viseme coefficient editor
├── data/                 # Runtime data
│   ├── viseme_table.json # Viseme coefficient table
│   └── web/              # Exported web buffers
├── voca/                 # VOCA submodule + reprojected data
│   ├── model/            # VOCA model / checkpoints
│   └── trainingdata/     # Raw VOCASET data
├── vendor/               # Git submodules
│   ├── GNM/              # Google GNM Head
│   └── voca/             # VOCA (TimoBolkart)
├── output/               # Generated outputs (OBJ, WAV, frames)
├── assets/               # Logos, images, screenshots
├── wiki/                 # Project wiki (this directory)
└── dev-docs/             # Immutable source documents
```
