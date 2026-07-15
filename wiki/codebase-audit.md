# Codebase Audit — Gaps, Issues & Production Plan

Comprehensive audit after implementing all production audit phases. Covers remaining gaps, broken code paths, missing functionality, and actionable improvements.

---

## Current State (Resolved)

| Item | Status |
|---|---|
| Keras "not compiled" warnings | **Fixed** — `_SilentExpressionSampler`, `_SilentIdentitySampler` with `compile=False` + absl/TF logger suppression |
| `log_request()` crash on GET | **Fixed** — base class compatibility restored |
| Speaker-specific data split | **Fixed** — VOCA-compatible 8/2/2 split |
| Edge loss in training | **Fixed** — added to loss function |
| Speaker ID in reprojected data | **Fixed** — embedded in NPZ files |
| Tongue animation coefficients | **Fixed** — 8 visemes with tongue dimension coefficients |
| Per-vertex body-part materials | **Fixed** — vertex colors in web renderer |
| Independent eye gaze | **Fixed** — look-at-target tracking |
| Natural blink timing | **Fixed** — jittered 1.2–7s interval |
| Identity PCA sliders | **Fixed** — 10-component bank in UI |
| Streaming audio | **Fixed** — `/api/speak/stream` with sentence chunking |
| Eager import chains | **Fixed** — `__init__.py` files cleared of eager imports |
| All unit tests | **Fixed** — 25 tests passing |
| README screenshot | **Fixed** — added |

---

## Remaining Gaps

### Gap 1: Export pipeline not integrated into setup

**Issue:** `tools/export_basis.py` must be run manually to generate `data/web/*.bin` buffers. If a new developer runs `setup.sh` + `src/server.py` without running export first, the web client hangs at "Loading GNM Basis Buffers..." with no meaningful error.

**Files:** `setup.sh`, `web/renderer.js`, `tools/export_basis.py`

**Fix:**
```bash
# Add to end of setup.sh
echo "=== Step 7: Exporting web buffers ==="
./venv/bin/python tools/export_basis.py
```

**Also:** Add a meaningful error message in `web/renderer.js` when buffer loading fails — currently just logs to console with no visual feedback beyond the loading overlay.

### Gap 2: No request timeout on synthesis endpoints

**Issue:** `/api/speak` and `/api/speak/stream` can hang indefinitely if Piper TTS or Wav2TextGrid blocks. The HTTP server runs single-threaded, so a hung request blocks all subsequent requests.

**Files:** `src/server.py`

**Fix:** Use `signal.alarm()` or a threading timeout wrapper:
```python
import signal
class TimeoutError(Exception): pass
def handler(signum, frame): raise TimeoutError()
signal.signal(signal.SIGALRM, handler)
signal.alarm(30)  # 30 second timeout
try:
    audio, sr, visemes = pipeline.process(text)
finally:
    signal.alarm(0)
```

### Gap 3: No asset generation in CI/onboarding

**Issue:** The `data/web/` directory is gitignored. New clones have no binary buffers. There's no script to regenerate them, no CI check that they exist, and no fallback if they're missing.

**Files:** `.gitignore`, `setup.sh`, `tools/export_basis.py`

**Fix:**
1. Add `tools/export_basis.py` call to `setup.sh`
2. Add a `make web-buffers` target or similar
3. Consider tracking a small test buffer to validate format

### Gap 4: Wav2TextGrid model cache not validated

**Issue:** `Wav2TextGridAligner.__init__()` downloads a ~1GB pretrained model to `pretrained_models/` on first run. If the download is interrupted, the half-downloaded directory exists but the model is unusable. There is no validation.

**Files:** `src/alignment/mfa_aligner.py`

**Fix:** Add download validation (checksum or model load test) with retry logic.

### Gap 5: Animation `runtime_loop.py` is standalone only

**Issue:** `animate_utterance()` in `runtime_loop.py` is a standalone CLI script. It's not integrated into the server or the training pipeline. It generates OBJ frames offline but there's no way to:
- Preview animation frames in the browser without exporting
- Export animations from the server API
- Use the runtime loop as part of the training validation pipeline

**Files:** `src/animation/runtime_loop.py`, `src/server.py`

**Fix:** Expose as `/api/animate` endpoint on the server:
```python
# POST /api/animate -> returns {frames: [base64_objs]}
```

### Gap 6: Training scripts assume data exists

**Issue:** `train.py` crashes with `FileNotFoundError` if reprojected data doesn't exist. But `reproject_vocaset.py` crashes if VOCASET training files don't exist. There's no end-to-end setup script that:
1. Checks for VOCASET data
2. Runs reprojection if needed
3. Runs training

**Files:** `src/training/reproject_vocaset.py`, `src/training/train.py`

**Fix:** Create a `src/training/run_pipeline.py` orchestrator:
```python
def run_pipeline():
    if not has_reprojected_data():
        reproject_vocaset.main()
    if not has_trained_model():
        train()
```

### Gap 7: Mind layer is unused in production

**Issue:** `src/mind/conversation.py` and `src/mind/local_provider.py` implement the LLM cognitive layer, but they are not wired into `server.py`. The server only handles TTS → alignment → animation, with no LLM generating responses. The user must type text manually.

**Files:** `src/server.py`, `src/mind/`

**Fix:** Add an `/api/chat` endpoint that:
1. Accepts user text
2. Calls `LocalMindProvider.generate()` 
3. Passes response text to the acoustic pipeline
4. Returns synthesized audio + visemes + response text

```python
elif self.path == "/api/chat":
    user_text = data.get("text", "")
    response = mind_provider.generate(user_text, conversation.get_history())
    conversation.add_user_message(user_text)
    conversation.add_assistant_message(response.text)
    audio, sr, visemes = pipeline.process(response.text)
    # encode and return
```

### Gap 8: Evaluate.py missing speaker_ids in model forward pass

**Issue:** In `evaluate.py`, the evaluation loop calls `model(features, src_key_padding_mask=padding_mask)` without `speaker_ids`. The model was trained with speaker conditioning, so evaluation without it produces suboptimal results.

**File:** `src/training/evaluate.py:47`

```python
# Current (missing speaker conditioning):
preds = model(features, src_key_padding_mask=padding_mask)
# Fix:
preds = model(features, speaker_ids=speaker_ids, src_key_padding_mask=padding_mask)
```

### Gap 9: Reprojection script error handling

**Issue:** `reproject_vocaset.py` has no partial progress saving. If it crashes mid-way through processing 480+ files, it must restart from scratch. Processing all 12 speakers takes significant time.

**File:** `src/training/reproject_vocaset.py`

**Fix:** Check for existing output files and skip already-reprojected sentences:
```python
for sentence in sentences:
    out_file = os.path.join(out_dir, f"{speaker}_{sentence}.npz")
    if os.path.exists(out_file):
        continue  # skip already reprojected
```

### Gap 10: No JavaScript tests

**Issue:** The web client (`web/renderer.js`, `web/animation_controller.js`, `web/audio_sync.js`) has zero tests. The deformation math, viseme interpolation, and audio sync logic are all untested.

**Files:** `web/*.js`

**Fix:** Add Jest/Puppeteer tests for:
- `AnimationController.getSpeechCoefficients()` edge cases
- `AnimationController.blend()` anatomical correctness
- `AudioSync` timing accuracy
- `deformMesh()` with known coefficient vectors

---

## Missing Features (Not Yet Implemented)

### Feature 1: Multi-subject speaking style training

**Status:** Architecture is in place (speaker embedding, style selector UI, style_id in API) but the model has NOT been trained with proper style conditioning. The current checkpoint (if any) was trained with the old broken data split.

**Action:** Re-run training with the new VOCA-compatible split and validate that style 0 vs style 11 produce different animations.

### Feature 2: Pose correctives in web renderer

**Status:** `export_basis.py` attempts to export pose correctives, but the GNM v3.0 HEAD model may not include them. The web renderer doesn't apply pose correctives during LBS.

**Action:** Verify if pose correctives exist in the GNM model data. If not, the production LBS will have skinning artifacts at extreme angles.

### Feature 3: GPU-accelerated deformation (WebGPU/TSL)

**Status:** Mesh deformation runs on CPU in JS. For 17,821 vertices × 383 expression dims, this is slow — especially on mobile. The architecture doc recommends WebGPU + TSL DataTextures.

**Action:** Port `deformMesh()` to a WebGPU compute shader using Three.js TSL.

### Feature 4: Expression blending beyond simple additive

**Status:** `EmotionBlender.blend()` uses simple additive blending (speech + 0.3× emotion). GNM supports expression coefficient interpolation and CVAE latent-space blending.

**Action:** Implement CVAE latent-space expression blending for more natural emotion mixtures.

### Feature 5: Texture mapping

**Status:** GNM provides an edgeflow texture map (`data/textures/edgeflow_bw_4k.png`). The web renderer uses flat vertex colors instead of UV texture mapping.

**Action:** Export UV coordinates from GNM, apply texture in Three.js.

---

## Implementation Plan

### Phase A — Critical (Next Sprint)

| # | Task | Effort | Files |
|---|---|---|---|
| A1 | Add buffer export to setup.sh | 15min | `setup.sh` |
| A2 | Add request timeout to synthesis endpoints | 1hr | `src/server.py` |
| A3 | Fix evaluate.py — add speaker_ids to forward pass | 15min | `src/training/evaluate.py` |
| A4 | Add reprojection skip-if-exists check | 30min | `src/reproject_vocaset.py` |
| A5 | Wire mind layer into server as /api/chat | 4hr | `src/server.py`, `src/mind/` |
| A6 | Add web client error message for missing buffers | 30min | `web/renderer.js` |
| A7 | Retrain model with VOCA-compatible split | 8hr | `src/training/train.py` |

### Phase B — Quality (Next Sprint)

| # | Task | Effort | Files |
|---|---|---|---|
| B1 | Add JS unit tests (Jest) | 4hr | `web/*.js`, `tests/` |
| B2 | Add server integration tests | 4hr | `tests/test_server.py` |
| B3 | Add partial progress saving to reprojection | 2hr | `src/reproject_vocaset.py` |
| B4 | Add Wav2TextGrid download validation | 1hr | `src/alignment/mfa_aligner.py` |
| B5 | Create training pipeline orchestrator | 2hr | `src/training/run_pipeline.py` |
| B6 | Add structured logging to training scripts | 2hr | `src/training/*.py` |

### Phase C — Performance (Future)

| # | Task | Effort | Files |
|---|---|---|---|
| C1 | WebGPU compute shader deformation | 40hr | `web/renderer.js` |
| C2 | Texture mapping from GNM UVs | 8hr | `web/renderer.js`, `tools/export_basis.py` |
| C3 | CVAE latent-space expression blending | 8hr | `src/animation/emotion_blender.py` |
| C4 | Pose corrective validation + LBS integration | 4hr | `web/renderer.js` |

---

## Summary of All Issues Found

| Category | Count | Key Examples |
|---|---|---|
| Fixed in this session | 12 | Keras warnings, log_request crash, data split, eager imports, broken tests |
| Critical remaining | 7 | Buffer export not in setup, no request timeout, mind layer unwired, evaluate.py missing speaker_ids, no JS tests, no partial reprojection, Wav2TextGrid no download validation |
| Performance | 4 | CPU deformation, no texture mapping, additive-only blending, pose correctives |
| Testing | 2 | No JS tests, no server integration tests |
