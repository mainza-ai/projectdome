# Production Audit & Integration Plan

A comprehensive gap analysis comparing Project Dome's current implementation against the full capabilities of [google/GNM](https://github.com/google/GNM) and [TimoBolkart/voca](https://github.com/TimoBolkart/voca), with a phased remediation plan for production readiness.

---

## Part 1: GNM Integration Gaps

google/GNM provides far more than what Project Dome currently uses. Below is a feature-by-feature audit.

### 1.1 Multi-Framework Backends

| GNM Feature | Status | Notes |
|---|---|---|
| NumPy backend (primary) | **Integrated** | Used in `GNMDriver`, `reproject_vocaset.py`, `gnm_sanity_check.py` |
| PyTorch backend | **Available, unused** | `gnm_pytorch.py` exists but not used — we have our own `SpeechToCoefficientsModel` |
| JAX backend | **Not integrated** | `gnm_jax.py` — could accelerate training |
| TensorFlow backend | **Not integrated** | `gnm_tensorflow.py` — GNM's ExpressionSampler/IdentitySampler use TF internally via `.h5` |

**Action:** Evaluate PyTorch GNM backend for direct GPU-based mesh deformation in training pipeline, eliminating numpy bridge.

### 1.2 Full Expression Parameter Space

| Parameter Group | Indices | Status |
|---|---|---|
| Left eye region | 0–99 | **Partially used** — blink only (CVAE), no fine-grained eye control |
| Right eye region | 100–199 | **Partially used** — blink only |
| Lower face region | 200–349 | **Integrated** — speech viseme mapping |
| Tongue | 350–381 | **Partially used** — basic coefficients, no tongue animation |
| Pupils | 382 | **Not integrated** — pupil dilation/constriction |

**Gap:** We only actively drive lower face + tongue. The 200 eye region parameters and pupil control are barely used beyond blink pre-caching.

**Action:**
- Map additional GNM expression channels to semantic controls (eyebrow raises, nostril flares, cheek raises)
- Implement pupil dilation response to emotion/lighting
- Add tongue animation during speech (currently has placeholder coefficients)

### 1.3 Identity Sampling (CVAE)

| GNM Feature | Status |
|---|---|
| `IdentitySampler` with Gender (Female, Male) | **Partially integrated** — `/api/identity` endpoint exists, UI dropdowns for gender/ethnicity |
| `IdentitySampler` with Ethnicity (Middle Eastern, Asian, White, Black) | **Partially integrated** — endpoint exists |
| 253 identity components (170 head, 3 eyeball, 80 teeth) | **Partially integrated** — coefficients are generated but identity basis is loaded in browser |
| Identity blending / interpolation | **Not integrated** — no smooth transitions between identities |
| Identity component PCA visualization | **Not integrated** — no UI for individual PCA slider manipulation |

**Action:**
- Add identity PCA slider bank in UI for fine-grained character sculpting
- Implement smooth identity morphing (lerp between identity vectors)
- Persist identity selection across sessions

### 1.4 Joint Kinematics & Linear Blend Skinning

| GNM Feature | Status |
|---|---|
| 4-joint skeleton (Neck, Head, Left Eye, Right Eye) | **Partially integrated** — skinning weights + joint regressor exported, LBS implemented in renderer.js |
| Axis-angle joint rotations | **Partially integrated** — neck yaw/pitch and eye gaze sliders exist |
| Pose correctives (blendshapes correcting LBS artifacts at extreme angles) | **Not integrated** — not exported from GNM, not applied in shader |
| Joint hierarchy with forward kinematics | **Partially integrated** — FK chain implemented with Matrix4 transforms |
| Template joint positions | **Partially integrated** — exported in metadata.json |

**Gap:** Pose correctives are completely missing. At extreme head rotations (e.g., looking down 20°), skinning artifacts will appear around the neck and eyes. GNM provides pose-corrective blendshapes specifically to fix this.

**Action:**
- Export `pose_correctives` from GNM basis
- Implement pose-corrective blendshape application in renderer's deformation loop
- Validate joint range limits against GNM specification

### 1.5 Facial Landmarks

| GNM Feature | Status |
|---|---|
| 68-point sparse landmark definition (`head_sparse_68.txt`) | **Not integrated** |
| Landmark regression from mesh vertices | **Not integrated** — `gnm_landmarks.py` provides this |

**Action:** Integrate landmark regression for:
- Eye gaze target validation
- Viseme accuracy measurement
- Future ARKit-style blendshape extraction pipeline

### 1.6 Visualization & Rendering Pipeline

| GNM Feature | Status |
|---|---|
| `render_gnm.py` — high-quality pyrender-based rendering with textures | **Not integrated** — irrelevant for web target |
| `vertex_colors.py` — per-vertex color mapping (skin, scleras, teeth, tongue) | **Not integrated** |
| Texture support (edgeflow texture map at `data/textures/edgeflow_bw_4k.png`) | **Not integrated** |
| Camera projection utilities (`camera_conversions.py`) | **Not integrated** |

**Action:** Although the browser renderer is the primary target, the Python visualization tools are valuable for:
- Offline quality validation of training output
- Generating reference renders for comparison
- Texture baking for the web client

### 1.7 GNM Model Anatomy — Separate Mesh Parts

| GNM Feature | Status |
|---|---|
| Skin mesh (primary head geometry) | **Integrated** — 17,821 vertices rendered |
| Eyeball geometry (left + right scleras + irises) | **Not rendered** — part of the mesh but not visually separated |
| Teeth geometry | **Not rendered** — part of the mesh but not visually separated |
| Tongue geometry | **Not rendered** — part of the mesh, only coefficients exist |
| Mouth sock/interior geometry | **Not rendered** |

**Gap:** The GNM model includes internal mouth geometry, teeth, eyeballs, and tongue as distinct vertex groups. The current renderer treats the entire 17,821-vertex mesh as a single surface with a uniform material. Without separate materials/shading for teeth, eyes, and tongue, the avatar will look like a featureless clay bust.

**Action:**
- Use `vertex_colors.py` or GNM body part labels to segment the mesh
- Apply different materials: glossy for eyes, enamel for teeth, diffuse for skin
- Implement transparency for eye scleras over irises
- Add separate eyelid geometry if present (may be part of eye region parameters)

---

## Part 2: VOCA Integration Gaps

### 2.1 Audio Feature Encoder

| VOCA Feature | Status |
|---|---|
| DeepSpeech features (Mozilla DeepSpeech v0.1.0, 29-dim per frame) | **Not used** — Project Dome uses 80-dim log-mel-spectrograms |
| SpeechEncoder CNN architecture (4 conv layers with temporal striding) | **Not used** — Project Dome uses a Linear projection + Transformer |
| Conditioned speech features (concatenating subject style with audio features) | **Partially integrated** — speaker embedding is added, not concatenated |

**Gap:** VOCA's DeepSpeech features are trained on a massive speech corpus and capture phonetic content robustly. Our log-mel approach is simpler but may not generalize as well to noisy environments or diverse voices.

**Action:**
- Benchmark log-mel vs DeepSpeech vs HuBERT features for GNM coefficient prediction accuracy
- Consider adding a HuBERT feature extraction path (as recommended in the research doc)
- Implement VOCA-style speech feature conditioning (concat, not add)

### 2.2 Subject Speaking Style Conditioning

| VOCA Feature | Status |
|---|---|
| 12-subject one-hot style vector | **Partially integrated** — `SpeechToCoefficientsModel` has `num_speakers=12` embedding |
| `condition_idx` exposed at inference | **Partially integrated** — `/api/speak` accepts `style_id`, UI has style dropdown (0–11) |
| Style-specific articulation amplitude, jaw limits, pacing | **Not validated** — model must be trained with proper style conditioning to learn these differences |

**Gap:** The current training pipeline (`train.py`) passes `speaker_ids` to the model, but:
1. The VOCASET reprojected data may not properly preserve speaker labels for all files
2. The UI style selector exists but the trained model weights may not have learned distinct speaking styles
3. No validation that style 0 and style 11 produce visibly different animations

**Action:**
- Verify speaker ID consistency across all reprojected NPZ files
- Train a multi-style model and validate style separation
- Add style interpolation (blend between two speaking styles)
- Document which VOCASET subjects correspond to which style indices

### 2.3 Training Loss Functions

| VOCA Loss | Project Dome Equivalent | Status |
|---|---|---|
| Reconstruction loss (L1 vertex positions) | L1 coefficient loss | **Integrated** |
| Velocity loss (temporal smoothness) | `compute_velocity_loss()` | **Integrated** |
| Acceleration loss (jitter reduction) | `compute_acceleration_loss()` | **Integrated** |
| Edge loss (edge length preservation) | **Not implemented** | Missing |
| Vertex weight masking | **Not implemented** | Missing |
| Wing loss (robust L1) | **Not implemented** | Missing |

**Gap:** Although acceleration loss code exists in `train.py`, it may not be properly weighted. VOCA's edge loss preserves mesh topology during deformation — without it, the model can learn to produce non-smooth surfaces. Vertex weight masking lets the model focus on important regions (mouth) vs. less important ones (forehead).

**Action:**
- Add edge loss to training (compute edge length differences between predicted and target meshes)
- Implement vertex-weighted loss (higher weight on mouth/lower face vertices vs. upper face)
- Tune loss weighting: position (1.0), velocity (0.5), acceleration (0.2), edge (0.1), sparsity (1e-4)
- Validate that acceleration loss is actually reducing jitter (compare with/without)

### 2.4 VOCA Model Architecture

| VOCA Component | Project Dome Equivalent | Status |
|---|---|---|
| DeepSpeech → Conv1D → FC → GRU | `Linear` → `TransformerEncoder` | Different architecture |
| Subject conditioning (concatenation) | Speaker embedding (addition) | Different approach |
| Output: vertex offsets (15069-dim) | Output: GNM coefficients (182-dim) | Different output space |
| Per-vertex decoder (mapping features to 5023 vertices) | Linear head (mapping features to 182 coeffs) | Different approach |

**Gap:** VOCA decodes speech features directly to per-vertex offsets (FLAME topology). Project Dome decodes to GNM expression coefficients, which is a more compact representation but relies on GNM's PCA basis quality. The advantage of the coefficient approach is that it guarantees valid face shapes (within PCA space). The disadvantage is that it can never represent details outside the GNM basis.

**Action:**
- No architecture change needed — the GNM coefficient approach is architecturally superior for our target
- But validate that the 182-dim coefficient space captures enough detail for photorealistic speech
- Consider residual vertex offset prediction on top of GNM coefficients for fine detail

### 2.5 VOCA Data Preprocessing Pipeline

| VOCA Step | Project Dome Equivalent | Status |
|---|---|---|
| FLAME fitting to raw 4D scans | VOCASET pre-processed data | **Relies on pre-processed data** |
| DeepSpeech feature extraction (from raw audio) | Log-mel-spectrogram extraction | **Reimplemented in dataset.py** |
| Vertex normalization (translation/rotation alignment) | ICP alignment in `reproject_vocaset.py` | **Integrated** |
| Train/val/test speaker split | 80/10/10 random split | **Different approach** — VOCA uses subject-specific splits |

**Gap:** VOCA uses a *subject-specific* train/val/test split (8 train + 2 val + 2 test subjects). Project Dome uses a random 80/10/10 split across all files, which means the same speaker's data may appear in both train and test sets. This invalidates speaker generalization metrics.

**Action:**
- Re-implement VOCA-style speaker-specific data split
- Ensure test subjects are completely unseen during training (FaceTalk_170809_00138_TA, FaceTalk_170731_00024_TA)

---

## Part 3: Production Enhancement Plan

### Phase 1 — Core Architecture Hardening (Current Sprint)

| # | Task | Files Affected | Priority |
|---|---|---|---|
| 1.1 | Implement speaker-specific data split (VOCA-compatible) | `src/training/dataset.py` | Critical |
| 1.2 | Add edge loss to training pipeline | `src/training/train.py` | High |
| 1.3 | Tune loss weights — validate accel/vel losses | `src/training/train.py` | High |
| 1.4 | Fix subject style consistency in reprojected data | `src/training/reproject_vocaset.py` | Critical |
| 1.5 | Train multi-style model, validate style separation | `src/training/train.py` | High |

### Phase 2 — Full GNM Expression Control

| # | Task | Files Affected | Priority |
|---|---|---|---|
| 2.1 | Add emotion-driven eyebrow and cheek expressions | `src/animation/emotion_blender.py` | High |
| 2.2 | Implement pupil dilation (emotion/light response) | `web/renderer.js`, `web/animation_controller.js` | Medium |
| 2.3 | Add tongue animation synchronized with viseme timeline | `src/animation/viseme_table.py`, `data/viseme_table.json` | Medium |
| 2.4 | Implement per-vertex body-part materials (eyes, teeth, skin, tongue) | `web/renderer.js`, `tools/export_basis.py` | High |
| 2.5 | Export and apply GNM texture map (edgeflow) | `tools/export_basis.py`, `web/renderer.js` | Medium |

### Phase 3 — Rig & Kinematics Production

| # | Task | Files Affected | Priority |
|---|---|---|---|
| 3.1 | Export pose correctives from GNM and implement in web renderer | `tools/export_basis.py`, `web/renderer.js` | High |
| 3.2 | Add head rotation damping (inertia) for natural motion | `web/animation_controller.js` | Medium |
| 3.3 | Implement independent left/right eye gaze with look-at target | `web/renderer.js` | High |
| 3.4 | Add eyelid blink with natural timing distribution | `web/renderer.js` (already partial) | Medium |
| 3.5 | Implement micro-saccades (subtle eye jitter) | `web/animation_controller.js` | Low |

### Phase 4 — Identity & Character System

| # | Task | Files Affected | Priority |
|---|---|---|---|
| 4.1 | Add identity PCA slider bank to UI | `web/index.html`, `web/renderer.js` | Medium |
| 4.2 | Implement smooth identity morphing transitions | `web/animation_controller.js` | Medium |
| 4.3 | Persist identity across sessions (localStorage) | `web/index.html`, `web/renderer.js` | Low |
| 4.4 | Add random character generator (combine gender + ethnicity + random PCA) | `src/server.py` | Low |

### Phase 5 — Performance & Latency

| # | Task | Files Affected | Priority |
|---|---|---|---|
| 5.1 | Implement streaming audio (chunked Piper synthesis) | `src/server.py`, `src/voice/piper_provider.py` | High |
| 5.2 | Chunked viseme delivery (timeline segments instead of full) | `src/server.py`, `web/audio_sync.js` | High |
| 5.3 | Server-sent events (SSE) for real-time push | `src/server.py` | Medium |
| 5.4 | WebAssembly-accelerated mesh deformation (optional) | `web/renderer.js` | Low |
| 5.5 | Basis buffer quantization (reduce download size) | `tools/export_basis.py` | Medium |

### Phase 6 — Testing & Validation

| # | Task | Files Affected | Priority |
|---|---|---|---|
| 6.1 | Unit tests for all animation components | `tests/` (new) | High |
| 6.2 | Integration test for full speech→viseme→render pipeline | `tests/` (new) | High |
| 6.3 | GNM model validation suite (displacement, joint range, expression range) | `tests/` (new) | Medium |
| 6.4 | Per-frame L1 coefficient error benchmark | `src/training/evaluate.py` | Medium |
| 6.5 | Subjective visual quality assessment (MOS) | External | Low |

---

## Part 4: Detailed Implementation Specifications

### 4.1 Speaker-Specific Data Split

Current (broken):
```python
rng = np.random.default_rng(42)
indices = np.arange(len(self.files))
rng.shuffle(indices)  # Shuffles across all speakers — data leakage!
```

Required (VOCA-compatible):
```python
# Fixed split from VOCA paper
TRAIN_SUBJECTS = [
    'FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA',
    'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA',
    'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA',
    'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA',
]
VAL_SUBJECTS = ['FaceTalk_170811_03275_TA', 'FaceTalk_170908_03277_TA']
TEST_SUBJECTS = ['FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA']
```

### 4.2 Edge Loss Implementation

```python
def compute_edge_loss(pred, target, edges):
    """L1 loss on edge lengths to preserve mesh topology.
    
    edges: (E, 2) tensor of vertex index pairs
    """
    pred_edges = pred[:, :, edges[:, 0]] - pred[:, :, edges[:, 1]]
    target_edges = target[:, :, edges[:, 0]] - target[:, :, edges[:, 1]]
    return nn.functional.l1_loss(pred_edges, target_edges)
```

### 4.3 Pose Corrective Application

Pose correctives are additional expression coefficients that depend on joint angles. In GNM:

```python
# GNM forward pass with pose correctives
pose_correctives = compute_pose_correctives(joint_rotations)
corrected_expression = expression_coeffs + pose_correctives
vertices = gnm(identity, corrected_expression, joint_rotations, translation)
```

In the browser renderer, this means:
1. Compute joint rotations from UI sliders / tracking
2. Evaluate pose corrective blendshapes based on joint angles
3. Add corrective coefficients to the expression vector before deformation

### 4.4 Streaming Audio Pipeline

```python
# Current: monolithic
def process(self, text):
    voice_result = self.voice.synthesize(text)  # blocks until full sentence done
    phonemes = self.aligner.align(voice_result.audio, ...)
    visemes = self.mapper.map_phonemes(phonemes)
    return audio, sr, visemes

# Target: streaming
def process_streaming(self, text):
    for sentence in split_sentences(text):
        yield self.process_sentence(sentence)  # emit audio chunks + viseme segments
```

---

## Part 5: Quick Reference — All Missing Features

### From GNM (8 gaps)

| Feature | Location in GNM | Effort |
|---|---|---|
| Pose correctives | `gnm.npz` basis data | 2 days |
| Eyeball geometry as separate mesh | Vertex group labels | 2 days |
| Teeth material/shaders | Vertex group labels | 1 day |
| Tongue animation coefficients | Lower face region | 1 day |
| 68-point landmarks + tracking | `gnm_landmarks.py` | 1 day |
| Pupil dilation (emotion response) | Expression param 382 | 0.5 day |
| PyTorch GNM backend | `gnm_pytorch.py` | 1 day |
| Texture mapping | `data/textures/edgeflow_bw_4k.png` | 2 days |

### From VOCA (6 gaps)

| Feature | Location in VOCA | Effort |
|---|---|---|
| Speaker-specific data split | `run_training.py` | 0.5 day |
| Edge loss | `utils/losses.py` | 0.5 day |
| Vertex weight masking | `utils/losses.py` | 0.5 day |
| DeepSpeech/HuBERT feature comparison | `utils/speech_encoder.py` | 3 days |
| Subject style validation | `run_training.py` | 1 day |
| Training config system | `config_parser.py` | 1 day |

### Infrastructure (8 gaps)

| Gap | Effort | Priority |
|---|---|---|
| Streaming audio delivery | 2 days | High |
| Streaming viseme timeline | 1 day | High |
| Separate mesh materials (skin/eyes/teeth) | 3 days | High |
| Loss function validation | 1 day | High |
| Unit tests for animation pipeline | 2 days | Medium |
| Speaker-specific data split fix | 0.5 day | Critical |
| Keras model loading warnings (ExpressionSampler not compiled) | 0.5 day | High |
| Server error handling & resource leak validation | 1 day | High |

## Part 6: Server Warnings & Stability

### 6.1 Keras Model Loading Warnings

Two warnings appear on every server start:

```
WARNING:absl:No training configuration found in the save file,
so the model was *not* compiled. Compile it manually.
```

**Root cause:** GNM's `ExpressionSampler` and `IdentitySampler` load Keras `.h5` model files (`expression_decoder_model.h5`, `identity_decoder_model.h5`). These files contain only the model architecture and weights, not the training configuration (optimizer state, loss function, metrics). Keras emits this warning because `load_model()` expects a full training config by default.

**Impact:** The models load and run inference correctly. The warning is cosmetic during inference — it only matters if you wanted to resume training.

**Fix options:**
1. **Suppress warning** (low effort): Use `tf.keras.models.load_model(filepath, compile=False)` — explicitly tells Keras not to expect training config
2. **Re-save with compile=True** (lower effort): Load the model once, call `model.compile()`, re-save — but this requires knowing the original loss/optimizer
3. **Patch semantic_sampler.py** (recommended): Modify GNM's `ExpressionSampler.__init__` to pass `compile=False`

```python
# In gnm.shape.semantic_sampler, replace:
self.model = tf.keras.models.load_model(model_path)
# With:
self.model = tf.keras.models.load_model(model_path, compile=False)
```

### 6.2 Server Error Boundaries

Current server has no:
- Request timeout handling (blocked TTS/alignment hangs the server forever)
- Memory leak detection (each `/api/speak` creates new tensors — are they freed?)
- Graceful degradation when subprocesses fail (Piper, Wav2TextGrid)
- Structured logging (all output is `print()`, no request IDs, no timing)

**Required for production:**
- `asyncio` timeout wrapper around long-running synthesis
- Tensor garbage collection after each request (`.detach()`, `gc.collect()`)
- Try/except around every subprocess call with meaningful fallback
- Replace `print()` with structured logging (JSON lines with timestamps + request IDs)
