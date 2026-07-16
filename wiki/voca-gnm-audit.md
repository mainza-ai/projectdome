# VOCA vs GNM: Architecture Audit & Correct Integration Path

## Executive Summary

VOCA works correctly for speech-driven mouth animation because it outputs **per-vertex 3D displacements** added directly to the template mesh, completely bypassing any model parameterization. Our current code tries to drive speech through GNM's **expression PCA basis** — a space designed for subtle emotional expressions, not speech articulation. The PCA basis attenuates the signal by ~100× relative to what VOCA produces, making mouth movement imperceptible regardless of coefficient values.

## Source Files Read (22 total)

**VOCA repo** (`vendor/voca/`):
| File | Lines | Purpose |
|------|-------|---------|
| `run_voca.py` | 60 | CLI entry point |
| `compute_FLAME_params.py` | 120 | Offline FLAME parameter extraction from mesh sequences |
| `utils/inference.py` | 175 | Runtime inference: audio → DeepSpeech → model → vertex meshes |
| `utils/voca_model.py` | 308 | Full model definition: graph, training, evaluation, rendering |
| `utils/expression_layer.py` | 45 | **Critical**: FC layer outputting 3×V vertex offsets |
| `utils/speech_encoder.py` | 93 | 4-layer Conv1D + 2 FC layers, conditioned on speaker ID |
| `utils/ops.py` | — | FC, conv2d, BatchNorm building blocks |
| `utils/losses.py` | — | Reconstruction, velocity, acceleration losses |
| `utils/audio_handler.py` | — | DeepSpeech feature extraction from WAV |
| `utils/data_handler.py` | — | VOCASET data loading, batching |

**GNM repo** (`vendor/GNM/gnm/shape/`):
| File | Lines | Purpose |
|------|-------|---------|
| `gnm_xnp.py` | 738 | Backend-agnostic GNM model definition |
| `gnm_common.py` | 505 | Core math: PCA basis, LBS, forward kinematics, pose correctives |

**Project Dome** (`src/` + `web/`):
| File | Lines | Purpose |
|------|-------|---------|
| `training/model.py` | 74 | Current Path B model (coefficient output, wrong architecture) |
| `animation/gnm_driver.py` | 36 | GNM forward pass wrapper |
| `animation/emotion_blender.py` | 97 | CVAE emotion + speech blend |
| `animation/interpolator.py` | 60 | Viseme timeline interpolation |
| `animation/viseme_table.py` | 116 | Hand-tuned viseme coefficients |
| `web/renderer.js` | 536 | Three.js deformation, LBS, render loop |
| `web/animation_controller.js` | 128 | Client-side viseme + emotion blend |

---

## Finding 1: VOCA outputs raw vertex displacements, not model parameters

### VOCA's architecture (`voca_model.py` + `expression_layer.py`)

```
Audio → DeepSpeech (29-dim features, 16-frame windows)
     → SpeechEncoder (4× Conv1D stride-2 → FC → 128 → FC → 50)
     → ExpressionLayer (FC: 50 → 3×num_vertices)
     → output_decoder = expression_offset + input_template
```

The `ExpressionLayer` (`expression_layer.py:40-44`):
```python
exp_offset = fc_layer(parameters, num_units_in=50, num_units_out=3*num_vertices)
return tf.reshape(exp_offset, [-1, num_vertices, 3, 1])
```

**The model outputs 3×V values — raw per-vertex offsets in XYZ.** These are added to the template mesh vertices. There is NO intermediate model parameterization (no FLAME shape/pose/expression coefficients). The FLAME model is used ONLY for offline training data generation (`compute_FLAME_params.py`).

### Our Path B architecture (`training/model.py:63`)

```python
self.output_head = nn.Linear(hidden_dim, output_dim)  # output_dim = 182
```

Our model outputs **182 GNM expression PCA coefficients**. These coefficients are then multiplied by GNM's `expression_basis` (shape: 383×17821×3) to produce vertex displacements. This introduces a **bottleneck**: the PCA basis magnitude per unit coefficient is only 0.008.

**Impact:** VOCA directly predicts vertex positions at full precision. Our model predicts coefficients that get attenuated by the PCA basis → ~100× less displacement for the same numerical value.

---

## Finding 2: GNM's expression PCA basis is too weak for speech

Measured by applying each of the 383 expression components to the template and computing maximum vertex displacement:

| Region | Components | Max displacement per unit coefficient |
|--------|-----------|--------------------------------------|
| Eye region (0-199) | 200 | 0.003 |
| Lower face (200-349) | 150 | 0.008 |
| Tongue (350-381) | 32 | 0.012 |
| Pupil (382) | 1 | 0.001 |

For comparison, a **−5.7° head rotation** (neck/head joints) produces **0.018 units** of mouth-region displacement — more than any single expression component.

For VISIBLE jaw opening (≈0.05 units), we'd need expression coefficient values of 6-12. But the CVAE only outputs values in range [−2, 2], and our hand-tuned viseme table uses 2.5. Scaling coefficients to 25-50 risks non-anatomical PCA artifacts.

**VOCA avoids this entirely by not using PCA at all.**

---

## Finding 3: GNM has no jaw joint; FLAME has explicit jaw rotation

### FLAME skeleton
- Full head model with **jaw joint** (pose[6:9] = 3-axis rotation)
- VOCA's training data includes this jaw rotation as part of the vertex targets
- When VOCA predicts per-vertex offsets, jaw movement is embedded naturally

### GNM skeleton
```
Joint 0 (neck):  parent=-1  position=(0.000, 0.134, -0.007)
Joint 1 (head):  parent=0   position=(0.000, 0.210,  0.018)
Joint 2 (left_eye):  parent=1  position=(0.031, 0.303, 0.099)
Joint 3 (right_eye): parent=1  position=(-0.031, 0.303, 0.099)
```

**4 joints, no jaw joint.** The neck and head joints control the entire head as a rigid unit. There is no mechanism to rotate the jaw independently of the skull.

### The LBS system cannot produce mouth opening

GNM's `linear_blend_skinning` (`gnm_common.py:302-376`) applies joint rotations through forward kinematics. Since no joint controls the jaw, rotating any existing joint moves the entire head or neck rigidly. The 0.018 units of mouth displacement from head rotation is from the chin moving as the whole head tilts — not jaw opening.

**Pose correctives** (`pose_correctives_regressor`, shape 36×53463) are **all zeros** in GNM Head v3.0. Even if we had a jaw joint, there are no corrective blendshapes for skin deformation.

---

## Finding 4: The ExpressionLayer initialization reveals the bridge

VOCA's `ExpressionLayer.__init__` (`expression_layer.py:34-37`):
```python
init_exp_basis = np.zeros((3*num_vertices, expression_dim))
if self.init_expression:
    init_exp_basis[:, :min(expression_dim, 100)] = np.load(expression_basis_fname)
```

The FC layer's weight matrix is **initialized** from FLAME's expression basis (100 components × 3×V). This means:
- **Before training**: the model outputs FLAME-like vertex offsets
- **During training**: the weights are refined to produce speech-optimized offsets
- **After training**: the model has learned to produce correct speech animation directly

**This is the pattern we should follow** for Path B: initialize a linear layer with GNM's expression PCA basis, then train on (audio → vertex displacement) pairs.

---

## Finding 5: VOCA's training loss operates on vertices, not coefficients

VOCA loss terms (`voca_model.py:82-87`):
```python
self.rec_loss = reconstruction_loss(predicted=self.output_decoder, real=self.target_vertices)
self.velocity_loss = velocity_loss(predicted=..., real=...)
self.acceleration_loss = acceleration_loss(predicted=..., real=...)
```

All losses are computed on **vertex positions** (3×V). Our training pipeline (`src/training/`) should compute losses on vertices too, not on coefficients. The current reprojection pipeline (`tools/reproject_vocaset.py`) projects FLAME vertex sequences to GNM PCA coefficients via ICP + least-squares fitting, which inherently loses information.

---

## Finding 6: The correct separation of concerns

| Component | Mechanism | Drives | Should affect | Implementation |
|-----------|-----------|--------|---------------|----------------|
| **Speech model** | Per-vertex 3D offsets added to GNM template | Audio features | Lower face + jaw + tongue | `GNM_template + speech_offset` before LBS |
| **Emotion CVAE** | 383-dim PCA coefficients × expression_basis | Emotion label | Upper face (0-199) only | Applied through GNM's existing `vertex_positions_bind_pose` |
| **Identity CVAE** | 253-dim PCA coefficients × identity_basis | Gender/ethnicity | Permanent face shape | Applied through GNM's existing `vertex_positions_bind_pose` |
| **LBS** | Joint rotations (neck, head, eyes) | Head/gaze sliders | Head pose only | Applied through GNM's existing `vertex_positions_world` |
| **Blink** | Pre-cached blink coefficients | Timer | Eye region (0-199) | Added to emotion coefficients |

**No blend function is needed** because speech and emotion use different mechanisms:
- Speech modifies vertices directly (bypasses PCA entirely)
- Emotion uses PCA coefficients (affects only upper face)
- Both combine additively: `final_vertices = template + speech_offset + identity_delta + emotion_delta`

---

## Finding 7: What our current code does wrong

| Component | Current implementation | Correct implementation |
|-----------|----------------------|----------------------|
| Speech model output | 182 PCA coefficients → attenuated by basis | 3×V vertex offsets directly |
| Blend function | Binary switch between speech/emotion PCA coefficients | Not needed — speech and emotion use different mechanisms |
| Viseme coefficients | Hand-tuned, scaled to 10-20× emotion range | Learned end-to-end from data |
| Training target | PCA coefficients (projected from FLAME via least-squares) | Raw vertex positions (direct from FLAME) |
| Loss function | Coefficient MSE | Vertex reconstruction + velocity + acceleration (VOCA-style) |
| Jaw control | None (no jaw joint, PCA too weak) | Embedded in vertex offsets (speech model learns it) |
| Emotion-speech separation | Index-based channel split (0-199 vs 200-349) | Mechanism-based (PCA vs vertex offsets) |

---

## Correct Architecture: Path B Rewrite Plan

### Component 1: Speech-to-Vertex Model (replaces current `SpeechToCoefficientsModel`)

```
Audio → HuBERT features (80-dim)
     → ConvFrontend (4× Conv1D stride-2 → same as VOCA's)
     → FC → encoding_dim (50)
     → VertexLayer (Linear: 50 → 3×V, initialized with GNM expression basis)
     → Reshape → (T, V, 3) vertex offsets
```

Initialization of `VertexLayer.weight`: set to `gnm_common.expression_basis` reshaped to (E, 3×V) and transposed to (3×V, E), then take first `encoding_dim` columns. This mirrors VOCA's ExpressionLayer initialization.

### Component 2: Training Pipeline (replaces current `reproject_vocaset.py`)

Instead of:
```
FLAME vertex sequence → PCA project → (audio, GNM_coefficient) pairs → train coefficient model
```

Do:
```
FLAME vertex sequence → (audio, vertex_target) pairs → train vertex-offset model
```

Where `vertex_target = FLAME_vertices - GNM_template` (the per-vertex offset needed).

### Component 3: Runtime Integration (modifies web renderer)

```javascript
// 1. Speech model predicts vertex offsets (inference on server or in browser)
const speechOffset = await predictVertexOffset(audioFeatures);

// 2. Emotion from CVAE (unchanged, but only affects upper face)
const emotionCoeffs = await fetchEmotion("HAPPY", 1.0);
const emotionDelta = applyExpressionBasis(emotionCoeffs, 0.199); // upper face only

// 3. Identity (unchanged)
const identityDelta = applyIdentityBasis(identityCoeffs);

// 4. Combine and skin
const bindPose = template + speechOffset + identityDelta + emotionDelta;
const finalVerts = linearBlendSkinning(bindPose, joints, rotations);
```

**No `blend()` function.** Speech and emotion operate on different mechanisms and combine naturally through additive vertex displacement.

### Component 4: Remove the viseme lookup table entirely

`data/viseme_table.json`, `src/animation/viseme_table.py`, `web/animation_controller.js:getDefaultVisemeTable()` — all become unused once the neural model is trained. The viseme table is a Path A artifact that should be deprecated when Path B is operational.

---

## Summary of Required Code Changes

| File | Change | Priority |
|------|--------|----------|
| `src/training/model.py` | Replace `SpeechToCoefficientsModel` with `SpeechToVertexModel` (output 3×V, not 182) | P1 |
| `src/training/dataset.py` | Load vertex targets instead of PCA coefficient targets | P1 |
| `src/animation/gnm_driver.py` | Add `apply_vertex_offset()` method for direct vertex manipulation | P1 |
| `tools/reproject_vocaset.py` | Save vertex displacement targets instead of PCA coefficients | P1 |
| `web/renderer.js` | Remove blend function; apply speech offset, emotion PCA, identity PCA independently | P1 |
| `web/animation_controller.js` | Remove `blend()`, remove viseme table dependency | P1 |
| `data/viseme_table.json` | Deprecate (no longer needed) | P2 |
| `src/animation/viseme_table.py` | Deprecate | P2 |
| `src/animation/emotion_blender.py` | Limit emotion PCA to channels 0-199 only (upper face) | P0 |
