# Deep Dive Audit: GNM & VOCA Integration Gaps

Comprehensive analysis of the GNM and VOCA codebases vs. current Project Dome implementation. Every discrepancy, missing feature, and architectural gap identified through direct source-code analysis.

---

## Part 1: GNM Model — What We're Missing

Source: `vendor/GNM/gnm/shape/gnm_xnp.py`, `gnm_common.py`, `gnm_numpy.py`, `gnm_data_loader.py`

### 1.1 Pose Correctives (CRITICAL)

**GNM has:** `pose_correctives_regressor` — a matrix of shape `(9*J, 3*V)` that maps joint rotation differences from identity to per-vertex corrective offsets. Computed in `gnm_common.compute_pose_correctives()`.

**Current state:** We attempt to export pose_correctives.bin in `export_basis.py`, but they're checked via `hasattr(model, 'pose_correctives')` which looks for the wrong attribute name. The actual attribute is `pose_correctives_regressor` (a 2D matrix), not `pose_correctives` (a 3D array). We never load or apply pose correctives in the web renderer.

**Impact:** At any non-zero head rotation, the LBS produces inaccurate vertex positions around the neck, jaw, and eyes. The GNM forward pass (`__call__`) computes them unconditionally:

```python
# GNM's __call__ (gnm_xnp.py:347-349):
pose_correctives = self.compute_pose_correctives(rotations)
vertices = vertices + pose_correctives
```

**Fix:** Export the `pose_correctives_regressor` correctly (shape: `[36, 53463]`) and apply in the web renderer:
```javascript
const poseFeatures = new Float32Array(numJoints * 9);
for (let j = 0; j < numJoints; j++) {
    const R = axisAngleToRotationMatrix(rotations[j]);
    for (let k = 0; k < 9; k++) {
        poseFeatures[j * 9 + k] = R[k] - (k % 4 === 0 ? 1.0 : 0.0);
    }
}
// Apply: delta_v = sum_f poseFeatures[f] * correctivesRegressor[f, v*3 + coord]
```

### 1.2 Joint Identity Basis (HIGH)

**GNM has:** `joint_identity_basis` of shape `(I, J, 3)` — identity PCA coefficients affect joint positions, not just vertex positions. The full forward pass computes:
1. `joints_bind = template_joints + sum(identity[i] * joint_identity_basis[i])`
2. Then LBS uses these identity-dependent joint positions

**Current state:** Our `export_basis.py` does not export `joint_identity_basis`. The web renderer computes joints from the `joint_regressor` applied to deformed vertices, which is a different method (regressing joints from deformed vertices rather than applying identity to template joints).

**Impact:** When identity changes (gender/ethnicity), the joint positions don't change accordingly. The skeleton stays in the template position regardless of the face shape.

**Fix:** Export `joint_identity_basis` and apply identity to joints:
```javascript
const joints = templateJoints.slice();
for (let i = 0; i < identityDim; i++) {
    const w = identityCoeffs[i];
    if (Math.abs(w) > 1e-4) {
        const offset = i * numJoints * 3;
        for (let j = 0; j < numJoints * 3; j++) {
            joints[j] += jointIdentityBasis[offset + j] * w;
        }
    }
}
```

### 1.3 Vertex Groups for Mesh Parts (HIGH)

**GNM has:** `vertex_groups` of shape `(G, V)` with `vertex_group_names` — soft assignments of each vertex to anatomical parts (skin, left_eye, right_eye, teeth, tongue, mouth_sock, gums, etc.).

**Current state:** Our `export_basis.py` tries `model.vertex_body_parts` which doesn't exist. We fall back to a heuristic that produces meaningless data. The web renderer treats the entire 17,821-vertex mesh as a single surface with uniform material.

**Impact:** No visual distinction between skin, eyes, teeth, and tongue. The avatar looks like a clay bust.

**Fix:** Export vertex groups properly:
```python
# In export_basis.py (actual GNM attribute names):
vertex_groups = model.vertex_groups  # shape (G, V)
group_names = model.vertex_group_names  # list of strings

# Convert to per-vertex body part label (argmax)
body_parts = np.argmax(vertex_groups, axis=0).astype(np.int32)
# Save
body_parts.tofile(os.path.join(out_dir, "vertex_body_parts.bin"))
metadata["group_names"] = group_names
```

### 1.4 UV Texture Coordinates (MEDIUM)

**GNM has:** `quad_uvs` of shape `(Q, 4, 2)` and `triangle_uvs` of shape `(T, 3, 2)` for texture mapping. Also `vertex_uvs` (computed, approximate).

**Current state:** Not exported at all. The web renderer uses flat vertex colors (when body parts are available) or a uniform color.

**Impact:** Can't apply textures (skin detail, eye texture, teeth color). The edgeflow texture at `data/textures/edgeflow_bw_4k.png` exists but is unused.

**Fix:** Export `triangle_uvs` and apply texture in Three.js:
```javascript
const uvAttr = new THREE.BufferAttribute(uvs, 2);
geometry.setAttribute('uv', uvAttr);
```

### 1.5 Mirror Indices (LOW)

**GNM has:** `mirror_indices` — maps each vertex to its symmetric counterpart. Useful for ensuring symmetric deformation.

**Current state:** Not used.

### 1.6 LBS Implementation Differences (HIGH)

**GNM's LBS** (`gnm_common.linear_blend_skinning`):
```python
# Uses homogeneous coordinates and proper joint transform hierarchy:
joint_transforms = joint_transforms_world(joints, rotations, translation, parents)
deltas = T_world[..., :3, :3] @ joints
offset = concat([zeros_3x3, deltas], axis=-1)
joint_transforms = joint_transforms - offset
vertices_h = concat([vertices, ones], axis=-1)
skinned = einsum('jv,...jmn,...vn->...vm', weights, joint_transforms, vertices_h)
```

**Our JS LBS** uses a simplified `invBind` approach:
```javascript
const invBind = new THREE.Matrix4().makeTranslation(-joints[j].x, -joints[j].y, -joints[j].z);
const skinningMatrix = T_world[j].clone().multiply(invBind);
```

This works but doesn't handle rotation correctly — the `invBind` matrix should include rotation, not just translation. For the 4-joint GNM skeleton with small rotations this is approximately correct but not accurate.

**Fix:** Port GNM's exact LBS math to JS:
```javascript
// Build 4x4 joint transforms in world space
// Apply skinning: v_skinned = sum_j w_j * T_world_j * inv(T_bind_j) * v_bind
// Where inv(T_bind_j) accounts for both rotation and translation of the bind pose
```

---

## Part 2: VOCA — What We're Missing

Source: `vendor/voca/run_voca.py`, `utils/voca_model.py`, `utils/losses.py`, `utils/speech_encoder.py`

### 2.1 Audio Feature Pipeline (HIGH)

**VOCA uses:** A pretrained DeepSpeech model (Mozilla DeepSpeech v0.1.0) that produces 29-dim audio features at 50fps. Before feeding to the model, these are windowed into context windows of 16 frames (±8 frames around each timestep), producing a 464-dim input (16×29).

**We use:** 80-dim log-mel-spectrograms at 100fps (10ms hop), projected linearly to hidden_dim.

**Impact:** DeepSpeech features are phonetically-aware (trained on 5,000+ hours of speech-to-text). Mel-spectrograms are generic acoustic features. VOCA's windowed context approach captures temporal dynamics better than our per-frame linear projection.

**Fix:** Add a DeepSpeech or HuBERT feature extraction pipeline:
```python
# Option A: DeepSpeech (VOCA-compatible)
deepspeech = DeepSpeech(model_path='deepspeech-0.1.0-models/output_graph.pb')
features = deepspeech.extract_features(audio)  # (T, 29)
# Window: ±8 frames → 16*29 = 464 per timestep
windowed = [features[max(0,i-8):i+8].flatten() for i in range(T)]

# Option B: HuBERT (modern alternative, recommended in architecture doc)
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
features = hubert(audio).last_hidden_state  # 50fps, 768-dim
```

### 2.2 Model Architecture Differences (MEDIUM)

**VOCA:**
```
DeepSpeech(29-dim) → Conv1D×4(stride=2) → FC → GRU(512) → FC → VertexOffsets(15069-dim)
SubjectConditioning: concatenated at multiple layers
```

**We have:**
```
LogMel(80-dim) → Linear(256) → TransformerEncoder(4 layers, 8 heads) → Linear → GNM Coeffs(182-dim)
SubjectConditioning: added via embedding
```

**Impact:** VOCA's Conv1D downsampling naturally handles the temporal structure. Our Transformer may be overkill for the small VOCASET dataset (~29 minutes) and could overfit. Subject conditioning in VOCA is stronger (concatenated at input and intermediate layers vs. our single additive embedding).

**Fix:** Add a VOCA-style Conv1D frontend before the Transformer, and strengthen subject conditioning:
```python
self.conv1 = nn.Conv1d(audio_dim, 32*hidden_factor, 3, stride=2)
self.conv2 = nn.Conv1d(32*hidden_factor, 32*hidden_factor, 3, stride=2)
self.conv3 = nn.Conv1d(32*hidden_factor, 64*hidden_factor, 3, stride=2)
self.conv4 = nn.Conv1d(64*hidden_factor, 64*hidden_factor, 3, stride=2)
# Then Transformer on conv output + speaker embedding
```

### 2.3 Loss Function Differences (MEDIUM)

**VOCA losses:**
1. Vertex reconstruction loss (L1 on 5023×3 = 15069-dim vertex offsets)
2. Edge loss (edge length preservation)
3. Orthogonality loss (only for certain weight matrices)

**We have:**
1. Coefficient L1 loss (on 182-dim GNM coefficients)  
2. Velocity loss (temporal, 1st order)
3. Acceleration loss (temporal, 2nd order)
4. Edge loss (on coefficient space, not vertex space)
5. L1 sparsity regularizer

**Key difference:** VOCA's vertex-space loss is more direct — it penalizes incorrect vertex positions. Our coefficient-space loss relies on GNM's PCA basis being a good proxy for visual quality. If a coefficient error of 0.1 produces a visible artifact, our loss catches it indirectly. VOCA's vertex loss catches it directly.

**We should add:** After training the coefficient model, add a fine-tuning stage that projects predicted coefficients through GNM and computes vertex-space loss:
```python
gnm = GNM(...)
pred_coeffs = model(audio)
pred_vertices = gnm(identity, pred_coeffs, rotations, translation)
target_vertices = gnm(identity, target_coeffs, rotations, translation)
vertex_loss = L1Loss(pred_vertices, target_vertices)
```

### 2.4 Training Configuration System (MEDIUM)

**VOCA has:** `config_parser.py` — generates a JSON config with all training hyperparameters. Controls model architecture, loss weights, learning rates, data paths, etc.

**We have:** Hardcoded parameters with argparse overrides. No experiment tracking, no config file.

**Fix:** Add a YAML/JSON-based config system for experiment reproducibility.

### 2.5 VOCA Demos Not Integrated (LOW)

VOCA provides:
- `visualize_sequence.py` — renders animation sequences to video
- `edit_sequences.py` — adds eye blinks, shape variation, head pose to VOCA output
- `sample_templates.py` — generates FLAME templates
- `compute_FLAME_params.py` — extracts FLAME parameters from meshes

None of these are integrated or have Project Dome equivalents.

---

## Part 3: Current Server & Pipeline Issues

### 3.1 Wav2TextGrid model loads every startup

**Issue:** `mfa_aligner.py` loads a SpeechBrain x-vector extractor and Wav2TextGrid model at import time. This adds ~3-5s to server startup. If the pretrained models aren't cached, it downloads them.

### 3.2 Piper TTS model download not validated

**Issue:** `piper_provider.py` downloads the ONNX model from HuggingFace on first run but doesn't validate the download. A corrupted download silently fails.

### 3.3 Temp files in /api/speak

**Issue:** `mfa_aligner.py` writes `output/temp_alignment.wav` for every synthesis request. This is a disk I/O bottleneck and a concurrency hazard.

### 3.4 No caching of synthesis results

**Issue:** Every `/api/speak` call re-synthesizes the same text. No LRU cache for frequent phrases.

### 3.5 /api/chat uses HTTP timeout but mind layer has no connection pool

**Issue:** If the OMLX endpoint is down, `LocalMindProvider` waits for the full 10s timeout on every `/api/chat` call. No connection pooling or health checking.

---

## Part 4: Comprehensive Implementation Plan

### Phase A — GNM Web Renderer Completeness (2 weeks)

| # | Task | Files | Priority | Effort |
|---|---|---|---|---|
| A1 | Export `pose_correctives_regressor` correctly + apply in web renderer | `export_basis.py`, `renderer.js` | Critical | 2d |
| A2 | Export `joint_identity_basis` + apply identity to joints | `export_basis.py`, `renderer.js` | High | 1d |
| A3 | Export vertex groups properly via `argmax(vertex_groups)` | `export_basis.py` | High | 0.5d |
| A4 | Add per-body-part materials (eyes glossy, teeth enamel, skin diffuse) | `renderer.js` | High | 2d |
| A5 | Export `triangle_uvs` + apply GNM texture map | `export_basis.py`, `renderer.js` | Medium | 1d |
| A6 | Port GNM's exact LBS math to JS (homogeneous coordinates) | `renderer.js` | High | 1d |
| A7 | Port GNM's vertex normal computation to JS | `renderer.js` | Medium | 0.5d |

### Phase B — VOCA Training Pipeline (2 weeks)

| # | Task | Files | Priority | Effort |
|---|---|---|---|---|
| B1 | Add DeepSpeech/HuBERT feature extraction | `src/training/dataset.py`, new module | High | 3d |
| B2 | Add VOCA-style Conv1D frontend to model | `src/training/model.py` | Medium | 2d |
| B3 | Add vertex-space loss (project coefficients through GNM) | `src/training/train.py` | High | 2d |
| B4 | Add YAML-based config system | `src/training/config.py` (new) | Medium | 1d |
| B5 | Re-train with VOCA-compatible feature pipeline | `src/training/run_pipeline.py` | High | 1d (compute) |

### Phase C — Server Production Hardening (1 week)

| # | Task | Files | Priority | Effort |
|---|---|---|---|---|
| C1 | Remove temp file writes from alignment (use in-memory) | `mfa_aligner.py` | High | 0.5d |
| C2 | Add LRU cache for synthesis results | `server.py` | Medium | 1d |
| C3 | Add connection pooling + timeout reduction for mind layer | `local_provider.py` | Medium | 0.5d |
| C4 | Add model download validation with checksums | `piper_provider.py`, `mfa_aligner.py` | High | 0.5d |
| C5 | Add gRPC or WebSocket support for streaming | `server.py` | Low | 3d |

### Phase D — Testing & Validation (1 week)

| # | Task | Files | Priority | Effort |
|---|---|---|---|---|
| D1 | Add GNM forward pass regression tests | `tests/test_gnm.py` (new) | High | 1d |
| D2 | Add server integration tests (all endpoints) | `tests/test_server.py` (new) | High | 2d |
| D3 | Add JS renderer unit tests (Jest) | `web/*.js`, `tests/` | High | 3d |
| D4 | Add perceptual quality evaluation pipeline | `src/training/evaluate.py` | Medium | 2d |

---

## Part 5: Key Architectural Decisions

### Should we port the full GNM forward pass to JS?

**Yes, but incrementally.** The GNM forward pass is:
```python
vertices = template + identity_basis·id + expression_basis·expr  # (1)
joints = template_joints + joint_identity_basis·id                # (2)
pose_correctives = compute_pose_correctives(rotations)            # (3)
vertices += pose_correctives                                      # (4)
skinned = lbs(vertices, joints, rotations, skinning_weights)      # (5)
```

Currently we do (1) and (5) in JS. Missing (2), (3), (4). Adding (3) and (4) is critical for production-quality deformation.

### Should we switch to DeepSpeech features?

**Benchmark first, switch if better.** DeepSpeech requires a ~188MB model download and a complex dependency chain. HuBERT is a more modern alternative that might perform as well or better. Add a feature abstraction layer and benchmark both against log-mel.

### Should we use the GNM PyTorch backend?

**Yes for training, no for inference.** The PyTorch backend (`gnm_pytorch.py`) enables GPU-accelerated forward passes during training, which is critical for vertex-space loss. But for web inference, the exported JS buffers are more efficient.

### Should we keep the additive coefficient blend?

**Replace with per-channel max.** The current max-based blend (fixed in the previous session) is correct. Keep it.

---

## Part 6: Quick Reference — Attribute Map

GNM's actual model attributes (from `gnm_xnp.py:140-206`) vs. what we access:

| GNM Attribute | Shape | We Export? | We Use? | Notes |
|---|---|---|---|---|
| `template_vertex_positions` | (17821, 3) | ✅ .bin | ✅ renderer | |
| `vertex_identity_basis` | (253, 17821, 3) | ✅ .bin | ✅ renderer | |
| `joint_identity_basis` | (253, 4, 3) | ❌ | ❌ | Identity affects joints — MISSING |
| `expression_basis` | (383, 17821, 3) | ✅ .bin | ✅ renderer | |
| `skinning_weights` | (4, 17821) | ✅ .bin | ✅ renderer | |
| `joint_regressor` | (4, 17821) | ✅ .bin | ✅ renderer | |
| `pose_correctives_regressor` | (36, 53463) | ❌ | ❌ | CRITICAL — prevents LBS artifacts |
| `triangles` | (35344, 3) | ✅ .bin | ✅ renderer | |
| `quads` | (17672, 4) | ❌ | ❌ | Higher-quality topology |
| `quad_uvs` | (17672, 4, 2) | ❌ | ❌ | Texture coordinates |
| `triangle_uvs` | (35344, 3, 2) | ❌ | ❌ | Texture coordinates |
| `vertex_groups` | (G, 17821) | ❌ | ❌ | Body part labels |
| `vertex_group_names` | (G,) | ❌ | ❌ | Name per group |
| `mirror_indices` | (17821,) | ❌ | ❌ | Symmetry |
| `joint_parent_indices` | (4,) | ✅ JSON | ✅ renderer | |
| `template_joint_positions` | (4, 3) | ✅ JSON | ✅ renderer | |
| `expression_names` | (383,) | ✅ JSON | ❌ | Not displayed in UI |
| `identity_names` | (253,) | ✅ JSON | ❌ | Not displayed in UI |
| `joint_names` | (4,) | ✅ JSON | ❌ | Not displayed in UI |

---

## Summary of All Issues Found

| Severity | Count | Key Issues |
|---|---|---|
| **CRITICAL** | 3 | Pose correctives missing, wrong attribute accessed, joint identity basis missing |
| **HIGH** | 6 | LBS math simplified, vertex groups/body parts wrong, no UV export, DeepSpeech features, no vertex-space loss, temp files in server path |
| **MEDIUM** | 5 | No config system, subject conditioning weak, no synthesis cache, no model download validation, no server integration tests |
| **LOW** | 4 | Mirror indices unused, VOCA demos not ported, normal computation, WebSocket/gRPC streaming |
