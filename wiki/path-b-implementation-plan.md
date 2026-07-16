# Path B: Speech-to-Vertex Model — Implementation Plan

Based on the VOCA-vs-GNM audit (`wiki/voca-gnm-audit.md`). This plan rewrites the speech animation pipeline from a PCA-coefficient predictor to a **vertex-offset predictor**, mirroring VOCA's proven architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PATH B (NEW)                             │
│                                                             │
│  Audio → HuBERT features → SpeechToVertexModel              │
│                              ↓                              │
│                    per-vertex offsets (3×V)                  │
│                              ↓                              │
│  GNM_template + speech_offset → bind_pose_vertices          │
│                              +                              │
│  emotion_delta (PCA, upper face only)                       │
│                              +                              │
│  identity_delta (PCA)                                       │
│                              ↓                              │
│                    LBS → final mesh                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    PATH A (CURRENT, DEPRECATE)              │
│                                                             │
│  Viseme lookup table → 182-dim coefficients                 │
│                     → PCA expression_basis (attenuated)     │
│                     → blend() → LBS                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Quick Wins — Fix Emotion to Upper Face Only

**Goal:** Immediately stop emotion CVAE from interfering with the lower face, without waiting for the full model rewrite.

### 0.1 Limit emotion PCA channels in `emotion_blender.py`

**File:** `src/animation/emotion_blender.py`

Current `blend()` zeros out lower face channels from speech. But the emotion CVAE itself produces full-face coefficients. Fix: after sampling emotion from the CVAE, zero out channels 200-382 so emotion never touches the lower face.

```python
def get_emotion_coefficients(self, emotion_name, intensity=1.0):
    coeffs = self.sampler.sample_expression(enum_val).squeeze(0) * intensity
    coeffs[200:] = 0.0  # ← NEW: emotion only affects upper face (0-199)
    return coeffs
```

### 0.2 Remove `blend()` from server-side `emotion_blender.py`

The server's `blend()` method is no longer needed — speech and emotion don't share a coefficient space anymore. Remove or deprecate.

### 0.3 Remove `blend()` from client-side `animation_controller.js`

Replace with independent application:
```javascript
// BEFORE:
blended = animController.blend(speechCoeffs, currentEmotionCoeffs);

// AFTER:
const blended = new Float32Array(383);
for (let i = 0; i < 200; i++) blended[i] = currentEmotionCoeffs[i];  // emotion: upper face only
for (let i = 200; i < 383; i++) blended[i] = 0;  // speech handled separately via vertex offsets
```

**Note:** This breaks current Path A speech (which relied on `blend()` putting speech into channels 200-349). That's intentional — Path A is being deprecated. During the transition, Path A speech will stop working until the vertex-offset model is deployed. The UI should show a clear warning.

### 0.4 Update `/api/emotion` to return zeroed lower face

**File:** `src/server.py:210`

The endpoint already calls `blender.get_emotion_coefficients()`. With the Phase 0.1 change, this will automatically return upper-face-only coefficients.

---

## Phase 1: Training Data Pipeline

**Goal:** Convert VOCASET FLAME-topology vertex sequences into (audio_features, GNM_vertex_offset) training pairs.

### 1.1 Understand the data flow

VOCASET provides:
- Audio WAV files (16kHz)
- FLAME-topology vertex sequences (5023 vertices per frame, 30fps)
- Phoneme annotations

We need:
- Audio features (HuBERT embeddings, 80-dim, 50Hz)
- GNM-topology vertex offsets (17821 vertices × 3, relative to GNM template)

### 1.2 Vertex topology transfer: FLAME → GNM

**New file:** `tools/transfer_topology.py`

FLAME and GNM have different vertex counts and topologies:
- FLAME face: 5023 vertices (or full head: 11731)
- GNM Head: 17821 vertices

We need a correspondence mapping. Two approaches:

**Approach A (ICP-based, preferred):**
```python
# For each FLAME mesh in the training sequence:
flame_vertices = load_flame_sequence()  # (T, 5023, 3)
gnm_template = model.template_vertex_positions  # (17821, 3)

# Fit GNM expression coefficients to match FLAME vertices:
# This is what the current reproject_vocaset.py does — keep it
# but ALSO save the residual error as vertex offsets
for t in range(T):
    coeffs = fit_gnm_expression(flame_vertices[t], gnm_template)
    gnm_approx = evaluate_gnm(coeffs)
    residual = flame_vertices[t] - gnm_approx  # what PCA CAN'T represent
    vertex_offset = apply_gnm_basis(coeffs) + residual  # full displacement
    save(vertex_offset)  # (17821, 3)
```

**Approach B (Learned, more accurate):**
Train a small MLP that maps FLAME vertex positions (5023×3) to GNM vertex offsets (17821×3). This is a dense-to-dense regression problem. The MLP can be trained on a subset of matched frames.

**Decision:** Start with Approach A (ICP + residual) since the infrastructure exists in `tools/reproject_vocaset.py`. Upgrade to Approach B if residuals are too large.

### 1.3 Audio feature extraction

**File:** `src/training/features.py` (already exists)

VOCA uses DeepSpeech (29-dim). We use HuBERT (80-dim). HuBERT provides better phoneme discrimination. Keep HuBERT but verify the feature rate matches:

- VOCA: DeepSpeech at 50Hz (16-frame window, 1-frame stride, 16kHz audio)
- Current: HuBERT at 50Hz (320-sample window, 160-sample stride, 16kHz audio)

Both produce 50 features/second. ✓

### 1.4 Build the training dataset

**File:** `tools/build_vertex_training_data.py`

```
Input:  VOCASET directory (audio WAVs + FLAME vertex sequences)
Output: npz files with:
  - audio_features: (T, 80) HuBERT embeddings
  - vertex_targets: (T, 17821, 3) GNM vertex offsets
  - speaker_id: int (0-11)

Processing pipeline:
1. For each subject/sentence pair:
   a. Load audio (16kHz mono WAV)
   b. Extract HuBERT features → (T, 80) at 50Hz
   c. Load FLAME vertex sequence → (T', 5023, 3) at 30fps
   d. Upsample FLAME vertices to 50Hz (linear interpolation) → (T, 5023, 3)
   e. For each frame:
      - Fit GNM expression coefficients (ICP-style)
      - Compute residual = FLAME_vertex - GNM_approximation
      - vertex_target = GNM_basis_delta + residual  # (17821, 3)
   f. Save npz with audio_features, vertex_targets, speaker_id
```

**Key implementation detail — fitting GNM to FLAME:**
```python
def flame_to_gnm_offset(flame_vertices, gnm_model, gnm_template):
    """
    Converts FLAME-topology vertices to GNM-topology vertex offsets.
    """
    # Step 1: Compute GNM expression coefficients that best match FLAME
    # Use least-squares fitting: min ||GNM(expr_coeffs) - FLAME_vertices||^2
    # This is the existing ICP approach in reproject_vocaset.py
    
    # Step 2: Evaluate GNM at those coefficients
    gnm_approx = gnm_model(expression=expr_coeffs)  # (17821, 3)
    
    # Step 3: The residual is what PCA CAN'T represent
    # For FLAME, we need a correspondence between FLAME and GNM vertices.
    # Use nearest-neighbor in vertex space or the registered template.
    
    # Step 4: vertex_target = gnm_approx - gnm_template + upsampled_residual
    # Where upsampled_residual maps FLAME residual to GNM vertices
    
    return vertex_target
```

### 1.5 Update the dataset loader

**File:** `src/training/dataset.py`

```python
class VertexOffsetDataset(Dataset):
    def __init__(self, npz_files):
        self.samples = []  # list of (audio_feats, vertex_targets, speaker_id)
        
    def __getitem__(self, idx):
        audio, vertices, speaker = self.samples[idx]
        # Return random window of 16 consecutive frames (matching VOCA)
        start = random.randint(0, len(audio) - 16)
        return {
            'audio': audio[start:start+16],      # (16, 80)
            'vertices': vertices[start:start+16], # (16, 17821, 3)
            'speaker': speaker
        }
```

---

## Phase 2: Model Architecture — SpeechToVertexModel

**Goal:** Replace `SpeechToCoefficientsModel` with a VOCA-inspired `SpeechToVertexModel` that outputs per-vertex offsets.

### 2.1 Model definition

**File:** `src/training/model.py` (rewrite)

```python
class ConvFrontend(nn.Module):
    """4-layer Conv1D with stride-2, identical to VOCA's SpeechEncoder."""
    def __init__(self, in_dim=80, encoding_dim=50, size_factor=1.0):
        super().__init__()
        f = size_factor
        self.conv1 = nn.Conv1d(in_dim, int(32*f), 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(int(32*f), int(32*f), 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(int(32*f), int(64*f), 3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(int(64*f), int(64*f), 3, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(int(64*f))
        # After 4 convs with stride 2: input T frames → output T//16 frames
        # VOCA handles this by having T=16 (single window) → 1 output frame
        # For sequence training, we process overlapping windows
    
    def forward(self, x):
        # x: (B, T, 80)
        x = x.transpose(1, 2)  # (B, 80, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn(x)
        x = x.transpose(1, 2)  # (B, T//16, 64*f)
        return x


class VertexLayer(nn.Module):
    """
    Linear layer producing per-vertex offsets.
    Initialized from GNM's expression PCA basis.
    """
    def __init__(self, encoding_dim, num_vertices, expression_basis):
        super().__init__()
        # expression_basis: (E, V, 3) → reshape to (E, V*3) → transpose to (V*3, E)
        E, V, _ = expression_basis.shape
        assert V == num_vertices
        
        # Initialize weight from PCA basis (first encoding_dim components)
        basis_flat = expression_basis.reshape(E, V*3)  # (E, V*3)
        init_weight = basis_flat[:encoding_dim].T       # (V*3, encoding_dim)
        
        self.weight = nn.Parameter(init_weight.clone().float())
        self.bias = nn.Parameter(torch.zeros(V*3))
        
    def forward(self, x):
        # x: (B, encoding_dim)
        offset = F.linear(x, self.weight.T, self.bias)  # (B, V*3)
        return offset.reshape(-1, V, 3)  # (B, V, 3)


class SpeechToVertexModel(nn.Module):
    """
    Maps audio features to GNM vertex displacements.
    Architecture mirrors VOCA's: Conv1D frontend → FC → VertexLayer.
    """
    def __init__(self, audio_dim=80, encoding_dim=50, num_vertices=17821,
                 num_speakers=12, expression_basis=None, size_factor=1.0):
        super().__init__()
        
        self.frontend = ConvFrontend(audio_dim, encoding_dim, size_factor)
        
        # Speaker conditioning (VOCA uses one-hot + FC concat)
        self.speaker_embed = nn.Embedding(num_speakers, encoding_dim)
        
        # Audio encoding → vertex offset
        self.vertex_layer = VertexLayer(encoding_dim, num_vertices, expression_basis)
        
        # Post-processing FC (to refine PCA-initialized output)
        self.refine_fc = nn.Linear(encoding_dim, encoding_dim)
        
    def forward(self, audio_features, speaker_ids=None):
        # audio_features: (B, T, 80)
        encoding = self.frontend(audio_features)  # (B, T//16, 64*f)
        
        # Take last frame encoding (or mean pool)
        encoding = encoding[:, -1, :]  # (B, 64*f)
        
        # Project to encoding_dim
        encoding = self.refine_fc(encoding)  # (B, encoding_dim)
        
        if speaker_ids is not None:
            embed = self.speaker_embed(speaker_ids)
            encoding = encoding + embed
        
        vertex_offset = self.vertex_layer(encoding)  # (B, V, 3)
        return vertex_offset
```

### 2.2 Training setup

**File:** `src/training/train.py` (update)

```python
# VOCA-style losses
criterion_rec = nn.L1Loss()  # reconstruction loss (vertex L1)
criterion_vel = nn.L1Loss()  # velocity loss (frame-to-frame difference)
criterion_acc = nn.L1Loss()  # acceleration loss (second derivative)

def vertex_loss(predicted, target):
    """
    Loss on vertex positions, matching VOCA's three-term loss.
    predicted, target: (B, V, 3)
    """
    # Reconstruction loss
    rec_loss = criterion_rec(predicted, target)
    
    # Velocity loss (frame-to-frame difference)
    vel_pred = predicted[:, 1:] - predicted[:, :-1]
    vel_target = target[:, 1:] - target[:, :-1]
    vel_loss = criterion_vel(vel_pred, vel_target)
    
    # Acceleration loss (second derivative)  
    acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
    acc_target = vel_target[:, 1:] - vel_target[:, :-1]
    acc_loss = criterion_acc(acc_pred, acc_target)
    
    return rec_loss + 0.5 * vel_loss + 0.25 * acc_loss
```

### 2.3 Hyperparameters (matching VOCA)

| Parameter | VOCA | Ours |
|-----------|------|------|
| Audio features | DeepSpeech 29-dim | HuBERT 80-dim |
| Window size | 16 frames | 16 frames |
| Window stride | 1 frame | 1 frame |
| Encoding dim | 50 | 50 |
| Conv1D channels | 32/32/64/64 | 32/32/64/64 |
| Conv1D stride | 2 | 2 |
| Num speakers | 8 | 12 |
| Learning rate | 0.001 | 0.001 |
| Batch size | 50 | 32 (GPU memory) |
| Epochs | 100 | 100 |
| Optimizer | Adam | Adam |

---

## Phase 3: Server-Side Inference

**Goal:** Deploy the trained model on the server and serve vertex offsets to the web client.

### 3.1 Model loading

**File:** `src/server.py`

```python
def init_vertex_model():
    checkpoint = torch.load('voca/model/checkpoints/speech_to_vertex.pt')
    model = SpeechToVertexModel(
        audio_dim=80,
        encoding_dim=50,
        num_vertices=17821,
        num_speakers=12,
        expression_basis=gnm_model.expression_basis
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
```

### 3.2 New API endpoint: `/api/speak/v2`

**File:** `src/server.py`

```python
elif self.path == "/api/speak/v2":
    text = data.get("text", "")
    audio, sr, _ = pipeline.process(text)  # Piper TTS only (no alignment needed)
    
    # Extract HuBERT features
    features = extract_hubert_features(audio, sr)  # (T, 80)
    
    # Run model inference (sliding window, stride=1)
    T = features.shape[0]
    vertex_offsets = []
    for t in range(T - 16 + 1):
        window = features[t:t+16].unsqueeze(0)  # (1, 16, 80)
        with torch.no_grad():
            offset = vertex_model(window)  # (1, 17821, 3)
        vertex_offsets.append(offset.squeeze(0))
    
    # Convert to list for JSON
    vertex_offsets = torch.stack(vertex_offsets)  # (T-15, 17821, 3)
    
    response = {
        "audio_base64": audio_base64,
        "vertex_offsets": vertex_offsets.cpu().numpy().tolist(),
        "sample_rate": sr,
        "fps": 50  # model runs at 50Hz (matching HuBERT frame rate)
    }
```

**Performance note:** Sliding window inference at 50Hz on CPU may be slow. Optimizations:
- Process in chunks of 16 frames at stride 1 → 50 evaluations per second of audio
- Each eval: (1, 16, 80) → 4 Conv1D layers + 2 FC layers → (1, 17821, 3) ≈ ~1ms on GPU
- Total: ~50ms per second of audio on GPU. On CPU (M3 Max): ~200-500ms per second → a 3-second utterance takes 1-1.5 seconds. Acceptable for non-realtime.
- For streaming: process in parallel batches.

### 3.3 Fallback for real-time: sliding window with batch processing

```python
# Process all windows at once using unfold
windows = features.unfold(0, 16, 1)  # (T-15, 16, 80)
windows = windows.transpose(0, 1)     # (16, T-15, 80) — not ideal
# Better: process in batches
batch_size = 64
all_offsets = []
for i in range(0, T - 15, batch_size):
    batch = features[i:i+batch_size+15]  # need 16-frame context
    # ... create windows and run model
```

---

## Phase 4: Client-Side Rendering Update

**Goal:** Apply vertex offsets in the web renderer, remove blend function, separate emotion and speech mechanisms.

### 4.1 New data structures

**File:** `web/renderer.js`

```javascript
let vertexOffsetBuffer = null;  // Float32Array for speech vertex offsets
let vertexOffsetTimeline = null;  // array of {time, offset} keyframes
let currentSpeechOffset = new Float32Array(17821 * 3);  // interpolated at current time
```

### 4.2 Updated render loop

**File:** `web/renderer.js` (modify `renderLoop`)

```javascript
function renderLoop(time) {
    requestAnimationFrame(renderLoop);
    
    // 1. Start with template
    currentPos.set(meanPositions);
    
    // 2. Apply identity basis
    applyIdentityBasis(currentPos, currentIdentityCoeffs);  // unchanged
    
    // 3. Apply SPEECH offset (per-vertex, bypasses PCA)
    if (activeUtterancePlaying && vertexOffsetTimeline) {
        const ts = audioSync.getCurrentTime();
        const speechOffset = interpolateVertexOffset(ts, vertexOffsetTimeline);
        for (let i = 0; i < currentPos.length; i++) {
            currentPos[i] += speechOffset[i];
        }
    }
    
    // 4. Apply EMOTION (PCA, upper face only)
    applyExpressionBasis(currentPos, currentEmotionCoeffs);  // coeffs already zeroed below 200
    
    // 5. LBS (unchanged)
    computeJoints(currentPos);
    applySkinning(currentPos);
    
    // 6. Update geometry
    faceMesh.geometry.attributes.position.array.set(currentPos);
    faceMesh.geometry.attributes.position.needsUpdate = true;
    faceMesh.geometry.computeVertexNormals();
    
    controls.update();
    renderer.render(scene, camera);
}
```

### 4.3 Vertex offset interpolation

**File:** `web/renderer.js` (new function)

```javascript
function interpolateVertexOffset(timeS, timeline) {
    // Timeline is an array of {time, offset_array} keyframes at 50fps
    // offset_array is a flat Float32Array of length 17821*3
    
    if (!timeline || timeline.length === 0) return null;
    if (timeS <= timeline[0].time) return timeline[0].offset;
    if (timeS >= timeline[timeline.length - 1].time) return timeline[timeline.length - 1].offset;
    
    // Find surrounding keyframes
    let i = 0;
    while (i < timeline.length - 1 && timeline[i+1].time < timeS) i++;
    
    const t0 = timeline[i].time;
    const t1 = timeline[i+1].time;
    const frac = (timeS - t0) / (t1 - t0);
    
    const off0 = timeline[i].offset;
    const off1 = timeline[i+1].offset;
    const result = new Float32Array(17821 * 3);
    for (let j = 0; j < result.length; j++) {
        result[j] = off0[j] + frac * (off1[j] - off0[j]);
    }
    return result;
}
```

### 4.4 Updated `/api/speak` client handler

**File:** `web/renderer.js` (modify `speakBtn` click handler)

```javascript
speakBtn.addEventListener("click", async () => {
    // Use Path B endpoint
    const r = await fetch("/api/speak/v2", { method: "POST", ... });
    const d = await r.json();
    
    // Convert vertex offsets from JSON to Float32Array[]
    vertexOffsetTimeline = d.vertex_offsets.map((offset, i) => ({
        time: i / d.fps,  // 50fps
        offset: new Float32Array(offset.flat())
    }));
    
    await audioSync.loadAudioFromBase64(d.audio_base64);
    audioSync.play();
    activeUtterancePlaying = true;
});
```

### 4.5 Remove blend function

**File:** `web/animation_controller.js`

Delete the `blend()` method entirely. The class becomes:
```javascript
class AnimationController {
    constructor() {
        this.visemeTable = null;  // deprecated, kept for Path A fallback
        this.rampDuration = 0.04;
    }
    async loadVisemeTable() { ... }  // deprecated, keep for fallback
    getDefaultVisemeTable() { ... }  // deprecated, keep for fallback
    getSpeechCoefficients(timeS, timeline, audioDuration) { ... }  // deprecated
    // blend() — REMOVED
}
```

---

## Phase 5: Deprecate Path A Artifacts

**Goal:** Clean up unused code after Path B is verified.

### 5.1 Files to deprecate

| File | Action | When |
|------|--------|------|
| `data/viseme_table.json` | Remove | After Path B deployed |
| `src/animation/viseme_table.py` | Remove | After Path B deployed |
| `src/animation/interpolator.py` | Remove | After Path B deployed |
| `web/animation_controller.js:getSpeechCoefficients()` | Remove | After Path B deployed |
| `web/animation_controller.js:getDefaultVisemeTable()` | Remove | After Path B deployed |
| `web/animation_controller.js:blend()` | Remove | Phase 4 |
| `src/animation/emotion_blender.py:blend()` | Remove | Phase 0 |
| `src/alignment/viseme_mapper.py` | Keep | Still needed for phoneme info |
| `src/alignment/pipeline.py:VisemeEvent` | Keep | Still used by Path A fallback |

### 5.2 UI changes

- Remove "speaking style" dropdown (was Path A only, needs trained model)
- Add Path B status indicator ("Neural: active" / "Neural: not available")

---

## Phase 6: Testing

### 6.1 Model tests

**File:** `tests/test_vertex_model.py`

```python
def test_model_output_shape():
    model = SpeechToVertexModel(audio_dim=80, encoding_dim=50, num_vertices=17821)
    audio = torch.randn(2, 16, 80)  # batch=2, window=16, features=80
    output = model(audio)
    assert output.shape == (2, 17821, 3)

def test_vertex_layer_initialization():
    basis = torch.randn(383, 17821, 3) * 0.01  # simulate PCA basis
    layer = VertexLayer(encoding_dim=50, num_vertices=17821, expression_basis=basis)
    x = torch.randn(4, 50)
    offset = layer(x)
    assert offset.shape == (4, 17821, 3)
    # Initial output should be reasonable (not NaN)
    assert not torch.any(torch.isnan(offset))
    assert torch.max(torch.abs(offset)) < 1.0
```

### 6.2 Training data tests

```python
def test_flame_to_gnm_mapping():
    flame_verts = load_test_flame_vertices()
    gnm_offset = flame_to_gnm_offset(flame_verts, gnm_model, gnm_template)
    assert gnm_offset.shape == (17821, 3)
    # Offset should not be all zeros
    assert torch.norm(gnm_offset) > 0.001
```

### 6.3 Integration tests

**File:** `tests/test_server_endpoints.py` (add)

```python
def test_speak_v2_endpoint():
    response = requests.post("http://localhost:8080/api/speak/v2", 
                           json={"text": "Hello world", "emotion": "NEUTRAL"})
    assert response.status_code == 200
    data = response.json()
    assert "vertex_offsets" in data
    assert "audio_base64" in data
    assert len(data["vertex_offsets"]) > 0
    assert len(data["vertex_offsets"][0]) == 17821  # first frame
    assert len(data["vertex_offsets"][0][0]) == 3    # first vertex XYZ
```

---

## Implementation Order (Dependency Graph)

```
Phase 0 (emotion fix) ───── no dependencies
    │
    ▼
Phase 1 (training data) ─── needs Phase 0 (for clean data gen)
    │
    ▼
Phase 2 (model) ─────────── needs Phase 1 (training data)
    │
    ├──▶ Phase 3 (server) ── needs Phase 2 (trained model)
    │       │
    │       ▼
    │   Phase 4 (client) ─── needs Phase 3 (server endpoint)
    │
    └──▶ Phase 5 (cleanup) ─ needs Phase 4 (verified)
    
Phase 6 (testing) ────────── continuous throughout
```

## Timeline Estimate

| Phase | Effort | Dependencies | Estimated time |
|-------|--------|-------------|----------------|
| Phase 0 | 3 files, ~20 lines | None | 1 hour |
| Phase 1 | 2 new files, ~300 lines | Phase 0 | 2-3 days |
| Phase 2 | 1 rewrite, ~150 lines | Phase 1 | 1 day |
| Phase 3 | 1 modified file, ~80 lines | Phase 2 | 1 day |
| Phase 4 | 2 modified files, ~100 lines | Phase 3 | 1 day |
| Phase 5 | Multiple files, ~50 deletions | Phase 4 | 0.5 day |
| Phase 6 | 2 test files, ~100 lines | Ongoing | 1 day |

**Total: ~7-10 days for full Path B deployment.**
