# Web Renderer

`web/renderer.js` — the 3D rendering engine. Loads GNM basis buffers into the browser and performs real-time mesh deformation using Three.js with CPU-side Linear Blend Skinning.

## Buffer Pipeline

The GNM model is too large to ship as 383 morph targets (WebGL limit is 4–8 active). Instead, the entire identity and expression basis is packed into binary buffers and decoded in JavaScript:

| Buffer | Format | Size | Contents |
|---|---|---|---|
| `mean_positions.bin` | float32 | 214KB | Template vertex positions (17821 × 3) |
| `identity_basis.bin` | float16 | ~10MB | Identity PCA basis (253 × 17821 × 3) |
| `expression_basis.bin` | float16 | ~15MB | Expression PCA basis (383 × 17821 × 3) |
| `face_indices.bin` | uint32 | 209KB | Triangle indices |
| `skinning_weights.bin` | float32 | 285KB | LBS weights (4 joints × 17821 verts) |
| `joint_regressor.bin` | float32 | 285KB | Joint position regressor (4 × 17821) |

**float16 decoding:** browsers don't natively support float16 typed arrays. The renderer includes a manual IEEE 754 float16 decoder that unpacks each 16-bit value to float32.

## Deformation Pipeline

The `deformMesh()` function runs every frame on the CPU:

1. **Start with template positions** — copy `meanPositions` to working buffer
2. **Sparse identity deformation** — iterate active (|w| > 1e-4) identity coefficients, accumulate vertex offsets
3. **Sparse expression deformation** — same for expression coefficients
4. **Joint regression** — regress 4 joint positions from deformed vertices using `jointRegressor`
5. **Auto gaze tracking** — if enabled, compute eye joint rotations to track camera position
6. **Forward kinematics** — compute local-to-world transforms for the 4-joint skeleton (neck → head → left eye / right eye)
7. **Linear Blend Skinning** — for each vertex, blend the 4 joint transforms weighted by `skinningWeights`
8. **Update Three.js BufferGeometry** — set position attribute, recompute vertex normals

## Scene Setup

- Camera: perspective, 45° FOV, positioned facing the head
- Lighting: ambient (0.25) + key directional (0.85) + cool fill (0.4) + rim light (0.45)
- Controls: OrbitControls with damping, constrained polar angle and distance
- Material: MeshStandardMaterial, light clay finish (roughness 0.4, metalness 0.1)
- Background: Apple-style light grey (#f5f5f7)

## Blink Simulation

Automatic eyelid blinking every 2–6 seconds using a 150ms sine-curve blink. Pre-cached blink coefficients from the GNM CVAE (`WINK_LEFT` + `WINK_RIGHT`) blend into eye region channels (0–199) during each blink.

## Stats Overlay

Real-time display of FPS, render time (ms), and active blendshape count.

## File reference

| File | Role |
|---|---|
| `web/renderer.js` | Main renderer: buffer loading, deformation, LBS, scene, render loop |
| `tools/export_basis.py` | Generates binary buffers from GNM model |
