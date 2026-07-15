# Web Renderer

`web/renderer.js` — the 3D rendering engine. Loads GNM basis buffers into the browser and performs real-time mesh deformation using Three.js with CPU-side Linear Blend Skinning.

## Buffer Pipeline

The GNM model is too large to ship as 383 morph targets (WebGL limit is 4–8 active). Instead, the entire identity and expression basis is packed into binary buffers and decoded in JavaScript:

| Buffer | Format | Size | Contents |
|---|---|---|---|
| `mean_positions.bin` | float32 | 214KB | Template vertex positions (17821 × 3) |
| `identity_basis.bin` | float16 | ~10MB | Identity PCA basis (253 × 17821 × 3) |
| `joint_identity_basis.bin` | float16 | ~24KB | Joint offset PCA basis (253 × 4 × 3) |
| `expression_basis.bin` | float16 | ~15MB | Expression PCA basis (383 × 17821 × 3) |
| `face_indices.bin` | uint32 | 209KB | Triangle indices |
| `skinning_weights.bin` | float32 | 285KB | LBS weights (4 joints × 17821 verts) |
| `joint_regressor.bin` | float32 | 285KB | Joint position regressor (4 × 17821) |
| `vertex_body_parts.bin` | int32 | 71KB | Per-vertex body-part group index (46 groups) |
| `mirror_indices.bin` | int32 | 71KB | Vertex mirror symmetry mapping |
| `metadata.json` | JSON | ~2KB | Model dimensions, joint parents, group names |

**float16 decoding:** browsers don't natively support float16 typed arrays. The renderer includes a manual IEEE 754 float16 decoder that unpacks each 16-bit value to float32.

## Deformation Pipeline

The `deformMesh()` function runs every frame on the CPU:

1. **Start with template positions** — copy `meanPositions` to working buffer
2. **Sparse identity deformation** — iterate active (|w| > 1e-4) identity coefficients, accumulate vertex offsets using `identityBasis`
3. **Joint identity deformation** — apply identity-dependent joint offsets using `jointIdentityBasis`
4. **Sparse expression deformation** — same for expression coefficients
5. **Joint regression** — regress 4 joint positions from deformed vertices using `jointRegressor`
6. **Auto gaze tracking** — if enabled, compute eye joint rotations to track camera position (clamped to ±25° yaw, ±15° pitch)
7. **Forward kinematics** — compute local-to-world transforms for the 4-joint skeleton (neck → head → left eye / right eye) using `buildJointTransforms()`
8. **Linear Blend Skinning** — for each vertex, blend the 4 joint skinning matrices weighted by `skinningWeights`. Each skinning matrix = `T × R × invBind` where `T` = world translation, `R` = world rotation, `invBind` = inverse rest pose
9. **Update Three.js BufferGeometry** — set position attribute, recompute vertex normals

## Scene Setup

**Camera:** PerspectiveCamera, 45° FOV, positioned dynamically from the mesh bounding box:

```javascript
const maxDim = Math.max(size.x, size.y, size.z);
const distance = (maxDim / 2) / Math.tan(fovRad / 2) * 1.4;
camera.position.set(center.x, center.y, center.z + distance);
controls.target.copy(center);
```

This computes the camera distance so the head fills ~71% of the viewport with equal top/bottom margins. For the current GNM Head model this yields distance ≈ 0.577 with camera at Y = 0.236 (head center).

**OrbitControls r128 caveat:** The OrbitControls constructor calls `update()` internally, locking spherical coordinates relative to the default target `(0,0,0)`. To avoid double-offset, both `controls.target` and `camera.position` are set to their final values before the first manual `controls.update()`.

- Lighting: ambient (0.25) + key directional (0.85) + cool fill (0.4) + rim light (0.45)
- Controls: OrbitControls with damping enabled
- Material: MeshStandardMaterial, light clay finish (roughness 0.4, metalness 0.1), vertex colors from 46 body-part groups
- Background: Apple-style light grey (#f5f5f7)

## Per-Vertex Body-Part Colors

GNM's 46 vertex groups are mapped to semantic color palettes:
- Skin/ears: warm clay (#c7b8a0)
- Sclera/iris/pupil: white (#f2f2f7)
- Teeth/gum: off-white (#ebdcc8)
- Tongue: pink (#cc807c)
- Mouth sock: dark cavity (#593333)

## Blink Simulation

Automatic eyelid blinking every 2–6 seconds using a 120ms sine-curve blink. Pre-cached blink coefficients from the GNM CVAE (`WINK_LEFT` + `WINK_RIGHT`) blend into eye region channels (0–199) during each blink.

## Stats Overlay

Real-time display of FPS, render time (ms), and active blendshape count.

## File reference

| File | Role |
|---|---|
| `web/renderer.js` | Main renderer: buffer loading, deformation, LBS, scene, render loop |
| `tools/export_basis.py` | Generates binary buffers from GNM model |
