# Animation Engine

`src/animation/` — the core mapping layer. Converts viseme timelines and emotion tags into GNM coefficient vectors, then drives the GNM model to produce deformed 3D meshes.

## VisemeTable

`src/animation/viseme_table.py` — stores 182-dimensional coefficient vectors for each of the 13 viseme categories.

- **182 dims:** 150 for lower face region + 32 for tongue
- Persisted to/from `data/viseme_table.json`
- Default placeholders for jaw opening (aa), lip rounding (OO), smile (EE), lip closure (PP), tongue protrusion (TH)

```python
table = VisemeTable()
coeffs = table.get_coefficients("aa")  # returns (182,) array
```

## VisemeInterpolator

`src/animation/interpolator.py` — samples coefficient vectors at any point in a viseme timeline with smooth transitions.

- Inside an event: returns the viseme's coefficient vector
- Near event boundaries: linear ramp over 40ms to the next viseme
- Between events (gaps): linear interpolation between neighboring visemes
- Before/after timeline: returns IDLE (neutral)

## EmotionBlender

`src/animation/emotion_blender.py` — bridges the 182-dim speech space with the 383-dim GNM expression space.

**Blending strategy:**
- Upper face + eyes (0–199): driven fully by emotion
- Lower face (200–349): speech + 30% emotion overlap (jaw/lips influenced by both)
- Tongue (350–381): driven fully by speech
- Pupils (382): driven fully by emotion

Uses GNM's built-in ExpressionSampler CVAE to convert emotion label strings (e.g. "HAPPY", "SURPRISE") to 383-dim coefficient vectors. "SAD" is mapped to "CORNERS_DOWN".

## GNMDriver

`src/animation/gnm_driver.py` — wraps the GNM numpy forward pass.

```python
driver = GNMDriver()
vertices = driver.evaluate(identity_coeffs, expression_coeffs)
# Returns (17821, 3) vertex positions
driver.save_mesh(vertices, "output/frame.obj")
```

- Loads GNM v3.0 Head model
- Forward pass: `vertices = template + identity_basis·id + expression_basis·expr`
- Exports to OBJ format with proper face indices

## Runtime Loop

`src/animation/runtime_loop.py` — offline batch animation pipeline.

```
text + emotion + intensity
  → AcousticPipeline → audio + visemes
  → VisemeInterpolator (per-frame)
  → EmotionBlender.blend() (speech + emotion)
  → GNMDriver.evaluate()
  → OBJ files at output/frames/frame_XXXX.obj
```

Key parameters: `fps` (default 30), `output_dir`

## File reference

| File | Role |
|---|---|
| `src/animation/__init__.py` | Exports all animation components |
| `src/animation/viseme_table.py` | 182-dim viseme lookup table |
| `src/animation/interpolator.py` | Timeline sampling with ramp transitions |
| `src/animation/emotion_blender.py` | Additive speech + emotion blending |
| `src/animation/gnm_driver.py` | GNM forward pass wrapper |
| `src/animation/runtime_loop.py` | Offline frame-by-frame animation pipeline |
