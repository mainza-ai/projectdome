# Troubleshooting

Common issues encountered while developing and running the Project Dome avatar engine.

## Head appears at the bottom of the viewport / not centered

**Symptoms:** The 3D head model renders near the bottom of the canvas, with the chin partially or fully below the visible area. The crown may be visible but the face appears "sunk."

**Root cause:** LBS skinning matrix double-invBind bug in `deformMesh()`. The skinning matrix formula was:

```javascript
const T = makeTranslation(jw.x - Rj.x, jw.y - Rj.y, jw.z - Rj.z);  // BUG
skinningMat = T * R * invBind;
```

The `-Rj` subtraction in `T` doubles with `invBind`'s `-jr` translation, producing `-2*R*jr` instead of `-R*jr`. For zero rotation this shifts every vertex by `-sum(w_j × joint_position_j)` — approximately −0.207 in Y for the GNM Head model.

**Fix** (`web/renderer.js:314-317`): Remove the `-Rj` subtraction:

```javascript
const T = makeTranslation(jw.x, jw.y, jw.z);  // CORRECT
skinningMat = T * R * invBind;
```

**Verification:** With the fix applied and all sliders at zero, the head center should match the camera target Y. Open the browser console and check the diagnostic logs:
- `Head center` should be approximately `(0, 0.236, 0.027)`
- `Distance` should be approximately `0.577`
- The head should be centered with equal padding above the crown and below the chin

## Head is too close / too far

**Symptoms:** The head fills either too much or too little of the viewport.

**Fix:** Adjust the camera distance formula in `initScene()`:

```javascript
const distance = (maxDim / 2) / Math.tan(fovRad / 2) * 1.4;
```

Increase `1.4` for more padding (smaller head), decrease for tighter framing. The default 1.4× margin produces ~71% viewport fill for the GNM Head model (height ≈ 0.342) at 45° FOV.

**Diagnostic:** The `Distance:` console log prints the computed value. Compare against the head's bounding box size (also logged).

## Model fails to load ("HTTP 404: ... not found")

**Symptoms:** Browser shows "Loading Failed!" with a 404 error for one or more binary buffers.

**Fix:** The GNM basis buffers must be exported before the web server starts:

```bash
python tools/export_basis.py
```

This generates files in `data/web/`. The server serves these as static files from the `/data/web/` path.

**Diagnostic:** Check `data/web/` directory — all 10 files (metadata.json, 7 binary buffers, 2 optional) should exist with non-zero sizes.

## Mouth does not animate during speech

**Symptoms:** Audio plays but the 3D model's mouth stays still or barely moves.

**Root cause 1 (viseme coefficients):** The GNM expression basis is not organized by anatomical region. Setting `lower_face_region_000` (index 0) does NOT open the jaw — it raises the upper lip. The true jaw-opening component is `lower_face_region_001` (index 1). Viseme coefficients must be tuned against the actual GNM model.

**Root cause 2 (emotion blend interference):** The emotion CVAE outputs coefficients that span the entire lower face with magnitudes up to 4.7 (BLOW). If the emotion blender uses additive superposition, emotion can overwhelm speech. The speech-energy gate fixes this: when `sum(|speechCoeffs|) > 0.01`, the lower face (channels 200–349) is driven by speech only; emotion only affects the upper face (0–199).

**Fix:** Both issues were resolved in `animation_controller.js` (speech-energy gate) and the viseme table (`data/viseme_table.json`) with hand-tuned GNM-expression-basis coefficients.

## Head appears with missing triangles or holes

**Symptoms:** Parts of the mesh are invisible, or triangle-shaped holes appear.

**Root cause:** The exported triangle index buffer uses GNM's `triangles_group("~eye_exteriors")` which excludes interior eye triangles. Additionally, the mouth interior and eye sockets are separate mesh components in GNM that are not rendered.

**This is expected behavior** — the model renders as an external head mesh with eye cavities and mouth socket opening. The eyes, teeth, and tongue are internal components that could be rendered in a future pass.

## OrbitControls causes camera to jump on interaction

**Symptoms:** The first click-drag on the model causes the camera to suddenly reposition.

**Root cause:** Three.js r128's OrbitControls constructor calls `update()` internally, locking spherical coordinates relative to the default target `(0,0,0)`. If `controls.target` is changed after construction without also updating the camera position, the next `controls.update()` recomputes camera.position as `new_target + old_spherical_offset`, doubling the offset.

**Fix** (`web/renderer.js:194-200`): Both `controls.target` and `camera.position` must be set to their desired final values before calling `controls.update()`:

```javascript
controls.target.copy(center);
camera.position.set(center.x, center.y, center.z + distance);
controls.update();
```

## Emotion blend feels weak or absent

**Symptoms:** Changing the emotion selector has no visible effect on the face.

**Root cause:** The speech-energy gate (see "Mouth does not animate" above) suppresses lower-face emotion whenever `sum(|speechCoeffs|) > 0.01`. If the idle speech coefficients contain noise above this threshold, emotion in the lower face is permanently muted.

**Diagnostic:** Check the browser console — `blended` array values in channels 200–349. If speech energy is > 0.01 during idle (no utterance playing), the threshold may need adjustment in `animation_controller.js:99`:

```javascript
const speechEnergy = speechCoeffs.reduce((a, b) => a + Math.abs(b), 0);
const isSpeaking = speechEnergy > 0.01;  // <-- adjust this threshold
```

## Audio plays but visemes don't sync

**Symptoms:** The mouth moves but the timing is noticeably off from the audio.

**Root cause:** The `AudioSync` class uses Web Audio API's `currentTime` which runs on a separate clock from `requestAnimationFrame`. Small drift is normal (< 50ms). Large drift indicates the audio buffer sample rate doesn't match the viseme timeline's reference clock.

**Diagnostic:** Compare `audioSync.getCurrentTime()` output (displayed in the UI) against the viseme timeline's `start_time`/`end_time` fields from the server response.

## Server fails to start

**Symptoms:** `python src/server.py` exits with an error.

**Common fixes:**
- Ensure the virtual environment is activated: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Download the Piper voice model: `python tools/download_models.py`
- Check port availability: port 8080 must be free

## All 31 unit tests pass but integration tests fail

**Symptoms:** Server integration tests (9 endpoints) return errors even though unit tests pass.

**Root cause:** The server must be running in a separate process:

```bash
# Terminal 1
python src/server.py &

# Terminal 2
python tests/test_server_endpoints.py
```

Integration tests connect to `http://localhost:8080` and will fail if the server is not running.
