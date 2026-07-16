# Speech Animation Production Plan

Comprehensive implementation plan for production-quality speech-driven mouth animation on the GNM Head model. Based on a full architecture audit of the GNM library, the web renderer, the animation pipeline, and the VOCA reference implementation.

## Architecture Audit Summary

The GNM Head v3.0 model expresses mesh deformation through two independent mechanisms:

| Mechanism | Components | Max displacement per unit | Used for speech? |
|-----------|-----------|--------------------------|-----------------|
| Expression PCA basis | 383 components (150 lower-face) | **0.008** (lower face) | **Currently** — but 0.008 is far too small |
| Joint rotations (LBS) | 4 joints (neck, head, eyes) | **0.018** at 5.7° pitch | **No** — only neck/head pose, not speech |
| Pose correctives | 36×53463 regressor | All zeros | N/A — not shipped in v3.0 HEAD |

**The expression PCA basis produces 0.006 units of jaw displacement at coefficient value 2.5 — this is 1.7% of head height, barely 2-3 pixels at the current camera framing.** A −5.7° head pitch rotation produces 3× more mouth-region displacement through LBS alone.

## Gap Analysis (8 gaps identified)

### Gap 1: No jaw joint in the skeleton
**File:** `vendor/GNM/gnm/shape/gnm_xnp.py` (model definition)
**Impact:** Jaw opening must be approximated via expression PCA. Coefficients need values of 6-12 for visible opening, but the CVAE only outputs in range [−2, 2] and the viseme table uses 2.5.
**Options:**
- **A (quick):** Scale lower-face viseme coefficients 10-20× in `data/viseme_table.json`. Test for non-anatomical artifacts visually.
- **B (architectural):** Use differential neck/head joint rotation to open the mouth via LBS. Rotating neck +0.1 rad while counter-rotating head −0.05 rad produces 3× more mouth displacement than PCA. Requires the speech model to output joint axis-angle values.

### Gap 2: Speech-energy gate is a binary switch
**File:** `web/animation_controller.js:96-109`
**Impact:** The lower face switches entirely between "speech" or "emotion" states, removing any emotion contribution during speech. Combined with the tiny PCA magnitude (Gap 1), the remaining pure-speech displacement is imperceptible. The `||` operator on the tongue line (line 106) is also a correctness bug — falsy values silently fall through to emotion.
**Fix:** Revert to the proven additive blend:
```javascript
blended[200 + i] = speechCoeffs[i] + 0.3 * emotionCoeffs[200 + i];
blended[350 + i] = speechCoeffs[150 + i] + 0.3 * emotionCoeffs[350 + i];
```

### Gap 3: Pose correctives are all zeros
**File:** `vendor/GNM/gnm/shape/gnm_common.py:448-497`
**Impact:** Joint rotations produce rigid-body deformation via LBS but no skin-bulging or muscle deformation. Limits realism of neck/head movement.
**Fix:** GNM model data limitation — v3.0 HEAD does not ship pose correctives. Would need to be learned from 3D scans. Accept limitation; prioritize Gap 1B (joint-rotation-based mouth opening) instead.

### Gap 4: PCA coefficient scale mismatch with speech
**File:** `data/viseme_table.json`
**Impact:** Speech requires coefficient values 5-20× larger than emotion CVAE outputs. Scaling up risks non-anatomical deformations since PCA is trained on natural face variation.
**Fix:** Short-term: multiply lower-face coefficients by 10 and test. Long-term: train Path B neural model on speech data — it learns the correct coefficient range for speech in GNM space.

### Gap 5: Web renderer LBS formula deviation (resolved)
**File:** `web/renderer.js:311-321`
**Impact:** Previously had double-invBind bug shifting all vertices by −0.207 in Y. Fixed in commit `8178cdf`. Should now match GNM library formula.
**Fix:** Verify with regression test comparing web LBS output vs GNM Python output for the same input coefficients and joint rotations.

### Gap 6: No neutral-speech identity calibration
**File:** `web/renderer.js:18` (identity coefficients), `web/animation_controller.js` (viseme table)
**Impact:** Viseme coefficients calibrated for mean identity. Different face shapes (from "Generate Identity") may require different coefficient magnitudes for equivalent lip movement.
**Fix:** Add identity-aware viseme scaling: for each identity, render the "aa" viseme and measure actual vertex displacement, then scale coefficients proportionally to maintain consistent lip opening.

### Gap 7: Alignment pipeline has no fallback
**File:** `web/animation_controller.js:42-44`
**Impact:** If Wav2TextGrid forced alignment fails (silence, noise, long pause), the viseme timeline is empty and `getSpeechCoefficients()` always returns IDLE (all zeros) — no mouth movement with no user-visible error.
**Fix:** When alignment returns < 2 phonemes, estimate viseme duration from audio duration and average English phoneme rate (~10 phonemes/sec). Distribute viseme events evenly across the audio.

### Gap 8: No GPU compute for deformation
**File:** `web/renderer.js:233-347`
**Impact:** Identity + expression basis application + LBS runs on CPU in JavaScript. 17821 vertices × 383 components via sparse iteration can take 8-15ms per frame, risking frame drops.
**Fix:** Implement WebGPU compute shader (WGSL). Or pre-compute identity deformation (changes rarely). Or select top-K expression components per frame.

## Implementation Plan (Priority Order)

| Priority | Gap | File(s) | Effort | Fix | Status |
|----------|-----|---------|--------|-----|--------|
| **P0** | **#2: Binary blend switch** | `web/animation_controller.js` | 3 lines | Revert to additive blend. Fix `\|\|` bug. | **DONE** |
| **P0** | **#1A: PCA scale too small** | `data/viseme_table.json` | Coefficient tuning | Scale lower-face coefficients 10×. Test for artifacts. | **DONE** |
| **P0** | **#7: Alignment fallback** | `web/animation_controller.js` | ~20 lines | Add phoneme-rate fallback in `getSpeechCoefficients()` | **DONE** |
| **P1** | **#4: Neural training (Path B)** | `src/training/` | Major | Train `SpeechToCoefficientsModel` on VOCASET reprojected to GNM space |
| **P2** | **#5: LBS verification** | `web/renderer.js`, tests | 1 test | Unit test comparing web LBS output vs GNM Python output |
| **P2** | **#6: Identity calibration** | `web/animation_controller.js` | 1 module | Identity-aware viseme coefficient scaling |
| **P3** | **#1B: Joint-rotation speech** | `web/renderer.js`, `src/training/` | Major | Add axis-angle output to neural model; apply neck/head differential rotation at runtime |
| **P3** | **#8: WebGPU compute** | `web/` | Major | WGSL compute shader for deformation pipeline |
| **P3** | **#3: Pose correctives** | Model data | N/A | Await GNM v3.1+ or learn from 3D scans |

## Detailed Fix Specifications

### P0-1: Revert blend function (`web/animation_controller.js`)

Current (broken):
```javascript
if (isSpeaking) {
    for (let i = 0; i < 150; i++) blended[200 + i] = speechCoeffs[i];
} else {
    for (let i = 0; i < 150; i++) blended[200 + i] = emotionCoeffs[200 + i];
}
for (let i = 0; i < 32; i++) blended[350 + i] = speechCoeffs[150 + i] || emotionCoeffs[350 + i];
```

Fixed:
```javascript
const speechEnergy = speechCoeffs.reduce((a, b) => a + Math.abs(b), 0);
const isSpeaking = speechEnergy > 0.01;
if (isSpeaking) {
    for (let i = 0; i < 150; i++) blended[200 + i] = speechCoeffs[i] + 0.3 * emotionCoeffs[200 + i];
} else {
    for (let i = 0; i < 150; i++) blended[200 + i] = emotionCoeffs[200 + i];
}
for (let i = 0; i < 32; i++) blended[350 + i] = speechCoeffs[150 + i] + 0.3 * emotionCoeffs[350 + i];
```

### P0-2: Scale viseme coefficients (`data/viseme_table.json`)

Multiply all lower-face region coefficients (indices 0-149 of the 182-dim viseme space, which map to GNM channels 200-349) by 10. For example:

| Viseme | Current `[1]` (jaw open) | After 10× |
|--------|--------------------------|-----------|
| aa     | 2.5                      | 25.0      |
| EE     | 1.0                      | 10.0      |
| PP     | −1.5                     | −15.0     |
| OO     | −1.0                     | −10.0     |

Leave tongue coefficients (indices 150-181) at current scale since they produce 0.012 displacement per unit — already reasonable for tongue movement.

**Verification:** After scaling, load the page and speak "Hello world." The jaw should visibly open during vowel sounds (aa, EE, OO). If the mesh shows non-anatomical stretching or pinching, reduce the multiplier.

### P0-3: Alignment fallback (`web/animation_controller.js`)

In `getSpeechCoefficients()`, if `timeline` is empty or contains fewer than 2 events, synthesize a fallback timeline at ~10 visemes/second:

```javascript
if (!timeline || timeline.length < 2) {
    // Fallback: distribute visemes at average phoneme rate
    const duration = audioSync.getDuration();
    if (duration > 0.1) {
        const fallback = [];
        const visemeNames = ["aa", "EE", "OO", "PP", "FF", "TH", "DD", "CH", "kk", "SS", "RR", "schwa"];
        const avgInterval = 0.1; // ~10 visemes/sec
        let t = 0;
        while (t < duration) {
            fallback.push({
                name: visemeNames[Math.floor(Math.random() * visemeNames.length)],
                start_time: t,
                end_time: Math.min(t + avgInterval, duration)
            });
            t += avgInterval;
        }
        return this.getSpeechCoefficients(timeS, fallback);
    }
    return new Array(182).fill(0.0);
}
```

### P3-1: Joint-rotation speech output (architectural)

The neural model (`SpeechToCoefficientsModel`) currently outputs 383 expression coefficients. To add joint-rotation-based mouth opening:

1. Extend the model output to include 4 joint axis-angle vectors: `383 expression + 4×3 rotation = 395 outputs`
2. In the training data, compute joint rotations that produce the target mouth shapes. For jaw opening: rotate neck forward and head backward.
3. At runtime, apply the predicted rotations to the neck and head joints before LBS.
4. The PCA expression coefficients then only need to add fine detail on top of the joint-based mouth opening.

This requires the joint regressor to produce good neck/head positions for the rotation pivot. The existing regressor already outputs joint positions at rest — the rotation happens around these points.
