# GNM Head

**Google GNM Head** — an Apache 2.0 licensed parametric 3D Morphable Model (3DMM) serving as the geometric foundation for Project Dome.

## Specification

| Property | Value |
|---|---|
| Vertices | 17,821 |
| Identity parameters | 253 |
| Expression parameters | 383 |
| Joint skeleton | 4-joint (linear blend skinning + pose correctives) |
| Compute cost | ~41 MFLOPs per frame |
| License | Apache 2.0 |

## Expression parameter groups

| Group | Count | Range |
|---|---|---|
| Left eye region | 100 | `left_eye_region_000`–`099` |
| Right eye region | 100 | `right_eye_region_000`–`099` |
| Lower face region | 150 | `lower_face_region_000`–`149` |
| Tongue | 32 | `tongue_mean` + `tongue_000`–`030` |
| Pupils | 1 | `pupils_000` |

## ExpressionSampler (CVAE)

The built-in `expression_decoder_model.h5` is a Conditional Variational Autoencoder that maps 20 semantic labels to 383-dimensional expression coefficients:

`HAPPY`, `SURPRISE`, `DISGUST`, `SQUINT`, `SMILE_WIDE`, `SAD`, `CORNERS_DOWN`, `PUCKER`, `LIPS_ROLL_IN`, `SNARL`, `SUCK`, `COMPRESS_FACE`, `STRETCH_FACE`, `PLATYSMA`, `BLOW`, `FUNNELER`, `WINK_LEFT`, `WINK_RIGHT`, `MOUTH_LEFT`, `MOUTH_RIGHT`, `TONGUE_CENTER`

## Key property: linear additivity

GNM is a linear model — expression coefficient vectors can be **additively blended**. This means the emotion signal (from ExpressionSampler) and the speech signal (from viseme mapping) can be summed without interfering:

```
final_vertices = template + identity_basis·id + expression_basis·expr
```

## Fitting utility

`fitting_utils/project_on_pca.py` projects external mesh geometry onto GNM's PCA basis. This is the critical bridge for re-projecting datasets like VOCASET into GNM coefficient space.

## Source

- [github.com/google/GNM](https://github.com/google/GNM) — Apache 2.0
