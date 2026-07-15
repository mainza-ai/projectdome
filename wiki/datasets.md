# Datasets

All datasets listed here are **free to download** but carry **research-only / non-commercial licenses**. None is natively in GNM's coordinate space — each requires re-projection via `fitting_utils/project_on_pca.py` before it can be used for training.

## VOCASET (MPI-IS) — primary candidate

| Property | Value |
|---|---|
| Content | ~29 min of 4D head scans at 60fps, 12 speakers |
| Topology | FLAME (shared parametric-scan lineage with GNM) |
| License | Research-only |
| Access | voca.is.tue.mpg.de (registration required) |
| Re-projection | Easiest — FLAME→GNM PCA projection is most tractable |

## BIWI 3D Audiovisual Corpus

| Property | Value |
|---|---|
| Content | 40 sentences × 14 subjects, emotional + neutral, 25fps |
| Topology | Raw scan (23,370 vertices), not FLAME/GNM |
| License | Research use |
| Value | Emotional speech variation (VOCASET is mostly neutral) |
| Re-projection | Heavier — ICP-style mesh alignment before PCA |

## Multiface (Meta)

| Property | Value |
|---|---|
| Content | 13 identities, 40–160 camera views per frame, ~65TB |
| Topology | Tracked meshes |
| License | Meta research license |
| Value | Highest fidelity — for fine detail refinement |
| Note | Too large for initial development; revisit in later phases |

## MEAD

| Property | Value |
|---|---|
| Content | 60 actors, 8 emotions × 3 intensities, 7 viewing angles |
| Topology | 2D video (not 3D geometry) |
| License | Research use |
| Value | Emotion-in-speech reference; can extract ARKit blendshapes for auxiliary supervision |
| Note | Cannot feed mesh regressor directly — use for emotional signal validation |
