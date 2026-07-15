# VOCASET Reprojection

`src/training/reproject_vocaset.py` — bridges the gap between external 3D face datasets and GNM's native coordinate space.

## The problem

GNM is a PCA-based parametric model. VOCASET meshes are in FLAME topology (5,023 vertices), not GNM topology (17,821 vertices). You can't train a GNM regressor directly on VOCASET data.

## The solution: ICP alignment + PCA projection

### Step 1: ICP alignment per speaker

For each of the 12 VOCASET speakers, compute an ICP (Iterative Closest Point) alignment between their FLAME template mesh and the GNM template mesh:

1. Centroid alignment
2. Iterative closest-point matching (up to 50 iterations)
3. SVD-based rotation estimation
4. Returns: rotation matrix `R`, translation vector `t`, vertex correspondence indices

This maps the 5,023 FLAME vertices onto a subset of the 17,821 GNM vertices.

### Step 2: PCA batch projection

For each sentence sequence:

1. Read target FLAME meshes from `data_verts.npy` (memory-mapped)
2. Align the entire sequence: `aligned = (flame_meshes @ R.T) + t`
3. Create a PCA basis projector using GNM's expression basis, scoped to the aligned vertex subset
4. Project the sequence to get 383-dim GNM coefficients
5. Extract speech-relevant subset: indices 200–382 (lower face + tongue) → **182-dim coefficient vectors**

### Output

Each sentence is saved as an `.npz` file in `voca/reprojected/`:
- `audio` — raw audio waveform
- `sample_rate` — original sample rate
- `coefficients` — (seq_len, 182) GNM speech coefficients

Format: `FaceTalk_YYMMDD_XXXXX_TA_sentence.npz`

## Dependencies

- GNM v3.0 model (for expression basis and template)
- VOCASET training data (`voca/trainingdata/`): templates, sequence indices, raw audio, vertex data
- SciPy KDTree for ICP matching
