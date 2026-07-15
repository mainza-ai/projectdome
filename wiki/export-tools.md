# Export & Tuning Tools

`tools/` — utility scripts for preparing web assets and tuning viseme coefficients.

## Basis Exporter

`tools/export_basis.py` — extracts GNM model data and exports it as binary buffers for the web renderer.

**Outputs to `data/web/`:**

| File | Type | Content |
|---|---|---|
| `mean_positions.bin` | float32 | Template vertex positions |
| `identity_basis.bin` | float16 | 253 identity PCA basis vectors |
| `expression_basis.bin` | float16 | 383 expression PCA basis vectors |
| `face_indices.bin` | uint32 | Triangle connectivity |
| `skinning_weights.bin` | float32 | 4-joint LBS weights |
| `joint_regressor.bin` | float32 | Joint position regressor |
| `metadata.json` | JSON | Model spec (vertex count, joint names, etc.) |

Identity and expression bases are stored as float16 to reduce download size. The web renderer decodes them at load time.

## Viseme Tuner

`tools/tune_visemes.py` — interactive CLI tool for editing viseme coefficients and previewing results as OBJ files.

**Commands:**
- `list` — show all 13 viseme names
- `show <viseme>` — display non-zero coefficients with human-readable labels
- `set <viseme> <idx> <val>` — set coefficient index (0–181) to a value
- `export <viseme>` — export viseme as OBJ to `output/tune_<viseme>.obj`
- `save` — persist changes to `data/viseme_table.json`
- `help` / `exit` — standard controls

Coefficient labels map to GNM expression names (lower_face_region_000–149, tongue_000–031).
