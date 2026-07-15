# GNM Sanity Check

`src/gnm_sanity_check.py` — verifies that the GNM model loads and deforms correctly.

## Purpose

A standalone script that:
1. Loads GNM v3.0 Head model
2. Generates a neutral mesh (all-zero expression) → saves as `output/neutral.obj`
3. Generates a HAPPY expression mesh → saves as `output/happy.obj`
4. Computes max/mean vertex displacement between neutral and happy

**Pass criteria:** max displacement > 0.005 units

## Usage

```bash
./venv/bin/python src/gnm_sanity_check.py
```

Expected output:
```
Neutral mesh saved to output/neutral.obj (17821 vertices)
Happy mesh saved to output/happy.obj (17821 vertices)
Max vertex displacement: 0.012345 units — PASSED
```
