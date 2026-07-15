# Path A — Deterministic Viseme Mapping

The fastest path to a working lip-sync pipeline. No ML training required — pure procedural animation.

## Overview

~40 English phonemes are collapsed into ~12–15 **viseme categories** (visually distinct mouth shapes). A hand-authored lookup table maps each viseme to GNM coefficient vectors. The forced-alignment timeline triggers lookups with linear interpolation.

## Steps

1. **Collapse phonemes to visemes**
   - /p/, /b/, /m/ → closed-lip viseme
   - /t/, /d/, /s/ → shared tongue/jaw position
   - ~15 categories total

2. **Author GNM coefficient vectors per viseme**
   - Use Blender GNM Head Importer add-on (free, open source)
   - Manually pose each viseme using lower face (150) and tongue (32) sliders
   - Export as JSON lookup table

3. **Runtime pipeline**
   ```
   forced-alignment timestamps
         ↓
   viseme category per frame
         ↓
   lookup table → 383-dim coefficient vector
         ↓
   linear/slerp interpolation between keyframes
         ↓
   additively blend with ExpressionSampler emotion vector
         ↓
   final_vertices = template + identity_basis·id + expression_basis·expr
   ```

## Advantages

- Instant — no training, no dataset collection
- Predictable — deterministic output, easy to debug
- Low cost — operates on CPU in milliseconds
- Proven — classic game/VTuber lip-sync for over a decade

## Limitations

- No **coarticulation** — mouth shape for a phoneme doesn't account for neighboring phonemes
- ~15 discrete categories → limited expressiveness
- Hand-authored → doesn't improve over time

## When to use

Phase 1 of the roadmap. Sufficient for an MVP and potentially a v1 product. Path A is the foundation — [[mapping-path-b|Path B]] is the upgrade path.
