# Path B — Neural Regression

A trained neural network that predicts continuous GNM mesh deformations directly from audio. Captures coarticulation and subtle dynamics that [[mapping-path-a|Path A]] cannot.

## Architecture (FaceDiffuser-inspired)

```
Audio → HuBERT embeddings → GRU / non-causal Transformer → Diffusion denoiser → GNM coefficients
```

1. **HuBERT** — self-supervised audio feature extractor (robust to noise, captures prosody)
2. **Temporal decoder** — GRU or Transformer processes the feature sequence
3. **Diffusion denoiser** — iteratively refines random noise into a coefficient sequence conditioned on audio features
4. **Output layer** — linear layer projecting to 182 dimensions (lower face + tongue; ExpressionSampler handles upper face)

## Why diffusion over deterministic regression

Deterministic models always produce the same output for a given input → rigid, uncanny. Diffusion introduces controlled stochasticity → subtle micro-expressions and natural variation that cross the uncanny valley.

## Training data: re-projection pipeline

External datasets don't ship in GNM's coordinate space. The bridging step:

1. Acquire a 3D speech dataset (e.g. [[datasets#vocaset|VOCASET]])
2. Run `fitting_utils/project_on_pca.py` to re-project each frame into GNM's 383-dim expression space
3. Result: `(audio, GNM_coefficient_sequence)` pairs for training

## Synthetic dataset generation (commercial-safe)

All public 3D speech datasets (VOCASET, BIWI, Multiface, MEAD) use research-only licenses. To build a commercially-safe model:

1. Aggregate public-domain text corpus
2. Generate audio with Kokoro TTS
3. Forced-align with Wav2TextGrid
4. Generate GNM coefficients via enhanced Path A (with automated coarticulation smoothing)
5. Inject emotional variance via ExpressionSampler

This produces an infinitely scalable, legally unencumbered synthetic dataset. Train the FaceDiffuser on this, then fine-tune on a small amount of self-captured 4D data.

## When to use

Phases 3–5 of the [[roadmap|roadmap]]. Start after Path A is working end-to-end.
