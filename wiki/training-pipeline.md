# Training Pipeline

`src/training/` — the Path B neural regressor that learns to predict GNM coefficients directly from audio.

## Model Architecture

`src/training/model.py` — `SpeechToCoefficientsModel`

```
audio_features (seq_len, 80) → Linear(hidden_dim) → + SpeakerEmbedding
  → PositionalEncoding → TransformerEncoder (4 layers, 8 heads)
  → Linear(output_dim) → coefficients (seq_len, 182)
```

- Input: 80-dim log-mel-spectrogram frames at 10ms hop
- Output: 182-dim GNM coefficient vectors (lower face + tongue)
- Speaker conditioning: 12-speaker embedding for VOCASET multi-style support
- Positional encoding for sequence order awareness
- Dropout: 0.1

## Dataset

`src/training/dataset.py` — `VocasetDataset`

- Loads reprojected `.npz` files from `voca/reprojected/`
- Train/val/test split: 80%/10%/10% (seeded shuffle)
- On-the-fly log-mel-spectrogram extraction (80 mel bands, 1024 FFT, 25ms window, 10ms hop)
- Resamples audio to 16kHz
- Linear interpolation to align audio features with coefficient sequence length
- Speaker ID parsing from filename convention: `FaceTalk_YYMMDD_XXXXX_TA`
- `collate_fn` — pads variable-length sequences to batch alignment

## Training

`src/training/train.py` — full training loop.

**Loss function (composite):**
- L1 position loss (weight: 1.0) — per-frame coefficient accuracy
- L1 velocity loss (weight: 0.5) — smooth temporal transitions
- L1 acceleration loss (weight: 0.2) — reduce jitter
- L1 sparsity regularizer (weight: 1e-4) — prevent drift

**Optimizer:** AdamW with weight decay 1e-4
**Learning rate:** 1e-4
**Batch size:** 8 (default)
**Epochs:** 30–50 (configurable)
**Device:** CUDA → MPS → CPU auto-detection

Checkpoints saved to `voca/model/checkpoints/best_model.pt` when validation loss improves.

## Evaluation

`src/training/evaluate.py` — loads the best checkpoint and computes L1 position error on the test split.

## File reference

| File | Role |
|---|---|
| `src/training/model.py` | SpeechToCoefficientsModel + PositionalEncoding |
| `src/training/dataset.py` | VocasetDataset + collate_fn |
| `src/training/train.py` | Training loop (loss functions, checkpointing) |
| `src/training/evaluate.py` | Test split evaluation |
