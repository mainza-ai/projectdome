import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

class VocasetDataset(Dataset):
    def __init__(self, data_dir: str = "voca/reprojected", split: str = "train", train_ratio: float = 0.8, val_ratio: float = 0.1):
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        if not self.files:
            raise FileNotFoundError(f"No reprojected NPZ files found in {data_dir}. Run reproject_vocaset.py first.")

        # Seed split splits
        rng = np.random.default_rng(42)
        indices = np.arange(len(self.files))
        rng.shuffle(indices)

        n_total = len(self.files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if split == "train":
            self.indices = indices[:n_train]
        elif split == "val":
            self.indices = indices[n_train:n_train + n_val]
        else:
            self.indices = indices[n_train + n_val:]

        self.files = [self.files[i] for i in self.indices]

        # Mel-spectrogram transformation parameters
        self.n_mels = 80
        self.sample_rate = 16000 # Wav2TextGrid uses 16kHz, Piper can be resampled

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        audio = data["audio"]
        sr = int(data["sample_rate"])
        coeffs = data["coefficients"] # shape (seq_len, 182)

        # 1. Convert audio to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        # 2. Resample to 16kHz if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio_tensor = resampler(audio_tensor)

        # 3. Extract log-mel-spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_length=400, # 25ms window
            hop_length=160, # 10ms hop
            n_mels=self.n_mels
        )
        # shape (1, n_mels, time_steps)
        mel_spec = mel_transform(audio_tensor).squeeze(0)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))

        # Transpose to (time_steps, n_mels)
        log_mel = log_mel.T

        # 4. Interpolate audio features to match coefficient sequence length
        # coeffs shape is (seq_len, 182)
        seq_len = coeffs.shape[0]
        
        # Grid sample / interpolation expects (Batch, Channel, Height, Width) or (Batch, Channel, Width)
        # Let's do a simple 1D linear interpolation
        log_mel_input = log_mel.unsqueeze(0).transpose(1, 2) # (1, n_mels, time_steps)
        log_mel_resampled = torch.nn.functional.interpolate(
            log_mel_input,
            size=seq_len,
            mode='linear',
            align_corners=False
        )
        log_mel = log_mel_resampled.squeeze(0).transpose(0, 1) # (seq_len, n_mels)

        return {
            "audio_features": log_mel,            # (seq_len, 80)
            "coefficients": torch.tensor(coeffs, dtype=torch.float32),  # (seq_len, 182)
            "seq_len": seq_len
        }

def collate_fn(batch):
    """Pads sequences in the batch to the same length."""
    audio_features = [item["audio_features"] for item in batch]
    coefficients = [item["coefficients"] for item in batch]
    seq_lens = [item["seq_len"] for item in batch]

    # Pad sequences
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio_features, batch_first=True, padding_value=-20.0)
    coeffs_padded = torch.nn.utils.rnn.pad_sequence(coefficients, batch_first=True, padding_value=0.0)

    return {
        "audio_features": audio_padded,      # (batch, max_len, 80)
        "coefficients": coeffs_padded,        # (batch, max_len, 182)
        "seq_lens": torch.tensor(seq_lens)
    }
