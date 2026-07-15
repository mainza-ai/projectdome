import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

VOCA_TRAIN_SUBJECTS = [
    'FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA',
    'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA',
    'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA',
    'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA',
]
VOCA_VAL_SUBJECTS = [
    'FaceTalk_170811_03275_TA', 'FaceTalk_170908_03277_TA',
]
VOCA_TEST_SUBJECTS = [
    'FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA',
]
VOCA_ALL_SUBJECTS = VOCA_TRAIN_SUBJECTS + VOCA_VAL_SUBJECTS + VOCA_TEST_SUBJECTS

class VocasetDataset(Dataset):
    def __init__(self, data_dir: str = "voca/reprojected", split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No reprojected NPZ files found in {data_dir}. Run reproject_vocaset.py first.")
        subject_map = {}
        for f in all_files:
            basename = os.path.basename(f)
            parts = basename.split("_")
            speaker = "_".join(parts[:5])
            subject_map.setdefault(speaker, []).append(f)
        if split == "train":
            allowed = VOCA_TRAIN_SUBJECTS
        elif split == "val":
            allowed = VOCA_VAL_SUBJECTS
        elif split == "test":
            allowed = VOCA_TEST_SUBJECTS
        else:
            raise ValueError(f"Unknown split: {split}")
        self.files = []
        for speaker in allowed:
            if speaker in subject_map:
                self.files.extend(subject_map[speaker])
        if not self.files:
            available = list(subject_map.keys())
            raise FileNotFoundError(
                f"No files for split '{split}'. Expected speakers: {allowed}. "
                f"Available speakers: {available}"
            )
        self.n_mels = 80
        self.sample_rate = 16000

    def __len__(self):
        return len(self.files)

    def _parse_speaker_id(self, filename):
        basename = os.path.basename(filename)
        parts = basename.split("_")
        speaker = "_".join(parts[:5])
        try:
            return VOCA_ALL_SUBJECTS.index(speaker)
        except ValueError:
            return 0

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        audio = data["audio"]
        sr = int(data["sample_rate"])
        coeffs = data["coefficients"]
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio_tensor = resampler(audio_tensor)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024,
            win_length=400, hop_length=160, n_mels=self.n_mels
        )
        mel_spec = mel_transform(audio_tensor).squeeze(0)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
        log_mel = log_mel.T
        seq_len = coeffs.shape[0]
        log_mel_input = log_mel.unsqueeze(0).transpose(1, 2)
        log_mel_resampled = torch.nn.functional.interpolate(
            log_mel_input, size=seq_len, mode='linear', align_corners=False
        )
        log_mel = log_mel_resampled.squeeze(0).transpose(0, 1)
        speaker_idx = self._parse_speaker_id(self.files[idx])
        return {
            "audio_features": log_mel,
            "coefficients": torch.tensor(coeffs, dtype=torch.float32),
            "seq_len": seq_len,
            "speaker_id": speaker_idx,
        }

def collate_fn(batch):
    audio_features = [item["audio_features"] for item in batch]
    coefficients = [item["coefficients"] for item in batch]
    seq_lens = [item["seq_len"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio_features, batch_first=True, padding_value=-20.0)
    coeffs_padded = torch.nn.utils.rnn.pad_sequence(coefficients, batch_first=True, padding_value=0.0)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
    return {
        "audio_features": audio_padded,
        "coefficients": coeffs_padded,
        "seq_lens": torch.tensor(seq_lens),
        "speaker_ids": speaker_ids,
    }
