import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Optional

class LogMelFeatures(nn.Module):
    def __init__(self, n_mels: int = 80, sample_rate: int = 16000):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024,
            win_length=400, hop_length=160, n_mels=n_mels
        )

    def forward(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)
        mel = self.transform(audio).squeeze(0)
        return torch.log(torch.clamp(mel, min=1e-5)).T


class HubertFeatures(nn.Module):
    def __init__(self, model_name: str = "facebook/hubert-base-ls960", layer: int = 9):
        super().__init__()
        try:
            from transformers import HubertModel
            self.model = HubertModel.from_pretrained(model_name)
            self.layer = layer
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")
        self.sample_rate = 16000

    def forward(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(audio)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(audio, output_hidden_states=True)
            features = outputs.hidden_states[self.layer]
        return features.squeeze(0)


FEATURE_REGISTRY = {
    'logmel': LogMelFeatures,
    'hubert': HubertFeatures,
}

def get_feature_extractor(name: str = 'logmel', **kwargs) -> nn.Module:
    cls = FEATURE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown feature extractor: {name}. Options: {list(FEATURE_REGISTRY.keys())}")
    return cls(**kwargs)
