import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ConvFrontend(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, size_factor: float = 1.0):
        super().__init__()
        f = size_factor
        self.conv1 = nn.Conv1d(in_dim, int(32 * f), kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(int(32 * f), int(32 * f), kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(int(32 * f), int(64 * f), kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(int(64 * f), int(64 * f), kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(int(64 * f))
        self.final_proj = nn.Linear(int(64 * f), hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.final_proj(x)
        return x


class SpeechToCoefficientsModel(nn.Module):
    def __init__(self, audio_dim: int = 80, output_dim: int = 182, hidden_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 8, num_speakers: int = 12,
                 use_conv_frontend: bool = False, conv_size_factor: float = 1.0):
        super().__init__()
        self.use_conv_frontend = use_conv_frontend
        self.input_proj_dim = hidden_dim

        if use_conv_frontend:
            self.frontend = ConvFrontend(audio_dim, hidden_dim, size_factor=conv_size_factor)
        else:
            self.frontend = nn.Linear(audio_dim, hidden_dim)

        self.speaker_embed = nn.Embedding(num_speakers, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, audio_features: torch.Tensor, speaker_ids: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.frontend(audio_features)
        if speaker_ids is not None:
            embed = self.speaker_embed(speaker_ids).unsqueeze(1)
            x = x + embed
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        out = self.output_head(x)
        return out
