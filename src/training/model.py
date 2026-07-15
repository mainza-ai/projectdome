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
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class SpeechToCoefficientsModel(nn.Module):
    def __init__(self, audio_dim: int = 80, output_dim: int = 182, hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 8, num_speakers: int = 12):
        super().__init__()
        self.audio_project = nn.Linear(audio_dim, hidden_dim)
        self.speaker_embed = nn.Embedding(num_speakers, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, audio_features: torch.Tensor, speaker_ids: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        audio_features: (batch, seq_len, audio_dim)
        speaker_ids: (batch,) integer tensor specifying speaker condition index
        src_key_padding_mask: (batch, seq_len) boolean mask for padded items (True means ignore)
        """
        # Project log-mel to hidden dimension
        x = self.audio_project(audio_features)
        
        # Add speaker conditioning embedding if provided
        if speaker_ids is not None:
            embed = self.speaker_embed(speaker_ids).unsqueeze(1) # (batch, 1, hidden_dim)
            x = x + embed
            
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Project to viseme coefficients
        out = self.output_head(x)
        return out
