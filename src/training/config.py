"""Training configuration system.

Usage:
    config = TrainingConfig.from_yaml('configs/default.yaml')
    # or use defaults:
    config = TrainingConfig()
"""

import os
import yaml
import copy
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-4
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    feature_type: str = 'logmel'
    feature_kwargs: dict = field(default_factory=lambda: {'n_mels': 80})

    loss_position_weight: float = 1.0
    loss_velocity_weight: float = 0.5
    loss_acceleration_weight: float = 0.2
    loss_edge_weight: float = 0.1
    loss_vertex_weight: float = 0.0
    loss_regularization_weight: float = 1e-4

    data_dir: str = "voca/reprojected"
    checkpoint_dir: str = "voca/model/checkpoints"
    checkpoint_name: str = "best_model.pt"

    use_vertex_loss: bool = False
    gnm_model_path: str = "vendor/GNM"

    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(copy.deepcopy(self.__dict__), f, default_flow_style=False)


DEFAULT_CONFIG_PATH = "configs/default_config.yaml"


def ensure_default_config():
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
        TrainingConfig().to_yaml(DEFAULT_CONFIG_PATH)
        print(f"Created default config at {DEFAULT_CONFIG_PATH}")
