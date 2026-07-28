from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    id_column: str
    ecg_path_column: str
    ecg_mat_key: str
    time_column: str
    event_column: str
    fold_column: str | None
    tabular_features: list[str]
    auxiliary_continuous_columns: list[str]
    auxiliary_binary_columns: list[str]

    @property
    def auxiliary_columns(self) -> list[str]:
        return self.auxiliary_continuous_columns + self.auxiliary_binary_columns

    @classmethod
    def from_json(cls, path: str | Path) -> "DataConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**payload)
        if not config.tabular_features:
            raise ValueError("tabular_features must not be empty")
        if not config.auxiliary_columns:
            raise ValueError("At least one auxiliary biomarker column is required")
        duplicates = sorted(
            {name for name in config.tabular_features if config.tabular_features.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"Duplicate tabular feature names: {duplicates}")
        return config

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    n_folds: int = 5
    random_seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    bottleneck_dim: int = 32
    tabular_hidden_dim: int = 64
    fusion_hidden_dim: int = 64
    dropout: float = 0.3
    auxiliary_weight: float = 0.1
    l1_weight: float = 1e-3
    phase1_lr: float = 3e-4
    phase1_weight_decay: float = 1e-2
    phase1_max_epochs: int = 150
    phase1_patience: int = 25
    phase2_tabular_lr: float = 1e-4
    phase2_ecg_lr: float = 3e-4
    phase2_weight_decay: float = 1e-2
    phase2_max_epochs: int = 100
    phase2_patience: int = 20
    warmup_epochs: int = 5

    def to_dict(self) -> dict:
        return asdict(self)
