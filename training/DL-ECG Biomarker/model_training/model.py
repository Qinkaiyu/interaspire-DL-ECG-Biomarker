from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .backbone import ECGFounderBackbone


class CoxHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


class ECGBottleneck(nn.Module):
    def __init__(self, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TabularWarmupModel(nn.Module):
    """Phase 1 tabular survival model."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = TabularEncoder(input_dim, hidden_dim, dropout)
        self.head = CoxHead(hidden_dim)

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(tabular))


class Model5(nn.Module):
    """Final two-phase multimodal Model 5 architecture.

    Raw 12-lead ECG is encoded by a pretrained, frozen ECGFounder backbone and
    compressed to 32 dimensions. The ECG and tabular embeddings are fused for
    Cox survival prediction. A second head predicts ECG biomarkers from the ECG
    bottleneck during training.
    """

    def __init__(
        self,
        tabular_dim: int,
        n_biomarkers: int,
        backbone_checkpoint: str | Path,
        bottleneck_dim: int = 32,
        tabular_hidden_dim: int = 64,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.ecg_backbone = ECGFounderBackbone(backbone_checkpoint, freeze=True)
        self.ecg_bottleneck = ECGBottleneck(bottleneck_dim, dropout)
        self.tab_encoder = TabularEncoder(
            tabular_dim,
            tabular_hidden_dim,
            dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(bottleneck_dim + tabular_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.surv_head = CoxHead(fusion_hidden_dim)
        self.biomarker_head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, n_biomarkers),
        )

    def forward(
        self,
        ecg: torch.Tensor,
        tabular: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ecg_embedding = self.ecg_backbone(ecg)
        ecg_features = self.ecg_bottleneck(ecg_embedding)
        tabular_features = self.tab_encoder(tabular)
        fused = self.fusion(torch.cat([ecg_features, tabular_features], dim=1))
        log_risk = self.surv_head(fused)
        biomarkers = self.biomarker_head(ecg_features)
        return log_risk, biomarkers

    def predict_risk(self, ecg: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self(ecg, tabular)[0]

    def initialize_tabular_branch(self, warmup_model: TabularWarmupModel) -> None:
        self.tab_encoder.load_state_dict(warmup_model.encoder.state_dict())

    def tabular_parameters(self):
        return self.tab_encoder.parameters()

    def ecg_head_parameters(self):
        return self.ecg_bottleneck.parameters()

    def fusion_parameters(self):
        yield from self.fusion.parameters()
        yield from self.surv_head.parameters()
        yield from self.biomarker_head.parameters()
