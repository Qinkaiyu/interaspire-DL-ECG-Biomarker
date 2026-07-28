from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .config import DataConfig


def load_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata table not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Metadata must be CSV or Parquet")


def validate_metadata(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    required = {
        config.ecg_path_column,
        config.time_column,
        config.event_column,
        *config.tabular_features,
        *config.auxiliary_columns,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    clean = df.copy()
    clean[config.time_column] = pd.to_numeric(clean[config.time_column], errors="coerce")
    clean[config.event_column] = pd.to_numeric(clean[config.event_column], errors="coerce")
    clean = clean.loc[
        clean[config.time_column].notna()
        & clean[config.event_column].isin([0, 1])
        & (clean[config.time_column] > 0)
        & clean[config.ecg_path_column].notna()
    ].reset_index(drop=True)
    if clean.empty:
        raise ValueError("No valid rows remain after outcome/path validation")
    if clean[config.event_column].sum() == 0:
        raise ValueError("The analysis table contains no events")
    return clean


class ECGSurvivalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tabular_matrix: np.ndarray,
        config: DataConfig,
        ecg_root: str | Path | None,
        load_ecg: bool,
        auxiliary_targets: np.ndarray | None = None,
        auxiliary_mask: np.ndarray | None = None,
        expected_ecg_length: int = 5000,
    ) -> None:
        if len(frame) != len(tabular_matrix):
            raise ValueError("frame and tabular_matrix have different lengths")
        self.frame = frame.reset_index(drop=True)
        self.tabular = np.asarray(tabular_matrix, dtype=np.float32)
        self.config = config
        self.ecg_root = Path(ecg_root) if ecg_root is not None else None
        self.load_ecg = load_ecg
        self.auxiliary_targets = auxiliary_targets
        self.auxiliary_mask = auxiliary_mask
        self.expected_ecg_length = expected_ecg_length

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_ecg_path(self, index: int) -> Path:
        path = Path(str(self.frame.at[index, self.config.ecg_path_column]))
        if not path.is_absolute() and self.ecg_root is not None:
            path = self.ecg_root / path
        return path

    def _load_ecg(self, path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"ECG file not found: {path}")
        if path.suffix.lower() == ".mat":
            payload = loadmat(path)
            if self.config.ecg_mat_key not in payload:
                raise KeyError(
                    f"{path} does not contain MAT key {self.config.ecg_mat_key!r}"
                )
            ecg = np.asarray(payload[self.config.ecg_mat_key], dtype=np.float32)
        elif path.suffix.lower() == ".npy":
            ecg = np.asarray(np.load(path), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported ECG format: {path.suffix}")

        if ecg.ndim != 2:
            raise ValueError(f"Expected a 2-D ECG array, got {ecg.shape} in {path}")
        if ecg.shape[0] != 12 and ecg.shape[1] == 12:
            ecg = ecg.T
        if ecg.shape != (12, self.expected_ecg_length):
            raise ValueError(
                f"Expected ECG shape (12, {self.expected_ecg_length}), "
                f"got {ecg.shape} in {path}"
            )
        return np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.load_ecg:
            ecg = self._load_ecg(self._resolve_ecg_path(index))
        else:
            ecg = np.zeros((12, self.expected_ecg_length), dtype=np.float32)

        item = {
            "ecg": torch.from_numpy(ecg),
            "tabular": torch.from_numpy(self.tabular[index]),
            "time": torch.tensor(
                float(self.frame.at[index, self.config.time_column]),
                dtype=torch.float32,
            ),
            "event": torch.tensor(
                float(self.frame.at[index, self.config.event_column]),
                dtype=torch.float32,
            ),
        }
        if self.auxiliary_targets is not None:
            item["auxiliary_targets"] = torch.from_numpy(
                self.auxiliary_targets[index].astype(np.float32)
            )
            item["auxiliary_mask"] = torch.from_numpy(
                self.auxiliary_mask[index].astype(np.float32)
            )
        return item


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    output = {
        key: torch.stack([item[key] for item in batch])
        for key in ("ecg", "tabular", "time", "event")
    }
    if "auxiliary_targets" in batch[0]:
        output["auxiliary_targets"] = torch.stack(
            [item["auxiliary_targets"] for item in batch]
        )
        output["auxiliary_mask"] = torch.stack(
            [item["auxiliary_mask"] for item in batch]
        )
    return output
