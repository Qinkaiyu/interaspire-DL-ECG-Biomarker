from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import DataConfig, TrainingConfig
from .data import ECGSurvivalDataset, collate_batch
from .losses import cox_partial_likelihood_loss
from .metrics import harrell_c_index
from .model import Model5, TabularWarmupModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CosineWarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1,
                self.total_epochs - self.warmup_epochs,
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale


def _safe_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    try:
        return harrell_c_index(time, event, risk)
    except Exception:
        return float("nan")


def _fit_tabular_preprocessor(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    train_raw = train_frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    validation_raw = validation_frame[feature_names].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float32)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_matrix = imputer.fit_transform(train_raw)
    validation_matrix = imputer.transform(validation_raw)
    train_matrix = scaler.fit_transform(train_matrix).astype(np.float32)
    validation_matrix = scaler.transform(validation_matrix).astype(np.float32)
    return train_matrix, validation_matrix, imputer, scaler


def _prepare_auxiliary_targets(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    data_config: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    columns = data_config.auxiliary_columns
    train = train_frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    validation = validation_frame[columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    train_mask = np.isfinite(train)
    validation_mask = np.isfinite(validation)
    train = np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0)
    validation = np.nan_to_num(validation, nan=0.0, posinf=0.0, neginf=0.0)

    statistics: dict[str, dict[str, float]] = {}
    for index, column in enumerate(data_config.auxiliary_continuous_columns):
        observed = train_mask[:, index]
        if observed.sum() < 2:
            mean, scale = 0.0, 1.0
        else:
            mean = float(train[observed, index].mean())
            scale = float(train[observed, index].std())
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
        train[:, index] = (train[:, index] - mean) / scale
        validation[:, index] = (validation[:, index] - mean) / scale
        statistics[column] = {"mean": mean, "scale": scale}

    return train, validation, train_mask, validation_mask, statistics


def _tabular_loader(
    matrix: np.ndarray,
    frame: pd.DataFrame,
    data_config: DataConfig,
    training_config: TrainingConfig,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(matrix),
        torch.from_numpy(frame[data_config.time_column].to_numpy(dtype=np.float32)),
        torch.from_numpy(frame[data_config.event_column].to_numpy(dtype=np.float32)),
    )
    drop_last = shuffle and len(dataset) % training_config.batch_size == 1
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
    )


def _ecg_loader(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    data_config: DataConfig,
    training_config: TrainingConfig,
    ecg_root: str | Path | None,
    device: torch.device,
    shuffle: bool,
    auxiliary_targets: np.ndarray | None,
    auxiliary_mask: np.ndarray | None,
    expected_ecg_length: int,
) -> DataLoader:
    dataset = ECGSurvivalDataset(
        frame=frame,
        tabular_matrix=matrix,
        config=data_config,
        ecg_root=ecg_root,
        load_ecg=True,
        auxiliary_targets=auxiliary_targets,
        auxiliary_mask=auxiliary_mask,
        expected_ecg_length=expected_ecg_length,
    )
    drop_last = shuffle and len(dataset) % training_config.batch_size == 1
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        num_workers=training_config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
        drop_last=drop_last,
    )


def _evaluate_tabular(
    model: TabularWarmupModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for tabular, time, event in loader:
            risk = model(tabular.to(device))
            risks.append(risk.cpu())
            times.append(time)
            events.append(event)
    risk_array = torch.cat(risks).numpy()
    time_array = torch.cat(times).numpy()
    event_array = torch.cat(events).numpy()
    loss = cox_partial_likelihood_loss(
        torch.from_numpy(risk_array),
        torch.from_numpy(time_array),
        torch.from_numpy(event_array),
    ).item()
    return {
        "loss": float(loss),
        "c_index": _safe_c_index(time_array, event_array, risk_array),
    }


def _evaluate_model5(
    model: Model5,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for batch in loader:
            risk = model.predict_risk(
                batch["ecg"].to(device),
                batch["tabular"].to(device),
            )
            risks.append(risk.cpu())
            times.append(batch["time"])
            events.append(batch["event"])
    risk_array = torch.cat(risks).numpy()
    time_array = torch.cat(times).numpy()
    event_array = torch.cat(events).numpy()
    loss = cox_partial_likelihood_loss(
        torch.from_numpy(risk_array),
        torch.from_numpy(time_array),
        torch.from_numpy(event_array),
    ).item()
    return {
        "loss": float(loss),
        "c_index": _safe_c_index(time_array, event_array, risk_array),
        "risk": risk_array,
    }


def train_phase1(
    train_matrix: np.ndarray,
    validation_matrix: np.ndarray,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    data_config: DataConfig,
    training_config: TrainingConfig,
    device: torch.device,
) -> tuple[TabularWarmupModel, dict[str, Any]]:
    train_loader = _tabular_loader(
        train_matrix,
        train_frame,
        data_config,
        training_config,
        shuffle=True,
    )
    validation_loader = _tabular_loader(
        validation_matrix,
        validation_frame,
        data_config,
        training_config,
        shuffle=False,
    )
    model = TabularWarmupModel(
        input_dim=train_matrix.shape[1],
        hidden_dim=training_config.tabular_hidden_dim,
        dropout=training_config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.phase1_lr,
        weight_decay=training_config.phase1_weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        training_config.warmup_epochs,
        training_config.phase1_max_epochs,
    )

    best_c = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience = 0
    history = {"epoch": [], "train_loss": [], "validation_loss": [], "validation_c": []}

    progress = tqdm(range(training_config.phase1_max_epochs), desc="phase 1", leave=False)
    for epoch in progress:
        model.train()
        batch_losses = []
        for tabular, time, event in train_loader:
            optimizer.zero_grad()
            risk = model(tabular.to(device))
            loss = cox_partial_likelihood_loss(
                risk,
                time.to(device),
                event.to(device),
            )
            if training_config.l1_weight > 0:
                loss = loss + training_config.l1_weight * model.head.fc.weight.abs().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        scheduler.step(epoch)

        metrics = _evaluate_tabular(model, validation_loader, device)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["validation_loss"].append(metrics["loss"])
        history["validation_c"].append(metrics["c_index"])
        progress.set_postfix(validation_c=f"{metrics['c_index']:.4f}")

        if np.isfinite(metrics["c_index"]) and metrics["c_index"] > best_c + 1e-5:
            best_c = metrics["c_index"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= training_config.phase1_patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_c": best_c,
        "history": history,
    }


def train_phase2(
    warmup_model: TabularWarmupModel,
    train_matrix: np.ndarray,
    validation_matrix: np.ndarray,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    auxiliary_train: np.ndarray,
    auxiliary_validation: np.ndarray,
    auxiliary_train_mask: np.ndarray,
    auxiliary_validation_mask: np.ndarray,
    data_config: DataConfig,
    training_config: TrainingConfig,
    backbone_checkpoint: str | Path,
    ecg_root: str | Path | None,
    expected_ecg_length: int,
    device: torch.device,
) -> tuple[Model5, np.ndarray, dict[str, Any]]:
    train_loader = _ecg_loader(
        train_frame,
        train_matrix,
        data_config,
        training_config,
        ecg_root,
        device,
        shuffle=True,
        auxiliary_targets=auxiliary_train,
        auxiliary_mask=auxiliary_train_mask,
        expected_ecg_length=expected_ecg_length,
    )
    validation_loader = _ecg_loader(
        validation_frame,
        validation_matrix,
        data_config,
        training_config,
        ecg_root,
        device,
        shuffle=False,
        auxiliary_targets=auxiliary_validation,
        auxiliary_mask=auxiliary_validation_mask,
        expected_ecg_length=expected_ecg_length,
    )

    model = Model5(
        tabular_dim=train_matrix.shape[1],
        n_biomarkers=len(data_config.auxiliary_columns),
        backbone_checkpoint=backbone_checkpoint,
        bottleneck_dim=training_config.bottleneck_dim,
        tabular_hidden_dim=training_config.tabular_hidden_dim,
        fusion_hidden_dim=training_config.fusion_hidden_dim,
        dropout=training_config.dropout,
    ).to(device)
    model.initialize_tabular_branch(warmup_model)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.tabular_parameters()),
                "lr": training_config.phase2_tabular_lr,
            },
            {
                "params": list(model.ecg_head_parameters()),
                "lr": training_config.phase2_ecg_lr,
            },
            {
                "params": list(model.fusion_parameters()),
                "lr": training_config.phase2_ecg_lr,
            },
        ],
        weight_decay=training_config.phase2_weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        training_config.warmup_epochs,
        training_config.phase2_max_epochs,
    )
    mse = nn.MSELoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    n_continuous = len(data_config.auxiliary_continuous_columns)
    continuous_indices = list(range(n_continuous))
    binary_indices = list(range(n_continuous, len(data_config.auxiliary_columns)))

    best_c = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience = 0
    history = {"epoch": [], "train_loss": [], "validation_loss": [], "validation_c": []}

    progress = tqdm(range(training_config.phase2_max_epochs), desc="phase 2", leave=False)
    for epoch in progress:
        model.train()
        batch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            risk, biomarker_logits = model(
                batch["ecg"].to(device),
                batch["tabular"].to(device),
            )
            loss = cox_partial_likelihood_loss(
                risk,
                batch["time"].to(device),
                batch["event"].to(device),
            )
            targets = batch["auxiliary_targets"].to(device)
            mask = batch["auxiliary_mask"].to(device)
            auxiliary_loss = torch.tensor(0.0, device=device)

            if continuous_indices:
                index = torch.tensor(continuous_indices, device=device)
                weighted = mse(
                    biomarker_logits[:, index],
                    targets[:, index],
                ) * mask[:, index]
                auxiliary_loss = auxiliary_loss + weighted.sum() / (
                    mask[:, index].sum() + 1e-8
                )
            if binary_indices:
                index = torch.tensor(binary_indices, device=device)
                weighted = bce(
                    biomarker_logits[:, index],
                    targets[:, index],
                ) * mask[:, index]
                auxiliary_loss = auxiliary_loss + weighted.sum() / (
                    mask[:, index].sum() + 1e-8
                )
            loss = loss + training_config.auxiliary_weight * auxiliary_loss

            if training_config.l1_weight > 0:
                l1 = sum(
                    parameter.abs().sum()
                    for name, parameter in model.named_parameters()
                    if ("head" in name or "bottleneck" in name) and "weight" in name
                )
                loss = loss + training_config.l1_weight * l1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        scheduler.step(epoch)

        metrics = _evaluate_model5(model, validation_loader, device)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["validation_loss"].append(metrics["loss"])
        history["validation_c"].append(metrics["c_index"])
        progress.set_postfix(validation_c=f"{metrics['c_index']:.4f}")

        if np.isfinite(metrics["c_index"]) and metrics["c_index"] > best_c + 1e-5:
            best_c = metrics["c_index"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= training_config.phase2_patience:
                break

    model.load_state_dict(best_state)
    final_validation = _evaluate_model5(model, validation_loader, device)
    return model, final_validation["risk"], {
        "best_epoch": best_epoch,
        "best_validation_c": best_c,
        "history": history,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _split_plan(
    frame: pd.DataFrame,
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    fold_column = data_config.fold_column
    if fold_column and fold_column in frame.columns:
        if frame[fold_column].isna().any():
            raise ValueError(f"Fixed fold column {fold_column!r} contains missing values")
        labels = sorted(frame[fold_column].unique().tolist(), key=str)
        if len(labels) < 2:
            raise ValueError("Fixed fold column must contain at least two fold labels")
        all_indices = np.arange(len(frame))
        return [
            (
                str(label),
                all_indices[frame[fold_column].to_numpy() != label],
                all_indices[frame[fold_column].to_numpy() == label],
            )
            for label in labels
        ]

    event = frame[data_config.event_column].to_numpy(dtype=int)
    if event.sum() < training_config.n_folds:
        raise ValueError(
            f"At least {training_config.n_folds} events are required for stratified CV"
        )
    splitter = StratifiedKFold(
        n_splits=training_config.n_folds,
        shuffle=True,
        random_state=training_config.random_seed,
    )
    return [
        (str(fold), train_index, validation_index)
        for fold, (train_index, validation_index) in enumerate(splitter.split(frame, event))
    ]


def run_cross_validation(
    frame: pd.DataFrame,
    data_config: DataConfig,
    training_config: TrainingConfig,
    backbone_checkpoint: str | Path,
    output_dir: str | Path,
    ecg_root: str | Path | None,
    expected_ecg_length: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(training_config.random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = _split_plan(frame, data_config, training_config)
    oof_risk = np.full(len(frame), np.nan, dtype=float)
    oof_fold = np.full(len(frame), "", dtype=object)
    fold_summaries = []

    _save_json(
        output_dir / "run_config.json",
        {
            "data": data_config.to_dict(),
            "training": training_config.to_dict(),
            "expected_ecg_length": expected_ecg_length,
            "device": str(device),
            "n_samples": len(frame),
            "n_events": int(frame[data_config.event_column].sum()),
            "fold_labels": [label for label, _, _ in plan],
        },
    )

    for fold_index, (fold_label, train_index, validation_index) in enumerate(plan):
        print(
            f"\nFold {fold_label}: train={len(train_index)}, "
            f"validation={len(validation_index)}"
        )
        train_frame = frame.iloc[train_index].reset_index(drop=True)
        validation_frame = frame.iloc[validation_index].reset_index(drop=True)
        train_matrix, validation_matrix, imputer, scaler = _fit_tabular_preprocessor(
            train_frame,
            validation_frame,
            data_config.tabular_features,
        )
        (
            auxiliary_train,
            auxiliary_validation,
            auxiliary_train_mask,
            auxiliary_validation_mask,
            auxiliary_statistics,
        ) = _prepare_auxiliary_targets(train_frame, validation_frame, data_config)

        warmup_model, phase1_history = train_phase1(
            train_matrix,
            validation_matrix,
            train_frame,
            validation_frame,
            data_config,
            training_config,
            device,
        )
        model, validation_risk, phase2_history = train_phase2(
            warmup_model,
            train_matrix,
            validation_matrix,
            train_frame,
            validation_frame,
            auxiliary_train,
            auxiliary_validation,
            auxiliary_train_mask,
            auxiliary_validation_mask,
            data_config,
            training_config,
            backbone_checkpoint,
            ecg_root,
            expected_ecg_length,
            device,
        )
        oof_risk[validation_index] = validation_risk
        oof_fold[validation_index] = fold_label

        fold_c = _safe_c_index(
            validation_frame[data_config.time_column].to_numpy(dtype=float),
            validation_frame[data_config.event_column].to_numpy(dtype=float),
            validation_risk,
        )
        fold_summary = {
            "fold_index": fold_index,
            "fold_label": fold_label,
            "n_validation": len(validation_frame),
            "n_validation_events": int(validation_frame[data_config.event_column].sum()),
            "validation_c_index": fold_c,
            "phase1": phase1_history,
            "phase2": phase2_history,
        }
        fold_summaries.append(fold_summary)
        _save_json(output_dir / f"fold_{fold_index}_history.json", fold_summary)

        checkpoint = {
            "model_state_dict": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "fold_index": fold_index,
            "fold_label": fold_label,
            "tabular_features": data_config.tabular_features,
            "auxiliary_columns": data_config.auxiliary_columns,
            "imputer_statistics": imputer.statistics_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "auxiliary_statistics": auxiliary_statistics,
            "model_parameters": {
                "bottleneck_dim": training_config.bottleneck_dim,
                "tabular_hidden_dim": training_config.tabular_hidden_dim,
                "fusion_hidden_dim": training_config.fusion_hidden_dim,
                "dropout": training_config.dropout,
            },
        }
        torch.save(checkpoint, output_dir / f"fold_{fold_index}.pt")
        del model, warmup_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.isnan(oof_risk).any():
        raise RuntimeError("OOF predictions are incomplete")

    oof = pd.DataFrame(
        {
            "row_index": np.arange(len(frame)),
            "fold": oof_fold,
            "time": frame[data_config.time_column].to_numpy(),
            "event": frame[data_config.event_column].to_numpy(),
            "oof_log_risk": oof_risk,
        }
    )
    if data_config.id_column in frame.columns:
        oof.insert(1, data_config.id_column, frame[data_config.id_column].to_numpy())
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)

    pooled_c = _safe_c_index(
        frame[data_config.time_column].to_numpy(dtype=float),
        frame[data_config.event_column].to_numpy(dtype=float),
        oof_risk,
    )
    metrics = {
        "n_samples": len(frame),
        "n_events": int(frame[data_config.event_column].sum()),
        "n_folds": len(plan),
        "pooled_harrell_c_index": pooled_c,
        "folds": [
            {
                key: value
                for key, value in summary.items()
                if key not in {"phase1", "phase2"}
            }
            for summary in fold_summaries
        ],
    }
    _save_json(output_dir / "metrics.json", metrics)
    return metrics
