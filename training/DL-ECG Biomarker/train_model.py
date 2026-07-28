from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from model5_training.config import DataConfig, TrainingConfig
from model5_training.data import load_metadata, validate_metadata
from model5_training.training import run_cross_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the two-phase multimodal DL-ECG Model 5",
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ecg-root",
        type=Path,
        default=None,
        help="Optional root prepended to relative ECG paths in the metadata table",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ecg-length", type=int, default=5000)
    parser.add_argument("--phase1-max-epochs", type=int, default=150)
    parser.add_argument("--phase2-max-epochs", type=int, default=100)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main() -> None:
    args = parse_args()
    data_config = DataConfig.from_json(args.config)
    training_config = replace(
        TrainingConfig(),
        n_folds=args.n_folds,
        random_seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        phase1_max_epochs=args.phase1_max_epochs,
        phase2_max_epochs=args.phase2_max_epochs,
    )
    device = resolve_device(args.device)
    frame = validate_metadata(load_metadata(args.metadata), data_config)

    print("Model 5 training")
    print(f"  samples: {len(frame)}")
    print(f"  events:  {int(frame[data_config.event_column].sum())}")
    print(f"  device:  {device}")
    print(f"  output:  {args.output_dir}")

    metrics = run_cross_validation(
        frame=frame,
        data_config=data_config,
        training_config=training_config,
        backbone_checkpoint=args.backbone_checkpoint,
        output_dir=args.output_dir,
        ecg_root=args.ecg_root,
        expected_ecg_length=args.ecg_length,
        device=device,
    )
    print("\nTraining complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
