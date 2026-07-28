from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dSame(nn.Module):
    """One-dimensional convolution with TensorFlow-style SAME padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_length = x.shape[-1]
        output_length = math.ceil(input_length / self.stride)
        total_padding = max(
            0,
            (output_length - 1) * self.stride + self.kernel_size - input_length,
        )
        left = total_padding // 2
        right = total_padding - left
        return self.conv(F.pad(x, (left, right)))


class MaxPool1dSame(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_padding = max(0, self.kernel_size - 1)
        left = total_padding // 2
        right = total_padding - left
        return self.pool(F.pad(x, (left, right)))


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: bool,
        first_block: bool,
        use_batch_norm: bool,
        use_dropout: bool,
    ) -> None:
        super().__init__()
        middle_channels = int(out_channels * ratio)
        effective_stride = stride if downsample else 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.first_block = first_block
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = Swish()
        self.drop1 = nn.Dropout(0.5)
        self.conv1 = Conv1dSame(in_channels, middle_channels, 1, 1)

        self.bn2 = nn.BatchNorm1d(middle_channels)
        self.act2 = Swish()
        self.drop2 = nn.Dropout(0.5)
        self.conv2 = Conv1dSame(
            middle_channels,
            middle_channels,
            kernel_size,
            effective_stride,
            groups,
        )

        self.bn3 = nn.BatchNorm1d(middle_channels)
        self.act3 = Swish()
        self.drop3 = nn.Dropout(0.5)
        self.conv3 = Conv1dSame(middle_channels, out_channels, 1, 1)

        self.se_fc1 = nn.Linear(out_channels, out_channels // 2)
        self.se_fc2 = nn.Linear(out_channels // 2, out_channels)
        self.se_activation = Swish()
        self.pool = MaxPool1dSame(effective_stride) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x

        if not self.first_block:
            if self.use_batch_norm:
                out = self.bn1(out)
            out = self.act1(out)
            if self.use_dropout:
                out = self.drop1(out)
        out = self.conv1(out)

        if self.use_batch_norm:
            out = self.bn2(out)
        out = self.act2(out)
        if self.use_dropout:
            out = self.drop2(out)
        out = self.conv2(out)

        if self.use_batch_norm:
            out = self.bn3(out)
        out = self.act3(out)
        if self.use_dropout:
            out = self.drop3(out)
        out = self.conv3(out)

        squeeze = out.mean(dim=-1)
        squeeze = self.se_fc2(self.se_activation(self.se_fc1(squeeze)))
        out = out * torch.sigmoid(squeeze).unsqueeze(-1)

        if self.pool is not None:
            identity = self.pool(identity)
        if self.out_channels != self.in_channels:
            left = (self.out_channels - self.in_channels) // 2
            right = self.out_channels - self.in_channels - left
            identity = F.pad(identity.transpose(-1, -2), (left, right)).transpose(-1, -2)
        return out + identity


class ResidualStage1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        kernel_size: int,
        stride: int,
        groups: int,
        stage_index: int,
        n_blocks: int,
        use_batch_norm: bool,
        use_dropout: bool,
    ) -> None:
        super().__init__()
        blocks = []
        for block_index in range(n_blocks):
            blocks.append(
                ResidualBlock1D(
                    in_channels=in_channels if block_index == 0 else out_channels,
                    out_channels=out_channels,
                    ratio=ratio,
                    kernel_size=kernel_size,
                    stride=stride if block_index == 0 else 1,
                    groups=groups,
                    downsample=block_index == 0,
                    first_block=stage_index == 0 and block_index == 0,
                    use_batch_norm=use_batch_norm,
                    use_dropout=use_dropout,
                )
            )
        # Keep the original attribute name so public ECGFounder checkpoints
        # load without key remapping (stage_list.N.block_list.N.*).
        self.block_list = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)
        return x


class Net1D(nn.Module):
    """ECGFounder-compatible one-dimensional residual network."""

    def __init__(self) -> None:
        super().__init__()
        filter_list = [64, 160, 160, 400, 400, 1024, 1024]
        blocks_per_stage = [2, 2, 2, 3, 3, 4, 4]
        self.first_conv = Conv1dSame(12, 64, kernel_size=16, stride=2)
        self.first_bn = nn.BatchNorm1d(64)
        self.first_activation = Swish()
        self.use_batch_norm = False

        stages = []
        in_channels = 64
        for stage_index, (out_channels, n_blocks) in enumerate(
            zip(filter_list, blocks_per_stage)
        ):
            stages.append(
                ResidualStage1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    ratio=1,
                    kernel_size=16,
                    stride=2,
                    groups=out_channels // 16,
                    stage_index=stage_index,
                    n_blocks=n_blocks,
                    use_batch_norm=False,
                    use_dropout=False,
                )
            )
            in_channels = out_channels
        self.stage_list = nn.ModuleList(stages)
        self.dense = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        if self.use_batch_norm:
            x = self.first_bn(x)
        x = self.first_activation(x)
        for stage in self.stage_list:
            x = stage(x)
        features = x.mean(dim=-1)
        return self.dense(features)


def _read_checkpoint(path: str | Path) -> dict:
    """Load legacy checkpoints under both old and new PyTorch defaults."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_ecgfounder_backbone(checkpoint_path: str | Path) -> Net1D:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ECGFounder checkpoint not found: {checkpoint_path}")

    model = Net1D()
    checkpoint = _read_checkpoint(checkpoint_path)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a state_dict key")

    cleaned = {}
    for name, value in state.items():
        while name.startswith("module."):
            name = name.removeprefix("module.")
        if name.startswith("dense."):
            continue
        cleaned[name] = value

    incompatible = model.load_state_dict(cleaned, strict=False)
    non_head_missing = [name for name in incompatible.missing_keys if not name.startswith("dense.")]
    if non_head_missing:
        preview = ", ".join(non_head_missing[:5])
        raise RuntimeError(f"Checkpoint is incompatible; missing backbone keys: {preview}")
    model.dense = nn.Identity()
    return model


class ECGFounderBackbone(nn.Module):
    def __init__(self, checkpoint_path: str | Path, freeze: bool = True) -> None:
        super().__init__()
        self.net = load_ecgfounder_backbone(checkpoint_path)
        if freeze:
            for parameter in self.net.parameters():
                parameter.requires_grad = False
        self.output_dim = 1024

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        return self.net(ecg)
