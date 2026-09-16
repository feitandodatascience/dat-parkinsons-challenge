"""Build a model from config, so architecture choices live in YAML like everything else.

Not vendored into the submission: inference constructs its one architecture directly, and every
extra module in the zip is another thing that can fail on a platform you cannot debug on.
"""

from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .resnet3d import ResNet3D


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate the configured architecture.

    Adding a MONAI backbone later (DenseNet121, SEResNet - docs/03 Phase 3, item 3) means adding
    a branch here and a `name` in the config, with nothing else in the training loop changing.
    """
    if cfg.name == "resnet3d":
        return ResNet3D(
            widths=tuple(cfg.widths),
            dropout=cfg.dropout,
            in_channels=cfg.in_channels,
        )
    raise ValueError(f"unknown model name {cfg.name!r}")
