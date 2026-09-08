"""Typed experiment configuration.

Every number that affects a result lives in a YAML file under `conf/`, never inline in
training code. The reason is reproducibility: an experiment is then fully described by
(config file, git commit, seed), which is exactly what the competition requires winners to
document, and what makes `reports/experiments/log.md` meaningful rather than decorative.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/raw"
    labels_csv: str = "data/raw/train_labels.csv"
    nifti_dir: str = "data/raw/niftis"
    folds_csv: str = "data/interim/folds.csv"
    target_spacing_mm: tuple[float, float, float] = (2.0, 2.0, 2.0)
    target_shape: tuple[int, int, int] = (96, 96, 96)
    normalization: str = "reference_region"
    crop: str = "center"
    cache_dir: str | None = "data/processed/cache"


@dataclass
class ModelConfig:
    name: str = "resnet3d"
    widths: tuple[int, ...] = (16, 32, 64, 128)
    dropout: float = 0.3
    in_channels: int = 1


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    # ReduceLROnPlateau mirrors the schedule used in the author's MNIST work; for short runs
    # cosine is usually the stronger default. Both are supported by train.py.
    scheduler: str = "cosine"
    plateau_patience: int = 3
    plateau_factor: float = 0.3
    warmup_epochs: int = 2
    amp: bool = True
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    num_workers: int = 4
    # Write a resume checkpoint after every epoch, and pick up from it if one is found.
    # Costs ~20 MB of disk churn per epoch; buys the ability to lose one epoch rather than a
    # whole fold when a run is interrupted. Worth it whenever a fold takes longer than the
    # interval between interruptions — i.e. always, on CPU.
    resume: bool = True
    fold_column: str = "fold_strat"
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass
class AugConfig:
    """Augmentation. Keep it anatomically plausible.

    Left-right flipping is DISABLED by default and that is a domain decision, not an
    oversight: DaT abnormality is frequently *asymmetric*, and the more-affected side carries
    clinical meaning. Flipping teaches the model to ignore laterality, which may discard
    signal. Test it explicitly before enabling.
    """

    flip_lr: bool = False
    rotate_degrees: float = 10.0
    translate_mm: float = 5.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    intensity_jitter: float = 0.1
    noise_std: float = 0.01
    mixup_alpha: float = 0.0


@dataclass
class Config:
    run_name: str = "baseline"
    seed: int = 42
    output_dir: str = "models"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aug: AugConfig = field(default_factory=AugConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, values: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, tolerating missing keys, rejecting unknown ones."""
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> Config:
    """Load a YAML config, filling any absent section with its defaults."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sections = {
        "data": DataConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "aug": AugConfig,
    }
    kwargs: dict[str, Any] = {
        k: v for k, v in raw.items() if k not in sections
    }
    for name, cls in sections.items():
        kwargs[name] = _build(cls, raw.get(name) or {})
    return Config(**kwargs)
