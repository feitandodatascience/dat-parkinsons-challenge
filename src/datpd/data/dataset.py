"""Torch Dataset over the examinations, with a preprocessing cache and 3D augmentation.

Two things here earn their complexity.

**The cache.** Resampling a volume to a common grid costs far more than a forward pass, and
without caching it would be repeated every epoch and dominate wall-clock time (docs/02 Delta 1).
Preprocessed arrays are written once, keyed by a hash of the preprocessing parameters, so
changing `target_spacing` or `normalization` invalidates the cache automatically rather than
silently training on stale arrays. The cache holds patient-derived voxels, so `cache_dir` must
stay under `data/` where the repository's guards already cover it.

**Augmentation in millimetres, not voxels.** A 5-voxel shift means different things at 2.0 mm
and 3.9 mm spacing. Because augmentation runs after resampling to a fixed grid, the conversion
is a single division - but it has to happen, or the augmentation strength silently varies with
the source scanner.

Note what is *absent*: left-right flipping is off by default. DaT abnormality is frequently
asymmetric and the more-affected side is clinically meaningful, so flipping teaches the model to
discard laterality. That is a domain decision to test, not to assume (see `AugConfig`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from ..config import AugConfig, DataConfig
from .io import build_volume_index, load_volume
from .preprocess import preprocess

# Suffix for cached preprocessed arrays. Payload is plain NPY written through a file object.
CACHE_SUFFIX = ".prep"


def _cache_key(cfg: DataConfig) -> str:
    """Short hash of every preprocessing parameter that changes the resulting array."""
    parts = (
        tuple(cfg.target_spacing_mm),
        tuple(cfg.target_shape),
        cfg.normalization,
        cfg.crop,
    )
    return hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()[:10]


def random_affine(
    array: np.ndarray,
    rng: np.random.Generator,
    rotate_degrees: float,
    translate_voxels: float,
    scale_range: tuple[float, float],
) -> np.ndarray:
    """Random rigid-plus-scale warp about the volume centre.

    One `affine_transform` call for the composed rotation, scale and shift rather than three
    sequential resamplings: each interpolation blurs, and blurring a small structure three times
    is a real loss of the signal the task depends on.
    """
    center = np.array(array.shape, dtype=np.float64) / 2.0

    matrix = np.eye(3)
    if rotate_degrees > 0:
        for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
            theta = np.deg2rad(rng.uniform(-rotate_degrees, rotate_degrees))
            rot = np.eye(3)
            cos, sin = np.cos(theta), np.sin(theta)
            rot[axis_a, axis_a] = cos
            rot[axis_a, axis_b] = -sin
            rot[axis_b, axis_a] = sin
            rot[axis_b, axis_b] = cos
            matrix = matrix @ rot

    if scale_range and scale_range[0] != scale_range[1]:
        # Dividing by the scale means scale > 1 magnifies, which is the intuitive direction.
        matrix = matrix / float(rng.uniform(*scale_range))

    shift = (
        rng.uniform(-translate_voxels, translate_voxels, size=3)
        if translate_voxels > 0
        else np.zeros(3)
    )
    offset = center - matrix @ center + shift

    return ndimage.affine_transform(
        array, matrix, offset=offset, order=1, mode="nearest"
    ).astype(np.float32)


def augment(array: np.ndarray, cfg: AugConfig, spacing_mm: float, rng: np.random.Generator) -> np.ndarray:
    """Apply the configured augmentations to an already-preprocessed array."""
    out = array
    if cfg.rotate_degrees or cfg.translate_mm or cfg.scale_range != (1.0, 1.0):
        out = random_affine(
            out,
            rng,
            cfg.rotate_degrees,
            cfg.translate_mm / max(spacing_mm, 1e-6),
            tuple(cfg.scale_range),
        )

    if cfg.flip_lr and rng.random() < 0.5:
        # Canonical RAS puts left-right on axis 0 (see io.load_volume).
        out = np.ascontiguousarray(out[::-1])

    if cfg.intensity_jitter:
        out = out * float(rng.normal(1.0, cfg.intensity_jitter))

    if cfg.noise_std:
        # Noise scaled by the array's own spread, so its strength does not depend on which
        # normalisation produced the array.
        out = out + rng.normal(0.0, cfg.noise_std * (out.std() or 1.0), out.shape)

    return out.astype(np.float32)


class DaTScanDataset(Dataset):
    """Examinations as (volume tensor, label) pairs.

    `train=True` enables augmentation; `train=False` is deterministic, which is what makes an
    OOF prediction a fixed number rather than a sample from a distribution.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_cfg: DataConfig,
        aug_cfg: AugConfig,
        train: bool = False,
        label_col: str = "is_pathologic",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data_cfg = data_cfg
        self.aug_cfg = aug_cfg
        self.train = train
        self.label_col = label_col

        self.index = build_volume_index(data_cfg.nifti_dir)
        missing = [u for u in self.df["uid"] if u not in self.index]
        if missing:
            # Count only. A uid is a per-sample identifier and must not reach a log.
            raise FileNotFoundError(
                f"{len(missing)} of {len(self.df)} uids have no volume in {data_cfg.nifti_dir}"
            )

        self.cache_dir: Path | None = None
        if data_cfg.cache_dir:
            self.cache_dir = Path(data_cfg.cache_dir) / _cache_key(data_cfg)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_preprocessed(self, uid: str) -> np.ndarray:
        cache_path = self.cache_dir / f"{uid}{CACHE_SUFFIX}" if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            with cache_path.open("rb") as handle:
                return np.load(handle)

        array = preprocess(
            load_volume(self.index[uid]),
            tuple(self.data_cfg.target_spacing_mm),
            tuple(self.data_cfg.target_shape),
            self.data_cfg.normalization,
            self.data_cfg.crop,
        )

        if cache_path is not None:
            # Write to a temporary name first: a DataLoader worker killed mid-write would
            # otherwise leave a truncated file that every later epoch would happily load.
            tmp_path = cache_path.with_suffix(CACHE_SUFFIX + ".tmp")
            with tmp_path.open("wb") as handle:
                np.save(handle, array)
            tmp_path.replace(cache_path)

        return array

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        uid = str(row["uid"])
        array = self._load_preprocessed(uid)

        if self.train:
            # Seeded from the epoch-independent worker RNG so augmentation differs per sample
            # per epoch, while the run as a whole stays reproducible from cfg.seed.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            array = augment(array, self.aug_cfg, float(self.data_cfg.target_spacing_mm[0]), rng)

        x = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unsqueeze(0)
        label = float(row[self.label_col]) if self.label_col in row.index else 0.0
        return x, torch.tensor(label, dtype=torch.float32)
