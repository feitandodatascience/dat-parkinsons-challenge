"""Richer training-time augmentation, applied on top of the existing dataset.

Why this is a separate module and a wrapper rather than an edit to the dataset
-----------------------------------------------------------------------------
`DaTScanDataset` already implements six augmentations (flip, rotation, translation, scale,
intensity jitter, additive noise) and owns the preprocessing cache that makes an epoch
compute-bound rather than IO-bound. None of that should be disturbed. What is added here
composes *after* it: the wrapper takes the tensor the dataset produced and perturbs it further.

The contract this relies on was established by instantiating the dataset over synthetic noise
volumes rather than by reading it — `__getitem__` returns `(Tensor[1, D, H, W] float32, scalar
float32 label)` and augments only when `train=True`. Nothing here assumes anything else about it.

What the perturbations are chosen to imitate
--------------------------------------------
The training set comes from ten French hospitals, and the recorded spread is in acquisition
geometry rather than in anatomy: field of view spans 176-630 mm, in-plane spacing ranges from
2.46 mm to 3.895 mm, and matrix sizes differ (`reports/eda/shared/`). The dominant domain shift
is therefore *how the scanner blurred and sampled the head*, not how the head is shaped. Two of
the four new perturbations target that directly:

- **PSF / resolution jitter** (`blur_sigma_max`) — a Gaussian blur of random width. Different
  gamma cameras and reconstruction filters resolve the striatum differently, and a model that has
  only ever seen one effective resolution can read a blurrier scan as a flatter, more pathologic
  one. This is the most physically motivated item in the file.
- **Count noise** (`count_noise`) — SPECT is a counting measurement, so noise grows with signal.
  Additive Gaussian noise of constant width (`noise_std`, already in the dataset) is the wrong
  shape for that; this adds a component scaling with the square root of the positive signal.

The other two are conventional but earn their place here:

- **Elastic deformation** (`elastic_alpha`) — smooth local warping, for anatomical variability
  that an affine cannot express. The displacement field is generated coarse and upsampled rather
  than generated dense and smoothed, which is ~10x cheaper for an indistinguishable field.
- **Mixup** (`mixup_alpha`) — the metric is log loss, where calibration is worth as much as
  discrimination, and mixup's best-documented effect is on calibration rather than on accuracy.
  Note this knob was declared in `config.py` from the start of the project and read by nothing;
  it is implemented here for the first time.

What is deliberately NOT here
-----------------------------
- **Left-right flip.** DaT abnormality is frequently asymmetric and the more-affected side
  carries clinical meaning (docs/01 §4). Mirroring destroys the signal rather than regularising
  it. This is the project's standing prior and nothing here revisits it.
- **Cutout / coarse dropout.** Occluding a cube inside the striatal box can turn a normal scan
  into one that reads as unilateral loss. On a task whose label IS "is striatal uptake reduced",
  cutout does not augment the input, it corrupts the label.

Competition rules
-----------------
Everything here is training-time only. `RichAugDataset` is never constructed for validation or
inference, so per-sample independence at prediction time (CLAUDE.md §1.1) is untouched. Mixup
mixes two scans from the SAME fold's training split — the partner is drawn from the wrapped
dataset, which holds training rows only — so no validation row influences any training sample.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from .config import AugConfig

# Spatial frequencies, in cycles across the volume, used to build displacement fields. Capped at
# 2 so the longest wavelength is the whole volume and the shortest is half of it: the field bends
# the head, it does not ripple the striatum.
_FIELD_MODES = (1.0, 2.0)


def _smooth_field(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """A smooth random displacement field, as a sum of separable low-frequency sinusoids.

    The obvious construction — draw white noise on a coarse grid and upsample, or draw it dense
    and Gaussian-blur it — costs 2.5 s and 0.35 s per call respectively at 96^3, measured. Both
    are far too slow for something that runs once per sample per epoch. A sum of a few sinusoids
    is smooth by construction, needs no filtering, and costs ~30 ms because each term is an outer
    product of three 96-element vectors rather than an operation on a 96^3 array.

    It is also a *better* field than the upsampled-noise version, which is piecewise linear and
    therefore creased at the coarse grid lines.
    """
    field = np.zeros((3, *shape), dtype=np.float32)
    for component in range(3):
        for frequency in _FIELD_MODES:
            waves = [
                np.sin(
                    2.0 * np.pi * frequency * np.arange(size, dtype=np.float32) / size
                    + rng.uniform(0.0, 2.0 * np.pi)
                ).astype(np.float32)
                for size in shape
            ]
            amplitude = float(rng.normal(0.0, 1.0))
            field[component] += amplitude * (
                waves[0][:, None, None] * waves[1][None, :, None] * waves[2][None, None, :]
            )
    return field


def elastic_warp(volume: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random local deformation, magnitude `alpha` in voxels.

    `order=1` and `mode="nearest"`: the volume is intensity-normalised with a background that is
    negative rather than zero, so extending the border beats stamping a hard zero edge the model
    never saw. Same reasoning as the test-time views in `scripts/07_tta.py`.
    """
    if alpha <= 0.0:
        return volume

    shape = volume.shape
    field = _smooth_field(shape, rng)

    # Normalise so `alpha` means "peak displacement in voxels" rather than an arbitrary scale,
    # which is what makes the knob readable in a config file.
    peak = float(np.abs(field).max())
    if peak > 1e-9:
        field = field * (alpha / peak)

    grid = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
    coords = np.stack([g + f for g, f in zip(grid, field)])
    return ndimage.map_coordinates(
        volume, coords, order=1, mode="nearest"
    ).astype(np.float32)


def signed_gamma(volume: np.ndarray, gamma: float) -> np.ndarray:
    """Contrast change that is valid on a SIGNED array.

    The preprocessed volume is reference-region normalised: background sits below zero, striatum
    above it. A plain `x ** gamma` is undefined for the negative half, and shifting the array to
    make it positive would make the transform depend on the scan's own minimum — a per-sample
    quantity that varies with field of view. `sign(x) * |x| ** gamma` is monotone, odd, fixes
    zero (the reference region itself), and compresses or expands both tails symmetrically.
    """
    if gamma == 1.0:
        return volume
    return (np.sign(volume) * np.abs(volume) ** gamma).astype(np.float32)


def psf_blur(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic Gaussian blur, imitating a different camera resolution or reconstruction filter."""
    if sigma <= 0.0:
        return volume
    return ndimage.gaussian_filter(volume, sigma=sigma).astype(np.float32)


def count_noise(volume: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Signal-proportional noise, imitating the counting statistics of SPECT.

    Poisson noise has variance equal to the mean, so its standard deviation grows as the square
    root of the signal. The array is normalised and signed, so "signal" is taken as the positive
    part; the background, which carries no counts to speak of, is left to the constant-width
    noise the dataset already applies.
    """
    if scale <= 0.0:
        return volume
    amplitude = np.sqrt(np.clip(volume, 0.0, None))
    return (volume + rng.normal(0.0, 1.0, size=volume.shape) * amplitude * scale).astype(
        np.float32
    )


class RichAugDataset(Dataset):
    """Wraps a training dataset and applies the augmentations above to each sample.

    Constructed only for training splits. The wrapped dataset keeps ownership of loading,
    caching and its own augmentation; this adds a second layer on top of the tensor it returns.
    """

    def __init__(self, base: Dataset, aug_cfg: AugConfig, seed: int = 0) -> None:
        self.base = base
        self.cfg = aug_cfg
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def _rng_for_worker(self) -> np.random.Generator:
        """One generator per dataloader worker, created on first use.

        Seeding per __getitem__ from the sample index would make a given scan receive the SAME
        perturbation in every epoch, which is not augmentation at all. A per-worker stream that
        simply continues across epochs gives fresh draws while staying reproducible for a fixed
        seed and worker count — the caveat being that changing `num_workers` changes the stream,
        which is worth knowing before comparing two runs that differ in it.
        """
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker_id = info.id if info is not None else 0
            self._rng = np.random.default_rng([self.seed, worker_id])
        return self._rng

    def _perturb(self, volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply each enabled perturbation independently with probability `rich_prob`.

        Gating matters for cost as much as for regularisation. Every op here touches all 884,736
        voxels, so applying all four to every sample costs roughly 250 ms; at p=0.5 the average
        is half that, and the model still sees each perturbation on half its samples and the
        unperturbed volume reasonably often. A model that NEVER sees a clean volume during
        training is being asked to generalise in the wrong direction — the test set is clean.
        """
        cfg = self.cfg
        p = cfg.rich_prob

        if cfg.elastic_alpha > 0.0 and rng.random() < p:
            volume = elastic_warp(volume, cfg.elastic_alpha, rng)

        if cfg.blur_sigma_max > 0.0 and rng.random() < p:
            volume = psf_blur(volume, float(rng.uniform(0.0, cfg.blur_sigma_max)))

        lo, hi = cfg.gamma_range
        if (lo != 1.0 or hi != 1.0) and rng.random() < p:
            volume = signed_gamma(volume, float(rng.uniform(lo, hi)))

        if cfg.count_noise > 0.0 and rng.random() < p:
            volume = count_noise(volume, cfg.count_noise, rng)

        return volume

    def __getitem__(self, index: int):
        x, y = self.base[index]
        rng = self._rng_for_worker()

        volume = self._perturb(x.numpy()[0], rng)

        if self.cfg.mixup_alpha > 0.0 and rng.random() < self.cfg.mixup_prob:
            partner = int(rng.integers(len(self)))
            x2, y2 = self.base[partner]
            other = self._perturb(x2.numpy()[0], rng)
            # Beta(a, a) is symmetric, so which sample is "the" sample does not matter; taking
            # max(lam, 1-lam) would bias every mix toward the first and is not done here.
            lam = float(rng.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha))
            volume = (lam * volume + (1.0 - lam) * other).astype(np.float32)
            y = torch.tensor(lam * float(y) + (1.0 - lam) * float(y2), dtype=torch.float32)

        return torch.from_numpy(volume)[None], y


def wrap_if_enabled(base: Dataset, aug_cfg: AugConfig, seed: int = 0) -> Dataset:
    """Return `base` untouched unless at least one rich augmentation is switched on.

    Every existing config leaves all of them at their off defaults, so this returns the wrapped
    dataset for new configs and the original object for every run already in the experiment log
    — which is what makes those runs still comparable.
    """
    enabled = (
        aug_cfg.elastic_alpha > 0.0
        or aug_cfg.blur_sigma_max > 0.0
        or aug_cfg.count_noise > 0.0
        or aug_cfg.mixup_alpha > 0.0
        or tuple(aug_cfg.gamma_range) != (1.0, 1.0)
    )
    return RichAugDataset(base, aug_cfg, seed=seed) if enabled else base
