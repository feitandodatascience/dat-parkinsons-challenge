"""Turn heterogeneous scans into one fixed-size, scanner-invariant tensor.

VENDORED INTO submission.zip. This module is the highest-risk piece of the whole solution,
because any difference between how a volume is prepared at training time and at inference time
shows up as "great CV, terrible leaderboard" and is invisible in both logs. Vendoring this
exact file is how that divergence is prevented (docs/02, debugging step 3).

The pipeline is three *separate* steps, in this order; conflating them is the classic beginner
error (docs/02 Delta 1):

    1. reorient   -> canonical RAS                 (done in io.load_volume)
    2. resample   -> fixed spacing, e.g. 2.0 mm    a voxel now has a constant physical size
    3. crop / pad -> fixed array shape, e.g. 96^3  the tensor shape is now constant

Intensity normalisation runs last, deliberately: it is then computed on exactly the voxels the
network will see, so the statistics cannot drift with the field of view.

**Every normalisation here is scale-invariant by construction** - multiply the input by any
positive constant and the output is unchanged. That is not a nicety. SPECT counts depend on
injected dose, scanner sensitivity and acquisition time, none of which carry diagnostic
information and all of which differ across the ten centres.

HOW MUCH OF THE ARRAY IS ACTUALLY HEAD
--------------------------------------
The real data spans a 3.6x range of field of view: 176 mm to 630 mm per axis. A head is roughly
180 mm across, so the fraction of the array occupied by anatomy runs from nearly all of it to
about **one percent** (the 630 mm acquisitions, which are a third of the dataset). Anything here
that identifies "where the head is" therefore has to work at both extremes, which is why the
foreground test is a fraction of a high quantile rather than a percentile of the array.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .io import Volume

# Voxels below this fraction of the volume's 99th percentile are treated as air/background.
# A fraction rather than an absolute count, so the mask is unchanged by rescaling.
BRAIN_THRESHOLD_FRACTION = 0.10

# Quantile standing in for "a bright voxel inside the head". Chosen above 99 because in the
# widest-field acquisitions the head is only ~1% of the array, which puts the 99th percentile
# right on the head boundary.
FOREGROUND_QUANTILE = 99.5

# Percentile band of in-brain intensity used as the non-specific-binding reference. Occipital
# cortex has essentially no dopamine transporter, so it sits low in the histogram but well
# above air - which is what this band selects without needing an atlas (docs/01 section 5).
REFERENCE_BAND = (20.0, 60.0)

NORMALIZATIONS = ("reference_region", "zscore_brain", "max")
CROPS = ("center", "com")


def _brain_mask(a: np.ndarray, hi: float) -> np.ndarray:
    """Coarse foreground mask: everything above a fixed fraction of a high quantile.

    Defined relative to a quantile of the volume itself, so the mask is identical under any
    rescaling of the input - which is what makes the normalisations below scale-invariant, and
    what keeps it meaningful whether the head fills the array or occupies one percent of it.
    """
    if hi <= 0.0:
        return np.ones(a.shape, dtype=bool)
    mask = a > BRAIN_THRESHOLD_FRACTION * hi
    if not mask.any():
        mask = a > 0.0
    if not mask.any():
        mask = np.ones(a.shape, dtype=bool)
    return mask


def resample_to_spacing(
    volume: Volume,
    target_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    order: int = 1,
) -> np.ndarray:
    """Resample voxels so that one voxel spans a fixed physical size.

    Returns the volume's own array unchanged when the spacing already matches, which keeps the
    common case free of an interpolation pass and of the blur it would introduce.

    Linear interpolation (`order=1`) rather than cubic: SPECT is already smooth and low
    resolution, so a higher-order kernel adds ringing around the striatum for no gain.
    """
    src = np.asarray(volume.spacing_mm, dtype=np.float64)
    tgt = np.asarray(target_spacing, dtype=np.float64)

    if np.allclose(src, tgt, rtol=0.0, atol=1e-6):
        return volume.array

    # The new shape preserves physical extent: n_voxels * spacing is invariant.
    factors = src / tgt
    new_shape = tuple(max(1, int(round(n * f))) for n, f in zip(volume.array.shape, factors))
    # Ask ndimage for the exact integer shape rather than trusting it to round the same way.
    exact = tuple(ns / n for ns, n in zip(new_shape, volume.array.shape))
    return ndimage.zoom(
        volume.array.astype(np.float32, copy=False),
        exact,
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )


def _window_slices(
    shape: tuple[int, ...], target_shape: tuple[int, int, int], centres: np.ndarray | None
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Source and destination slices placing a `target_shape` window at `centres`.

    Axes are handled independently, so a volume that is too large on one axis and too small on
    another is fine - the normal case with variable fields of view. `centres=None` means the
    geometric centre of the array.
    """
    src_slices, dst_slices = [], []
    for axis, (n, t) in enumerate(zip(shape, target_shape)):
        t = int(t)
        if n >= t:
            centre = n / 2.0 if centres is None else float(centres[axis])
            start = int(round(centre - t / 2.0))
            start = max(0, min(start, n - t))  # keep the window inside the array
            src_slices.append(slice(start, start + t))
            dst_slices.append(slice(0, t))
        else:
            start = (t - n) // 2
            src_slices.append(slice(0, n))
            dst_slices.append(slice(start, start + n))
    return tuple(src_slices), tuple(dst_slices)


def center_crop_or_pad(array: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Crop or zero-pad each axis about the array centre to reach exactly `target_shape`."""
    a = np.asarray(array)
    out = np.zeros(tuple(int(t) for t in target_shape), dtype=np.float32)
    src, dst = _window_slices(a.shape, target_shape, None)
    out[dst] = a[src].astype(np.float32, copy=False)
    return out


def center_of_mass_crop(
    array: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Crop about the intensity centre of mass rather than the array centre.

    Heads are not always centred in the field of view, and a geometric centre crop can clip the
    striatum on one side - destroying exactly the asymmetry the task depends on.

    The foreground test is `_brain_mask`, a fraction of a high quantile, **not a percentile of
    the array**. That distinction is the whole correctness of this function on real data: in the
    630 mm acquisitions the head is about one percent of the volume, so any percentile below
    ~99 selects air, the centre of mass lands at the middle of the array, and this function
    silently degenerates into `center_crop_or_pad` on a third of the dataset.
    """
    a = np.asarray(array, dtype=np.float32)
    mask = _brain_mask(a.astype(np.float64), float(np.percentile(a, FOREGROUND_QUANTILE)))
    if not mask.any():
        return center_crop_or_pad(a, target_shape)

    centres = np.asarray(ndimage.center_of_mass(mask.astype(np.float32)), dtype=np.float64)
    if not np.isfinite(centres).all():
        return center_crop_or_pad(a, target_shape)

    out = np.zeros(tuple(int(t) for t in target_shape), dtype=np.float32)
    src, dst = _window_slices(a.shape, target_shape, centres)
    out[dst] = a[src]
    return out


def normalize_intensity(array: np.ndarray, method: str = "reference_region") -> np.ndarray:
    """Remove scanner gain, dose and acquisition-time effects. Always scale-invariant.

    - `reference_region` - the principled default. Approximates the specific binding ratio
      clinicians use: subtract and divide by non-specific binding, estimated from a low
      percentile band of in-brain intensity (docs/01 section 5). The output reads as "how many
      times background is this voxel", which is the quantity that means the same thing at every
      centre.
    - `zscore_brain` - generic whitening over the brain mask. Cheap and assumption-free, but it
      discards the absolute ratio that carries the diagnosis.
    - `max` - divide by a robust maximum. Simplest; keeps the histogram shape but leaves the
      background level uncorrected.

    Ablate all three on OOF (docs/03 Phase 2). The domain argument favours the first; the data
    decides.
    """
    if method not in NORMALIZATIONS:
        raise ValueError(f"unknown normalization {method!r}; expected one of {NORMALIZATIONS}")

    # float64 for the statistics: the guarantee is a ratio of two reductions, and float32
    # accumulation over millions of voxels loses enough precision to be visible.
    a = np.asarray(array, dtype=np.float64)
    hi = float(np.percentile(a, 99.0))
    brain = a[_brain_mask(a, hi)]

    if method == "max":
        scale = float(np.percentile(a, 99.5))
        if scale <= 0.0:
            scale = float(np.abs(a).max()) or 1.0
        out = a / scale

    elif method == "zscore_brain":
        mean = float(brain.mean())
        std = float(brain.std())
        if std <= 0.0:
            std = abs(hi) or 1.0
        out = (a - mean) / std

    else:  # reference_region
        lo_q, hi_q = np.percentile(brain, REFERENCE_BAND)
        band = brain[(brain >= lo_q) & (brain <= hi_q)]
        ref = float(band.mean()) if band.size else float(brain.mean())
        # The fallback also scales linearly with the input, so invariance survives the
        # degenerate case rather than being quietly broken by it.
        denom = ref if ref > 0.0 else (abs(hi) or 1.0)
        out = (a - ref) / denom

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def preprocess(
    volume: Volume,
    target_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    target_shape: tuple[int, int, int] = (96, 96, 96),
    normalization: str = "reference_region",
    crop: str = "center",
) -> np.ndarray:
    """The full training-and-inference preprocessing pipeline. One tensor shape out, always.

    These defaults mirror `conf/baseline.yaml`. `05_package_submission.py` bakes the actual
    values from the trained run into `main.py`, so changing the config cannot silently
    desynchronise inference from training.
    """
    if crop not in CROPS:
        raise ValueError(f"unknown crop {crop!r}; expected one of {CROPS}")

    resampled = resample_to_spacing(volume, target_spacing)
    cropped = (
        center_of_mass_crop(resampled, target_shape)
        if crop == "com"
        else center_crop_or_pad(resampled, target_shape)
    )
    return normalize_intensity(cropped, normalization)
