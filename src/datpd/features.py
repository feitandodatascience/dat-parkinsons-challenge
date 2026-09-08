"""SBR-style geometric features, computed from a preprocessed volume.

These are the quantities a nuclear medicine physician actually reads off a DaT scan
(docs/01 §§4-5): striatal uptake against a reference region, left-right asymmetry, the
anterior-posterior gradient that separates caudate from putamen, and the shape descriptors
that distinguish a comma from a dot.

They exist to give a gradient-boosted model something to disagree with the CNN about. On
out-of-fold predictions the two logit vectors correlate 0.797, and stacking them is worth
-0.0175 log loss — several times any other change measured in this project.

**This module lives in the package rather than in the script that first used it because it is
vendored into the submission.** Inference has to compute exactly these numbers, from exactly
this code; a second implementation in `main.py` would be free to drift, and the drift would be
invisible until the leaderboard score failed to match CV.

Per-sample by construction (CLAUDE.md §1.1): every function here reads one volume and nothing
else, so a model built on these features never pools information across the test set.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

# Half-width of the central box, in voxels, that should contain the striatum after
# centre-of-mass cropping. At 2.0 mm this is a 66 mm cube; the striatum spans roughly 60-70 mm
# (docs/01 §4), so this holds it with a little margin and excludes most of the rest of the brain.
BOX_HALF = 16

# Fractions of the box maximum defining "supra-threshold" uptake. Two thresholds because the
# shape of the bright region changes with where you cut it, and that change is itself
# informative: a comma thins out gradually, a dot does not.
THRESHOLDS = (0.5, 0.7)


def central_box(array: np.ndarray, half: int = BOX_HALF) -> np.ndarray:
    """The central cube where the striatum should sit after cropping."""
    centre = [s // 2 for s in array.shape]
    return array[
        tuple(
            slice(max(0, c - half), min(s, c + half + 1))
            for c, s in zip(centre, array.shape)
        )
    ]


def signed_ratio(lo: float, hi: float) -> float:
    """Signed relative difference, safe when both sides are ~0."""
    total = lo + hi
    return float((lo - hi) / total) if abs(total) > 1e-9 else 0.0


def inertia_shape(mask: np.ndarray) -> tuple[float, float, float]:
    """Eigenvalue-derived shape of a binary region: elongation, flatness, and its size.

    A healthy striatum images as a comma — elongated. Dopaminergic loss erodes the putaminal
    tail first, so the region rounds toward a dot, and elongation states that numerically.

    CAUTION: applied to a region containing BOTH striata this measures their separation rather
    than either one's shape, and a pair of compact dots then scores as more elongated than a
    pair of commas. Use it per hemisphere for shape; the whole-box value is named `separation`
    downstream for that reason.
    """
    coords = np.argwhere(mask)
    if len(coords) < 8:
        return 0.0, 0.0, 0.0
    centred = coords - coords.mean(axis=0)
    cov = (centred.T @ centred) / len(centred)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    e0, e1, e2 = np.maximum(eigenvalues, 1e-9)
    return float(np.sqrt(1.0 - e1 / e0)), float(np.sqrt(1.0 - e2 / e0)), float(len(coords))


def extract_features(array: np.ndarray) -> dict[str, float]:
    """All features for one preprocessed volume. A pure function of that volume alone.

    On not assuming which axis is which: volumes are canonical RAS, so axis 0 should be
    left-right. Laterality and gradient features are nonetheless computed for every axis and
    every ordered axis pair, and the model is left to find the ones that carry signal. The cost
    is a few redundant columns; the benefit is that a wrong orientation assumption degrades into
    noise rather than into a silently meaningless feature. (The two-lobe diagnostic in
    `scripts/08_sbr_features.py` confirms axis 0 empirically: 77.2% of scans show two lobes
    along it, against 4.0% and 2.6% for the other two.)
    """
    features: dict[str, float] = {}
    box = central_box(array)

    # --- Intensity, globally and in the striatal box -------------------------------------
    for name, region in (("global", array), ("box", box)):
        flat = region.ravel()
        features[f"{name}_mean"] = float(flat.mean())
        features[f"{name}_std"] = float(flat.std())
        for q in (50, 75, 90, 99):
            features[f"{name}_p{q}"] = float(np.percentile(flat, q))
        features[f"{name}_max"] = float(flat.max())

    # The classic SBR shape: target region against everything else. `preprocess` already
    # divided by a reference region, so this is a second, geometry-based view of the same idea.
    shell_mean = float((array.sum() - box.sum()) / max(array.size - box.size, 1))
    features["box_to_shell"] = float(box.mean() / shell_mean) if abs(shell_mean) > 1e-9 else 0.0
    features["box_to_global"] = (
        float(box.mean() / array.mean()) if abs(float(array.mean())) > 1e-9 else 0.0
    )
    features["box_peak_to_shell"] = (
        float(box.max() / shell_mean) if abs(shell_mean) > 1e-9 else 0.0
    )

    box_max = float(box.max())

    # --- Supra-threshold extent and shape -------------------------------------------------
    # Named `separation`, not `elongation`: over the whole box the inertia tensor is dominated
    # by the gap between the two striata, so a pair of dots scores higher than a pair of commas.
    # The per-hemisphere `elong{axis}_*` features below are what measure comma-versus-dot.
    for threshold in THRESHOLDS:
        mask = box >= threshold * box_max
        tag = f"t{int(threshold * 100)}"
        features[f"{tag}_frac"] = float(mask.mean())
        separation, flatness, size = inertia_shape(mask)
        features[f"{tag}_separation"] = separation
        features[f"{tag}_flatness"] = flatness
        features[f"{tag}_size"] = size

    # --- Laterality, computed for every axis ----------------------------------------------
    for axis in range(3):
        lo, hi = np.split(box, [box.shape[axis] // 2], axis=axis)
        features[f"asym{axis}_mean"] = signed_ratio(float(lo.mean()), float(hi.mean()))
        features[f"asym{axis}_max"] = signed_ratio(float(lo.max()), float(hi.max()))
        features[f"asym{axis}_p99"] = signed_ratio(
            float(np.percentile(lo, 99)), float(np.percentile(hi, 99))
        )
        mask_lo = lo >= THRESHOLDS[0] * box_max
        mask_hi = hi >= THRESHOLDS[0] * box_max
        features[f"asym{axis}_supra"] = signed_ratio(float(mask_lo.sum()), float(mask_hi.sum()))
        # Absolute asymmetry: which side is worse is arbitrary, how unequal they are is not.
        features[f"asym{axis}_abs"] = abs(features[f"asym{axis}_mean"])

        elong_lo, _, _ = inertia_shape(mask_lo)
        elong_hi, _, _ = inertia_shape(mask_hi)
        features[f"elong{axis}_lo"] = elong_lo
        features[f"elong{axis}_hi"] = elong_hi
        features[f"elong{axis}_min"] = min(elong_lo, elong_hi)

    # --- Gradient within each half: the caudate-to-putamen direction ------------------------
    # Putaminal uptake is lost before caudate uptake, so the gradient along the anterior-
    # posterior axis, measured separately in each hemisphere, is one of the earliest signs.
    for lateral, gradient in permutations(range(3), 2):
        halves = np.split(box, [box.shape[lateral] // 2], axis=lateral)
        ratios = []
        for half in halves:
            near, far = np.split(half, [half.shape[gradient] // 2], axis=gradient)
            ratios.append(signed_ratio(float(near.mean()), float(far.mean())))
        features[f"grad{lateral}{gradient}_mean"] = float(np.mean(ratios))
        features[f"grad{lateral}{gradient}_diff"] = float(abs(ratios[0] - ratios[1]))

    return features


def feature_names() -> list[str]:
    """The feature column order, derived from the function rather than hard-coded.

    A gradient-boosted model is fitted on a column ORDER, and at inference the same order has to
    be reproduced exactly. Deriving it from a zero volume means the two cannot disagree.
    """
    return list(extract_features(np.zeros((96, 96, 96), dtype=np.float32)))


def feature_vector(array: np.ndarray) -> np.ndarray:
    """Features as a row vector in the canonical column order, for inference."""
    values = extract_features(array)
    return np.array([values[name] for name in feature_names()], dtype=np.float64)
