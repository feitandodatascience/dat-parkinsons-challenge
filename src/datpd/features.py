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

The set was later extended, and the extension is APPENDED rather than merged in: the original 61
columns remain an exact prefix of the current list. Two reasons. Removing or reordering a column
invalidates every artifact fitted on it, and the first three contrast features
(`box_to_shell`, `box_to_global`, `box_peak_to_shell`) are known to be broken — they divide by a
mean of a SIGNED volume, which sits near zero and changes sign between scans, so they flip
rather than rank. They are kept only because something is fitted on them. `sbr_diff_*` and
`sbr_z_*` are what those three were trying to be.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

# Half-width of the central box, in voxels, that should contain the striatum after
# centre-of-mass cropping. At 2.0 mm this is a 66 mm cube; the striatum spans roughly 60-70 mm
# (docs/01 §4), so this holds it with a little margin and excludes most of the rest of the brain.
BOX_HALF = 16

# Fractions of the box maximum defining "supra-threshold" uptake. Two thresholds because the
# shape of the bright region changes with where you cut it, and that change is itself
# informative: a comma thins out gradually, a dot does not.
THRESHOLDS = (0.5, 0.7)

# More cut levels, kept in a separate constant so that THRESHOLDS — and therefore the `t50_*`
# and `t70_*` column names and values that fitted artifacts depend on — stay exactly as they
# were. Five levels turn a two-point comparison into a curve (see `extent_slope` below).
EXTRA_THRESHOLDS = (0.4, 0.6, 0.8)
ALL_THRESHOLDS = tuple(sorted(THRESHOLDS + EXTRA_THRESHOLDS))

# Floor for every denominator added below. Reference-normalised units put real spreads orders of
# magnitude above this, so the floor only ever fires on a degenerate volume (all-zero, constant,
# empty region) — where the numerator is zero too, and the feature comes out 0.0 rather than nan.
# That matters because `feature_names()` derives the column order from an all-zero volume.
EPS = 1e-6

# Ceiling on any added ratio. Nothing realistic comes near it; it exists so that "finite for
# every plausible input" is a property of the code rather than a property of the data.
RATIO_CLIP = 1e6


def _box_slices(array: np.ndarray, half: int) -> tuple[slice, ...]:
    centre = [s // 2 for s in array.shape]
    return tuple(
        slice(max(0, c - half), min(s, c + half + 1)) for c, s in zip(centre, array.shape)
    )


def central_box(array: np.ndarray, half: int = BOX_HALF) -> np.ndarray:
    """The central cube where the striatum should sit after cropping."""
    return array[_box_slices(array, half)]


def shell_values(array: np.ndarray, half: int = BOX_HALF) -> np.ndarray:
    """Everything outside the striatal box, flattened — the reference side of a contrast."""
    outside = np.ones(array.shape, dtype=bool)
    outside[_box_slices(array, half)] = False
    return array[outside]


def robust_spread(values: np.ndarray) -> float:
    """Interquartile range, floored strictly positive.

    This is the fix for the defect in `box_to_shell` / `box_to_global` / `box_peak_to_shell`.
    Those divide by a mean of a SIGNED, reference-normalised volume, which sits near zero and
    changes sign from scan to scan — so the ratio flips sign as the denominator crosses zero and
    is not monotone in the contrast it claims to measure. Measured over the training set they
    have standard deviations of 147, 88 and 1109. A quantile spread cannot flip: it is
    non-negative by construction, so everything built on it stays ordered.
    """
    if values.size == 0:
        return EPS
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return max(float(q75 - q25), EPS)


def safe_ratio(numerator: float, denominator: float) -> float:
    """A ratio whose denominator is known non-negative: floored, then clipped."""
    return float(
        np.clip(float(numerator) / max(float(denominator), EPS), -RATIO_CLIP, RATIO_CLIP)
    )


def standardised_moments(values: np.ndarray) -> tuple[float, float]:
    """Skewness and excess kurtosis of an intensity distribution.

    A healthy striatum puts a long bright tail on the box histogram; as uptake falls the
    distribution collapses toward the background and the tail goes with it.

    Written out rather than taken from `scipy.stats` because those return nan on a constant
    region, and every feature here has to be finite on an all-zero volume.
    """
    if values.size < 4:
        return 0.0, 0.0
    centred = values.astype(np.float64) - float(values.mean())
    variance = float(np.mean(centred**2))
    if variance <= EPS:
        return 0.0, 0.0
    return (
        float(np.mean(centred**3) / variance**1.5),
        float(np.mean(centred**4) / variance**2 - 3.0),
    )


def mask_components(mask: np.ndarray) -> tuple[float, float, float]:
    """Component count of a supra-threshold region, and how the mass splits between the two
    largest.

    This is the "are both striata still there" question asked directly. Two components is the
    normal picture; one means either that they have merged or that one side has faded below the
    cut entirely, and the size ratio of the two largest says which.
    """
    if not mask.any():
        return 0.0, 0.0, 0.0
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0.0, 0.0, 0.0
    sizes = np.sort(np.bincount(labels.ravel())[1:].astype(np.float64))[::-1]
    largest = float(sizes[0])
    second = float(sizes[1]) if sizes.size > 1 else 0.0
    return float(count), safe_ratio(largest, float(sizes.sum())), safe_ratio(second, largest)


def mask_surface(mask: np.ndarray) -> tuple[float, float]:
    """Surface-to-volume ratio and sphericity of a binary region.

    Complementary to `inertia_shape`: the eigenvalues describe how the region is oriented and
    stretched, this describes how convoluted its boundary is. A comma has far more surface per
    unit volume than the dot it degenerates into, and unlike the eigenvalues that statement
    survives the two striata being measured together.
    """
    volume = float(mask.sum())
    if volume < 1.0:
        return 0.0, 0.0
    # Pad with False so that a region touching the box edge is counted as bounded there rather
    # than wrapping around; np.roll on the padded array then never brings real voxels across.
    padded = np.pad(mask, 1, constant_values=False)
    faces = 0.0
    for axis in range(3):
        for shift in (1, -1):
            neighbour = np.roll(padded, shift, axis=axis)[1:-1, 1:-1, 1:-1]
            faces += float(np.count_nonzero(mask & ~neighbour))
    sphericity = safe_ratio((36.0 * np.pi * volume**2) ** (1.0 / 3.0), faces)
    return safe_ratio(faces, volume), sphericity


def split_centre(region: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Near and far halves along one axis, dropping the central slice when there is one.

    `np.split` at `shape // 2` gives the far half one extra slice on the odd-sized box, so every
    near-minus-far quantity carries a constant offset and a perfectly symmetric volume does not
    score zero. Equal halves make the zero mean what it says.
    """
    n = region.shape[axis] // 2
    near = [slice(None)] * region.ndim
    far = [slice(None)] * region.ndim
    near[axis] = slice(0, n)
    far[axis] = slice(region.shape[axis] - n, None)
    return region[tuple(near)], region[tuple(far)]


def hemisphere_gradient(
    half: np.ndarray, half_mask: np.ndarray, axis: int, scale: float
) -> tuple[float, float, float]:
    """The caudate-to-putamen gradient inside one hemisphere, measured three ways.

    Putaminal uptake goes before caudate uptake (docs/01 §4), so within one striatum the comma
    loses its posterior tail first. The `grad*` features above test that with a mean over a
    half-box that is mostly background, which mostly measures how much background the box
    happened to include. These are better-powered readings of the same question:

    - the intensity-weighted centroid of the supra-threshold mask, which slides toward the end
      that is still bright;
    - the supra-threshold VOXEL COUNT near against far, which is the tail shortening directly;
    - an upper quantile of intensity near against far, which is `grad*_mean` with an estimator
      that the background cannot drown.

    The last two are deliberately measured on the fixed half-box geometry rather than inside the
    mask. A mask-restricted mean is not a clean contrast: the mask reaches further into the
    Gaussian tail on the brighter side, so it collects more barely-supra-threshold voxels there,
    and that bias runs opposite to the effect and can outweigh it. Measured on the same geometry
    for both halves, it cannot.

    The centroid is rescaled to [-1, 1] so it does not depend on the box size; the count
    difference is a bounded balance; the intensity difference is divided by a strictly positive
    robust spread (never a mean — see `robust_spread`).
    """
    near_mask, far_mask = split_centre(half_mask, axis)
    near, far = split_centre(half, axis)

    n_near = float(np.count_nonzero(near_mask))
    n_far = float(np.count_nonzero(far_mask))
    supra = safe_ratio(n_near - n_far, max(n_near + n_far, 1.0))
    contrast = safe_ratio(float(np.percentile(near, 90.0) - np.percentile(far, 90.0)), scale)

    if not half_mask.any():
        return 0.0, supra, contrast
    positions = np.argwhere(half_mask)[:, axis].astype(np.float64)
    # Negative voxels are background that crept above the cut; they must not pull the centroid.
    weights = np.maximum(half[half_mask].astype(np.float64), 0.0)
    total = float(weights.sum())
    centroid = (
        float((positions * weights).sum() / total) if total > EPS else float(positions.mean())
    )
    span = max(float(half.shape[axis] - 1), 1.0)
    return float(2.0 * (centroid - 0.5 * span) / span), supra, contrast


def axis_profile(mask: np.ndarray, axis: int) -> np.ndarray:
    """Supra-threshold voxel count per slice along one axis."""
    return mask.sum(axis=tuple(a for a in range(3) if a != axis)).astype(np.float64)


def count_peaks(profile: np.ndarray) -> int:
    """Peaks in a supra-threshold profile, at the prominence the axis diagnostic uses."""
    if profile.size == 0 or profile.max() <= 0:
        return 0
    peaks, _ = find_peaks(profile, prominence=0.15 * float(profile.max()))
    return int(len(peaks))


def two_lobe_score(array: np.ndarray, axis: int) -> int:
    """Number of peaks in the supra-threshold profile along one axis.

    The lateral axis should show two lobes — left and right striatum — while the other two show
    one, which is how `scripts/08_sbr_features.py` identifies the anatomical left-right axis
    without anyone having to look at a scan. It was only ever a report column there; the same
    number is a feature here, because "did one striatum disappear" is the question the whole
    scan is asking. Kept under this name so the diagnostic can import it instead of duplicating.
    """
    box = central_box(array)
    return count_peaks(axis_profile(box >= THRESHOLDS[0] * float(box.max()), axis))


def lobe_dip(profile: np.ndarray) -> float:
    """Depth of the valley between the two tallest peaks, as a fraction of the shorter peak.

    `count_peaks` answers "are both striata still there" with 0, 1 or 2. This answers it
    continuously, so a tree can split part-way through the merge rather than only at its end.
    """
    if profile.size == 0 or profile.max() <= 0:
        return 0.0
    peaks, _ = find_peaks(profile, prominence=0.15 * float(profile.max()))
    if len(peaks) < 2:
        return 0.0
    tallest = peaks[np.argsort(profile[peaks])[::-1][:2]]
    left, right = int(min(tallest)), int(max(tallest))
    valley = float(profile[left : right + 1].min())
    lower = float(min(profile[left], profile[right]))
    return safe_ratio(lower - valley, lower) if lower > EPS else 0.0


def peak_distance(box: np.ndarray, axis: int) -> tuple[float, float]:
    """Distance between the brightest voxel of each half-box, in voxels.

    Two striata that have each shrunk to a dot sit further apart than two intact commas, whose
    bright centres are pulled toward each other along their tails. Reported both as a straight
    line and as the component along the splitting axis, because only the latter is pure
    separation.
    """
    lo, hi = np.split(box, [box.shape[axis] // 2], axis=axis)
    lo_peak = np.array(np.unravel_index(int(np.argmax(lo)), lo.shape), dtype=np.float64)
    hi_peak = np.array(np.unravel_index(int(np.argmax(hi)), hi.shape), dtype=np.float64)
    hi_peak[axis] += float(lo.shape[axis])
    return float(np.linalg.norm(lo_peak - hi_peak)), float(abs(lo_peak[axis] - hi_peak[axis]))


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

    # =====================================================================================
    # Everything below was added after the first fitted artifacts, and is deliberately
    # APPENDED rather than interleaved: the original column list stays an exact prefix of this
    # one, so a diff of `feature_names()` shows only additions and nothing fitted on the old
    # order can silently have its columns permuted underneath it.
    # =====================================================================================

    # --- Contrasts that do not flip sign ---------------------------------------------------
    # `box_to_shell`, `box_to_global` and `box_peak_to_shell` above divide by a near-zero,
    # sign-changing mean (see `robust_spread`), so they are scrambled rather than merely oddly
    # scaled. They are kept because removing a column invalidates fitted artifacts; nothing
    # should be read off them. These are the replacements. `preprocess` has already divided by a
    # reference region, so a DIFFERENCE in these units is already a specific binding ratio
    # (docs/01 §5) — the division is only there to make scans comparable, and a quantile spread
    # does that without ever crossing zero.
    shell = shell_values(array)
    shell_q = np.percentile(shell, [25.0, 50.0, 75.0]) if shell.size else np.zeros(3)
    shell_p50 = float(shell_q[1])
    shell_scale = max(float(shell_q[2] - shell_q[0]), EPS)
    box_scale = robust_spread(box.ravel())
    global_scale = robust_spread(array.ravel())
    box_mean = float(box.mean())
    box_p99 = features["box_p99"]
    global_p50 = features["global_p50"]

    features["shell_p50"] = shell_p50
    features["shell_iqr"] = shell_scale
    features["box_iqr"] = box_scale
    features["sbr_diff_mean"] = box_mean - shell_p50
    features["sbr_diff_p99"] = box_p99 - shell_p50
    features["sbr_diff_peak"] = box_max - shell_p50
    features["sbr_z_mean"] = safe_ratio(box_mean - shell_p50, shell_scale)
    features["sbr_z_p99"] = safe_ratio(box_p99 - shell_p50, shell_scale)
    features["sbr_z_peak"] = safe_ratio(box_max - shell_p50, shell_scale)
    features["sbr_gdiff_mean"] = box_mean - global_p50
    features["sbr_gz_mean"] = safe_ratio(box_mean - global_p50, global_scale)

    # --- Laterality on robust statistics ----------------------------------------------------
    # `asym*_mean` averages a half-box that is mostly background, so it is a weak estimator of
    # striatal uptake, and `asym*_max` rests on a single voxel. An upper quantile is dominated by
    # the striatum yet still averages over hundreds of voxels. The per-side values are contrasts
    # against the shell, so `hemi*_min` is directly "uptake on the worse side", which is the
    # number a reader quotes; the signed difference is scaled by the box spread so it stays
    # comparable across scans without the sign hazard of a sum in the denominator.
    for axis in range(3):
        lo, hi = np.split(box, [box.shape[axis] // 2], axis=axis)
        lo95, lo99 = np.percentile(lo, [95.0, 99.0])
        hi95, hi99 = np.percentile(hi, [95.0, 99.0])
        side95 = (
            safe_ratio(lo95 - shell_p50, shell_scale),
            safe_ratio(hi95 - shell_p50, shell_scale),
        )
        side99 = (
            safe_ratio(lo99 - shell_p50, shell_scale),
            safe_ratio(hi99 - shell_p50, shell_scale),
        )
        features[f"hemi{axis}_min_p95"] = min(side95)
        features[f"hemi{axis}_min_p99"] = min(side99)
        features[f"hemi{axis}_mean_p99"] = float(np.mean(side99))
        features[f"rasym{axis}_p95"] = safe_ratio(float(lo95 - hi95), box_scale)
        features[f"rasym{axis}_p99"] = safe_ratio(float(lo99 - hi99), box_scale)
        features[f"rasym{axis}_abs"] = abs(features[f"rasym{axis}_p99"])

    # --- Supra-threshold extent at five cut levels, and the decay across them ---------------
    # Two thresholds give two points; five give a curve, and the curve is the comma-versus-dot
    # question in numbers. A comma thins gradually as the cut rises because its putaminal tail is
    # dimmer than its caudate head; a dot is uniformly at peak brightness and then falls off a
    # cliff. The slope and curvature of log(count) against threshold say which happened.
    counts: dict[float, float] = {}
    for threshold in ALL_THRESHOLDS:
        mask = box >= threshold * box_max
        counts[threshold] = float(np.count_nonzero(mask))
        if threshold in THRESHOLDS:
            continue  # t50_* and t70_* already exist above, under their original names
        tag = f"t{int(round(threshold * 100))}"
        features[f"{tag}_frac"] = float(mask.mean())
        separation, _, size = inertia_shape(mask)
        features[f"{tag}_separation"] = separation
        features[f"{tag}_size"] = size

    # log1p, not log: the count is legitimately zero on a volume with no supra-threshold voxels.
    logs = np.log1p(np.array([counts[t] for t in ALL_THRESHOLDS], dtype=np.float64))
    levels = np.array(ALL_THRESHOLDS, dtype=np.float64)
    centred_levels = levels - levels.mean()
    features["extent_log_40_80"] = float(logs[0] - logs[-1])
    features["extent_log_50_70"] = float(np.log1p(counts[0.5]) - np.log1p(counts[0.7]))
    features["extent_slope"] = safe_ratio(
        float((centred_levels * (logs - logs.mean())).sum()), float((centred_levels**2).sum())
    )
    # Second difference across the three central levels: positive means the decay is slowing.
    features["extent_curve"] = float(logs[0] - 2.0 * logs[2] + logs[4])

    # --- Topology and boundary of the bright region ------------------------------------------
    for threshold in THRESHOLDS:
        mask = box >= threshold * box_max
        tag = f"t{int(threshold * 100)}"
        cc_count, cc_top_share, cc_second_ratio = mask_components(mask)
        features[f"{tag}_cc_count"] = cc_count
        features[f"{tag}_cc_top_share"] = cc_top_share
        features[f"{tag}_cc_second_ratio"] = cc_second_ratio
        surface_ratio, sphericity = mask_surface(mask)
        features[f"{tag}_surface_ratio"] = surface_ratio
        features[f"{tag}_sphericity"] = sphericity

    mask50 = box >= THRESHOLDS[0] * box_max

    # --- Caudate-versus-putamen, measured on the mask rather than the box --------------------
    # Same axis-agnostic stance as the `grad*` block: every ordered pair is computed and the
    # model decides which one is anterior-posterior within a hemisphere.
    for lateral, gradient in permutations(range(3), 2):
        halves = split_centre(box, lateral)
        half_masks = split_centre(mask50, lateral)
        centroids, supras, contrasts = [], [], []
        for half, half_mask in zip(halves, half_masks):
            centroid, supra, contrast = hemisphere_gradient(half, half_mask, gradient, box_scale)
            centroids.append(centroid)
            supras.append(supra)
            contrasts.append(contrast)
        features[f"mcent{lateral}{gradient}_mean"] = float(np.mean(centroids))
        features[f"mcent{lateral}{gradient}_diff"] = float(abs(centroids[0] - centroids[1]))
        features[f"msupra{lateral}{gradient}_mean"] = float(np.mean(supras))
        features[f"mgrad{lateral}{gradient}_mean"] = float(np.mean(contrasts))
        # A steeper gradient on one side than the other: loss is typically worse on one side
        # (docs/01 §4), and which side is arbitrary while how unequal they are is not.
        features[f"mgrad{lateral}{gradient}_diff"] = float(abs(contrasts[0] - contrasts[1]))

    # --- Are both striata still resolved? -----------------------------------------------------
    for axis in range(3):
        profile = axis_profile(mask50, axis)
        features[f"lobes{axis}"] = float(count_peaks(profile))
        features[f"lobedip{axis}"] = lobe_dip(profile)
        euclidean, along_axis = peak_distance(box, axis)
        features[f"peakdist{axis}"] = euclidean
        features[f"peaksep{axis}"] = along_axis

    # --- Shape of the box histogram ------------------------------------------------------------
    box_skew, box_kurtosis = standardised_moments(box.ravel())
    features["box_skew"] = box_skew
    features["box_kurtosis"] = box_kurtosis

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
