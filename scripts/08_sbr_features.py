#!/usr/bin/env python3
"""Extract SBR-style geometric features from every scan, for a model that is not a CNN.

Why bother, with a CNN already at 0.3186
----------------------------------------
Not to beat it — to disagree with it. A gradient-boosted model over a few dozen explicit
striatal measurements makes errors in different places than a 3D convolutional network, and a
blend of two models that disagree is worth more than either. On CPU this is also the only
Phase 3 item that buys model diversity without retraining anything: one pass over the volumes,
then seconds of tree fitting (docs/05).

The features are the quantities a nuclear medicine physician actually reads (docs/01 4-5):
striatal uptake against a reference region, left-right asymmetry, the anterior-posterior
gradient that separates caudate from putamen, and the shape descriptors that distinguish a
comma from a dot.

On not knowing which axis is which
----------------------------------
Volumes are canonical RAS, so axis 0 should be left-right. This script does not assume it.
Laterality and gradient features are computed for every axis and every ordered axis pair, and
the model is left to discover which ones carry signal. The cost is a few redundant columns; the
benefit is that a wrong assumption about orientation degrades into noise rather than into a
silently meaningless feature set.

The report includes a two-lobe diagnostic per axis, which identifies the lateral axis
empirically. Check it before reading too much into any individual feature.

Output
------
- `data/interim/sbr_features.csv` — one row per uid. Contains uids, so it is git-ignored and
  Claude-blocked, like the fold assignments.
- `reports/experiments/sbr_features.md` — distributions over the whole set. Aggregate only,
  no uids; safe for Claude to read.

Usage:
    python scripts/08_sbr_features.py --config conf/baseline.yaml
    python scripts/08_sbr_features.py --config conf/baseline.yaml --limit 50   # smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.config import load_config  # noqa: E402
from datpd.data.io import load_volume  # noqa: E402
from datpd.data.preprocess import preprocess  # noqa: E402

# Half-width of the central box, in voxels, that should contain the striatum after COM
# cropping. At 2.0 mm this is a 66 mm cube; the striatum spans roughly 60-70 mm (docs/01 4),
# so this holds it with a little margin and excludes most of the rest of the brain.
BOX_HALF = 16

# Fractions of the box maximum used to define "supra-threshold" uptake. Two thresholds because
# the shape of the bright region changes with where you cut it, and that change is itself
# informative: a comma thins out gradually, a dot does not.
THRESHOLDS = (0.5, 0.7)


def _central_box(array: np.ndarray, half: int = BOX_HALF) -> np.ndarray:
    centre = [s // 2 for s in array.shape]
    return array[
        tuple(
            slice(max(0, c - half), min(s, c + half + 1))
            for c, s in zip(centre, array.shape)
        )
    ]


def _ratio(lo: float, hi: float) -> float:
    """Signed relative difference, safe when both sides are ~0."""
    total = lo + hi
    return float((lo - hi) / total) if abs(total) > 1e-9 else 0.0


def _inertia_shape(mask: np.ndarray) -> tuple[float, float, float]:
    """Eigenvalue-derived shape of a binary region: elongation, flatness, and its size.

    A healthy striatum images as a comma — elongated. Dopaminergic loss erodes the putaminal
    tail first, so the region rounds toward a dot. Elongation is the direct numeric statement of
    that, and it is the reason shape features belong here at all rather than only intensity.
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
    """All features for one preprocessed volume. Pure function of that volume alone.

    Per-sample by construction: nothing here consults another scan, so a model built on these
    features stays inside the competition's independence rule (CLAUDE.md 1.1).
    """
    features: dict[str, float] = {}
    box = _central_box(array)

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
    #
    # CAUTION on interpreting these: computed over the WHOLE box, the inertia tensor is
    # dominated by the separation between the two striata, not by the shape of either one. A
    # pair of compact dots 14 voxels apart is *more* elongated by this measure than a pair of
    # commas, which is the opposite of the clinical reading. The whole-box numbers are therefore
    # a lobe-separation descriptor and are named accordingly; the per-hemisphere `elong{axis}_*`
    # features below are the ones that measure comma-versus-dot. Both are kept — separation is
    # plausibly informative too — but they must not be confused.
    for threshold in THRESHOLDS:
        mask = box >= threshold * box_max
        tag = f"t{int(threshold * 100)}"
        features[f"{tag}_frac"] = float(mask.mean())
        separation, flatness, size = _inertia_shape(mask)
        features[f"{tag}_separation"] = separation
        features[f"{tag}_flatness"] = flatness
        features[f"{tag}_size"] = size

    # --- Laterality, computed for every axis ----------------------------------------------
    # Only one of these is anatomical left-right; the two-lobe diagnostic in the report says
    # which. The others are near-symmetric by construction and should carry little signal.
    for axis in range(3):
        lo, hi = np.split(box, [box.shape[axis] // 2], axis=axis)
        features[f"asym{axis}_mean"] = _ratio(float(lo.mean()), float(hi.mean()))
        features[f"asym{axis}_max"] = _ratio(float(lo.max()), float(hi.max()))
        features[f"asym{axis}_p99"] = _ratio(
            float(np.percentile(lo, 99)), float(np.percentile(hi, 99))
        )
        mask_lo = lo >= THRESHOLDS[0] * box_max
        mask_hi = hi >= THRESHOLDS[0] * box_max
        features[f"asym{axis}_supra"] = _ratio(float(mask_lo.sum()), float(mask_hi.sum()))
        # Absolute asymmetry: which side is worse is arbitrary, how unequal they are is not.
        features[f"asym{axis}_abs"] = abs(features[f"asym{axis}_mean"])

        elong_lo, _, _ = _inertia_shape(mask_lo)
        elong_hi, _, _ = _inertia_shape(mask_hi)
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
            ratios.append(_ratio(float(near.mean()), float(far.mean())))
        features[f"grad{lateral}{gradient}_mean"] = float(np.mean(ratios))
        features[f"grad{lateral}{gradient}_diff"] = float(abs(ratios[0] - ratios[1]))

    return features


def two_lobe_score(array: np.ndarray, axis: int) -> int:
    """Number of peaks in the supra-threshold profile along one axis.

    The lateral axis should show two lobes — left and right striatum — while the other two show
    one. Aggregated over the dataset this identifies the anatomical left-right axis without
    anyone having to look at a scan.
    """
    box = _central_box(array)
    mask = box >= THRESHOLDS[0] * float(box.max())
    profile = mask.sum(axis=tuple(a for a in range(3) if a != axis)).astype(float)
    if profile.max() <= 0:
        return 0
    peaks, _ = find_peaks(profile, prominence=0.15 * profile.max())
    return int(len(peaks))


def write_report(features: pd.DataFrame, lobes: pd.DataFrame, out_path: Path) -> None:
    """Aggregate-only summary. No uids, no per-scan values."""
    lines = [
        "# SBR-style feature distributions",
        "",
        f"- n = {len(features)} scans, {features.shape[1]} features",
        "",
        "## Which axis is left-right?",
        "",
        "Peaks in the supra-threshold profile along each axis. The lateral axis should show two",
        "lobes in most scans; the others should show one.",
        "",
        "| axis | 1 peak | 2 peaks | 3+ peaks | mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis in range(3):
        counts = lobes[f"lobes{axis}"]
        lines.append(
            f"| {axis} | {(counts == 1).mean():.1%} | {(counts == 2).mean():.1%} | "
            f"{(counts >= 3).mean():.1%} | {counts.mean():.2f} |"
        )

    best = max(range(3), key=lambda a: (lobes[f"lobes{a}"] == 2).mean())
    lines += [
        "",
        f"**Axis {best} looks lateral.** Canonical RAS predicts axis 0; if this disagrees, the",
        "orientation handling is worth re-checking before trusting the asymmetry features.",
        "",
        "## Feature distributions",
        "",
        "| feature | mean | std | p05 | p50 | p95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for column in features.columns:
        values = features[column]
        lines.append(
            f"| {column} | {values.mean():.4f} | {values.std():.4f} | "
            f"{values.quantile(0.05):.4f} | {values.median():.4f} | "
            f"{values.quantile(0.95):.4f} |"
        )

    degenerate = [c for c in features.columns if features[c].std() < 1e-9]
    if degenerate:
        lines += [
            "",
            f"**{len(degenerate)} constant features** (no variance, will be ignored by the "
            f"model): {', '.join(degenerate)}",
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/baseline.yaml")
    parser.add_argument("--out", default="data/interim/sbr_features.csv")
    parser.add_argument("--limit", type=int, default=None, help="smoke test, not a result")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    df = pd.read_csv(cfg.data.folds_csv)
    if args.limit:
        df = df.head(args.limit)
        print(f"SMOKE RUN: {args.limit} scans. Do not blend on this.")

    nifti_dir = REPO_ROOT / cfg.data.nifti_dir
    rows, lobe_rows, failures = [], [], 0
    start = time.time()

    for i, uid in enumerate(df["uid"], 1):
        try:
            volume = load_volume(nifti_dir / f"{uid}.nii.gz")
            array = preprocess(
                volume,
                tuple(cfg.data.target_spacing_mm),
                tuple(cfg.data.target_shape),
                cfg.data.normalization,
                cfg.data.crop,
            )
            rows.append({"uid": uid, **extract_features(array)})
            lobe_rows.append({f"lobes{a}": two_lobe_score(array, a) for a in range(3)})
        except Exception:
            failures += 1
        if i % 100 == 0:
            rate = (time.time() - start) / i
            print(
                f"  ...{i}/{len(df)}  {rate:.2f}s/scan  "
                f"eta {rate * (len(df) - i) / 60:.0f} min",
                flush=True,
            )

    if not rows:
        raise SystemExit("No scans processed successfully.")
    if failures:
        print(f"\n{failures} scans failed and were skipped.")

    features = pd.DataFrame(rows)
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    write_report(
        features.drop(columns=["uid"]),
        pd.DataFrame(lobe_rows),
        REPO_ROOT / "reports" / "experiments" / "sbr_features.md",
    )

    print(f"\nWrote {out_path.relative_to(REPO_ROOT)} "
          f"({len(features)} rows x {features.shape[1] - 1} features) - contains uids")
    print("Wrote reports/experiments/sbr_features.md - aggregates only, safe for Claude")
    print(f"\nTotal {(time.time() - start) / 60:.1f} min")
    print("\nNext: python scripts/09_blend.py --run baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
