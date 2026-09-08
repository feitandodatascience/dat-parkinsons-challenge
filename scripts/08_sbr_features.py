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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.config import load_config  # noqa: E402
from datpd.data.io import load_volume  # noqa: E402
from datpd.data.preprocess import preprocess  # noqa: E402
from datpd.features import (  # noqa: E402
    THRESHOLDS,
    central_box,
    extract_features,
)


def two_lobe_score(array: np.ndarray, axis: int) -> int:
    """Number of peaks in the supra-threshold profile along one axis.

    The lateral axis should show two lobes — left and right striatum — while the other two show
    one. Aggregated over the dataset this identifies the anatomical left-right axis without
    anyone having to look at a scan.
    """
    box = central_box(array)
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
