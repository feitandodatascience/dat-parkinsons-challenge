#!/usr/bin/env python3
"""Break the SBR features down by acquisition family, to test whether cropping is failing.

The lead this exists to chase
-----------------------------
`reports/experiments/baseline_by_group.md` showed the largest acquisition family,
`256@2.5x2.5x2.5mm` (n=460, 34% of the data), performing worst of the large families — AUROC
0.9176 against 0.9560 for `128@2.4x2.4x2.4mm`, and log loss 0.3691 against 0.2339.

Spacing does not separate those two families; **field of view does**. A 256 matrix at 2.5 mm is
a 640 mm cube, against 307 mm for a 128 matrix at 2.4 mm. EDA already recorded that in the
widest acquisitions the head occupies about 1% of the array, which is precisely where
centre-of-mass cropping has the least margin for error. So the hypothesis is that the worst
family is worst because its crops are worse, not because its images are.

The measurement, and what confounds it
--------------------------------------
The indicator is already sitting in the feature table: **`box_max / global_max`**. The central
box is where the striatum should be after COM cropping, so if the crop landed correctly the
brightest voxel in the whole volume is inside it and the ratio is ~1. A ratio well below 1 means
the peak uptake is somewhere else entirely — the crop missed.

**That reading is wrong on its own, and the first run of this script showed why.** In a
pathologic scan striatal uptake is *reduced*, so the brightest voxel is more often outside the
box for reasons that have nothing to do with cropping. The ratio is therefore confounded with
the label, and the confound is large: across families the 5th-percentile ratio tracks positive
rate almost monotonically, from 1.000 at a 46.3% positive rate down to 0.504 at 62.1%.

So the ratio is reported **stratified by label**. A family whose crops are genuinely worse
should look worse within the negatives, where every striatum is bright and the peak belongs in
the box. A family that only looks worse overall is a family with sicker patients, and its lower
AUROC is not a preprocessing bug.

This runs on the CSV `08_sbr_features.py` already wrote. No volumes are re-read and nothing is
re-preprocessed, so it costs seconds.

Reading the result
------------------
If `peak_in_box` is uniformly high, cropping is fine and the family's weakness is about image
quality or resolution — which on this hardware is not fixable before the deadline, and the
right response is to accept it. If it is visibly low for the wide-field families, that is a
preprocessing bug with real log loss attached, and worth the retraining run it would take.

Output is aggregate only; families below `--min-group-n` are pooled.

Usage:
    python scripts/10_features_by_family.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SMALL_LABEL = "small families (pooled)"

# Feature, label, and whether a LOW value is the suspicious direction.
DIAGNOSTICS: tuple[tuple[str, str, bool], ...] = (
    ("peak_in_box", "peak in box", True),
    ("box_to_shell", "striatal contrast", True),
    ("t50_frac", "supra-threshold fraction", False),
    ("elong0_lo", "lobe elongation (low side)", True),
    ("asym0_abs", "absolute asymmetry", False),
)


def add_derived(features: pd.DataFrame) -> pd.DataFrame:
    """Add the crop-quality ratio, which is a function of two existing columns.

    Derived here rather than in `08_sbr_features.py` so this diagnostic can run against feature
    tables that were extracted before the question came up — which includes the only one that
    currently exists.
    """
    out = features.copy()
    denominator = out["global_max"].replace(0.0, np.nan)
    out["peak_in_box"] = (out["box_max"] / denominator).fillna(0.0)
    return out


def summarise_by_family(df: pd.DataFrame, min_group_n: int) -> pd.DataFrame:
    counts = df["acquisition_group"].value_counts()
    reportable = set(counts[counts >= min_group_n].index)
    df = df.copy()
    df["report_group"] = np.where(
        df["acquisition_group"].isin(reportable), df["acquisition_group"], SMALL_LABEL
    )

    rows = []
    for group, sub in df.groupby("report_group"):
        row: dict[str, object] = {"family": group, "n": len(sub)}
        for column, _, _ in DIAGNOSTICS:
            if column in sub:
                row[f"{column}_p50"] = float(sub[column].median())
                row[f"{column}_p05"] = float(sub[column].quantile(0.05))
        if "is_pathologic" in sub:
            for label, tag in ((0, "neg"), (1, "pos")):
                stratum = sub[sub["is_pathologic"] == label]
                row[f"n_{tag}"] = len(stratum)
                if len(stratum):
                    row[f"peak_{tag}_p50"] = float(stratum["peak_in_box"].median())
                    row[f"peak_{tag}_p05"] = float(stratum["peak_in_box"].quantile(0.05))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def build_report(table: pd.DataFrame, n_total: int, min_group_n: int) -> list[str]:
    lines = [
        "# SBR features by acquisition family",
        "",
        "Testing one hypothesis: is the worst-performing acquisition family worst because its",
        "**crops** are worse, rather than its images? See the module docstring of",
        "`scripts/10_features_by_family.py` for the reasoning.",
        "",
        f"- n = {n_total}, families reported at n >= {min_group_n}",
        "",
        "## Crop quality",
        "",
        "`peak in box` is the brightest voxel inside the central striatal box divided by the",
        "brightest voxel anywhere in the volume. After a correct centre-of-mass crop the",
        "striatum is the brightest thing present and sits in the box, so this should be at or",
        "near **1.00**. Values well below 1 mean the peak uptake is outside the box — the crop",
        "missed. The 5th percentile matters more than the median: a crop that fails on one scan",
        "in twenty still costs real log loss.",
        "",
        "| family | n | peak in box (median) | peak in box (p05) | striatal contrast (median) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['family']} | {int(row['n'])} | {row.get('peak_in_box_p50', float('nan')):.3f} "
            f"| {row.get('peak_in_box_p05', float('nan')):.3f} "
            f"| {row.get('box_to_shell_p50', float('nan')):.3f} |"
        )

    lines += [
        "",
        "## The same ratio, split by label",
        "",
        "The column above cannot be read on its own: a pathologic striatum is dimmer, so its",
        "peak leaves the box for reasons unrelated to cropping. **The negatives are the",
        "diagnostic** — every negative striatum is bright, so the peak belongs in the box, and a",
        "family whose crops genuinely fail should look worse here.",
        "",
        "| family | n- | peak in box, negatives (p05) | n+ | peak in box, positives (p05) |",
        "|---|---:|---:|---:|---:|",
    ]
    has_labels = "peak_neg_p05" in table.columns
    if has_labels:
        for _, row in table.iterrows():
            lines.append(
                f"| {row['family']} | {int(row.get('n_neg', 0))} "
                f"| {row.get('peak_neg_p05', float('nan')):.3f} "
                f"| {int(row.get('n_pos', 0))} "
                f"| {row.get('peak_pos_p05', float('nan')):.3f} |"
            )
        neg_spread = float(
            table["peak_neg_p05"].max() - table["peak_neg_p05"].min()
        )
        pos_spread = float(
            table["peak_pos_p05"].max() - table["peak_pos_p05"].min()
        )
        lines += [
            "",
            f"- spread across families within the negatives: **{neg_spread:.3f}**",
            f"- spread across families within the positives: **{pos_spread:.3f}**",
            "",
            "## Verdict",
            "",
        ]
        if neg_spread < 0.15:
            lines += [
                "**Cropping is not the explanation.** Within the negatives — where the striatum",
                "is bright and the peak belongs in the box — the families agree. The differences",
                "in the unstratified column are the label confound, not preprocessing. A weaker",
                "family's weaker AUROC is about its images, and on this hardware that is not",
                "fixable before the deadline. Do not spend a retraining run on the crop.",
                "",
            ]
        else:
            lines += [
                "**Worth investigating.** The families still differ within the negatives, where",
                "the label confound cannot explain it. Check whether the low ones are the",
                "wide-field acquisitions — a 256 matrix at 2.5 mm is a 640 mm cube against 307 mm",
                "for a 128 matrix at 2.4 mm — since that is the mechanism predicted here. A crop",
                "fix costs a retraining run, so confirm before spending it.",
                "",
            ]
    else:
        lines += [
            "| (labels unavailable) | | | | |",
            "",
            "**Cannot judge.** Without labels the ratio cannot be separated from disease",
            "severity. Pass a folds file that carries `is_pathologic`.",
            "",
        ]

    others = [(column, label) for column, label, _ in DIAGNOSTICS if column != "peak_in_box"]
    lines += [
        "## Other features",
        "",
        "| family | n | " + " | ".join(label for _, label in others) + " |",
        "|---|---:|" + "---:|" * len(others),
    ]
    for _, row in table.iterrows():
        cells = [f"{row.get(f'{column}_p50', float('nan')):.3f}" for column, _ in others]
        lines.append(f"| {row['family']} | {int(row['n'])} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Medians over each family. These are the model's raw inputs, so a family whose",
        "distribution sits far from the others is a family the network sees differently — which",
        "is worth knowing whether or not the crop turns out to be the cause.",
        "",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/interim/sbr_features.csv")
    parser.add_argument("--folds-csv", default="data/interim/folds.csv")
    parser.add_argument("--min-group-n", type=int, default=30)
    args = parser.parse_args()

    features_path = REPO_ROOT / args.features
    if not features_path.exists():
        raise SystemExit(
            f"Missing {features_path}.\n"
            "Run: python scripts/08_sbr_features.py --config conf/baseline.yaml"
        )

    features = add_derived(pd.read_csv(features_path))
    folds_all = pd.read_csv(REPO_ROOT / args.folds_csv)
    columns = ["uid", "acquisition_group"]
    if "is_pathologic" in folds_all.columns:
        columns.append("is_pathologic")
    folds = folds_all[columns]
    df = features.merge(folds, on="uid", how="left")

    missing = int(df["acquisition_group"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} feature rows have no acquisition_group; regenerate folds.")

    if len(df) < 200:
        print(
            f"WARNING: only {len(df)} rows. This looks like a --limit smoke extraction; "
            "per-family medians will be unreliable."
        )

    table = summarise_by_family(df, args.min_group_n)
    lines = build_report(table, len(df), args.min_group_n)

    out_path = REPO_ROOT / "reports" / "experiments" / "features_by_family.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Wrote {out_path.relative_to(REPO_ROOT)} - aggregates only, safe for Claude")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
