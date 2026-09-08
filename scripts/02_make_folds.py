#!/usr/bin/env python3
"""Build cross-validation folds and write them to data/interim/folds.csv.

Produces BOTH split types so every experiment can be scored under either regime:

  fold_strat  - stratified on the label. Lower variance, mildly optimistic.
  fold_group  - stratified AND grouped by acquisition signature (a scanner fingerprint,
                standing in for hospital centre, which is not given as a column).

Track both. A large gap between them means centre effects are real, and the grouped number is
the one that will predict your leaderboard score. See docs/02 Delta 7.

Output goes to data/interim/, which is git-ignored and Claude-blocked (it contains uids).
A safe aggregate summary is written to reports/eda/raw/ for you to vet and promote.

Usage:
    python scripts/02_make_folds.py
    python scripts/02_make_folds.py --n-splits 10 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.data.io import read_header_only  # noqa: E402
from datpd.data.splits import acquisition_signature, add_folds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/raw/train_labels.csv")
    parser.add_argument("--nifti-dir", default="data/raw/niftis")
    parser.add_argument("--out", default="data/interim/folds.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = pd.read_csv(REPO_ROOT / args.labels)
    nifti_dir = REPO_ROOT / args.nifti_dir
    print(f"Loaded {len(labels)} labels")

    print("Reading headers for acquisition signatures ...")
    signatures = []
    for i, uid in enumerate(labels["uid"], 1):
        info = read_header_only(nifti_dir / f"{uid}.nii.gz")
        signatures.append(acquisition_signature(info["shape"], info["spacing_mm"]))
        if i % 500 == 0:
            print(f"  ...{i}/{len(labels)}", flush=True)
    labels["acquisition_group"] = signatures

    df = add_folds(labels, n_splits=args.n_splits, seed=args.seed)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Aggregate-only summary, safe to promote after you check it.
    summary = ["# Fold summary", "", f"- n = {len(df)}", f"- folds = {args.n_splits}",
               f"- seed = {args.seed}",
               f"- distinct acquisition groups = {df['acquisition_group'].nunique()}", "",
               "| fold | n (strat) | pos rate (strat) | n (group) | pos rate (group) |",
               "|---|---:|---:|---:|---:|"]
    for f in range(args.n_splits):
        s = df[df["fold_strat"] == f]
        g = df[df["fold_group"] == f]
        summary.append(
            f"| {f} | {len(s)} | {s['is_pathologic'].mean():.1%} | "
            f"{len(g)} | {g['is_pathologic'].mean():.1%} |"
        )

    raw_out = REPO_ROOT / "reports" / "eda" / "raw" / "folds_summary.md"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"\nWrote {out_path.relative_to(REPO_ROOT)} (contains uids - do not share)")
    print(f"Wrote {raw_out.relative_to(REPO_ROOT)} (aggregates - vet, then promote to shared/)")
    print("\nCheck: if grouped fold sizes are very uneven, some acquisition groups are large")
    print("       relative to 1/k. Consider fewer folds, or merging near-identical signatures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
