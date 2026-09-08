#!/usr/bin/env python3
"""Decide whether one run genuinely beats another, by a PAIRED comparison of their OOF sets.

Why paired
----------
Two runs over the same cross-validation predict the *same scans* with the *same fold models*.
Comparing their headline log losses as if they were independent throws that away and asks a
much harder question than the one being posed. What matters is not "could these two numbers
have come from the same distribution" but "does switching from A to B improve this scan's
prediction, averaged over scans" — and per-scan differences are tiny and highly correlated,
which is exactly the situation a paired test is for.

Concretely: an unpaired standard error at n=1362 is around 0.016, so a 0.002 difference looks
hopeless. Paired, the standard error is of the *difference*, which can easily be an order of
magnitude smaller. A change can be real and worth shipping at a size an unpaired test would
never resolve.

Both sides are calibrated first
-------------------------------
A run ships with a temperature fitted on its own OOF predictions, so comparing raw predictions
would penalise whichever run happens to be more overconfident — a difference that calibration
removes for free. Each side therefore gets a cross-validated temperature before scoring, by the
same procedure, so neither gains an in-sample advantage (the 2026-09-08 isotonic lesson).

What it reports
---------------
- the paired mean difference and its standard error
- a bootstrap confidence interval over scans, which assumes nothing about the shape of the
  per-scan differences (they are heavily skewed, so this matters)
- the fraction of scans improved, as a sign test that ignores magnitude entirely

Usage:
    python scripts/11_compare_runs.py --run-a baseline --run-b baseline_tta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.calibrate import TemperatureScaler  # noqa: E402
from datpd.utils.metrics import auroc, clip_probs, competition_logloss  # noqa: E402


def cv_calibrated(y: np.ndarray, p: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Held-out temperature-scaled predictions, so no side is scored on its own fit."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.full(len(y), np.nan)
    for train, test in splitter.split(p.reshape(-1, 1), y):
        out[test] = TemperatureScaler().fit(y[train], p[train]).transform(p[test])
    assert not np.isnan(out).any()
    return out


def per_sample_logloss(y: np.ndarray, p: np.ndarray, eps: float) -> np.ndarray:
    p = clip_probs(p, eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap(
    diff: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """Percentile CI for the mean paired difference, resampling scans.

    Resamples the DIFFERENCES rather than the two runs separately, which is what keeps the
    pairing intact — the whole source of the test's power.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", default="baseline", help="the incumbent")
    parser.add_argument("--run-b", default="baseline_tta", help="the challenger")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--clip-eps", type=float, default=1e-6)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.002,
        help="gain below which the change is not worth the complexity it adds",
    )
    args = parser.parse_args()

    frames = {}
    for label, run in (("A", args.run_a), ("B", args.run_b)):
        path = REPO_ROOT / args.models_dir / run / "oof.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Has run '{run}' been scored?")
        frames[label] = pd.read_csv(path)[["uid", "is_pathologic", "oof_pred"]]

    merged = frames["A"].merge(frames["B"], on="uid", suffixes=("_a", "_b"))
    if len(merged) != len(frames["A"]) or len(merged) != len(frames["B"]):
        raise SystemExit(
            f"The two runs do not cover the same scans "
            f"(A={len(frames['A'])}, B={len(frames['B'])}, shared={len(merged)}). "
            "A paired comparison needs identical uid sets."
        )
    if not (merged["is_pathologic_a"] == merged["is_pathologic_b"]).all():
        raise SystemExit("The two runs disagree about the labels; check the fold assignment.")

    y = merged["is_pathologic_a"].to_numpy()
    pa = cv_calibrated(y, merged["oof_pred_a"].to_numpy(), args.n_splits, args.seed)
    pb = cv_calibrated(y, merged["oof_pred_b"].to_numpy(), args.n_splits, args.seed)

    la = per_sample_logloss(y, pa, args.clip_eps)
    lb = per_sample_logloss(y, pb, args.clip_eps)
    diff = lb - la                                   # negative = B is better

    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    lo, hi = paired_bootstrap(diff, args.n_boot, args.seed)
    improved = float((diff < 0).mean())

    # Unpaired standard error, purely to show what the pairing bought.
    unpaired_se = float(np.sqrt(la.var(ddof=1) / len(la) + lb.var(ddof=1) / len(lb)))

    # Two runs can be bit-identical - comparing a run with itself, or a change that turned out
    # to be a no-op, such as TTA configured with only the identity view. The difference then has
    # no variance at all, and both the sigma and the pairing-gain ratio divide by zero.
    identical = se == 0.0
    sigma_text = "n/a (no variation)" if identical else f"{abs(mean_diff) / se:.1f} sigma"
    ratio_text = "n/a" if identical else f"{unpaired_se / se:.0f}x larger"

    lines = [
        f"# Paired comparison: {args.run_b} against {args.run_a}",
        "",
        f"- n = {len(merged)} scans, matched on uid",
        "- both sides temperature-scaled by 5-fold cross-validation within the OOF set, so",
        "  neither is scored on a calibration fitted to it",
        "",
        "## Scores",
        "",
        "| run | OOF log loss | AUROC |",
        "|---|---:|---:|",
        f"| {args.run_a} | {competition_logloss(y, pa, args.clip_eps):.4f} | {auroc(y, pa):.4f} |",
        f"| {args.run_b} | {competition_logloss(y, pb, args.clip_eps):.4f} | {auroc(y, pb):.4f} |",
        "",
        "## The paired difference",
        "",
        f"- mean difference: **{mean_diff:+.5f}** log loss per scan (negative favours "
        f"`{args.run_b}`)",
        f"- paired standard error: **{se:.5f}**  ->  **{sigma_text}**",
        f"- bootstrap 95% CI: **[{lo:+.5f}, {hi:+.5f}]** over {args.n_boot:,} resamples",
        f"- scans improved: **{improved:.1%}** (a sign test, ignoring magnitude)",
        "",
        f"- unpaired standard error, for contrast: {unpaired_se:.5f} ({ratio_text})",
        "",
        "That last line is why this script exists. Comparing the two headline numbers as if",
        "they were independent would call a real difference unresolvable.",
        "",
        "## Verdict",
        "",
    ]

    significant = not (lo <= 0.0 <= hi)
    if identical:
        lines += [
            f"**The two runs made identical predictions.** Nothing to choose between them; "
            "check that the change under test was actually applied.",
            "",
        ]
    elif not significant and mean_diff < 0:
        lines += [
            f"**Direction favours `{args.run_b}`, but the interval crosses zero.** The gain is",
            "not distinguishable from no change on this data. Whether to ship it comes down to",
            "cost: if the change is already built and adds no failure modes, a small unproven",
            "gain in the right direction is a defensible bet. If it adds inference complexity,",
            "it is not.",
            "",
        ]
    elif not significant:
        lines += [
            f"**No detectable difference, and the point estimate favours `{args.run_a}`.**",
            "Do not ship the change.",
            "",
        ]
    elif mean_diff < -args.margin:
        lines += [
            f"**`{args.run_b}` is better, and by more than the {args.margin} margin.** The",
            "interval excludes zero and the effect clears the threshold below which the added",
            "complexity would not be worth it. Ship it, and refit the shipped temperature on",
            f"`{args.run_b}`'s own OOF predictions.",
            "",
        ]
    elif mean_diff < 0:
        lines += [
            f"**`{args.run_b}` is better, but by less than the {args.margin} margin.** The",
            "effect is real and it is small. Ship it only if it is already built and carries no",
            "new failure modes; do not spend a day of the remaining budget on it.",
            "",
        ]
    else:
        lines += [
            f"**`{args.run_a}` is better.** The interval excludes zero in the wrong direction.",
            "Revert the change and record it as a negative result.",
            "",
        ]

    out_path = REPO_ROOT / "reports" / "experiments" / f"compare_{args.run_a}_vs_{args.run_b}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Wrote {out_path.relative_to(REPO_ROOT)} - aggregates only, safe for Claude")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
