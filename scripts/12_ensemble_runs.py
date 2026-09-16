"""Average two or more runs' OOF predictions in logit space, and bank the result as a run.

    python scripts/12_ensemble_runs.py --runs baseline baseline_anneal

Why this exists, and why it is cheap. Ensembling separately-trained models is the last
CPU-feasible lever in this project: `submission_src/main.py` globs `assets/fold*.pt` and averages
whatever it finds, so shipping N runs' checkpoints together needs no inference-code change at all.
The question is only whether it pays.

The point of doing it at the OOF level first is that **the training is already sunk**. Both runs'
`oof.csv` are on disk, so this answers "would ensembling help?" in seconds, before committing an
hour of CPU to a TTA pass or a night to another training run.

The number to read is the **logit correlation**. `reports/experiments/baseline_tta_blend.md`
established the governing fact: the SBR tree model is much worse than the CNN alone (0.3919
against 0.3186) and still improved the ensemble by -0.0175, because its logits correlate only
0.797. Diversity paid, not quality. Two runs of the same architecture on the same folds will
correlate far higher, and the variance-reduction benefit scales roughly with (1 - rho) - so a
correlation of 0.97 predicts about a seventh of what the tree model bought.

Averaging is in LOGIT space, matching how folds and TTA views are already combined, so one
saturated member cannot drag the ensemble the way probability averaging would.

Outputs are aggregates only and safe for Claude to read: a report under `reports/experiments/`
and a run directory holding `oof.csv` so `scripts/11_compare_runs.py` can put the result through
the same paired per-scan test as every other change. **The pooled numbers printed here are
uncalibrated and are not the verdict** - run 04_evaluate and 11_compare_runs for that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.utils.metrics import competition_logloss, summarize  # noqa: E402

CLIP = 1e-6


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1.0 - CLIP)
    return np.log(p / (1.0 - p))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="run dirs under models/")
    parser.add_argument("--out-run", default=None, help="default: <run1>_<run2>_ens")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="optional per-run weights, normalised to sum to 1. Default: equal. Fitting these "
        "on the OOF they are scored against would be the in-sample calibration mistake again "
        "(see reports/experiments/log.md), so they are not fitted here.",
    )
    args = parser.parse_args()

    if len(args.runs) < 2:
        raise SystemExit("need at least two runs to ensemble")
    if args.weights and len(args.weights) != len(args.runs):
        raise SystemExit(f"{len(args.weights)} weights for {len(args.runs)} runs")

    models_dir = REPO_ROOT / args.models_dir
    frames: dict[str, pd.DataFrame] = {}
    for run in args.runs:
        path = models_dir / run / "oof.csv"
        if not path.exists():
            raise SystemExit(f"no oof.csv in {models_dir / run} - train or TTA that run first")
        frames[run] = pd.read_csv(path).set_index("uid").sort_index()

    # Every run must cover the same scans, or the ensemble is scored on a different sample than
    # its members and no comparison downstream means anything.
    reference = frames[args.runs[0]]
    for run, frame in frames.items():
        if not frame.index.equals(reference.index):
            raise SystemExit(
                f"{run} covers {len(frame)} scans against {len(reference)} in "
                f"{args.runs[0]}; ensembling partial runs would compare different samples"
            )
        if not np.array_equal(frame["is_pathologic"].to_numpy(), reference["is_pathologic"].to_numpy()):
            raise SystemExit(f"{run} disagrees with {args.runs[0]} on the labels")
        # Folds must match too: a scan validated by different folds in different runs has
        # predictions that are not comparable, and pooling them mixes the estimators.
        if "fold" in frame and "fold" in reference:
            if not np.array_equal(frame["fold"].to_numpy(), reference["fold"].to_numpy()):
                raise SystemExit(f"{run} used a different fold assignment than {args.runs[0]}")

    y = reference["is_pathologic"].to_numpy()
    logits = {run: to_logit(frame["oof_pred"].to_numpy()) for run, frame in frames.items()}

    weights = np.asarray(args.weights, dtype=np.float64) if args.weights else np.ones(len(args.runs))
    weights = weights / weights.sum()
    blended = sum(w * logits[run] for w, run in zip(weights, args.runs))
    ensemble_pred = 1.0 / (1.0 + np.exp(-blended))

    out_run = args.out_run or "_".join(args.runs) + "_ens"
    out_dir = models_dir / out_run
    out_dir.mkdir(parents=True, exist_ok=True)
    banked = pd.DataFrame(
        {
            "uid": reference.index,
            "is_pathologic": y,
            "oof_pred": ensemble_pred,
            "fold": reference["fold"].to_numpy() if "fold" in reference else 0,
        }
    )
    banked.to_csv(out_dir / "oof.csv", index=False)

    lines = [
        f"# Ensemble: {' + '.join(args.runs)}",
        "",
        f"- n = {len(y)} scans, matched on uid",
        f"- averaged in **logit** space, weights {np.round(weights, 4).tolist()}",
        "- **uncalibrated.** Averaging logits softens them, so the temperature differs from any",
        "  member's. Run `04_evaluate.py` for the calibrated number and `11_compare_runs.py`",
        "  for the paired verdict. Nothing below is a decision.",
        "",
        "## Members",
        "",
        "| run | OOF log loss | AUROC |",
        "|---|---:|---:|",
    ]
    for run in args.runs:
        m = summarize(y, frames[run]["oof_pred"].to_numpy())
        lines.append(f"| {run} | {m['logloss']:.4f} | {m['auroc']:.4f} |")

    ens = summarize(y, ensemble_pred)
    lines += [
        f"| **{out_run}** | **{ens['logloss']:.4f}** | **{ens['auroc']:.4f}** |",
        "",
        "## Logit correlation",
        "",
        "The number that predicts whether this pays. The SBR tree model bought -0.0175 at 0.797;",
        "the benefit scales roughly with (1 - rho), so 0.97 predicts about a seventh of that.",
        "",
        "| pair | Pearson r |",
        "|---|---:|",
    ]
    for i, a in enumerate(args.runs):
        for b in args.runs[i + 1 :]:
            r = float(np.corrcoef(logits[a], logits[b])[0, 1])
            lines.append(f"| {a} vs {b} | **{r:.4f}** |")

    best_member = min(args.runs, key=lambda r: competition_logloss(y, frames[r]["oof_pred"].to_numpy()))
    delta = competition_logloss(y, ensemble_pred) - competition_logloss(
        y, frames[best_member]["oof_pred"].to_numpy()
    )
    lines += [
        "",
        "## Uncalibrated change against the best member",
        "",
        f"- best member: `{best_member}`",
        f"- change: **{delta:+.5f}** log loss (negative favours the ensemble)",
        "",
        "Read this against the +-0.01 CV<->leaderboard spread in `log.md`, not against zero.",
        "A change inside that band is unresolved, not won.",
        "",
        "## Next",
        "",
        "```",
        f"python scripts/04_evaluate.py --run {out_run}",
        f"python scripts/11_compare_runs.py --run-a {best_member} --run-b {out_run}",
        "```",
        "",
    ]

    report = REPO_ROOT / "reports" / "experiments" / f"{out_run}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Wrote {report.relative_to(REPO_ROOT)} - aggregates only, safe for Claude")
    print(f"Banked {(out_dir / 'oof.csv').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
