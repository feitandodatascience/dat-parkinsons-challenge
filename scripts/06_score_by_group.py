#!/usr/bin/env python3
"""Decompose a run's OOF error by acquisition signature — without training anything.

Why this exists
---------------
`conf/grouped_cv.yaml` answers "does the score survive unseen centres?" honestly, but it costs
a full retraining run. This script answers a *narrower* question for free, from out-of-fold
predictions that already exist:

    Of the error this model makes, how much is bad ranking, and how much is bad calibration —
    and is either of them concentrated in particular acquisition families?

That distinction is the one `reports/experiments/log.md` flags as the thing to get right:
grouped log loss is penalised both by genuine model failure and by train/val prior mismatch,
and only the first is a defect worth fixing with architecture.

What it can and cannot tell you
-------------------------------
These predictions come from STRATIFIED folds, so the model saw scans from every acquisition
family during training. That is exactly the leak grouped CV is designed to expose, and nothing
here can substitute for it. What this does measure:

  * per-family discrimination (AUROC) — a real, leak-independent model property
  * per-family calibration — does one family's predictions sit systematically high or low
  * leave-one-group-out temperature transfer — does a temperature fitted on other centres
    still calibrate a centre it has never seen? This is the *calibration* half of the
    stratified-to-grouped gap, and it is the half that transfers to the leaderboard.

So: a bad LOGO-transfer number predicts trouble under grouped CV. A good one does not prove
safety, because the representation-learning half is untested.

Privacy (CLAUDE.md 0.3)
-----------------------
Output is aggregate only. No uids are read into the report, and any acquisition family smaller
than `--min-group-n` is pooled into a single "small families" row rather than reported on its
own. The report lands in reports/experiments/, which Claude may read.

Usage:
    python scripts/06_score_by_group.py --run baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.calibrate import TemperatureScaler  # noqa: E402
from datpd.utils.metrics import (  # noqa: E402
    auroc,
    baseline_logloss,
    clip_probs,
    competition_logloss,
    expected_calibration_error,
)

SMALL_LABEL = "small families (pooled)"


def _logit(p: np.ndarray) -> np.ndarray:
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _safe_auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    """AUROC is undefined when a group holds only one class."""
    if len(np.unique(y)) < 2:
        return None
    return auroc(y, p)


def _fmt(value: float | None, spec: str = ".4f") -> str:
    return "—" if value is None else format(value, spec)


def oracle_prior_shift(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Best log loss reachable by a constant logit offset, and the offset itself.

    This is an ORACLE: the offset is chosen using the group's own labels, so the score is a
    lower bound on that group's loss, not something achievable at inference. Its purpose is to
    size the question "how much of this family's error is a systematic over/under-prediction
    that a single shipped constant could have removed?"
    """
    z = _logit(p)
    grid = np.linspace(-3.0, 3.0, 601)
    losses = [competition_logloss(y, _sigmoid(z + d)) for d in grid]
    best = int(np.argmin(losses))
    return float(losses[best]), float(grid[best])


def logo_temperature(df: pd.DataFrame, group_col: str) -> tuple[np.ndarray, dict[str, float]]:
    """Leave-one-group-out temperature scaling.

    For each acquisition family, fit the temperature on every OTHER family's OOF predictions
    and apply it to this one. The pooled result estimates how well a calibration constant
    fitted at training time transfers to a centre the calibration never saw — which is exactly
    what the shipped scalar temperature has to do on the leaderboard.
    """
    y = df["is_pathologic"].to_numpy()
    p = df["oof_pred"].to_numpy()
    out = np.full(len(df), np.nan)
    per_group_t: dict[str, float] = {}

    for group in df[group_col].unique():
        held = (df[group_col] == group).to_numpy()
        if held.all():
            continue
        scaler = TemperatureScaler().fit(y[~held], p[~held])
        out[held] = scaler.transform(p[held])
        per_group_t[str(group)] = scaler.temperature

    return out, per_group_t


def build_report(df: pd.DataFrame, run_name: str, min_group_n: int) -> list[str]:
    y_all = df["is_pathologic"].to_numpy()
    p_all = df["oof_pred"].to_numpy()

    # Pool families below the reporting threshold so no small group is described on its own.
    counts = df["acquisition_group"].value_counts()
    reportable = set(counts[counts >= min_group_n].index)
    df = df.copy()
    df["report_group"] = np.where(
        df["acquisition_group"].isin(reportable), df["acquisition_group"], SMALL_LABEL
    )

    global_t = TemperatureScaler().fit(y_all, p_all)
    p_global = global_t.transform(p_all)
    p_logo, per_group_t = logo_temperature(df, "acquisition_group")

    lines = [
        f"# Error decomposition by acquisition family: {run_name}",
        "",
        "Computed from existing out-of-fold predictions — no training was run.",
        "These folds are STRATIFIED, so the model saw every family during training. Read the",
        "AUROC column as a real model property and the calibration columns as a lower bound on",
        "what grouped CV will show.",
        "",
        f"- n = {len(df)}, positive rate = {y_all.mean():.3f}",
        f"- distinct acquisition families = {df['acquisition_group'].nunique()}"
        f" ({len(reportable)} at or above the n >= {min_group_n} reporting threshold)",
        f"- global temperature fitted on all OOF = {global_t.temperature:.4f}",
        "",
        "## Per-family breakdown",
        "",
        "`prior floor` is the log loss of predicting that family's own base rate — the score to",
        "beat *within* the family. `oracle shift` is the best log loss reachable by adding one",
        "constant to the logits, chosen with the family's own labels; the gap between it and",
        "`logloss` is the part of the family's error that is a systematic offset rather than bad",
        "ranking. It is an oracle and therefore a bound, not an achievable score.",
        "",
        "| family | n | pos rate | mean pred | logloss | prior floor | oracle shift | offset | AUROC | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    order = (
        df.groupby("report_group").size().sort_values(ascending=False).index.tolist()
    )
    for group in order:
        sub = df[df["report_group"] == group]
        y = sub["is_pathologic"].to_numpy()
        p = sub["oof_pred"].to_numpy()
        shifted, offset = oracle_prior_shift(y, p)
        lines.append(
            f"| {group} | {len(sub)} | {y.mean():.1%} | {p.mean():.3f} | "
            f"{competition_logloss(y, p):.4f} | {baseline_logloss(y):.4f} | "
            f"{shifted:.4f} | {offset:+.2f} | {_fmt(_safe_auroc(y, p))} | "
            f"{expected_calibration_error(y, p):.4f} |"
        )

    # --- The headline number -------------------------------------------------------------
    ll_raw = competition_logloss(y_all, p_all)
    ll_global = competition_logloss(y_all, p_global)
    valid = ~np.isnan(p_logo)
    ll_logo = competition_logloss(y_all[valid], p_logo[valid])

    lines += [
        "",
        "## Does a temperature transfer to an unseen centre?",
        "",
        "Each family's predictions are calibrated with a temperature fitted on every *other*",
        "family, then pooled. The shipped submission carries one scalar temperature that must",
        "work on centres it never saw, so this is the closest free estimate of whether that",
        "assumption holds.",
        "",
        "| calibration | OOF log loss |",
        "|---|---:|",
        f"| none | {ll_raw:.4f} |",
        f"| global temperature (fitted on all OOF, in-sample) | {ll_global:.4f} |",
        f"| leave-one-family-out temperature | {ll_logo:.4f} |",
        "",
        f"- transfer penalty = **{ll_logo - ll_global:+.4f}**",
        "",
        "Read it like this: a penalty near zero means the calibration constant is a property of",
        "the model, not of the centres it was fitted on, and the shipped temperature is safe. A",
        "large positive penalty means the temperature is partly fitting centre composition, and",
        "the leaderboard will not reproduce the OOF calibration gain.",
        "",
        "### Temperature fitted without each family",
        "",
        "Spread here is the diagnostic: a family whose exclusion moves the fitted temperature a",
        "long way is a family that dominates the global calibration.",
        "",
        "| family | n | T fitted excluding it |",
        "|---|---:|---:|",
    ]

    shown = [g for g in order if g != SMALL_LABEL]
    for group in shown:
        sub = df[df["report_group"] == group]
        temps = [per_group_t[str(g)] for g in sub["acquisition_group"].unique()
                 if str(g) in per_group_t]
        if temps:
            lines.append(f"| {group} | {len(sub)} | {np.mean(temps):.4f} |")

    spread = [per_group_t[str(g)] for g in counts[counts >= min_group_n].index
              if str(g) in per_group_t]
    if spread:
        lines += [
            "",
            f"- range across reportable families: {min(spread):.4f} to {max(spread):.4f}"
            f" (global {global_t.temperature:.4f})",
        ]

    # --- How much could a prior correction ever buy? ---------------------------------------
    total_oracle = 0.0
    for group in df["acquisition_group"].unique():
        sub = df[df["acquisition_group"] == group]
        shifted, _ = oracle_prior_shift(
            sub["is_pathologic"].to_numpy(), sub["oof_pred"].to_numpy()
        )
        total_oracle += shifted * len(sub)
    ll_oracle = total_oracle / len(df)

    lines += [
        "",
        "## Ceiling on per-centre prior correction",
        "",
        "Give every family its own oracle logit offset, chosen with that family's own labels:",
        "",
        f"- pooled log loss with per-family oracle offsets: **{ll_oracle:.4f}**",
        f"- against uncalibrated {ll_raw:.4f} — headroom **{ll_raw - ll_oracle:.4f}**",
        "",
        "This is the absolute most that any per-centre prior correction could be worth, using",
        "information no inference-time model can have. If the headroom is small, centre prior",
        "shift is a second-order problem and the open question from row 0b can be closed. If it",
        "is large, a shipped logit offset is worth designing — but it would have to be predicted",
        "from the scan header alone, per sample, to stay inside the independence rule.",
        "",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--folds-csv", default="data/interim/folds.csv")
    parser.add_argument(
        "--min-group-n",
        type=int,
        default=30,
        help="acquisition families smaller than this are pooled into one row (privacy)",
    )
    args = parser.parse_args()

    run_dir = REPO_ROOT / args.models_dir / args.run
    oof_path = run_dir / "oof.csv"
    if not oof_path.exists():
        raise SystemExit(
            f"No {oof_path}. Run scripts/04_evaluate.py --run {args.run} first — it assembles "
            "oof.csv from completed folds."
        )

    oof = pd.read_csv(oof_path)
    folds = pd.read_csv(REPO_ROOT / args.folds_csv)
    df = oof.merge(
        folds[["uid", "acquisition_group", "fold_strat", "fold_group"]], on="uid", how="left"
    )

    missing = int(df["acquisition_group"].isna().sum())
    if missing:
        raise SystemExit(
            f"{missing} OOF rows have no acquisition_group in {args.folds_csv}. "
            "Regenerate folds before scoring by group."
        )

    lines = build_report(df, args.run, args.min_group_n)

    out_dir = REPO_ROOT / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run}_by_group.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    print("Aggregates only, no uids — safe for Claude to read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
