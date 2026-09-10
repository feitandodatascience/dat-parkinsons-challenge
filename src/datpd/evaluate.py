"""Out-of-fold evaluation and calibration selection.

Answers the only question that should drive decisions: **did this experiment reduce OOF log
loss, and how much of any gain is calibration rather than discrimination?**

Prints an aggregate report and writes it to `reports/experiments/<run>.md`, which is safe for
Claude to read (metrics only, no uids).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import TemperatureScaler, choose_calibrator
from .utils.metrics import competition_logloss, summarize

# Candidate clips to evaluate. Tighter clipping caps the worst-case cost of a confident error
# at the price of a small tax on every correct confident prediction.
CLIP_GRID = (1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05)


def _schedule_section(run_dir: Path) -> list[str]:
    """Summarise where each fold's training actually stopped, as report lines.

    Aggregate over a fold's whole validation set, so this belongs in `reports/experiments/`
    (CLAUDE.md §0.1) - epochs and learning rates describe the optimiser, not any scan.

    It exists because the question "did early stopping cut the run short while there was still
    something to gain?" was, for run 1, unanswerable: the curve lived only in a terminal that
    had since been closed. `lr at best / peak` is the number that answers it. With
    `T_max = cfg.train.epochs`, a fold that stops early selects its weights partway down the
    cosine curve; a ratio near 1 means the anneal barely started, and the low-rate endgame the
    schedule was written to provide never happened.

    Read from `history_fold*.json`, NOT from `metrics.json`'s schedule block. `run_cv` writes
    metrics.json only after the whole fold loop, so an interrupted run - the case the per-epoch
    banking exists for - has every curve on disk and no metrics.json to describe them. Reading
    the history files directly means this table appears for a partial run too.

    `history[0]["lr"]` is the configured peak: `train_fold` records the rate before stepping the
    scheduler, so epoch 0 carries `cfg.train.lr` exactly. That makes the ratio computable
    without the config, which a partial run may not have written.
    """
    histories: list[tuple[int, list[dict[str, float]]]] = []
    for path in sorted(run_dir.glob("history_fold*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if rows:
            histories.append((int(path.stem.removeprefix("history_fold")), rows))
    if not histories:
        return []

    # Only needed to say whether a fold stopped short of its schedule; absent on a partial run.
    epochs_configured: int | None = None
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        cfg_block = json.loads(metrics_path.read_text(encoding="utf-8")).get("config") or {}
        configured = (cfg_block.get("train") or {}).get("epochs")
        if isinstance(configured, int):
            epochs_configured = configured

    lines = [
        "## Training schedule",
        "",
        "| fold | best epoch | last epoch | of | stopped early | lr at best / peak |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    stopped: list[tuple[int, float]] = []
    for fold, rows in histories:
        best = min(rows, key=lambda h: h["val_logloss"])
        ratio = best["lr"] / rows[0]["lr"] if rows[0]["lr"] else 0.0
        last_epoch = rows[-1]["epoch"]
        if epochs_configured is None:
            of, early = "?", "?"
        else:
            of = str(epochs_configured)
            early = "yes" if last_epoch + 1 < epochs_configured else "no"
            if early == "yes":
                stopped.append((fold, ratio))
        lines.append(
            f"| {fold} | {best['epoch']} | {last_epoch} | {of} | {early} | {ratio:.2f} |"
        )

    if len(histories) > 1:
        lines += ["", f"- {len(histories)} folds trained."]
    else:
        (fold, _), = histories
        lines += [
            "",
            f"- **Only fold {fold} trained.** Every number in this report is that fold's "
            "validation set alone, not a five-fold OOF, and is not comparable to a full run.",
        ]

    if stopped:
        worst_fold, worst_ratio = max(stopped, key=lambda fr: fr[1])
        lines += [
            f"- **{len(stopped)}/{len(histories)} folds stopped early.** The worst (fold "
            f"{worst_fold}) selected its weights at **{worst_ratio:.0%} of the peak learning "
            "rate**, so the cosine anneal was still far from finished.",
            "- Raising `epochs` does NOT fix this: `T_max` is tied to it, so a longer schedule "
            "decays more slowly and lands the same epoch at a *higher* rate. The levers are a "
            "**shorter** schedule or a larger `early_stopping_patience`.",
        ]
    return lines + [""]


def evaluate_run(run_dir: Path, reports_dir: Path) -> dict[str, object]:
    """Score one training run's OOF predictions and pick a calibrator."""
    oof = pd.read_csv(run_dir / "oof.csv")
    y = oof["is_pathologic"].to_numpy()
    p = oof["oof_pred"].to_numpy()

    base = summarize(y, p)
    cal_name, calibrator, cal_scores = choose_calibrator(y, p)
    p_cal = calibrator.transform(p)
    calibrated = summarize(y, p_cal)

    clip_scores = {eps: competition_logloss(y, p_cal, eps) for eps in CLIP_GRID}
    best_clip = min(clip_scores, key=clip_scores.get)  # type: ignore[arg-type]

    temperature = getattr(calibrator, "temperature", None)

    lines = [
        f"# Evaluation: {run_dir.name}",
        "",
        f"- n = {base['n']}, positive rate = {base['positive_rate']:.3f}",
        f"- **prior baseline log loss = {base['baseline_logloss']:.4f}** (the floor to beat)",
        "",
    ]
    lines += _schedule_section(run_dir)
    lines += [
        "## Uncalibrated",
        f"- log loss **{base['logloss']:.4f}**",
        f"- AUROC {base['auroc']:.4f}",
        f"- ECE {base['ece']:.4f}",
        "",
        "## Calibration",
        "",
        "| method | OOF log loss |",
        "|---|---:|",
    ]
    for name, score in sorted(cal_scores.items(), key=lambda kv: kv[1]):
        mark = " **<- chosen**" if name == cal_name else ""
        lines.append(f"| {name} | {score:.4f}{mark} |")

    lines += [
        "",
        f"- chosen: **{cal_name}**"
        + (f" (temperature = {temperature:.4f})" if temperature is not None else ""),
        f"- ECE after calibration: {calibrated['ece']:.4f} (was {base['ece']:.4f})",
        f"- AUROC after calibration: {calibrated['auroc']:.4f} "
        "(calibration is monotonic, so this should be unchanged)",
        "",
        "## Prediction clipping",
        "",
        "| clip eps | OOF log loss |",
        "|---|---:|",
    ]
    for eps, score in clip_scores.items():
        mark = " **<- best**" if eps == best_clip else ""
        lines.append(f"| {eps:g} | {score:.4f}{mark} |")

    gain = base["logloss"] - clip_scores[best_clip]
    lines += [
        "",
        "## Summary",
        "",
        f"- final OOF log loss: **{clip_scores[best_clip]:.4f}**",
        f"- improvement from calibration + clipping: **{gain:.4f}**",
        f"- improvement over prior baseline: **{base['baseline_logloss'] - clip_scores[best_clip]:.4f}**",
        "",
    ]

    # `main.py` applies a single scalar temperature in logit space. That is the only calibration
    # the submission can carry, so the shipping command is built from the temperature scaler
    # whichever method won the comparison. Emitting `--temperature 1.0` because the winner had
    # no `.temperature` attribute would ship an UNCALIBRATED model while the report above claims
    # otherwise — a silent loss of the whole calibration gain.
    shippable = TemperatureScaler().fit(y, p)
    p_ship = shippable.transform(p)
    ship_clip = min(CLIP_GRID, key=lambda eps: competition_logloss(y, p_ship, eps))
    ship_logloss = competition_logloss(y, p_ship, ship_clip)

    lines += ["## To ship this", ""]
    if cal_name != "temperature":
        lines += [
            f"> The comparison above chose **{cal_name}**, but `submission_src/main.py` can only",
            "> carry a scalar temperature, so the command below ships **temperature scaling**",
            f"> (OOF log loss {ship_logloss:.4f}) rather than {cal_name}.",
            "",
        ]
        if cal_name == "isotonic":
            lines += [
                "> Shipping isotonic would mean embedding its full step function in the zip.",
                "> Worth doing only if its cross-validated margin over temperature is large and",
                "> stable across seeds; at this sample size it usually is not.",
                "",
            ]

    lines += [
        # One line, no continuations. This project is developed on Windows and the command is
        # pasted into PowerShell, where `\` is not a line continuation - it is a path separator,
        # and the shell fails with "Missing expression after unary operator '--'". A wrapped
        # command that cannot be pasted is worse than a long one that can.
        "```",
        "python scripts/05_package_submission.py"
        " --config conf/<the config these weights were trained with>"
        f" --temperature {shippable.temperature:.4f}"
        f" --fallback {base['positive_rate']:.4f}"
        f" --clip-eps {ship_clip:g}",
        "```",
        "",
        "Copy the fold checkpoints into `submission_src/assets/` first.",
        "",
    ]

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{run_dir.name}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {report_path}")

    return {
        "uncalibrated": base,
        "calibrated": calibrated,
        "calibration_method": cal_name,
        "temperature": temperature,
        "best_clip": best_clip,
        "final_logloss": clip_scores[best_clip],
    }
