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
        "```bash",
        "python scripts/05_package_submission.py \\",
        "    --config conf/<the config these weights were trained with> \\",
        f"    --temperature {shippable.temperature:.4f} \\",
        f"    --fallback {base['positive_rate']:.4f} \\",
        f"    --clip-eps {ship_clip:g}",
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
