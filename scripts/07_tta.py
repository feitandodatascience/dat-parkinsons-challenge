#!/usr/bin/env python3
"""Test-time augmentation over an existing run's fold checkpoints.

Why this is cheap
-----------------
No training. Each fold's checkpoint predicts its own validation fold several times, under small
geometric perturbations, and the logits are averaged. That is forward passes only — roughly 2%
of the cost of the run that produced the checkpoints — and it usually buys a little log loss
from variance reduction, plus a little more because averaging softens overconfident predictions.

Out-of-fold discipline
----------------------
Fold k's model predicts ONLY fold k's validation rows, exactly as during training. The result is
a genuine out-of-fold set, comparable row for row with `models/<run>/oof.csv`, so the two can be
scored against each other honestly.

Competition rules (CLAUDE.md 1.1)
---------------------------------
TTA augments a single scan and averages that scan's own predictions. Nothing is pooled across
samples, so per-sample independence holds and this is shippable. `model.eval()` throughout.

No left-right flip
------------------
The obvious TTA view is deliberately absent. DaT abnormality is frequently asymmetric and the
more-affected side carries clinical meaning (docs/01 4), so averaging a scan with its mirror
image destroys exactly the signal the task depends on. Every view here is small and
laterality-preserving.

Recalibrate afterwards
----------------------
Averaging shrinks logits toward zero, so the temperature fitted for the un-augmented run is
wrong for this one. Score with `scripts/04_evaluate.py --run <run>_tta`, which refits it.

Usage:
    python scripts/07_tta.py --config conf/baseline.yaml --run baseline
    python scripts/07_tta.py --config conf/baseline.yaml --run baseline --limit 40   # smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.config import load_config  # noqa: E402
from datpd.data.io import load_volume  # noqa: E402
from datpd.data.preprocess import preprocess  # noqa: E402
from datpd.models.factory import build_model  # noqa: E402
from datpd.utils.metrics import summarize  # noqa: E402
from datpd.utils.seed import set_seed  # noqa: E402

# One view = (rotation degrees, plane of rotation, shift in voxels, isotropic scale).
#
# Magnitudes are deliberately smaller than the training augmentation (10 deg, 5 mm, 0.9-1.1).
# Training augmentation should span more than the test distribution, to regularise; TTA should
# span less than it, because every view is a prediction that gets equal weight and a view far
# from the data manifold drags the average toward noise.
#
# The identity view is first and is what a single-view run would have produced, which makes
# `--views 1` an exact reproduction of the original OOF predictions and therefore a usable
# control.
VIEW_PRESETS: dict[str, tuple[tuple[float, tuple[int, int], tuple[int, int, int], float], ...]] = {
    "identity": (
        (0.0, (1, 2), (0, 0, 0), 1.0),
    ),
    # Rotations only. The cheapest set that still perturbs the model's spatial alignment.
    "rot4": (
        (0.0, (1, 2), (0, 0, 0), 1.0),
        (+4.0, (1, 2), (0, 0, 0), 1.0),
        (-4.0, (1, 2), (0, 0, 0), 1.0),
        (+4.0, (0, 2), (0, 0, 0), 1.0),
        (-4.0, (0, 2), (0, 0, 0), 1.0),
    ),
    # Default. Rotations in both off-lateral planes, a symmetric scale pair, and two
    # anterior-posterior shifts. Nothing displaces along axis 0 by more than a voxel, so
    # left-right structure stays where the model expects it.
    "default": (
        (0.0, (1, 2), (0, 0, 0), 1.00),
        (+4.0, (1, 2), (0, 0, 0), 1.00),
        (-4.0, (1, 2), (0, 0, 0), 1.00),
        (+4.0, (0, 2), (0, 0, 0), 1.00),
        (-4.0, (0, 2), (0, 0, 0), 1.00),
        (0.0, (1, 2), (0, 0, 0), 0.96),
        (0.0, (1, 2), (0, 0, 0), 1.04),
        (0.0, (1, 2), (0, +2, 0), 1.00),
        (0.0, (1, 2), (0, -2, 0), 1.00),
    ),
}


def make_view(
    array: np.ndarray,
    rotate_deg: float,
    axes: tuple[int, int],
    shift_vox: tuple[int, int, int],
    scale: float,
) -> np.ndarray:
    """Apply one deterministic geometric perturbation, preserving shape.

    Built as a single affine about the volume centre rather than a chain of ndimage calls, so
    the volume is interpolated exactly once. Repeated interpolation is its own kind of
    augmentation - a progressive blur - and it would confound the comparison this script exists
    to make.

    `mode="nearest"` rather than zero fill: the preprocessed volume is intensity-normalised and
    its border is background, so extending the border is closer to the truth than stamping a
    hard zero edge the model never saw in training.
    """
    if array.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {array.shape}")

    matrix = np.eye(3, dtype=np.float64) / float(scale)

    if rotate_deg:
        theta = np.deg2rad(rotate_deg)
        i, j = axes
        if i == j or not {i, j} <= {0, 1, 2}:
            raise ValueError(f"invalid rotation plane {axes}")
        rotation = np.eye(3, dtype=np.float64)
        rotation[i, i] = np.cos(theta)
        rotation[i, j] = -np.sin(theta)
        rotation[j, i] = np.sin(theta)
        rotation[j, j] = np.cos(theta)
        matrix = matrix @ rotation

    # affine_transform maps OUTPUT coordinates to INPUT ones, so the centre correction and the
    # shift both apply in output space: offset = centre - M @ centre - shift.
    centre = (np.asarray(array.shape, dtype=np.float64) - 1.0) / 2.0
    offset = centre - matrix @ centre - np.asarray(shift_vox, dtype=np.float64)

    return ndimage.affine_transform(
        array.astype(np.float32, copy=False),
        matrix,
        offset=offset,
        order=1,
        mode="nearest",
        output_shape=array.shape,
    ).astype(np.float32)


@torch.no_grad()
def predict_views(
    model: torch.nn.Module,
    array: np.ndarray,
    views: tuple[tuple[float, tuple[int, int], tuple[int, int, int], float], ...],
    device: torch.device,
) -> float:
    """Mean LOGIT across views for one scan.

    Averaging in logit space, not probability space, to match how the fold ensemble is combined
    in `submission_src/main.py`. The two differ: probability-space averaging pulls harder toward
    0.5 and would double-count the softening that calibration is about to handle properly.
    """
    batch = np.stack([make_view(array, *view) for view in views])
    tensor = torch.from_numpy(batch).float()[:, None].to(device)
    logits = model(tensor).float().squeeze(-1).cpu().numpy()
    return float(np.mean(logits))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/baseline.yaml")
    parser.add_argument("--run", default=None, help="run dir under models/ (default: config's)")
    parser.add_argument("--out-run", default=None, help="output run name (default: <run>_tta)")
    parser.add_argument("--views", default="default", choices=sorted(VIEW_PRESETS))
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only predict this many scans per fold - a smoke test, not a result",
    )
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    set_seed(cfg.seed)
    run_name = args.run or cfg.run_name
    out_name = args.out_run or f"{run_name}_tta"
    views = VIEW_PRESETS[args.views]
    folds = tuple(args.folds) if args.folds is not None else cfg.train.folds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = REPO_ROOT / cfg.output_dir / run_name
    out_dir = REPO_ROOT / cfg.output_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.data.folds_csv)
    nifti_dir = REPO_ROOT / cfg.data.nifti_dir
    fold_col = cfg.train.fold_column

    print(f"Device: {device} | views: {args.views} ({len(views)}) | fold column: {fold_col}")
    print(f"Reading checkpoints from {run_dir.relative_to(REPO_ROOT)}")
    if args.limit:
        print(f"SMOKE RUN: {args.limit} scans per fold. Do not log this as a result.")

    for fold in folds:
        ckpt_path = run_dir / f"fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"  fold {fold}: no checkpoint at {ckpt_path.name}, skipping")
            continue

        model = build_model(cfg.model).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()  # mandatory - see CLAUDE.md 1.1

        val_df = df[df[fold_col] == fold]
        if args.limit:
            val_df = val_df.head(args.limit)

        start = time.time()
        logits, failures = [], 0
        for uid in val_df["uid"]:
            try:
                volume = load_volume(nifti_dir / f"{uid}.nii.gz")
                array = preprocess(
                    volume,
                    tuple(cfg.data.target_spacing_mm),
                    tuple(cfg.data.target_shape),
                    cfg.data.normalization,
                    cfg.data.crop,
                )
                logits.append(predict_views(model, array, views, device))
            except Exception:
                # One unreadable scan should not cost the whole fold. Record NaN and report a
                # count; the row is dropped from the OOF rather than silently mispredicted.
                logits.append(np.nan)
                failures += 1

        fold_frame = val_df[["uid", "is_pathologic"]].copy()
        fold_frame["oof_pred"] = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
        fold_frame["fold"] = fold
        fold_frame = fold_frame.dropna(subset=["oof_pred"])
        fold_frame.to_csv(out_dir / f"oof_fold{fold}.csv", index=False)

        elapsed = time.time() - start
        metrics = summarize(
            fold_frame["is_pathologic"].to_numpy(), fold_frame["oof_pred"].to_numpy()
        )
        note = f"  ({failures} scans failed to load)" if failures else ""
        print(
            f"  fold {fold}: n={metrics['n']}  logloss {metrics['logloss']:.4f}  "
            f"auroc {metrics['auroc']:.4f}  {elapsed:.0f}s{note}",
            flush=True,
        )

    fold_files = sorted(out_dir.glob("oof_fold*.csv"))
    if not fold_files:
        raise SystemExit(f"No folds produced predictions in {out_dir}.")

    oof = pd.concat([pd.read_csv(p) for p in fold_files], ignore_index=True)
    duplicated = int(oof["uid"].duplicated().sum())
    if duplicated:
        raise SystemExit(
            f"{duplicated} uids predicted by more than one fold; the folds overlap and the "
            "OOF score would be optimistic. Delete oof_fold*.csv and rerun."
        )
    oof.to_csv(out_dir / "oof.csv", index=False)

    tta = summarize(oof["is_pathologic"].to_numpy(), oof["oof_pred"].to_numpy())
    print(f"\n=== TTA OOF ({args.views}, {len(views)} views) ===")
    print(f"  n                {tta['n']}")
    print(f"  logloss          {tta['logloss']:.4f}   (uncalibrated)")
    print(f"  auroc            {tta['auroc']:.4f}")
    print(f"  ece              {tta['ece']:.4f}")

    baseline_oof = run_dir / "oof.csv"
    if baseline_oof.exists() and not args.limit:
        before = pd.read_csv(baseline_oof)
        shared = before.merge(oof[["uid"]], on="uid")
        base = summarize(shared["is_pathologic"].to_numpy(), shared["oof_pred"].to_numpy())
        print(f"\n  without TTA      {base['logloss']:.4f}  auroc {base['auroc']:.4f}")
        print(f"  change           {tta['logloss'] - base['logloss']:+.4f} log loss, "
              f"{tta['auroc'] - base['auroc']:+.4f} auroc")
        print("\n  Both numbers are UNCALIBRATED. TTA softens logits, so its temperature")
        print("  differs from the original run's - compare after calibration, not here.")

    print(f"\nNext: python scripts/04_evaluate.py --run {out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
