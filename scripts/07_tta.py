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

Two axes of variation
---------------------
`identity`, `rot4`, `default` and `wide17` perturb geometry only. `photo5` and `photo13` add a
pointwise intensity perturbation - a gain and a signed gamma - which no geometric view can
reach: resampling a volume cannot change the fact that every view shares one intensity
realisation, so averaging them reduces no intensity-related variance. See the comment above
`photo5` for why a naive gamma is a domain error on this signed array.

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

# One view = (rotation degrees, plane of rotation, shift in voxels, isotropic scale) and,
# optionally, two more elements: (intensity gain, signed gamma). Both default to 1.0 = no
# change, so a four-element view still means exactly what it meant before this existed and
# every preset measured under the old signature reads the same today.
#
# Magnitudes are deliberately smaller than the training augmentation (10 deg, 5 mm, 0.9-1.1).
# Training augmentation should span more than the test distribution, to regularise; TTA should
# span less than it, because every view is a prediction that gets equal weight and a view far
# from the data manifold drags the average toward noise.
#
# The identity view is first and is what a single-view run would have produced, which makes
# `--views 1` an exact reproduction of the original OOF predictions and therefore a usable
# control.
GeometricView = tuple[float, tuple[int, int], tuple[int, int, int], float]
PhotometricView = tuple[float, tuple[int, int], tuple[int, int, int], float, float, float]
View = GeometricView | PhotometricView

VIEW_PRESETS: dict[str, tuple[View, ...]] = {
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
    # Wider and denser geometry, 17 views. A strict SUPERSET of `default`: the first nine
    # entries are `default` verbatim, so whatever this scores differs from the shipped number
    # only because of the eight added views, and the comparison needs no other control.
    #
    # What it adds over `default`, and why each:
    #   - a second, smaller rotation magnitude (+-2 deg). `default` samples one radius only, so
    #     its rotation average is a ring; adding an inner ring makes the average an area.
    #   - rotation in the third plane (0, 1), the only one `default` never touches. A rotation
    #     is a rigid motion, never a reflection, so rotating in a plane that contains the
    #     left-right axis does not mirror anything - it tilts, exactly as a differently
    #     positioned head would.
    #   - shifts along axis 2 as well as axis 1, so translation is sampled in a plane rather
    #     than on a line. Axis 0 (left-right) is still never shifted by more than a voxel; it
    #     is not shifted at all here.
    #   - scale composed WITH rotation rather than only on its own. Scale and rotation are the
    #     two transforms the model is least invariant to, and a set that only ever varies them
    #     one at a time samples the axes but not the corners.
    "wide17": (
        (0.0, (1, 2), (0, 0, 0), 1.00),
        (+4.0, (1, 2), (0, 0, 0), 1.00),
        (-4.0, (1, 2), (0, 0, 0), 1.00),
        (+4.0, (0, 2), (0, 0, 0), 1.00),
        (-4.0, (0, 2), (0, 0, 0), 1.00),
        (0.0, (1, 2), (0, 0, 0), 0.96),
        (0.0, (1, 2), (0, 0, 0), 1.04),
        (0.0, (1, 2), (0, +2, 0), 1.00),
        (0.0, (1, 2), (0, -2, 0), 1.00),
        (+2.0, (1, 2), (0, 0, 0), 1.00),
        (-2.0, (1, 2), (0, 0, 0), 1.00),
        (+3.0, (0, 1), (0, 0, 0), 1.00),
        (-3.0, (0, 1), (0, 0, 0), 1.00),
        (0.0, (1, 2), (0, 0, +2), 1.00),
        (0.0, (1, 2), (0, 0, -2), 1.00),
        (+4.0, (1, 2), (0, 0, 0), 0.97),
        (-4.0, (1, 2), (0, 0, 0), 1.03),
    ),
    # The intensity axis alone, 5 views: identity plus two gain and two gamma perturbations,
    # all on identity geometry. Every view in every other preset resamples geometry and nothing
    # else, which means they all share ONE intensity realisation - averaging them cannot reduce
    # any variance the model's sensitivity to intensity contributes. This is the cheapest
    # possible test of whether that variance is worth attacking: four of the five views touch
    # no interpolation at all, so it costs five forward passes and one preprocess.
    #
    # Why these two curves, and why a naive gamma would be wrong
    # ---------------------------------------------------------
    # The preprocessed array is REFERENCE-REGION normalised and therefore SIGNED: background
    # sits below zero, striatum above it, and zero means "same uptake as the reference region".
    # `x ** g` on that array is not a mild perturbation, it is a domain error - a negative base
    # with a fractional exponent has no real value, so the background would come back NaN and
    # the view would poison the whole average. Any intensity perturbation here has to be
    # defined on a signed quantity.
    #
    #   gain  g : x -> g * x. Scales the signed deviation about zero, i.e. it rescales the
    #             specific-binding ratio while leaving the reference level exactly where it is.
    #             That is the physically real perturbation - camera sensitivity, reconstruction
    #             and scatter correction all move this number between sites - and it is exactly
    #             what the multicentre data varies (CLAUDE.md 4).
    #   gamma g : x -> sign(x) * |x| ** g. The signed analogue of a gamma curve: monotone,
    #             sign-preserving, finite for every real input, with fixed points at 0 and at
    #             |x| = 1. It reshapes the striatum-to-background contrast rather than scaling
    #             it, which is the part a gain cannot reach.
    #
    # Both are applied pointwise AFTER the geometry, so they add no second interpolation, and
    # both come in pairs straddling 1.0 so the average stays centred on the identity.
    "photo5": (
        (0.0, (1, 2), (0, 0, 0), 1.00),
        (0.0, (1, 2), (0, 0, 0), 1.00, 0.95, 1.00),
        (0.0, (1, 2), (0, 0, 0), 1.00, 1.05, 1.00),
        (0.0, (1, 2), (0, 0, 0), 1.00, 1.00, 0.96),
        (0.0, (1, 2), (0, 0, 0), 1.00, 1.00, 1.04),
    ),
    # Geometry and intensity together, 13 views: `default` verbatim plus the four intensity
    # views of `photo5`. Again a strict superset of the shipped set, so the delta is
    # attributable. Run this one if `photo5` shows the intensity axis moves anything at all.
    "photo13": (
        (0.0, (1, 2), (0, 0, 0), 1.00),
        (+4.0, (1, 2), (0, 0, 0), 1.00),
        (-4.0, (1, 2), (0, 0, 0), 1.00),
        (+4.0, (0, 2), (0, 0, 0), 1.00),
        (-4.0, (0, 2), (0, 0, 0), 1.00),
        (0.0, (1, 2), (0, 0, 0), 0.96),
        (0.0, (1, 2), (0, 0, 0), 1.04),
        (0.0, (1, 2), (0, +2, 0), 1.00),
        (0.0, (1, 2), (0, -2, 0), 1.00),
        (0.0, (1, 2), (0, 0, 0), 1.00, 0.95, 1.00),
        (0.0, (1, 2), (0, 0, 0), 1.00, 1.05, 1.00),
        (0.0, (1, 2), (0, 0, 0), 1.00, 1.00, 0.96),
        (0.0, (1, 2), (0, 0, 0), 1.00, 1.00, 1.04),
    ),
}


def make_view(
    array: np.ndarray,
    rotate_deg: float,
    axes: tuple[int, int],
    shift_vox: tuple[int, int, int],
    scale: float,
    gain: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Apply one deterministic perturbation - geometry, then intensity - preserving shape.

    The geometry is built as a single affine about the volume centre rather than a chain of
    ndimage calls, so the volume is interpolated exactly once. Repeated interpolation is its own
    kind of augmentation - a progressive blur - and it would confound the comparison this script
    exists to make.

    `mode="nearest"` rather than zero fill: the preprocessed volume is intensity-normalised and
    its border is background, so extending the border is closer to the truth than stamping a
    hard zero edge the model never saw in training.

    `gain` and `gamma` are POINTWISE and are applied after the resampling, so they add no second
    interpolation and the single-affine property above is untouched. They are defined for a
    SIGNED array, because the reference-region normalised volume is one - background below zero,
    striatum above it. `x ** gamma` would be a domain error on the background; the signed form
    `sign(x) * |x| ** gamma` is monotone, sign-preserving and finite everywhere, with fixed
    points at 0 and |x| = 1. `gain` simply rescales the signed deviation from the reference
    level. Order is gamma then gain, and it matters only in that the shipped copy of this
    function must apply them the same way round.
    """
    if array.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {array.shape}")

    geometric = not (rotate_deg == 0.0 and scale == 1.0 and tuple(shift_vox) == (0, 0, 0))
    photometric = not (gain == 1.0 and gamma == 1.0)

    # Short-circuit the identity. Resampling a volume onto its own grid is a no-op in exact
    # arithmetic but not in floating point, and this path has to be *exactly* a no-op: it is
    # what `submission_src/main.py` runs when TTA is disabled, so turning TTA off must
    # reproduce the un-augmented predictions bit for bit rather than approximately.
    if not geometric and not photometric:
        return array.astype(np.float32, copy=False)

    out = array.astype(np.float32, copy=False)

    if geometric:
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

        # affine_transform maps OUTPUT coordinates to INPUT ones, so the centre correction and
        # the shift both apply in output space: offset = centre - M @ centre - shift.
        centre = (np.asarray(array.shape, dtype=np.float64) - 1.0) / 2.0
        offset = centre - matrix @ centre - np.asarray(shift_vox, dtype=np.float64)

        out = ndimage.affine_transform(
            out,
            matrix,
            offset=offset,
            order=1,
            mode="nearest",
            output_shape=array.shape,
        ).astype(np.float32)

    # np.float32 on the exponent and the multiplier rather than a bare Python float, so the
    # arithmetic cannot be promoted to float64 on one numpy version and not another. The
    # shipped copy has to reproduce this bit for bit.
    if gamma != 1.0:
        out = np.sign(out) * np.abs(out) ** np.float32(gamma)
    if gain != 1.0:
        out = out * np.float32(gain)

    return out.astype(np.float32, copy=False)


@torch.no_grad()
def predict_views(
    model: torch.nn.Module,
    array: np.ndarray,
    views: tuple[View, ...],
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
    views = VIEW_PRESETS[args.views]
    folds = tuple(args.folds) if args.folds is not None else cfg.train.folds

    # A smoke run writes somewhere else entirely. Sharing a directory with the real run is a
    # trap: per-fold predictions are banked as they finish, so a full run that dies partway
    # leaves the folds it never reached holding 40-row smoke files, and the pooled OOF silently
    # mixes the two. Separate directories make that impossible rather than merely unlikely.
    #
    # The preset goes in the directory name for the same reason, and it is the same trap one
    # step along: two presets both defaulting to `<run>_tta` would overwrite each other fold by
    # fold, and the completeness check below cannot see it - a `wide17` fold file and a
    # `photo13` fold file have exactly the same number of rows. A pass interrupted halfway
    # would leave an OOF built from two different view sets, scoring somewhere between them.
    # `default` keeps the bare name so the numbers already recorded under it still line up.
    default_out = f"{run_name}_tta"
    if args.views != "default":
        default_out += f"_{args.views}"
    if args.limit:
        default_out += "_smoke"
    out_name = args.out_run or default_out

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
        print(f"SMOKE RUN: {args.limit} scans per fold, sampled. Not a result.")

    # The original run's predictions, for a like-for-like comparison on the SAME rows. Without
    # this, a fold's TTA score can only be read against the other folds — and at smoke-run
    # sample sizes that comparison is dominated by which scans happened to be drawn, not by
    # anything TTA did. Paired against the same fold's own baseline, the sampling noise mostly
    # cancels, because both numbers are computed over the identical rows.
    baseline_oof = run_dir / "oof.csv"
    before = pd.read_csv(baseline_oof) if baseline_oof.exists() else None
    if before is None:
        print(f"NOTE: no {baseline_oof.name} to compare against; showing TTA scores alone.")

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
            # Sampled, not `.head()`. folds.csv inherits the label file's row order, so the
            # first N rows of a fold are a systematic slice of it — if that order carries any
            # structure at all, the smoke run's per-fold scores differ for reasons that have
            # nothing to do with the model, and the folds stop being comparable to each other.
            val_df = val_df.sample(
                n=min(args.limit, len(val_df)), random_state=cfg.seed + fold
            )

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

        if before is not None:
            paired = before.merge(fold_frame[["uid"]], on="uid")
            if len(paired) == len(fold_frame):
                was = summarize(
                    paired["is_pathologic"].to_numpy(), paired["oof_pred"].to_numpy()
                )
                print(
                    f"          without TTA on the same rows: "
                    f"logloss {was['logloss']:.4f} ({metrics['logloss'] - was['logloss']:+.4f})"
                    f"  auroc {was['auroc']:.4f} ({metrics['auroc'] - was['auroc']:+.4f})",
                    flush=True,
                )

    fold_files = sorted(out_dir.glob("oof_fold*.csv"))
    if not fold_files:
        raise SystemExit(f"No folds produced predictions in {out_dir}.")

    # Every banked fold file must cover its whole validation fold. A short one means it was
    # written by an earlier, smaller run - a smoke test, or an interrupted pass - and pooling it
    # with full folds produces an OOF that is quietly built on the wrong rows. Checked rather
    # than assumed, because the resulting number looks entirely reasonable.
    stale = []
    for path in fold_files:
        fold = int(path.stem.replace("oof_fold", ""))
        expected = int((df[fold_col] == fold).sum())
        actual = len(pd.read_csv(path))
        if actual < expected:
            stale.append(f"    fold {fold}: {actual} rows, expected {expected}")
    if stale and not args.limit:
        raise SystemExit(
            "Some banked fold predictions are incomplete:\n"
            + "\n".join(stale)
            + f"\n\nThese are left over from a smaller run. Delete "
            f"{out_dir.relative_to(REPO_ROOT)}/oof_fold*.csv and rerun, or pass --folds to "
            "redo just the short ones."
        )

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

    if before is not None:
        shared = before.merge(oof[["uid"]], on="uid")
        base = summarize(shared["is_pathologic"].to_numpy(), shared["oof_pred"].to_numpy())
        print(f"\n  without TTA      {base['logloss']:.4f}  auroc {base['auroc']:.4f}")
        print(f"  change           {tta['logloss'] - base['logloss']:+.4f} log loss, "
              f"{tta['auroc'] - base['auroc']:+.4f} auroc")
        print("\n  Both numbers are UNCALIBRATED. TTA softens logits, so its temperature")
        print("  differs from the original run's - compare after calibration, not here.")
        if args.limit:
            print("\n  Pooled over a SAMPLE. The paired per-fold deltas above are the thing")
            print("  to read on a smoke run; this total is still too small to act on.")

    print(f"\nNext: python scripts/04_evaluate.py --run {out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
