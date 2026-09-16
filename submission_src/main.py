"""Inference entry point for the DrivenData code-execution submission.

This file must sit at the ROOT of submission.zip. The platform runs it with
`/code_execution/src` as the working directory, `/code_execution/data` mounted read-only, and
no network access.

Hard requirements, each of which has failed a real submission before:

- Writes `submission.csv` (columns `uid`, `is_pathologic`) to the working directory.
- Imports only packages present in the official runtime (see CLAUDE.md section 3).
- Processes every examination **independently** — no statistic pooled across the test set.
  This is a competition rule, not a preference (CLAUDE.md 1.1).
- Never logs data values. Shapes and counts only, under 300 lines / 300 chars per line.
- Completes within 3 hours (6 minutes for the smoke test).
- Degrades gracefully: if a single scan fails to load, emit the prior rather than crashing and
  scoring nothing. One fallback prediction costs a little log loss; a crash costs everything.

Local test loop:
    python scripts/05_package_submission.py     # -> submission/submission.zip
    # then, in the runtime repo checkout:
    just pack-submission && just test-submission
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

SRC_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_ROOT))

DATA_DIR = Path("/code_execution/data")
NIFTI_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"
OUTPUT_PATH = Path("submission.csv")
ASSETS_DIR = SRC_ROOT / "assets"

# Fallback probability used when a scan cannot be processed. Set this to the TRAINING positive
# rate once EDA is done — the prior is the log-loss-optimal guess in the absence of evidence.
FALLBACK_PROB = 0.5

# Written by scripts/05_package_submission.py from the fitted OOF calibrator. 1.0 = no-op.
TEMPERATURE = 1.0

# Guard rails on the final probabilities. See datpd.utils.metrics for why this matters.
CLIP_EPS = 1e-6

# Progress heartbeat interval, in seconds. Time-based rather than row-based so the log never
# discloses how many rows the test set has (see the note in main()). At 3 hours' budget this
# yields ~12 lines, well inside the 300-line cap.
HEARTBEAT_SECONDS = 900

# ---------------------------------------------------------------------------------------
# Preprocessing and architecture, baked in by scripts/05_package_submission.py from the
# config the shipped weights were trained with.
#
# These are constants rather than defaults-by-omission on purpose. Inference silently using
# `preprocess`'s defaults while training used a different `target_shape` is the single most
# expensive failure available here: it costs nothing at package time, raises no error on the
# platform, and shows up only as a leaderboard score that does not match CV at all.
# ---------------------------------------------------------------------------------------
TARGET_SPACING = (2.0, 2.0, 2.0)
TARGET_SHAPE = (96, 96, 96)
NORMALIZATION = "reference_region"
CROP = "center"
MODEL_WIDTHS = (16, 32, 64, 128)
MODEL_DROPOUT = 0.3

# Test-time augmentation views, as (rotation degrees, plane, shift in voxels, scale) with two
# OPTIONAL trailing elements (intensity gain, signed gamma), both 1.0 meaning no change. A
# four-element view is still a complete view and means exactly what it always meant.
#
# DEFAULT IS OFF: a single identity view, which `_make_view` short-circuits to an untouched
# array, so predictions are bit-for-bit what they would be with no TTA code present at all.
# `scripts/05_package_submission.py --tta` bakes a measured view set in its place, and only
# after that set has been shown to help on OOF (scripts/07_tta.py).
#
# Every view is small and laterality-preserving. There is deliberately no left-right flip: DaT
# abnormality is frequently asymmetric and the more-affected side carries clinical meaning, so
# averaging a scan with its mirror would destroy the signal being measured.
#
# Rules note (CLAUDE.md 1.1): each view is derived from this scan alone and the results are
# averaged for this scan alone. Nothing is pooled across the test set, so per-sample
# independence holds.
TTA_VIEWS = ((0.0, (1, 2), (0, 0, 0), 1.0),)

# Blend with a gradient-boosted model over SBR-style geometric features.
#
# DEFAULT IS OFF. With USE_BLEND False nothing below runs, no booster is loaded, and the
# prediction is exactly the CNN path above. `scripts/05_package_submission.py --blend-from`
# turns it on and bakes the three stack coefficients, after `scripts/09_blend.py` has shown the
# features are worth their complexity.
#
# When ON, the stack REPLACES temperature scaling rather than composing with it: the fitted
# CNN coefficient already absorbs the calibration that TEMPERATURE would have applied, so
# dividing by TEMPERATURE as well would soften the logits twice and give back most of the gain.
# The packager refuses --temperature alongside --blend-from for that reason.
#
# Rules note (CLAUDE.md 1.1): the features are computed from this scan's voxels alone and the
# boosters are shipped weights, so per-sample independence holds exactly as it does for the CNN.
USE_BLEND = False
STACK_INTERCEPT = 0.0
STACK_COEF_CNN = 1.0
STACK_COEF_GBM = 0.0


def load_models() -> list[torch.nn.Module]:
    """Load every fold checkpoint shipped in `assets/`.

    Averaging fold models is a free variance reduction: it almost always improves log loss
    over any single fold, and costs only inference time (of which there is plenty in 3 hours).
    """
    from datpd_vendored.models.resnet3d import ResNet3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = sorted(ASSETS_DIR.glob("fold*.pt"))
    if not checkpoints:
        logger.warning("No checkpoints found in assets/ - falling back to prior for all rows")
        return []

    models = []
    for ckpt_path in checkpoints:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = ResNet3D(widths=MODEL_WIDTHS, dropout=MODEL_DROPOUT)
        model.load_state_dict(state)
        model.to(device)
        model.eval()  # mandatory: keeps norm layers from using batch statistics
        models.append(model)

    logger.info(f"Loaded {len(models)} fold models on {device.type}")
    return models


def _make_view(
    array: np.ndarray,
    rotate_deg: float,
    axes: tuple[int, int],
    shift_vox: tuple[int, int, int],
    scale: float,
    gain: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """One deterministic perturbation - geometry, then intensity - shape-preserving.

    Must stay numerically identical to `make_view` in scripts/07_tta.py — that is the function
    the OOF measurement was made with, and a divergence here would mean the shipped model is
    not the one that was validated. `tests/test_tta.py` asserts array equality between the two
    for every view of every preset.

    Geometry is a SINGLE affine, so the volume is interpolated exactly once. `gain` and `gamma`
    are pointwise and applied afterwards, adding no second interpolation. They are written for
    a SIGNED array: the reference-region normalised volume has negative background, so a plain
    power would be a domain error there; `sign(x) * |x| ** gamma` is monotone, sign-preserving
    and finite everywhere.
    """
    geometric = not (rotate_deg == 0.0 and scale == 1.0 and tuple(shift_vox) == (0, 0, 0))
    photometric = not (gain == 1.0 and gamma == 1.0)
    if not geometric and not photometric:
        return array.astype(np.float32, copy=False)

    from scipy import ndimage

    out = array.astype(np.float32, copy=False)

    if geometric:
        matrix = np.eye(3, dtype=np.float64) / float(scale)
        if rotate_deg:
            theta = np.deg2rad(rotate_deg)
            i, j = axes
            rotation = np.eye(3, dtype=np.float64)
            rotation[i, i] = np.cos(theta)
            rotation[i, j] = -np.sin(theta)
            rotation[j, i] = np.sin(theta)
            rotation[j, j] = np.cos(theta)
            matrix = matrix @ rotation

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

    # Same order and the same explicit float32 casts as the experiment script, so the two
    # produce identical arrays rather than merely similar ones.
    if gamma != 1.0:
        out = np.sign(out) * np.abs(out) ** np.float32(gamma)
    if gain != 1.0:
        out = out * np.float32(gain)

    return out.astype(np.float32, copy=False)


_GBM_BOOSTERS: list[object] | None = None


def _load_boosters() -> list[object]:
    """Load the shipped lightgbm boosters once, and check the feature order they expect.

    The order check is the important part. A gradient-boosted model is fitted on a column
    *position*, not a name, so a feature vector assembled in a different order than the one the
    trees were trained on raises nothing, produces plausible-looking probabilities, and is
    wrong. `stack.json` records the order used at fit time and this asserts against it.
    """
    global _GBM_BOOSTERS
    if _GBM_BOOSTERS is not None:
        return _GBM_BOOSTERS

    import json

    import lightgbm as lgb

    from datpd_vendored.features import feature_names

    meta = json.loads((ASSETS_DIR / "stack.json").read_text(encoding="utf-8"))
    expected = list(meta["feature_names"])
    actual = feature_names()
    if actual != expected:
        raise RuntimeError(
            "shipped feature order does not match the one the boosters were fitted on "
            f"({len(actual)} columns here, {len(expected)} recorded)"
        )

    paths = sorted(ASSETS_DIR.glob("gbm_fold*.txt"))
    if not paths:
        raise RuntimeError("USE_BLEND is on but no gbm_fold*.txt were shipped in assets/")

    _GBM_BOOSTERS = [lgb.Booster(model_file=str(p)) for p in paths]
    logger.info(f"Loaded {len(_GBM_BOOSTERS)} boosters")
    return _GBM_BOOSTERS


def _gbm_logit(array: np.ndarray) -> float:
    """Mean logit across the tree fold models, mirroring how the CNN folds are combined."""
    from datpd_vendored.features import feature_vector

    boosters = _load_boosters()
    row = feature_vector(array).reshape(1, -1)
    probs = np.array([b.predict(row)[0] for b in boosters], dtype=np.float64)
    probs = np.clip(probs, CLIP_EPS, 1.0 - CLIP_EPS)
    return float(np.mean(np.log(probs / (1.0 - probs))))


def predict_one(models: list[torch.nn.Module], uid: str) -> float:
    """Predict a calibrated probability for a single examination.

    Independent of every other test sample by construction — the only inputs are this uid's
    voxels and the shipped weights.
    """
    from datpd_vendored.data.io import load_volume
    from datpd_vendored.data.preprocess import preprocess

    volume = load_volume(NIFTI_DIR / f"{uid}.nii.gz")
    array = preprocess(volume, TARGET_SPACING, TARGET_SHAPE, NORMALIZATION, CROP)

    device = next(models[0].parameters()).device
    # One batch holding every view of this one scan. Batching rather than looping matters:
    # measured on CPU, nine views cost 3.9x a single view, not 9x.
    batch = np.stack([_make_view(array, *view) for view in TTA_VIEWS])
    tensor = torch.from_numpy(batch).float()[:, None].to(device)

    with torch.no_grad():
        # Mean logit across views, then across folds — both in logit space, so a single
        # saturated view or fold cannot drag the ensemble the way probability averaging would.
        logits = [float(model(tensor).float().mean().item()) for model in models]

    cnn_logit = float(np.mean(logits))

    if USE_BLEND:
        # The stack carries its own calibration in STACK_COEF_CNN; TEMPERATURE is deliberately
        # not applied here. See the note beside USE_BLEND.
        gbm_logit = _gbm_logit(array)
        z = STACK_INTERCEPT + STACK_COEF_CNN * cnn_logit + STACK_COEF_GBM * gbm_logit
        return float(1.0 / (1.0 + np.exp(-z)))

    # Average in logit space, then temperature-scale, then squash. Averaging logits rather
    # than probabilities keeps the ensemble from being dragged by a single saturated member.
    return float(1.0 / (1.0 + np.exp(-cnn_logit / TEMPERATURE)))


def main() -> None:
    # ---------------------------------------------------------------------------------------
    # LOGGING RULE, learned the hard way on 2026-09-08.
    #
    # The platform filtered three lines from a submission's log with "WARNING: line filtered"
    # and a disqualification warning. The three were exactly the three containing
    # `len(submission)`; a line reporting the number of fold models survived. **The number of
    # rows in the test set is information about the hidden test set, and must never be
    # logged** — not as a total, not as a denominator in a progress counter, not implied by a
    # row index.
    #
    # So: no counts derived from the test data, and no row indices. Progress is reported as
    # elapsed time, which is what actually matters for diagnosing a timeout anyway.
    # ---------------------------------------------------------------------------------------
    submission = pd.read_csv(SUBMISSION_FORMAT_PATH)
    logger.info("Loaded submission format")

    models = load_models()
    started = time.monotonic()

    predictions: list[float] = []
    failures = 0
    next_heartbeat = HEARTBEAT_SECONDS
    for uid in submission["uid"]:
        try:
            p = predict_one(models, uid) if models else FALLBACK_PROB
        except Exception as exc:  # noqa: BLE001 - never crash the whole run for one scan
            failures += 1
            # Exactly ONE line, on the first failure only.
            #
            # This used to log for `failures <= 5`, which meant the NUMBER OF TIMES the line
            # appeared was a lower bound on how many test scans failed - a count derived from
            # the test data, which is the category that drew the 2026-09-08 disqualification
            # warning. `check_logging` does not catch it: it scans for `len(...)` and row
            # indices, and multiplicity is neither. Nor had the platform ever exercised this
            # branch, since the 0.2973 run needed no fallbacks - so the only untested logging
            # path was the one that fires when something goes wrong on the final submission.
            #
            # Exception TYPE only, never the message, which could carry data-derived values.
            # The diagnostic value is in knowing *what* failed, not how often; the summary
            # line below already reports whether-not-how-many.
            if failures == 1:
                logger.warning(f"Prediction failed ({type(exc).__name__}); using fallback")
            p = FALLBACK_PROB
        predictions.append(p)

        elapsed = time.monotonic() - started
        if elapsed >= next_heartbeat:
            logger.info(f"Still running, {elapsed / 60:.0f} min elapsed")
            next_heartbeat += HEARTBEAT_SECONDS

    submission["is_pathologic"] = np.clip(predictions, CLIP_EPS, 1.0 - CLIP_EPS)
    submission.to_csv(OUTPUT_PATH, index=False)

    # Whether fallbacks happened, not how many: a count of failures is derived from the test
    # data, and in the worst case would equal its size.
    if failures:
        logger.warning("Fallback predictions were used for one or more rows")
    else:
        logger.info("No fallback predictions were needed")
    logger.success(f"Wrote predictions to {OUTPUT_PATH} in {(time.monotonic() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
