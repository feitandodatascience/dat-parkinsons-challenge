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

# Test-time augmentation views, as (rotation degrees, plane, shift in voxels, scale).
#
# DEFAULT IS OFF: a single identity view, which `_make_view` short-circuits to an untouched
# array, so predictions are bit-for-bit what they would be with no TTA code present at all.
# `scripts/05_package_submission.py --tta` bakes the nine-view set in its place, and only after
# TTA has been shown to help on OOF (scripts/07_tta.py).
#
# Every view is small and laterality-preserving. There is deliberately no left-right flip: DaT
# abnormality is frequently asymmetric and the more-affected side carries clinical meaning, so
# averaging a scan with its mirror would destroy the signal being measured.
#
# Rules note (CLAUDE.md 1.1): each view is derived from this scan alone and the results are
# averaged for this scan alone. Nothing is pooled across the test set, so per-sample
# independence holds.
TTA_VIEWS = ((0.0, (1, 2), (0, 0, 0), 1.0),)


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
) -> np.ndarray:
    """One deterministic geometric perturbation, shape-preserving.

    Must stay numerically identical to `make_view` in scripts/07_tta.py — that is the function
    the OOF measurement was made with, and a divergence here would mean the shipped model is
    not the one that was validated.
    """
    if rotate_deg == 0.0 and scale == 1.0 and tuple(shift_vox) == (0, 0, 0):
        return array.astype(np.float32, copy=False)

    from scipy import ndimage

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
    return ndimage.affine_transform(
        array.astype(np.float32, copy=False),
        matrix,
        offset=offset,
        order=1,
        mode="nearest",
        output_shape=array.shape,
    ).astype(np.float32)


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

    # Average in logit space, then temperature-scale, then squash. Averaging logits rather
    # than probabilities keeps the ensemble from being dragged by a single saturated member.
    mean_logit = float(np.mean(logits)) / TEMPERATURE
    return float(1.0 / (1.0 + np.exp(-mean_logit)))


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
            # Exception TYPE only, and no row index: the message could carry data-derived
            # values, and the index would disclose the test set's size.
            if failures <= 5:
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
