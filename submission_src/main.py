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
    tensor = torch.from_numpy(array).float()[None, None].to(device)

    with torch.no_grad():
        logits = [model(tensor).item() for model in models]

    # Average in logit space, then temperature-scale, then squash. Averaging logits rather
    # than probabilities keeps the ensemble from being dragged by a single saturated member.
    mean_logit = float(np.mean(logits)) / TEMPERATURE
    return float(1.0 / (1.0 + np.exp(-mean_logit)))


def main() -> None:
    submission = pd.read_csv(SUBMISSION_FORMAT_PATH)
    logger.info(f"Loaded submission format: {len(submission)} rows")

    models = load_models()

    predictions: list[float] = []
    failures = 0
    for i, uid in enumerate(submission["uid"]):
        try:
            p = predict_one(models, uid) if models else FALLBACK_PROB
        except Exception as exc:  # noqa: BLE001 - never crash the whole run for one scan
            failures += 1
            # Log the exception TYPE only. The message could contain data-derived values.
            if failures <= 5:
                logger.warning(f"Row {i}: {type(exc).__name__}; using fallback")
            p = FALLBACK_PROB
        predictions.append(p)

        if (i + 1) % 200 == 0:
            logger.info(f"Predicted {i + 1}/{len(submission)}")

    submission["is_pathologic"] = np.clip(predictions, CLIP_EPS, 1.0 - CLIP_EPS)
    submission.to_csv(OUTPUT_PATH, index=False)

    logger.info(f"Fallbacks used: {failures}/{len(submission)}")
    logger.success(f"Wrote {len(submission)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
