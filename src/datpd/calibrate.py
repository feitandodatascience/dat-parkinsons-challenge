"""Probability calibration — the cheapest log-loss win available in this competition.

A 3D CNN trained with BCE on a few thousand scans will almost always be overconfident: it
pushes predictions toward 0 and 1 harder than the evidence supports. AUROC does not notice
(ranking is unchanged) but log loss does, badly. Fitting a one-parameter temperature on
out-of-fold predictions typically recovers a meaningful chunk of loss for a few seconds of
compute and zero extra training.

Fit calibration on OOF predictions ONLY. Fitting on the training split gives an overconfident
temperature (the model is already near-perfect there) and makes things worse.

COMPETITION CONSTRAINT: the fitted parameter is a scalar learned at training time and shipped
as a literal in the submission. It is never re-fitted on the test set — that would violate the
per-sample independence rule (CLAUDE.md 1.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from .utils.metrics import clip_probs, competition_logloss


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


def _to_prob(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class TemperatureScaler:
    """Single-parameter calibration: p_calibrated = sigmoid(logit(p) / T).

    T > 1 softens predictions (fixes overconfidence — the common case).
    T < 1 sharpens them.
    """

    temperature: float = 1.0

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "TemperatureScaler":
        """Find the T minimising log loss on out-of-fold predictions."""
        logits = _to_logit(y_prob)
        y = np.asarray(y_true)

        def objective(log_t: float) -> float:
            return competition_logloss(y, _to_prob(logits / np.exp(log_t)))

        # Optimise in log space so T stays positive. Bounds allow T in [~0.05, ~20].
        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        return _to_prob(_to_logit(y_prob) / self.temperature)


@dataclass
class IsotonicCalibrator:
    """Non-parametric calibration via isotonic regression.

    More flexible than temperature scaling, and correspondingly easier to overfit. Prefer it
    only when OOF n is large (rule of thumb: >2000) and it beats temperature scaling on a
    held-out fold, not on the data it was fitted to.
    """

    model: IsotonicRegression | None = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "IsotonicCalibrator":
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model.fit(np.asarray(y_prob), np.asarray(y_true))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsotonicCalibrator.fit must be called before transform")
        return clip_probs(self.model.predict(np.asarray(y_prob)))


@dataclass
class IdentityCalibrator:
    """No-op calibrator, so "apply nothing" competes on the same footing as the others."""

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        return clip_probs(y_prob)


Calibrator = IdentityCalibrator | TemperatureScaler | IsotonicCalibrator

CALIBRATORS: dict[str, type] = {
    "none": IdentityCalibrator,
    "temperature": TemperatureScaler,
    "isotonic": IsotonicCalibrator,
}


def choose_calibrator(
    y_true: np.ndarray, y_prob: np.ndarray, n_splits: int = 5, seed: int = 42
) -> tuple[str, Calibrator, dict[str, float]]:
    """Compare no-op / temperature / isotonic by CROSS-VALIDATED log loss on OOF predictions.

    The cross-validation is the whole point, and omitting it produces a specific, expensive
    mistake. Scoring each calibrator on the same data it was fitted to rewards flexibility
    rather than accuracy: temperature scaling has one parameter and barely benefits, while
    isotonic regression has of order *n* knots and can drive in-sample log loss down almost
    arbitrarily. Isotonic then "wins" every comparison, reports an implausible ECE of ~0, and
    generalises worse than the method it displaced.

    So each candidate is fitted on k-1 folds of the OOF set and scored on the held-out fold,
    which is an honest estimate of what it will do to unseen test predictions. Only the winner
    is refitted on the full OOF set for shipping.

    Returns the winning name, the refitted object, and every score, so the decision is visible
    in the experiment log rather than hidden.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    held_out = {name: np.full(len(y), np.nan) for name in CALIBRATORS}

    for train_idx, test_idx in splitter.split(p.reshape(-1, 1), y):
        for name, factory in CALIBRATORS.items():
            fitted = factory().fit(y[train_idx], p[train_idx])
            held_out[name][test_idx] = fitted.transform(p[test_idx])

    scores = {name: competition_logloss(y, preds) for name, preds in held_out.items()}
    best = min(scores, key=scores.get)  # type: ignore[arg-type]

    # Refit the winner on everything: the CV was for *choosing*, not for producing the object
    # that ships.
    chosen = CALIBRATORS[best]().fit(y, p)
    return best, chosen, scores
