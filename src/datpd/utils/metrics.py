"""Metrics, framed around the fact that this competition scores **log loss**.

Log loss punishes confident mistakes without bound and rewards honest uncertainty. That makes
it a different game from accuracy or AUROC:

- A model can improve AUROC (better *ranking*) while worsening log loss (worse *calibration*).
- Clipping predictions away from 0 and 1 is not a hack, it is risk management: a single
  confidently-wrong prediction at p=1e-9 contributes ~20.7 to the sum, which can outweigh
  hundreds of good ones.
- The score to beat with zero modelling is the class prior. `baseline_logloss` computes it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

# Rationale for 1e-6: at p=1e-6 a wrong prediction costs ~13.8. Tighter clipping (say 0.02)
# caps the worst case at ~3.9 but taxes every correct confident prediction. Tune on OOF.
DEFAULT_CLIP = 1e-6


def clip_probs(p: np.ndarray, eps: float = DEFAULT_CLIP) -> np.ndarray:
    """Keep probabilities strictly inside (0, 1) so log loss stays finite."""
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def competition_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = DEFAULT_CLIP) -> float:
    """Log loss exactly as the leaderboard computes it (mean over samples)."""
    return float(log_loss(np.asarray(y_true), clip_probs(y_prob, eps), labels=[0, 1]))


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC. Reported for reference only; it does not affect ranking."""
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_prob)))


def baseline_logloss(y_true: np.ndarray) -> float:
    """Log loss of predicting the class prior for every sample.

    Any model must beat this or it is worse than knowing nothing but the base rate. Compute it
    first, and record it in reports/experiments/log.md as row zero.
    """
    y = np.asarray(y_true, dtype=np.float64)
    prior = float(y.mean())
    return competition_logloss(y, np.full_like(y, prior))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted confidence and observed frequency.

    Diagnostic, not the competition metric. High ECE with good AUROC is the signature of a
    model that ranks well but is overconfident — the exact case where calibration (see
    `datpd.calibrate`) buys a large log-loss win for free.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = clip_probs(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error, total = 0.0, len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi)
        if not mask.any():
            continue
        error += mask.sum() / total * abs(y[mask].mean() - p[mask].mean())
    return float(error)


def summarize(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """The standard block of numbers to log for every experiment."""
    return {
        "logloss": competition_logloss(y_true, y_prob),
        "auroc": auroc(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "baseline_logloss": baseline_logloss(y_true),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }
