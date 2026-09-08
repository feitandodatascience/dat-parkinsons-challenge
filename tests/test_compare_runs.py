"""Tests for scripts/11_compare_runs.py, on SYNTHETIC out-of-fold sets only.

The script's job is to say whether a small difference is real. So the tests give it cases where
the answer is known by construction — an identical run, a genuinely better one, a genuinely
worse one, and a pure-noise perturbation — and check it reaches the right verdict each time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "compare_runs", REPO_ROOT / "scripts" / "11_compare_runs.py"
)
assert _spec is not None and _spec.loader is not None
compare = importlib.util.module_from_spec(_spec)
sys.modules["compare_runs"] = compare
_spec.loader.exec_module(compare)


def make_pair(
    n: int = 1362, improvement: float = 0.0, noise: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two predictors of the same labels, B differing from A by a controlled amount.

    `improvement` shifts B's logits toward the truth (a real gain); `noise` perturbs them
    randomly (no gain, just churn). Both are expressed in logit space.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1.4, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-truth))).astype(int)

    za = truth + rng.normal(0, 0.9, n)
    zb = za + improvement * (truth - za) + rng.normal(0, noise, n)
    return y, 1 / (1 + np.exp(-za)), 1 / (1 + np.exp(-zb))


class TestPerSampleLogloss:
    def test_matches_the_pooled_metric(self) -> None:
        y, pa, _ = make_pair()
        per = compare.per_sample_logloss(y, pa, 1e-6)
        assert per.mean() == pytest.approx(
            compare.competition_logloss(y, pa, 1e-6), rel=1e-9
        )

    def test_is_always_positive(self) -> None:
        y, pa, _ = make_pair()
        assert (compare.per_sample_logloss(y, pa, 1e-6) > 0).all()


class TestCvCalibrated:
    def test_returns_a_prediction_for_every_row(self) -> None:
        y, pa, _ = make_pair(n=500)
        out = compare.cv_calibrated(y, pa, n_splits=5, seed=42)
        assert np.isfinite(out).all()
        assert ((out > 0) & (out < 1)).all()

    def test_improves_an_overconfident_predictor(self) -> None:
        y, pa, _ = make_pair(n=2000)
        sharp = 1 / (1 + np.exp(-2.5 * np.log(pa / (1 - pa))))
        calibrated = compare.cv_calibrated(y, sharp, n_splits=5, seed=42)
        assert compare.competition_logloss(y, calibrated, 1e-6) < compare.competition_logloss(
            y, sharp, 1e-6
        )


class TestPairedBootstrap:
    def test_interval_brackets_the_mean(self) -> None:
        rng = np.random.default_rng(0)
        diff = rng.normal(-0.01, 0.1, 1362)
        lo, hi = compare.paired_bootstrap(diff, n_boot=5000, seed=1)
        assert lo < diff.mean() < hi

    def test_a_zero_difference_interval_contains_zero(self) -> None:
        lo, hi = compare.paired_bootstrap(np.zeros(1362), n_boot=2000, seed=1)
        assert lo == 0.0 and hi == 0.0

    def test_narrows_as_n_grows(self) -> None:
        rng = np.random.default_rng(0)
        small = compare.paired_bootstrap(rng.normal(0, 0.1, 200), n_boot=4000, seed=1)
        large = compare.paired_bootstrap(rng.normal(0, 0.1, 5000), n_boot=4000, seed=1)
        assert (large[1] - large[0]) < (small[1] - small[0])


class TestVerdicts:
    """End-to-end, through `main()`, on synthetic runs whose truth is known."""

    def _run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, y, pa, pb
    ) -> str:
        for run, p in (("a_run", pa), ("b_run", pb)):
            d = tmp_path / "models" / run
            d.mkdir(parents=True)
            pd.DataFrame(
                {
                    "uid": [f"u{i:05d}" for i in range(len(y))],
                    "is_pathologic": y,
                    "oof_pred": p,
                }
            ).to_csv(d / "oof.csv", index=False)

        monkeypatch.setattr(compare, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "11_compare_runs.py",
                "--run-a", "a_run",
                "--run-b", "b_run",
                "--models-dir", str(tmp_path / "models"),
                "--n-boot", "4000",
            ],
        )
        assert compare.main() == 0
        return (
            tmp_path / "reports" / "experiments" / "compare_a_run_vs_b_run.md"
        ).read_text(encoding="utf-8")

    def test_identical_runs_are_reported_as_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zero-variance difference divides by zero unless it is special-cased.

        Not a contrived input: it happens when a change turns out to be a no-op — TTA
        configured with only the identity view, say — and "your change did nothing" is a more
        useful answer than a crash or a spurious 'no detectable difference'.
        """
        y, pa, _ = make_pair()
        text = self._run(tmp_path, monkeypatch, y, pa, pa.copy())
        assert "identical predictions" in text
        assert "n/a (no variation)" in text

    def test_a_near_no_op_change_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The neighbouring case: almost identical, but not exactly."""
        y, pa, _ = make_pair()
        nudged = np.clip(pa + 1e-9, 1e-9, 1 - 1e-9)
        text = self._run(tmp_path, monkeypatch, y, pa, nudged)
        assert "Verdict" in text

    def test_a_real_improvement_is_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        y, pa, pb = make_pair(improvement=0.25)
        text = self._run(tmp_path, monkeypatch, y, pa, pb)
        assert "is better" in text
        assert "a_run is better" not in text

    def test_a_real_regression_is_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        y, pa, pb = make_pair(improvement=-0.25)
        text = self._run(tmp_path, monkeypatch, y, pa, pb)
        assert "`a_run` is better" in text
        assert "Revert" in text

    def test_pure_noise_is_not_called_an_improvement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure that matters: churn must not be reported as a gain."""
        y, pa, pb = make_pair(noise=0.15)
        text = self._run(tmp_path, monkeypatch, y, pa, pb)
        assert "Ship it" not in text

    def test_pairing_beats_the_unpaired_error_bar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The premise of the script: the paired SE must be materially smaller."""
        y, pa, pb = make_pair(improvement=0.1)
        text = self._run(tmp_path, monkeypatch, y, pa, pb)
        line = next(ln for ln in text.splitlines() if "unpaired standard error" in ln)
        factor = float(line.split("(")[1].split("x")[0])
        assert factor > 2

    def test_mismatched_uid_sets_are_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        y, pa, pb = make_pair(n=300)
        for run, p, take in (("a_run", pa, 300), ("b_run", pb, 250)):
            d = tmp_path / "models" / run
            d.mkdir(parents=True)
            pd.DataFrame(
                {
                    "uid": [f"u{i:05d}" for i in range(take)],
                    "is_pathologic": y[:take],
                    "oof_pred": p[:take],
                }
            ).to_csv(d / "oof.csv", index=False)

        monkeypatch.setattr(compare, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "11_compare_runs.py",
                "--run-a", "a_run", "--run-b", "b_run",
                "--models-dir", str(tmp_path / "models"),
            ],
        )
        with pytest.raises(SystemExit, match="same scans"):
            compare.main()

    def test_report_contains_no_uids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        y, pa, pb = make_pair(n=400, improvement=0.2)
        text = self._run(tmp_path, monkeypatch, y, pa, pb)
        assert "u00000" not in text
