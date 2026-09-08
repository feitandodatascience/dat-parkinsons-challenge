"""Tests for the blend path in `submission_src/main.py`, on SYNTHETIC data only.

The blend adds a second model, a feature extractor and three coefficients to the file that
CLAUDE.md says to audit line by line. Everything here targets a way that can be wrong *without
raising anything*:

- the stack applied on top of temperature scaling instead of instead of it, softening twice
- a feature vector assembled in a different column order than the trees were fitted on
- the blend silently active, or silently inactive, in a build that did not ask for it

None of those produce an error. All of them produce plausible probabilities that are wrong.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.features import extract_features, feature_names, feature_vector  # noqa: E402


@pytest.fixture(autouse=True)
def _vendored_alias():
    """Make `datpd_vendored` resolve to `datpd` for the duration of a test.

    Inside the zip the vendored tree is a byte-identical copy of these modules — that is the
    property that guarantees inference preprocessing matches training. Aliasing here reproduces
    that relationship without having to build a zip for every test.
    """
    import datpd
    import datpd.features

    added = {"datpd_vendored": datpd, "datpd_vendored.features": datpd.features}
    for name, module in added.items():
        sys.modules.setdefault(name, module)
    yield
    for name in added:
        sys.modules.pop(name, None)


def load_main():
    spec = importlib.util.spec_from_file_location(
        "submission_main_blend", REPO_ROOT / "submission_src" / "main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_main_blend"] = module
    spec.loader.exec_module(module)
    return module


class TestDefaultsAreInert:
    def test_blend_is_off_by_default(self) -> None:
        main = load_main()
        assert main.USE_BLEND is False

    def test_default_coefficients_are_an_identity_on_the_cnn_logit(self) -> None:
        """Even if USE_BLEND were flipped without baking, the defaults must not distort."""
        main = load_main()
        assert main.STACK_INTERCEPT == 0.0
        assert main.STACK_COEF_CNN == 1.0
        assert main.STACK_COEF_GBM == 0.0


class TestStackArithmetic:
    """The combination rule, isolated from any model."""

    @staticmethod
    def _blend(intercept: float, a: float, b: float, z_cnn: float, z_gbm: float) -> float:
        return 1.0 / (1.0 + np.exp(-(intercept + a * z_cnn + b * z_gbm)))

    def test_zero_gbm_coefficient_reduces_to_scaled_cnn(self) -> None:
        z = 1.3
        assert self._blend(0.0, 0.8, 0.0, z, 99.0) == pytest.approx(
            1.0 / (1.0 + np.exp(-0.8 * z))
        )

    def test_temperature_is_not_applied_on_the_blend_path(self) -> None:
        """The trap: a stack coefficient of 1/T composed with T would soften twice.

        `STACK_COEF_CNN` is fitted on logits that were never temperature-scaled, so it already
        carries the calibration. Dividing by TEMPERATURE as well would halve the gain, and the
        result would still look like a reasonable probability.
        """
        main = load_main()
        source = (REPO_ROOT / "submission_src" / "main.py").read_text(encoding="utf-8")
        block = source[source.index("if USE_BLEND:") : source.index("# Average in logit")]
        # Comments only, stripped: the block deliberately *mentions* TEMPERATURE to explain why
        # it is absent, and matching on that would be checking the prose rather than the code.
        code = "\n".join(
            line for line in block.splitlines() if not line.strip().startswith("#")
        )
        assert "TEMPERATURE" not in code, (
            "the blend path must not apply TEMPERATURE; the stack carries the calibration"
        )
        assert main.USE_BLEND is False


class TestFeatureOrdering:
    """A permuted feature vector is the silent failure this guards."""

    def test_vector_matches_the_recorded_order(self) -> None:
        rng = np.random.default_rng(0)
        volume = rng.normal(0.5, 0.3, (96, 96, 96)).astype(np.float32)
        volume[40:56, 40:56, 40:56] += 3.0

        values = extract_features(volume)
        vector = feature_vector(volume)
        assert list(values) == feature_names()
        assert np.allclose(vector, [values[n] for n in feature_names()])

    def test_order_is_deterministic_across_calls(self) -> None:
        assert feature_names() == feature_names()

    def test_loader_rejects_a_mismatched_feature_order(self, tmp_path: Path) -> None:
        """`stack.json` records the fit-time order; a disagreement must be loud, not silent."""
        main = load_main()
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "stack.json").write_text(
            json.dumps({"feature_names": ["a", "b", "c"], "n_folds": 5}), encoding="utf-8"
        )
        (assets / "gbm_fold0.txt").write_text("not a real booster", encoding="utf-8")

        main.ASSETS_DIR = assets
        main._GBM_BOOSTERS = None
        with pytest.raises(RuntimeError, match="feature order"):
            main._load_boosters()

    def test_loader_rejects_missing_boosters(self, tmp_path: Path) -> None:
        main = load_main()
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "stack.json").write_text(
            json.dumps({"feature_names": feature_names(), "n_folds": 5}), encoding="utf-8"
        )

        main.ASSETS_DIR = assets
        main._GBM_BOOSTERS = None
        with pytest.raises(RuntimeError, match="no gbm_fold"):
            main._load_boosters()


class TestGbmLogit:
    """End to end through real lightgbm boosters, fitted here on synthetic features."""

    @staticmethod
    def _fit_boosters(tmp_path: Path, n_folds: int = 3):
        import lightgbm as lgb

        rng = np.random.default_rng(0)
        names = feature_names()
        n = 400
        x = rng.normal(0, 1, (n, len(names)))
        y = (x[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)

        assets = tmp_path / "assets"
        assets.mkdir(exist_ok=True)
        for k in range(n_folds):
            model = lgb.LGBMClassifier(
                n_estimators=20, max_depth=3, num_leaves=7, verbose=-1, random_state=k
            )
            model.fit(x, y)
            model.booster_.save_model(str(assets / f"gbm_fold{k}.txt"))
        (assets / "stack.json").write_text(
            json.dumps({"feature_names": names, "n_folds": n_folds}), encoding="utf-8"
        )
        return assets

    def test_returns_a_finite_logit(self, tmp_path: Path) -> None:
        main = load_main()
        main.ASSETS_DIR = self._fit_boosters(tmp_path)
        main._GBM_BOOSTERS = None

        rng = np.random.default_rng(1)
        volume = rng.normal(0.5, 0.3, (96, 96, 96)).astype(np.float32)
        assert np.isfinite(main._gbm_logit(volume))

    def test_boosters_are_loaded_once_and_cached(self, tmp_path: Path) -> None:
        main = load_main()
        main.ASSETS_DIR = self._fit_boosters(tmp_path)
        main._GBM_BOOSTERS = None

        first = main._load_boosters()
        assert main._load_boosters() is first

    def test_all_folds_contribute(self, tmp_path: Path) -> None:
        main = load_main()
        main.ASSETS_DIR = self._fit_boosters(tmp_path, n_folds=4)
        main._GBM_BOOSTERS = None
        assert len(main._load_boosters()) == 4

    def test_is_per_sample(self, tmp_path: Path) -> None:
        """Same volume, different predecessors, same logit (CLAUDE.md 1.1)."""
        main = load_main()
        main.ASSETS_DIR = self._fit_boosters(tmp_path)
        main._GBM_BOOSTERS = None

        rng = np.random.default_rng(2)
        a = rng.normal(0.5, 0.3, (96, 96, 96)).astype(np.float32)
        b = rng.normal(0.9, 0.5, (96, 96, 96)).astype(np.float32)

        first = main._gbm_logit(a)
        main._gbm_logit(b)
        assert main._gbm_logit(a) == first
