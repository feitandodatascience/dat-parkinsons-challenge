"""Tests for scripts/08_sbr_features.py and scripts/09_blend.py, on SYNTHETIC data only.

Two things are worth pinning here. First, that the features actually respond to the anatomy
they claim to measure — an asymmetry feature that does not notice asymmetry is worse than no
feature, because it looks like evidence. Second, that the blend's honest estimate really is
honest: the in-sample weight search must be demonstrably optimistic relative to it, or the
guard against the 2026-09-08 isotonic mistake is not doing anything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sbr = _load("sbr_features", "08_sbr_features.py")
blend = _load("blend", "09_blend.py")


def striatal_volume(
    shape: tuple[int, int, int] = (96, 96, 96),
    right_strength: float = 1.0,
    elongation: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """A DaT-like volume: two elongated lobes either side of centre on axis 0.

    `right_strength` scales the second lobe, standing in for asymmetric dopaminergic loss.
    `elongation` stretches both lobes along axis 1, standing in for the comma shape that
    degrades to a dot.
    """
    rng = np.random.default_rng(seed)
    array = rng.normal(0.2, 0.02, size=shape).astype(np.float32).clip(0)

    cx, cy, cz = (s // 2 for s in shape)
    xx, yy, zz = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for offset, strength in ((-7, 1.0), (+7, right_strength)):
        blob = np.exp(
            -(
                ((xx - cx - offset) ** 2) / (2 * 3.0**2)
                + ((yy - cy) ** 2) / (2 * (3.0 * elongation) ** 2)
                + ((zz - cz) ** 2) / (2 * 3.0**2)
            )
        )
        array += (strength * blob).astype(np.float32)
    return array


class TestFeatureExtraction:
    def test_returns_finite_values_only(self) -> None:
        features = sbr.extract_features(striatal_volume())
        assert features
        for name, value in features.items():
            assert np.isfinite(value), f"{name} is not finite"

    def test_feature_set_is_stable(self) -> None:
        """Blending merges on column names; a shifting feature set would break silently."""
        a = sbr.extract_features(striatal_volume(seed=1))
        b = sbr.extract_features(striatal_volume(right_strength=0.3, seed=2))
        assert a.keys() == b.keys()

    def test_handles_a_flat_volume_without_dividing_by_zero(self) -> None:
        features = sbr.extract_features(np.zeros((96, 96, 96), dtype=np.float32))
        for name, value in features.items():
            assert np.isfinite(value), f"{name} is not finite on an empty volume"

    def test_asymmetry_tracks_the_asymmetry_it_is_given(self) -> None:
        symmetric = sbr.extract_features(striatal_volume(right_strength=1.0))
        lopsided = sbr.extract_features(striatal_volume(right_strength=0.25))
        assert abs(symmetric["asym0_mean"]) < 0.02
        assert lopsided["asym0_mean"] > 0.05, "the intact side should dominate"
        assert lopsided["asym0_abs"] > symmetric["asym0_abs"]

    def test_asymmetry_is_signed_the_right_way_round(self) -> None:
        left_worse = sbr.extract_features(striatal_volume(right_strength=1.0, seed=3))
        # Mirror the volume: the sign of the axis-0 asymmetry must flip with it.
        mirrored = sbr.extract_features(
            np.flip(striatal_volume(right_strength=0.25, seed=3), axis=0).copy()
        )
        normal = sbr.extract_features(striatal_volume(right_strength=0.25, seed=3))
        assert np.sign(mirrored["asym0_mean"]) == -np.sign(normal["asym0_mean"])
        assert abs(left_worse["asym0_mean"]) < abs(normal["asym0_mean"])

    def test_per_hemisphere_elongation_separates_a_comma_from_a_dot(self) -> None:
        """The clinically meaningful shape feature, measured on one lobe at a time."""
        comma = sbr.extract_features(striatal_volume(elongation=2.0))
        dot = sbr.extract_features(striatal_volume(elongation=1.0))
        assert comma["elong0_lo"] > dot["elong0_lo"]
        assert comma["elong0_hi"] > dot["elong0_hi"]

    def test_whole_box_shape_measures_separation_not_lobe_shape(self) -> None:
        """Pins the documented caveat, so nobody reads `t50_separation` as elongation.

        Over the whole box the inertia tensor is dominated by the gap between the two striata,
        so a pair of compact dots scores HIGHER than a pair of commas — the reverse of the
        clinical reading. This is why the feature is not called `elongation`.
        """
        comma = sbr.extract_features(striatal_volume(elongation=2.0))
        dot = sbr.extract_features(striatal_volume(elongation=1.0))
        assert dot["t50_separation"] > comma["t50_separation"]

    def test_uptake_ratio_falls_when_the_striatum_fades(self) -> None:
        healthy = sbr.extract_features(striatal_volume(right_strength=1.0))
        faded = sbr.extract_features(striatal_volume(right_strength=0.1))
        assert faded["box_to_shell"] < healthy["box_to_shell"]

    def test_features_are_per_sample(self) -> None:
        """Same volume, different neighbours, identical features (CLAUDE.md 1.1)."""
        volume = striatal_volume(seed=7)
        first = sbr.extract_features(volume)
        sbr.extract_features(striatal_volume(right_strength=0.2, seed=99))
        assert sbr.extract_features(volume) == first


class TestTwoLobeDiagnostic:
    def test_finds_two_lobes_on_the_lateral_axis(self) -> None:
        volume = striatal_volume()
        assert sbr.two_lobe_score(volume, 0) == 2

    def test_finds_one_lobe_on_the_other_axes(self) -> None:
        volume = striatal_volume()
        assert sbr.two_lobe_score(volume, 1) == 1
        assert sbr.two_lobe_score(volume, 2) == 1

    def test_empty_volume_scores_zero(self) -> None:
        assert sbr.two_lobe_score(np.zeros((96, 96, 96), dtype=np.float32), 0) == 0


def synthetic_oof(n: int = 900, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two partially-independent predictors of the same labels."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-truth))).astype(int)
    z_a = truth + rng.normal(0, 0.8, n)
    z_b = truth + rng.normal(0, 0.8, n)
    return y, z_a, z_b


class TestStack:
    def test_gives_a_useless_column_almost_no_weight(self) -> None:
        y, z_good, _ = synthetic_oof()
        rng = np.random.default_rng(1)
        z_noise = rng.normal(0, 1, len(y))

        model = blend.fit_stack(np.column_stack([z_good, z_noise]), y)
        coef_good, coef_noise = model.coef_[0]
        assert abs(coef_noise) < 0.2 * abs(coef_good)

    def test_shares_weight_between_two_equally_good_columns(self) -> None:
        y, z_a, z_b = synthetic_oof()
        coef_a, coef_b = blend.fit_stack(np.column_stack([z_a, z_b]), y).coef_[0]
        assert coef_a > 0 and coef_b > 0
        assert 0.4 < coef_a / (coef_a + coef_b) < 0.6

    def test_absorbs_overconfidence_into_the_coefficient(self) -> None:
        """A sharpened logit should draw a coefficient below 1 — temperature scaling, restated."""
        y, z, _ = synthetic_oof()
        coef = blend.fit_stack((z * 2.5).reshape(-1, 1), y).coef_[0][0]
        assert 0.2 < coef < 0.8


class TestHonestComparison:
    def test_cross_validated_score_is_not_better_than_in_sample(self) -> None:
        """The guard against the isotonic mistake: honest must not beat optimistic."""
        y, z_a, z_b = synthetic_oof()
        both = np.column_stack([z_a, z_b])
        in_sample = blend.competition_logloss(
            y, blend.stack_predict(blend.fit_stack(both, y), both)
        )
        honest = blend.competition_logloss(y, blend.cv_score(both, y, n_splits=5, seed=42))
        assert honest >= in_sample - 1e-9

    def test_a_useless_second_model_does_not_produce_a_gain(self) -> None:
        """The failure that forced the convex weight to be replaced by a stack.

        A convex blend scored an apparent win here purely by shrinking an overconfident logit.
        Comparing two stacks — one with the noise column, one without — removes that route,
        because both sides get the same recalibration.
        """
        y, z_good, _ = synthetic_oof(n=1200)
        rng = np.random.default_rng(5)
        z_noise = rng.normal(0, 1, len(y))

        alone = blend.competition_logloss(
            y, blend.cv_score(z_good.reshape(-1, 1), y, n_splits=5, seed=42)
        )
        with_noise = blend.competition_logloss(
            y, blend.cv_score(np.column_stack([z_good, z_noise]), y, n_splits=5, seed=42)
        )
        assert with_noise >= alone - 0.005

    def test_a_genuinely_informative_column_does_produce_a_gain(self) -> None:
        """The other side of the guard: a real second signal must be detected."""
        y, z_a, z_b = synthetic_oof(n=1200)
        alone = blend.competition_logloss(
            y, blend.cv_score(z_a.reshape(-1, 1), y, n_splits=5, seed=42)
        )
        with_b = blend.competition_logloss(
            y, blend.cv_score(np.column_stack([z_a, z_b]), y, n_splits=5, seed=42)
        )
        assert with_b < alone - 0.01

    def test_returns_finite_predictions(self) -> None:
        y, z_a, z_b = synthetic_oof()
        out = blend.cv_score(np.column_stack([z_a, z_b]), y, n_splits=5, seed=42)
        assert np.isfinite(out).all()
        assert ((out > 0) & (out < 1)).all()


class TestBlendEndToEnd:
    """Exercise `main()` itself on synthetic inputs.

    The unit tests above cover the statistics; this covers the wiring — argument parsing, the
    three-way merge, and report generation — which is where a rename breaks things silently.
    """

    @staticmethod
    def _write_inputs(root: Path, n: int = 900, informative: bool = True) -> None:
        """Two independent drivers of the label, so the two verdicts are separable.

        `seen` drives the label and is what the CNN observes. `unseen` also drives the label
        but the CNN never sees it. In the informative case the features carry `unseen`, so the
        blend has something real to add; otherwise they are pure noise. Making the features a
        noisy copy of `seen` would test nothing — the CNN already has it, and the honest
        estimate correctly reports no gain.
        """
        rng = np.random.default_rng(0)
        uids = [f"u{i:04d}" for i in range(n)]
        seen = rng.normal(0, 1, n)
        unseen = rng.normal(0, 1, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(seen + unseen)))).astype(int)
        p_cnn = 1 / (1 + np.exp(-(seen * 1.5 + rng.normal(0, 0.3, n))))

        # Folds must be STRATIFIED, as `fold_strat` is. With arbitrary folds each fold's
        # training base rate is anti-correlated with its own, a tree model fitted on pure noise
        # learns that offset, and the blend shows a small gain from nothing at all. Stratifying
        # equalises the base rates and closes the route.
        fold = np.empty(n, dtype=int)
        for label in (0, 1):
            idx = np.flatnonzero(y == label)
            fold[idx] = np.arange(len(idx)) % 5

        (root / "models" / "fake").mkdir(parents=True)
        pd.DataFrame(
            {"uid": uids, "is_pathologic": y, "oof_pred": p_cnn, "fold": fold}
        ).to_csv(root / "models" / "fake" / "oof.csv", index=False)

        columns = {"uid": uids}
        for j in range(12):
            signal = unseen if (informative and j < 3) else 0.0
            columns[f"f{j}"] = signal + rng.normal(0, 0.4, n)
        pd.DataFrame(columns).to_csv(root / "features.csv", index=False)
        pd.DataFrame({"uid": uids, "fold_strat": fold}).to_csv(
            root / "folds.csv", index=False
        )

    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        monkeypatch.setattr(blend, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "09_blend.py",
                "--run", "fake",
                "--models-dir", str(tmp_path / "models"),
                "--features", str(tmp_path / "features.csv"),
                "--folds-csv", str(tmp_path / "folds.csv"),
            ],
        )
        assert blend.main() == 0
        report = tmp_path / "reports" / "experiments" / "fake_blend.md"
        assert report.exists()
        return report.read_text(encoding="utf-8")

    def test_runs_and_writes_a_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_inputs(tmp_path)
        text = self._run(tmp_path, monkeypatch)
        assert "cross-validated log loss" in text
        assert "fitted coefficients" in text
        assert "Verdict" in text

    def test_report_contains_no_uids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_inputs(tmp_path)
        text = self._run(tmp_path, monkeypatch)
        for uid in pd.read_csv(tmp_path / "features.csv")["uid"]:
            assert uid not in text, "a uid leaked into a Claude-readable report"

    def test_uninformative_features_reach_the_negative_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_inputs(tmp_path, informative=False)
        text = self._run(tmp_path, monkeypatch)
        assert "not worth shipping" in text

    def test_informative_features_reach_the_positive_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_inputs(tmp_path, informative=True)
        text = self._run(tmp_path, monkeypatch)
        assert "worth shipping" in text
        assert "not worth shipping" not in text


class TestGbmOof:
    def test_every_row_predicted_exactly_once(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        x = pd.DataFrame(rng.normal(0, 1, (n, 6)), columns=[f"f{i}" for i in range(6)])
        y = (x["f0"] + rng.normal(0, 0.5, n) > 0).astype(int).to_numpy()
        folds = np.tile(np.arange(5), n // 5)

        oof = blend.fit_gbm_oof(x, y, folds, seed=42)
        assert oof.shape == (n,)
        assert np.isfinite(oof).all()
        assert ((oof >= 0) & (oof <= 1)).all()

    def test_learns_a_real_signal(self) -> None:
        rng = np.random.default_rng(2)
        n = 600
        x = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=[f"f{i}" for i in range(4)])
        y = (x["f0"] * 2 + rng.normal(0, 0.5, n) > 0).astype(int).to_numpy()
        folds = np.tile(np.arange(5), n // 5)

        oof = blend.fit_gbm_oof(x, y, folds, seed=42)
        assert blend.competition_logloss(y, oof) < 0.5

    def test_finds_nothing_in_pure_noise(self) -> None:
        """Guards against a leak: unrelated features must not beat the prior."""
        rng = np.random.default_rng(3)
        n = 600
        x = pd.DataFrame(rng.normal(0, 1, (n, 8)), columns=[f"f{i}" for i in range(8)])
        y = (rng.random(n) < 0.5).astype(int)
        folds = np.tile(np.arange(5), n // 5)

        oof = blend.fit_gbm_oof(x, y, folds, seed=42)
        assert blend.competition_logloss(y, oof) > 0.65
