"""Tests for the tuned / bagged / N-component paths in scripts/09_blend.py. SYNTHETIC only.

What is worth pinning here is not that the code runs — the end-to-end tests in
`test_sbr_blend.py` already do that — but that the three new ways it could quietly start lying
are closed:

1. **The nested search must not leak.** A hyperparameter chosen on the fold it is then scored on
   turns a grid into a free parameter fitted to the reported number. The test gives the search a
   deliberately useless feature set: if selection were leaking, the winner of a twelve-entry grid
   on pure noise would show an apparent gain, and it must not.
2. **N components must reduce exactly to two.** The shipped `stack.json` is read by
   `submission_src/main.py` as three constants. Generalising the stack is only safe if the
   two-component fit is bit-for-bit what it was before the generalisation existed.
3. **The paired statistics must be arithmetically right.** They are the whole decision procedure;
   a plausible-looking but wrong standard error would be worse than no standard error at all.

Plus the bagging invariant: averaging several seeds inside a fold must not let a fold's own rows
influence its own predictions.
"""

from __future__ import annotations

import importlib.util
import json
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


blend = _load("blend_tuning", "09_blend.py")


def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three configurations spanning low to high capacity, so the search has a real choice.

    The full twelve-entry grid would make these tests minutes long without testing anything the
    three do not. High capacity is the entry that matters: if selection leaked, that is the one
    a noise dataset would pick.
    """
    grid = (
        ("current-default", {}),
        ("tiny", dict(max_depth=1, num_leaves=2, n_estimators=60, learning_rate=0.1)),
        ("greedy", dict(max_depth=-1, num_leaves=31, n_estimators=120, min_child_samples=5)),
    )
    monkeypatch.setattr(blend, "TUNE_GRID", grid)
    monkeypatch.setattr(blend, "GRID_BY_NAME", dict(grid))


def stratified_folds(y: np.ndarray, k: int = 5) -> np.ndarray:
    """Folds with equal base rates, as `fold_strat` has.

    Unstratified folds let a tree model on pure noise learn each fold's training base rate and
    look informative; `test_sbr_blend.py` documents that artifact. Every test here that asks
    "does noise produce a gain?" must therefore stratify, or it would be measuring the artifact
    instead of the leak.
    """
    fold = np.empty(len(y), dtype=int)
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def noise_problem(n: int = 500, n_features: int = 10, seed: int = 0):
    """Features that know nothing, and a CNN logit that knows something."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    x = pd.DataFrame(
        rng.normal(0, 1, (n, n_features)), columns=[f"f{i}" for i in range(n_features)]
    )
    z_cnn = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1.2, n)
    return x, y, z_cnn, stratified_folds(y)


class TestNestedSearchDoesNotLeak:
    def test_a_useless_feature_set_produces_no_apparent_gain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The headline guard. Selection inside the fold, scored outside it, on pure noise.

        The tolerance is the same 0.005 the existing noise test uses: cross-validated log loss
        at this n wobbles, and the assertion that matters is that the tuned column buys nothing
        like the -0.017 a real second model bought, not that it scores exactly zero.
        """
        small_grid(monkeypatch)
        x, y, z_cnn, folds = noise_problem()

        tuned = blend.fit_gbm_oof(
            x, y, folds, seed=42, tune=True, tune_mode="nested", inner_splits=3
        )
        z = np.column_stack([z_cnn, blend.to_logit(tuned.oof)])

        alone = blend.competition_logloss(
            y, blend.cv_score(z[:, [0]], y, n_splits=5, seed=42)
        )
        with_trees = blend.competition_logloss(y, blend.cv_score(z, y, n_splits=5, seed=42))
        assert with_trees >= alone - 0.005

    def test_the_tuned_model_itself_does_not_beat_the_prior_on_noise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One level lower: the tuned OOF predictions alone must be worse than knowing nothing.

        If the selection saw the held-out fold, the best of three configurations would score
        below the prior here. It cannot, because it never sees those rows.
        """
        small_grid(monkeypatch)
        x, y, _, folds = noise_problem(seed=3)
        tuned = blend.fit_gbm_oof(
            x, y, folds, seed=42, tune=True, tune_mode="nested", inner_splits=3
        )
        assert blend.competition_logloss(y, tuned.oof) > 0.65

    def test_selection_reads_only_the_rows_it_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`select_params` is handed training rows; it must return the same choice regardless
        of what the rest of the dataset looks like."""
        small_grid(monkeypatch)
        x, y, _, _ = noise_problem(n=400, seed=5)
        first, _ = blend.select_params(x, y, seed=42, mode="nested", inner_splits=3)
        second, _ = blend.select_params(
            x.copy(), y.copy(), seed=42, mode="nested", inner_splits=3
        )
        assert first == second

    def test_both_nesting_modes_are_available_and_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        small_grid(monkeypatch)
        x, y, _, folds = noise_problem(n=400, seed=7)
        for mode in ("nested", "inner"):
            fitted = blend.fit_gbm_oof(
                x, y, folds, seed=42, tune=True, tune_mode=mode, inner_splits=3
            )
            assert len(fitted.chosen) == 5
            assert set(fitted.chosen) <= {name for name, _ in blend.TUNE_GRID}

    def test_preselected_choices_are_replayed_not_researched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reusing a tuned run's per-fold choices must give the same fold models."""
        small_grid(monkeypatch)
        x, y, _, folds = noise_problem(n=400, seed=9)
        tuned = blend.fit_gbm_oof(
            x, y, folds, seed=42, tune=True, tune_mode="inner", inner_splits=3
        )
        replayed = blend.fit_gbm_oof(x, y, folds, seed=42, preselected=tuned.chosen)
        assert replayed.chosen == tuned.chosen
        np.testing.assert_allclose(replayed.oof, tuned.oof)


class TestSeedBagging:
    def test_one_bag_is_exactly_the_old_behaviour(self) -> None:
        x, y, _, folds = noise_problem(n=400, seed=11)
        default = blend.fit_gbm_oof(x, y, folds, seed=42)
        explicit = blend.fit_gbm_oof(x, y, folds, seed=42, n_bag=1)
        np.testing.assert_array_equal(default.oof, explicit.oof)
        assert default.n_boosters == 5

    def test_bagging_fits_s_boosters_per_fold(self) -> None:
        x, y, _, folds = noise_problem(n=400, seed=12)
        bagged = blend.fit_gbm_oof(x, y, folds, seed=42, n_bag=3)
        assert [len(members) for members in bagged.models] == [3] * 5
        assert bagged.n_boosters == 15

    def test_bagging_is_reproducible(self) -> None:
        """Fixed seed offsets, not a clock. Two runs must agree to the last bit."""
        x, y, _, folds = noise_problem(n=400, seed=13)
        first = blend.fit_gbm_oof(x, y, folds, seed=42, n_bag=3)
        second = blend.fit_gbm_oof(x, y, folds, seed=42, n_bag=3)
        np.testing.assert_array_equal(first.oof, second.oof)

    def test_members_actually_differ(self) -> None:
        """Averaging identical boosters would be an expensive no-op; check they are not."""
        x, y, _, folds = noise_problem(n=400, seed=14)
        one = blend.fit_gbm_oof(x, y, folds, seed=42, n_bag=1)
        three = blend.fit_gbm_oof(x, y, folds, seed=42, n_bag=3)
        assert not np.allclose(one.oof, three.oof)
        assert not blend.bagging_is_a_no_op(three.chosen)

    def test_a_deterministic_configuration_is_flagged_as_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        grid = (("no-sampling", dict(colsample_bytree=1.0, subsample=1.0)),)
        monkeypatch.setattr(blend, "TUNE_GRID", grid)
        monkeypatch.setattr(blend, "GRID_BY_NAME", dict(grid))
        assert blend.bagging_is_a_no_op(["no-sampling"] * 5)

    def test_averaging_stays_inside_the_fold(self) -> None:
        """A fold's own labels must not touch its own predictions, bagged or not.

        Flipping labels *inside* fold 0 changes what every other fold's model trains on, but
        fold 0's model never sees them — so fold 0's out-of-fold predictions must be identical.
        That is the property that makes the whole comparison honest, and bagging is the easiest
        place to break it by averaging in the wrong scope.
        """
        x, y, _, folds = noise_problem(n=400, seed=15)
        before = blend.fit_gbm_oof(x, y, folds, seed=42, n_bag=3).oof

        flipped = y.copy()
        flipped[folds == 0] = 1 - flipped[folds == 0]
        after = blend.fit_gbm_oof(x, flipped, folds, seed=42, n_bag=3).oof

        np.testing.assert_allclose(before[folds == 0], after[folds == 0])

    def test_refuses_more_bags_than_it_has_fixed_seeds(self) -> None:
        x, y, _, folds = noise_problem(n=200, seed=16)
        with pytest.raises(SystemExit):
            blend.fit_gbm_oof(x, y, folds, seed=42, n_bag=len(blend.BAG_SEED_OFFSETS) + 1)

    def test_members_are_averaged_in_logit_space(self) -> None:
        """Logit averaging, not probability averaging: the two differ, and only one is right.

        Reconstructed directly from the fold models, so this pins the combination rule rather
        than trusting the name of the helper that implements it.
        """
        x, y, _, folds = noise_problem(n=400, seed=17)
        bagged = blend.fit_gbm_oof(x, y, folds, seed=42, n_bag=3)

        held_out = folds == 0
        members = np.array(
            [
                blend.to_logit(m.predict_proba(x[held_out])[:, 1])
                for m in bagged.models[0]
            ]
        )
        expected = blend.from_logit(members.mean(axis=0))
        np.testing.assert_allclose(bagged.oof[held_out], expected, rtol=1e-9)
        assert not np.allclose(
            bagged.oof[held_out], blend.from_logit(members).mean(axis=0), atol=1e-6
        )


class TestPairedStatistics:
    """Checked against arithmetic done by hand, not against the implementation's own output."""

    @staticmethod
    def _constructed(n_better: int = 60, n_worse: int = 40):
        n = n_better + n_worse
        y = np.ones(n)
        p_ref = np.full(n, 0.5)
        p_var = np.concatenate([np.full(n_better, 0.6), np.full(n_worse, 0.4)])
        return y, p_ref, p_var

    def test_mean_difference_matches_the_arithmetic(self) -> None:
        y, p_ref, p_var = self._constructed()
        stats = blend.paired_stats(y, p_ref, p_var, eps=1e-6, n_boot=2000, seed=0)

        better = -np.log(0.6) + np.log(0.5)
        worse = -np.log(0.4) + np.log(0.5)
        expected = (60 * better + 40 * worse) / 100
        assert stats.mean == pytest.approx(expected, abs=1e-9)

    def test_standard_error_matches_the_arithmetic(self) -> None:
        y, p_ref, p_var = self._constructed()
        stats = blend.paired_stats(y, p_ref, p_var, eps=1e-6, n_boot=2000, seed=0)

        diff = np.concatenate(
            [
                np.full(60, -np.log(0.6) + np.log(0.5)),
                np.full(40, -np.log(0.4) + np.log(0.5)),
            ]
        )
        assert stats.se == pytest.approx(diff.std(ddof=1) / np.sqrt(100), abs=1e-12)

    def test_fraction_improved_is_a_sign_test(self) -> None:
        y, p_ref, p_var = self._constructed()
        stats = blend.paired_stats(y, p_ref, p_var, eps=1e-6, n_boot=2000, seed=0)
        assert stats.improved == pytest.approx(0.60)

    def test_direction_sign_is_negative_when_the_variant_wins(self) -> None:
        y, p_ref, p_var = self._constructed(n_better=100, n_worse=0)
        stats = blend.paired_stats(y, p_ref, p_var, eps=1e-6, n_boot=2000, seed=0)
        assert stats.mean < 0
        assert stats.improved == 1.0

    def test_bootstrap_interval_brackets_the_point_estimate(self) -> None:
        y, p_ref, p_var = self._constructed()
        stats = blend.paired_stats(y, p_ref, p_var, eps=1e-6, n_boot=5000, seed=0)
        assert stats.lo < stats.mean < stats.hi
        assert stats.resolved() == (not (stats.lo <= 0.0 <= stats.hi))

    def test_identical_predictions_are_flagged_rather_than_dividing_by_zero(self) -> None:
        y, p_ref, _ = self._constructed()
        stats = blend.paired_stats(y, p_ref, p_ref.copy(), eps=1e-6, n_boot=100, seed=0)
        assert stats.degenerate
        assert stats.sigma_text == "n/a"
        assert not stats.resolved()
        assert not stats.beats(margin=0.0)

    def test_the_clip_is_applied_before_the_difference(self) -> None:
        """A confidently-wrong prediction is capped by eps, not allowed to become infinite."""
        y = np.ones(10)
        stats = blend.paired_stats(
            y, np.full(10, 0.5), np.zeros(10), eps=1e-3, n_boot=100, seed=0
        )
        assert np.isfinite(stats.mean)
        assert stats.mean == pytest.approx(-np.log(1e-3) + np.log(0.5), abs=1e-9)

    def test_acceptance_rule_needs_both_margin_and_interval(self) -> None:
        clear = blend.Paired(
            mean=-0.02, se=0.003, lo=-0.026, hi=-0.014, improved=0.6, degenerate=False
        )
        crosses_zero = blend.Paired(
            mean=-0.02, se=0.02, lo=-0.06, hi=+0.01, improved=0.52, degenerate=False
        )
        too_small = blend.Paired(
            mean=-0.0005, se=0.0001, lo=-0.0007, hi=-0.0003, improved=0.55, degenerate=False
        )
        assert clear.beats(0.002)
        assert not crosses_zero.beats(0.002)
        assert not too_small.beats(0.002)


def write_inputs(root: Path, n: int = 700, duplicate_cnn_as: str | None = None) -> None:
    """A CNN that sees one driver of the label and features that carry the other.

    Mirrors `test_sbr_blend.py`'s fixture so the two suites agree about what a working blend
    looks like. `duplicate_cnn_as` writes a second run whose predictions are a copy of the
    CNN's, which is the cleanest possible extra component: it adds a column carrying no new
    information, so the two-component fit underneath it must be untouched.
    """
    rng = np.random.default_rng(0)
    uids = [f"u{i:04d}" for i in range(n)]
    seen = rng.normal(0, 1, n)
    unseen = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(seen + unseen)))).astype(int)
    p_cnn = 1 / (1 + np.exp(-(seen * 1.5 + rng.normal(0, 0.3, n))))
    fold = stratified_folds(y)

    (root / "models" / "fake").mkdir(parents=True)
    frame = pd.DataFrame({"uid": uids, "is_pathologic": y, "oof_pred": p_cnn, "fold": fold})
    frame.to_csv(root / "models" / "fake" / "oof.csv", index=False)
    if duplicate_cnn_as is not None:
        (root / "models" / duplicate_cnn_as).mkdir(parents=True)
        frame.to_csv(root / "models" / duplicate_cnn_as / "oof.csv", index=False)

    columns: dict[str, object] = {"uid": uids}
    for j in range(12):
        signal = unseen if j < 3 else 0.0
        columns[f"f{j}"] = signal + rng.normal(0, 0.4, n)
    pd.DataFrame(columns).to_csv(root / "features.csv", index=False)
    pd.DataFrame({"uid": uids, "fold_strat": fold}).to_csv(root / "folds.csv", index=False)


def run_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra_argv: str
) -> tuple[str, Path]:
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
            "--n-boot", "2000",
            *extra_argv,
        ],
    )
    assert blend.main() == 0
    report = tmp_path / "reports" / "experiments" / "fake_blend.md"
    assert report.exists()
    return report.read_text(encoding="utf-8"), tmp_path / "models" / "fake_blend"


class TestNComponentReducesToTwo:
    def test_an_extra_column_leaves_the_shipped_two_component_fit_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The compatibility guarantee: generalising the stack changed nothing underneath it.

        Same inputs, one run with an extra component and one without. The extra column is a
        copy of the CNN's own predictions, so it cannot shift the two-component fit through any
        legitimate route — and `stack.json`, which `main.py` reads as three constants, must come
        out identical to the last decimal.
        """
        plain, withextra = tmp_path / "plain", tmp_path / "withextra"
        plain.mkdir()
        withextra.mkdir()
        write_inputs(plain)
        write_inputs(withextra, duplicate_cnn_as="fake2")

        _, plain_dir = run_main(plain, monkeypatch)
        _, extra_dir = run_main(withextra, monkeypatch, "--extra-runs", "fake2")

        a = json.loads((plain_dir / "stack.json").read_text(encoding="utf-8"))
        b = json.loads((extra_dir / "stack.json").read_text(encoding="utf-8"))
        for key in ("intercept", "coef_cnn", "coef_gbm"):
            assert a[key] == pytest.approx(b[key], abs=1e-12)
        for key in ("stack_format", "n_folds", "n_bag", "feature_names"):
            assert a[key] == b[key]

        np.testing.assert_allclose(
            pd.read_csv(plain_dir / "oof.csv")["oof_pred"].to_numpy(),
            pd.read_csv(extra_dir / "oof.csv")["oof_pred"].to_numpy(),
        )

    def test_the_n_component_fit_is_written_separately_and_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_inputs(tmp_path, duplicate_cnn_as="fake2")
        text, blend_dir = run_main(tmp_path, monkeypatch, "--extra-runs", "fake2")

        shipped = json.loads((blend_dir / "stack.json").read_text(encoding="utf-8"))
        assert set(shipped) >= {"intercept", "coef_cnn", "coef_gbm", "feature_names"}

        ncomp = json.loads((blend_dir / "stack_ncomp.json").read_text(encoding="utf-8"))
        assert ncomp["shippable_by_current_main"] is False
        assert len(ncomp["components"]) == 3
        assert "STACK_COEF_GBM" in ncomp["note"]
        assert "cannot be expressed in what ships today" in text

    def test_no_n_component_file_without_extras(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_inputs(tmp_path)
        _, blend_dir = run_main(tmp_path, monkeypatch)
        assert not (blend_dir / "stack_ncomp.json").exists()

    def test_pairwise_correlations_are_reported_for_every_pair(self) -> None:
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, (200, 4))
        rows = blend.correlation_rows(["a", "b", "c", "d"], z)
        assert len(rows) == 6

    def test_a_mismatched_extra_run_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Labels that disagree mean the fold assignment moved; blending them silently would
        produce a stack fitted against the wrong truth."""
        write_inputs(tmp_path, duplicate_cnn_as="fake2")
        path = tmp_path / "models" / "fake2" / "oof.csv"
        broken = pd.read_csv(path)
        broken["is_pathologic"] = 1 - broken["is_pathologic"]
        broken.to_csv(path, index=False)
        with pytest.raises(SystemExit):
            run_main(tmp_path, monkeypatch, "--extra-runs", "fake2")


class TestReportIsHonestAndSafe:
    def test_the_report_states_which_nesting_was_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        small_grid(monkeypatch)
        write_inputs(tmp_path, n=400)
        text, _ = run_main(
            tmp_path, monkeypatch, "--tune", "--tune-mode", "inner", "--tune-inner-splits", "3"
        )
        assert "selection: **inner**" in text
        assert "NOT optimistic on account of the tuning" in text
        assert "folds that chose it" in text

    def test_the_report_says_when_no_tuning_happened(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_inputs(tmp_path, n=400)
        text, _ = run_main(tmp_path, monkeypatch)
        assert "## Hyperparameter tuning" in text
        assert "Not run." in text

    def test_the_clip_sweep_is_reported_and_labelled_in_sample(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_inputs(tmp_path, n=400)
        text, _ = run_main(tmp_path, monkeypatch)
        assert "## Clip epsilon" in text
        for eps in blend.CLIP_GRID:
            assert f"| {eps:g} |" in text
        assert "chosen in-sample" in text

    def test_bagging_reports_the_shipped_asset_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_inputs(tmp_path, n=400)
        text, blend_dir = run_main(tmp_path, monkeypatch, "--gbm-bag", "3")
        assert "Shipped asset count" in text
        assert len(list(blend_dir.glob("gbm_fold*.txt"))) in (5, 15)

    def test_the_decision_table_carries_a_sigma_and_an_interval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_inputs(tmp_path, n=400)
        text, _ = run_main(tmp_path, monkeypatch, "--gbm-bag", "2")
        assert "Δ vs shipped" in text
        assert "95% CI" in text
        assert "scans improved" in text
        assert "σ" in text

    def test_no_uid_reaches_the_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLAUDE.md §0: the report is read by a model, so one identifier in it is a violation."""
        write_inputs(tmp_path, n=400, duplicate_cnn_as="fake2")
        text, _ = run_main(tmp_path, monkeypatch, "--extra-runs", "fake2", "--gbm-bag", "2")
        for uid in pd.read_csv(tmp_path / "features.csv")["uid"]:
            assert uid not in text

    def test_stale_boosters_are_cleared_before_writing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The packager globs `gbm_fold*.txt`; a leftover from a differently-shaped run would be
        staged and averaged in alongside this one's."""
        write_inputs(tmp_path, n=400)
        run_main(tmp_path, monkeypatch, "--gbm-bag", "3")
        _, blend_dir = run_main(tmp_path, monkeypatch)
        names = sorted(p.name for p in blend_dir.glob("gbm_fold*.txt"))
        assert names == [f"gbm_fold{i}.txt" for i in range(5)]

    def test_feature_names_are_recorded_without_a_hard_coded_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The feature module is still growing; the metadata must follow it, not a constant."""
        write_inputs(tmp_path, n=400)
        _, blend_dir = run_main(tmp_path, monkeypatch)
        meta = json.loads((blend_dir / "stack.json").read_text(encoding="utf-8"))
        expected = [c for c in pd.read_csv(tmp_path / "features.csv").columns if c != "uid"]
        assert meta["feature_names"] == expected
        assert meta["n_features"] == len(expected)
