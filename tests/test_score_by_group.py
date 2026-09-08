"""Tests for scripts/06_score_by_group.py, on SYNTHETIC predictions only.

No competition data is involved: the "OOF predictions" here are drawn from a generative model
where the answer is known in advance, so the assertions can check that the decomposition
recovers the effect it is supposed to measure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The script is a numbered CLI entry point, not an importable module name.
_spec = importlib.util.spec_from_file_location(
    "score_by_group", REPO_ROOT / "scripts" / "06_score_by_group.py"
)
assert _spec is not None and _spec.loader is not None
score_by_group = importlib.util.module_from_spec(_spec)
sys.modules["score_by_group"] = score_by_group
_spec.loader.exec_module(score_by_group)


def make_oof(
    n_per_group: int = 300,
    group_priors: tuple[float, ...] = (0.45, 0.55, 0.65),
    overconfidence: float = 1.6,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic OOF predictions with a known per-group prior and known overconfidence.

    Construction, chosen so the optimal temperature is known exactly rather than approximately:
    each sample gets a true probability `p_true`, the label is drawn from it, and the recorded
    prediction is `p_true` sharpened by `overconfidence` in logit space. Dividing the predicted
    logit by `overconfidence` therefore recovers `p_true` sample for sample, so the optimal
    temperature *is* `overconfidence` — for any group composition. Each group's logits are
    centred differently, which gives the per-family base-rate spread the report is built around
    without disturbing that identity.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for gi, prior in enumerate(group_priors):
        centre = np.log(prior / (1.0 - prior))
        z_true = rng.normal(centre, 1.5, n_per_group)
        p_true = 1.0 / (1.0 + np.exp(-z_true))
        y = (rng.random(n_per_group) < p_true).astype(int)
        p = 1.0 / (1.0 + np.exp(-z_true * overconfidence))
        frames.append(
            pd.DataFrame(
                {
                    "uid": [f"g{gi}s{i}" for i in range(n_per_group)],
                    "is_pathologic": y,
                    "oof_pred": p,
                    "fold": gi,
                    "acquisition_group": f"fam{gi}",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestOraclePriorShift:
    def test_offset_is_zero_when_already_calibrated_in_the_large(self) -> None:
        rng = np.random.default_rng(1)
        p = rng.uniform(0.05, 0.95, 4000)
        y = (rng.random(4000) < p).astype(int)
        loss, offset = score_by_group.oracle_prior_shift(y, p)
        assert abs(offset) < 0.25
        assert loss <= score_by_group.competition_logloss(y, p) + 1e-9

    def test_offset_corrects_a_known_systematic_shift(self) -> None:
        rng = np.random.default_rng(2)
        p = rng.uniform(0.05, 0.95, 4000)
        y = (rng.random(4000) < p).astype(int)
        # Push every prediction down by 1.0 in logit space; the oracle should push back up.
        z = np.log(p / (1 - p)) - 1.0
        shifted = 1.0 / (1.0 + np.exp(-z))
        loss, offset = score_by_group.oracle_prior_shift(y, shifted)
        assert offset == pytest.approx(1.0, abs=0.3)
        assert loss < score_by_group.competition_logloss(y, shifted)

    def test_never_worse_than_no_shift(self) -> None:
        df = make_oof()
        for _, sub in df.groupby("acquisition_group"):
            y = sub["is_pathologic"].to_numpy()
            p = sub["oof_pred"].to_numpy()
            loss, _ = score_by_group.oracle_prior_shift(y, p)
            assert loss <= score_by_group.competition_logloss(y, p) + 1e-9


class TestLogoTemperature:
    def test_every_row_receives_a_prediction(self) -> None:
        df = make_oof()
        out, per_group = score_by_group.logo_temperature(df, "acquisition_group")
        assert not np.isnan(out).any()
        assert set(per_group) == {"fam0", "fam1", "fam2"}

    def test_recovers_overconfidence_when_groups_are_homogeneous(self) -> None:
        """With the same overconfidence in every group, LOGO temperature should transfer."""
        df = make_oof(overconfidence=1.6)
        out, per_group = score_by_group.logo_temperature(df, "acquisition_group")
        y = df["is_pathologic"].to_numpy()
        p = df["oof_pred"].to_numpy()

        logo_loss = score_by_group.competition_logloss(y, out)
        raw_loss = score_by_group.competition_logloss(y, p)
        assert logo_loss < raw_loss, "calibration fitted on other families should still help"

        # The fixture makes 1.6 the exactly-optimal temperature, so a fit that excludes one
        # family should still land on it. Drift here would mean the temperature is tracking
        # family composition rather than the model's sharpness — the failure this measures.
        for t in per_group.values():
            assert t == pytest.approx(1.6, rel=0.15)

    def test_single_group_yields_no_predictions(self) -> None:
        """Leave-one-out is undefined with one group; it must not fabricate a temperature."""
        df = make_oof(group_priors=(0.55,))
        out, per_group = score_by_group.logo_temperature(df, "acquisition_group")
        assert np.isnan(out).all()
        assert per_group == {}


class TestBuildReport:
    def test_report_is_aggregate_only(self) -> None:
        df = make_oof()
        lines = score_by_group.build_report(df, "synthetic", min_group_n=30)
        text = "\n".join(lines)
        for uid in df["uid"]:
            assert uid not in text, "a uid leaked into a Claude-readable report"

    def test_small_families_are_pooled(self) -> None:
        df = make_oof(n_per_group=300, group_priors=(0.5, 0.5))
        tiny = df.head(5).copy()
        tiny["uid"] = [f"tiny{i}" for i in range(5)]
        tiny["acquisition_group"] = "rare_family"
        df = pd.concat([df, tiny], ignore_index=True)

        lines = score_by_group.build_report(df, "synthetic", min_group_n=30)
        text = "\n".join(lines)
        assert "rare_family" not in text, "a family of 5 was described on its own"
        assert score_by_group.SMALL_LABEL in text

    def test_reports_every_large_family(self) -> None:
        df = make_oof()
        text = "\n".join(score_by_group.build_report(df, "synthetic", min_group_n=30))
        for family in ("fam0", "fam1", "fam2"):
            assert family in text

    def test_every_markdown_table_is_well_formed(self) -> None:
        """Ragged tables render silently and wrongly; one slipped into a sibling script."""
        df = make_oof()
        lines = score_by_group.build_report(df, "synthetic", min_group_n=30)

        blocks, current = [], []
        for line in lines:
            if line.startswith("|"):
                current.append(line)
            elif current:
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)

        assert blocks
        for block in blocks:
            widths = {line.count("|") for line in block}
            assert len(widths) == 1, f"ragged table: column counts {widths} in {block[0]!r}"

    def test_transfer_penalty_is_reported(self) -> None:
        df = make_oof()
        text = "\n".join(score_by_group.build_report(df, "synthetic", min_group_n=30))
        assert "transfer penalty" in text
        assert "per-family oracle offsets" in text
