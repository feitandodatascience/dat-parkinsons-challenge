"""Tests for scripts/10_features_by_family.py, on SYNTHETIC feature tables only.

The script exists to answer one question — is the crop failing on wide-field acquisitions? — so
the tests check that it reaches the right verdict when the answer is yes and when it is no, and
that it cannot describe a family small enough to identify anyone.
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
    "features_by_family", REPO_ROOT / "scripts" / "10_features_by_family.py"
)
assert _spec is not None and _spec.loader is not None
fbf = importlib.util.module_from_spec(_spec)
sys.modules["features_by_family"] = fbf
_spec.loader.exec_module(fbf)


def make_features(
    families: dict[str, int],
    peak_ratio: dict[str, float] | None = None,
    pos_rate: dict[str, float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """A synthetic feature table with controllable crop quality AND label confounding.

    `peak_ratio` degrades a family's crops for every scan, positive or negative — a genuine
    preprocessing failure. `pos_rate` instead varies how many positives a family has, and
    positives get a lower ratio on their own, reproducing the confound found in the real data:
    a dim pathologic striatum puts the peak outside the box without any crop having failed.
    """
    rng = np.random.default_rng(seed)
    peak_ratio = peak_ratio or {}
    pos_rate = pos_rate or {}
    rows = []
    i = 0
    for family, n in families.items():
        ratio = peak_ratio.get(family, 0.95)
        rate = pos_rate.get(family, 0.5)
        for _ in range(n):
            label = int(rng.random() < rate)
            # Disease dims the striatum, so positives lose peak-in-box regardless of cropping.
            effective = ratio * (0.75 if label else 1.0)
            global_max = float(rng.uniform(4, 9))
            rows.append(
                {
                    "uid": f"u{i:05d}",
                    "acquisition_group": family,
                    "global_max": global_max,
                    "is_pathologic": label,
                    "box_max": global_max * min(1.0, effective + rng.normal(0, 0.01)),
                    "box_to_shell": float(rng.normal(3.0, 0.2)),
                    "t50_frac": float(rng.normal(0.02, 0.003)),
                    "elong0_lo": float(rng.normal(0.6, 0.05)),
                    "asym0_abs": float(abs(rng.normal(0.05, 0.02))),
                }
            )
            i += 1
    return pd.DataFrame(rows)


class TestDerivedRatio:
    def test_peak_in_box_is_the_ratio_of_the_two_maxima(self) -> None:
        df = pd.DataFrame({"global_max": [8.0, 4.0], "box_max": [8.0, 2.0]})
        out = fbf.add_derived(df)
        assert out["peak_in_box"].tolist() == [1.0, 0.5]

    def test_a_zero_global_max_does_not_divide_by_zero(self) -> None:
        df = pd.DataFrame({"global_max": [0.0], "box_max": [0.0]})
        out = fbf.add_derived(df)
        assert np.isfinite(out["peak_in_box"]).all()
        assert out["peak_in_box"].iloc[0] == 0.0

    def test_does_not_mutate_its_input(self) -> None:
        df = pd.DataFrame({"global_max": [8.0], "box_max": [4.0]})
        fbf.add_derived(df)
        assert "peak_in_box" not in df.columns


class TestFamilySummary:
    def test_small_families_are_pooled(self) -> None:
        df = fbf.add_derived(make_features({"big": 200, "tiny": 4}))
        table = fbf.summarise_by_family(df, min_group_n=30)
        assert "tiny" not in set(table["family"])
        assert fbf.SMALL_LABEL in set(table["family"])

    def test_row_counts_add_up(self) -> None:
        df = fbf.add_derived(make_features({"a": 100, "b": 60, "c": 5}))
        table = fbf.summarise_by_family(df, min_group_n=30)
        assert table["n"].sum() == 165

    def test_sorted_by_size(self) -> None:
        df = fbf.add_derived(make_features({"a": 60, "b": 200, "c": 120}))
        table = fbf.summarise_by_family(df, min_group_n=30)
        assert list(table["n"]) == sorted(table["n"], reverse=True)


class TestVerdict:
    def test_uniform_crops_reach_the_no_problem_verdict(self) -> None:
        df = fbf.add_derived(
            make_features({"wide": 460, "narrow": 150, "mid": 200})
        )
        table = fbf.summarise_by_family(df, min_group_n=30)
        text = "\n".join(fbf.build_report(table, len(df), 30))
        assert "Cropping is not the explanation" in text

    def test_a_failing_family_is_flagged_and_named(self) -> None:
        df = fbf.add_derived(
            make_features(
                {"wide": 460, "narrow": 150, "mid": 200},
                peak_ratio={"wide": 0.55},
            )
        )
        table = fbf.summarise_by_family(df, min_group_n=30)
        text = "\n".join(fbf.build_report(table, len(df), 30))
        assert "Worth investigating" in text
        assert "wide" in text

    def test_a_pure_label_confound_is_not_called_a_crop_failure(self) -> None:
        """The mistake the first real run of this script invited.

        Every family here crops equally well; they differ only in how many positives they
        have. Because a pathologic striatum is dimmer, the unstratified ratio separates them
        anyway — and reading that as a preprocessing bug would have bought a retraining run,
        two days of the remaining budget, for nothing.
        """
        df = fbf.add_derived(
            make_features(
                {"sick": 460, "healthy": 200, "mid": 150},
                pos_rate={"sick": 0.62, "healthy": 0.38, "mid": 0.5},
            )
        )
        table = fbf.summarise_by_family(df, min_group_n=30)
        text = "\n".join(fbf.build_report(table, len(df), 30))
        assert "Cropping is not the explanation" in text

    def test_a_real_failure_survives_the_stratification(self) -> None:
        """The other side of it: a genuine crop failure hiding behind a confound is still caught."""
        df = fbf.add_derived(
            make_features(
                {"sick": 460, "healthy": 200, "mid": 150},
                peak_ratio={"sick": 0.5},
                pos_rate={"sick": 0.62, "healthy": 0.38, "mid": 0.5},
            )
        )
        table = fbf.summarise_by_family(df, min_group_n=30)
        text = "\n".join(fbf.build_report(table, len(df), 30))
        assert "Worth investigating" in text

    def test_every_markdown_table_is_well_formed(self) -> None:
        """Header, separator and body rows must all have the same number of columns.

        Caught a real bug: the "other features" header was built with a generator that rebound
        the same throwaway name twice, so its filter never fired and the header carried one
        more label than the rows had cells. Markdown renders that silently and wrongly.
        """
        df = fbf.add_derived(make_features({"wide": 200, "narrow": 100}))
        table = fbf.summarise_by_family(df, min_group_n=30)
        lines = fbf.build_report(table, len(df), 30)

        blocks, current = [], []
        for line in lines:
            if line.startswith("|"):
                current.append(line)
            elif current:
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)

        assert blocks, "the report should contain at least one table"
        for block in blocks:
            widths = {line.count("|") for line in block}
            assert len(widths) == 1, f"ragged table: column counts {widths} in {block[0]!r}"
            assert len(block) >= 3, "a table needs a header, a separator and a row"

    def test_report_contains_no_uids(self) -> None:
        df = fbf.add_derived(make_features({"wide": 200, "narrow": 100}))
        table = fbf.summarise_by_family(df, min_group_n=30)
        text = "\n".join(fbf.build_report(table, len(df), 30))
        for uid in df["uid"]:
            assert uid not in text


class TestEndToEnd:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs) -> str:
        df = make_features(
            kwargs.pop("families"),
            kwargs.pop("peak_ratio", None),
            kwargs.pop("pos_rate", None),
        )
        # The feature table carries no labels; they arrive from folds.csv, exactly as in the
        # real pipeline. Splitting them this way is what lets the script stratify at all.
        df.drop(columns=["acquisition_group", "is_pathologic"]).to_csv(
            tmp_path / "features.csv", index=False
        )
        df[["uid", "acquisition_group", "is_pathologic"]].to_csv(
            tmp_path / "folds.csv", index=False
        )

        monkeypatch.setattr(fbf, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "10_features_by_family.py",
                "--features", str(tmp_path / "features.csv"),
                "--folds-csv", str(tmp_path / "folds.csv"),
            ],
        )
        assert fbf.main() == 0
        return (tmp_path / "reports" / "experiments" / "features_by_family.md").read_text(
            encoding="utf-8"
        )

    def test_writes_a_report(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = self._run(tmp_path, monkeypatch, families={"wide": 300, "narrow": 200})
        assert "Crop quality" in text
        assert "peak in box" in text

    def test_detects_the_failure_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = self._run(
            tmp_path,
            monkeypatch,
            families={"wide": 300, "narrow": 200},
            peak_ratio={"wide": 0.5},
        )
        assert "Worth investigating" in text

    def test_missing_features_file_fails_with_a_usable_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fbf, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["10_features_by_family.py"])
        with pytest.raises(SystemExit, match="08_sbr_features"):
            fbf.main()
