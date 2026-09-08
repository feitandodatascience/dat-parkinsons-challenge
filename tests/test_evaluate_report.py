"""Tests for the shipping command that `datpd.evaluate` generates, on SYNTHETIC predictions.

The report tells the user exactly what to paste to build a submission. That command is followed
literally, usually late, so it has to work in the shell this project is actually developed in.
It did not: the first version wrapped its arguments with `\\` continuations, which is bash
syntax. PowerShell reads `\\` as a path separator and fails with "Missing expression after
unary operator '--'", so the command could not be pasted at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datpd.evaluate import evaluate_run


@pytest.fixture
def scored_run(tmp_path: Path) -> Path:
    """A run directory holding synthetic out-of-fold predictions."""
    rng = np.random.default_rng(0)
    n = 800
    truth = rng.normal(0, 1.4, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-truth))).astype(int)
    p = 1 / (1 + np.exp(-(truth * 1.8 + rng.normal(0, 0.5, n))))

    run_dir = tmp_path / "models" / "synthetic"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        {"uid": [f"u{i:04d}" for i in range(n)], "is_pathologic": y, "oof_pred": p}
    ).to_csv(run_dir / "oof.csv", index=False)
    return run_dir


def report_text(run_dir: Path) -> str:
    reports = run_dir.parent.parent / "reports"
    evaluate_run(run_dir, reports)
    return (reports / f"{run_dir.name}.md").read_text(encoding="utf-8")


class TestShippingCommand:
    def test_command_is_a_single_pasteable_line(self, scored_run: Path) -> None:
        text = report_text(scored_run)
        line = next(
            ln for ln in text.splitlines() if ln.startswith("python scripts/05_package")
        )
        assert "--config" in line
        assert "--temperature" in line
        assert "--fallback" in line
        assert "--clip-eps" in line

    def test_no_bash_line_continuations(self, scored_run: Path) -> None:
        """The regression: `\\` at end of line is bash, and breaks in PowerShell."""
        text = report_text(scored_run)
        for number, line in enumerate(text.splitlines(), 1):
            assert not line.rstrip().endswith("\\"), (
                f"line {number} ends with a bash continuation: {line!r}"
            )

    def test_temperature_is_the_shippable_one(self, scored_run: Path) -> None:
        """main.py can only carry a scalar temperature, whichever calibrator won."""
        text = report_text(scored_run)
        line = next(
            ln for ln in text.splitlines() if ln.startswith("python scripts/05_package")
        )
        temperature = float(line.split("--temperature ")[1].split()[0])
        assert temperature != 1.0, "a temperature of exactly 1.0 would ship uncalibrated"
        assert 0.2 < temperature < 5.0

    def test_fallback_matches_the_positive_rate(self, scored_run: Path) -> None:
        text = report_text(scored_run)
        line = next(
            ln for ln in text.splitlines() if ln.startswith("python scripts/05_package")
        )
        fallback = float(line.split("--fallback ")[1].split()[0])
        expected = pd.read_csv(scored_run / "oof.csv")["is_pathologic"].mean()
        assert fallback == pytest.approx(expected, abs=5e-5)

    def test_report_contains_no_uids(self, scored_run: Path) -> None:
        text = report_text(scored_run)
        for uid in pd.read_csv(scored_run / "oof.csv")["uid"]:
            assert uid not in text
