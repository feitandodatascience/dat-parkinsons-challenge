"""The submission must not log anything that describes the hidden test set.

On 2026-09-08 a submission came back with three lines replaced by `WARNING: line filtered` and
an explicit disqualification warning. The three filtered lines were exactly the three containing
`len(submission)`; a line reporting the number of fold models was left alone. So the rule being
enforced is not "log less" — it is **do not disclose facts about the test data**, and its row
count above all.

These tests pin the scanner that now runs at package time, and pin the shipped entry point
against regressing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "05_package_submission.py"
MAIN_PY = REPO_ROOT / "submission_src" / "main.py"


@pytest.fixture(scope="module")
def packager():
    spec = importlib.util.spec_from_file_location("_packager_logging", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestShippedEntryPoint:
    def test_main_py_is_clean(self, packager) -> None:
        """The file that actually ships must pass the scan."""
        assert packager.check_logging(MAIN_PY) == []

    def test_main_py_does_not_log_the_row_count(self) -> None:
        """A direct check, independent of the scanner's regexes."""
        text = MAIN_PY.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "logger." not in stripped:
                continue
            assert "len(submission" not in stripped, f"logs the test set size: {stripped}"


class TestScanner:
    @pytest.mark.parametrize(
        "snippet",
        [
            'logger.info(f"Loaded submission format: {len(submission)} rows")',
            'logger.info(f"Fallbacks used: {failures}/{len(submission)}")',
            'logger.success(f"Wrote {len(submission)} predictions")',
            'logger.info(f"Predicted {i + 1}/{len(submission)}")',
            'logger.warning(f"Row {i}: {type(exc).__name__}")',
        ],
    )
    def test_offending_patterns_are_caught(
        self, packager, tmp_path: Path, snippet: str
    ) -> None:
        target = tmp_path / "main.py"
        target.write_text(f"from loguru import logger\n\n\ndef f():\n    {snippet}\n", "utf-8")
        assert packager.check_logging(target), f"should have been flagged: {snippet}"

    @pytest.mark.parametrize(
        "snippet",
        [
            'logger.info("Loaded submission format")',
            'logger.info(f"Loaded {len(models)} fold models on {device.type}")',
            'logger.info(f"Still running, {elapsed / 60:.0f} min elapsed")',
            'logger.warning(f"Prediction failed ({type(exc).__name__}); using fallback")',
            'logger.warning("Fallback predictions were used for one or more rows")',
        ],
    )
    def test_safe_patterns_are_not_flagged(
        self, packager, tmp_path: Path, snippet: str
    ) -> None:
        """Counting our own shipped weights is fine — that line survived the real filter.

        A scanner that flags everything gets switched off, so the false-positive side matters
        as much as the catch.
        """
        target = tmp_path / "main.py"
        target.write_text(f"from loguru import logger\n\n\ndef f():\n    {snippet}\n", "utf-8")
        assert packager.check_logging(target) == [], f"false positive on: {snippet}"
