"""Verify the data guard hook blocks what CLAUDE.md section 0 says it blocks.

This is the most important test in the repo: it protects competition eligibility.
Run it after any edit to .claude/hooks/guard_data.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard_data.py"

# The hook lives in .claude/, which is kept local and not published; a public clone has no hook to test.
pytestmark = pytest.mark.skipif(not HOOK.exists(), reason="data guard hook is local-only (.claude/)")


def run_hook(tool_name: str, tool_input: dict) -> bool:
    """Return True if the hook denies the call."""
    payload = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return False
    out = json.loads(proc.stdout)
    return out["hookSpecificOutput"]["permissionDecision"] == "deny"


MUST_BLOCK_BASH = [
    "cat data/raw/train_labels.csv",
    "head -5 data/raw/train_labels.csv",
    "head -n 20 ./data/raw/niftis/xaji0y6d.nii.gz",
    "sed -n '1,10p' data/raw/train_labels.csv",
    "python -c \"import pandas as pd; print(pd.read_csv('data/raw/train_labels.csv').head())\"",
    "python -c \"import nibabel; print(nibabel.load('data/raw/niftis/a.nii.gz').get_fdata())\"",
    "strings data/raw/niftis/a.nii.gz",
    "xxd data/raw/niftis/a.nii.gz | head",
    "zcat data/raw/niftis/a.nii.gz",
    "cat D:/work/dat-parkinsons-challenge/data/raw/train_labels.csv",
    "cat data\\raw\\train_labels.csv",
    "awk -F, '{print $2}' data/raw/train_labels.csv",
    "cat reports/eda/raw/summary.json",
    "python -c \"import torch; print(torch.load('models/fold0.pt'))\"",
    "cat submission_src/assets/weights.safetensors",
    "ls data/raw | head && cat data/raw/train_labels.csv",
    "base64 data/raw/niftis/a.nii.gz",
    "cat submission_format.csv",
]

MUST_ALLOW_BASH = [
    "ls data/raw",
    "ls -la data/raw/niftis | wc -l",
    "find data/raw -name '*.nii.gz' | wc -l",
    "du -sh data/raw",
    "mkdir -p data/interim",
    "rm -rf data/interim/tmp",
    "git status",
    "git add data/.gitkeep",
    "cat src/datpd/data/io.py",
    "cat docs/00_competition_brief.md",
    "python scripts/01_inspect_data.py",
    "python scripts/03_train.py --config conf/baseline.yaml",
    "pytest -q",
    "cat reports/eda/shared/summary.md",
]

MUST_BLOCK_READ = [
    "data/raw/train_labels.csv",
    "D:/work/dat-parkinsons-challenge/data/raw/niftis/a.nii.gz",
    "reports/eda/raw/shapes.json",
    "models/fold0.ckpt",
    "submission_src/assets/model.safetensors",
    "./data/processed/cache.npy",
]

MUST_ALLOW_READ = [
    "src/datpd/train.py",
    "docs/00_competition_brief.md",
    "reports/eda/shared/summary.md",
    "reports/experiments/log.md",
    "conf/baseline.yaml",
    "submission_src/main.py",
]


@pytest.mark.parametrize("cmd", MUST_BLOCK_BASH)
def test_bash_blocked(cmd: str) -> None:
    assert run_hook("Bash", {"command": cmd}), f"should have been BLOCKED: {cmd}"


@pytest.mark.parametrize("cmd", MUST_ALLOW_BASH)
def test_bash_allowed(cmd: str) -> None:
    assert not run_hook("Bash", {"command": cmd}), f"should have been ALLOWED: {cmd}"


@pytest.mark.parametrize("path", MUST_BLOCK_READ)
def test_read_blocked(path: str) -> None:
    assert run_hook("Read", {"file_path": path}), f"should have been BLOCKED: {path}"


@pytest.mark.parametrize("path", MUST_ALLOW_READ)
def test_read_allowed(path: str) -> None:
    assert not run_hook("Read", {"file_path": path}), f"should have been ALLOWED: {path}"


def test_grep_into_data_blocked() -> None:
    assert run_hook("Grep", {"pattern": "1.0", "path": "data/raw"})


def test_grep_into_src_allowed() -> None:
    assert not run_hook("Grep", {"pattern": "def train", "path": "src"})


def test_malformed_input_fails_closed() -> None:
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input="not json at all",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
