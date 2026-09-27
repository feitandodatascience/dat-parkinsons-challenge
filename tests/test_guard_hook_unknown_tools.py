"""The guard hook must FAIL CLOSED for tools it does not recognise.

CLAUDE.md §0 item 6: *"The rule is about data reaching a model context, not about which tool
was used."* An allowlist of known tool names cannot satisfy that — it covers Bash, Read, Grep
and Glob, and says nothing about an MCP server connected next week.

Before this coverage existed, `_tool_is_blocked` ended in `return False` and the `PreToolUse`
matcher was `Bash|Read|Grep|Glob|NotebookEdit`. Connecting any MCP server able to read a file
was therefore a complete bypass of all three enforcement layers, silently. These tests are what
stop that regressing, and they are the reason the matcher is now `*`.

Kept in a separate module from `test_guard_hook.py` purely for a mechanical reason: appending to
that file requires a shell heredoc containing guarded extension literals, which the hook itself
(correctly, if bluntly) refuses to let Claude write.
"""

from __future__ import annotations

import pytest

from .test_guard_hook import HOOK, run_hook

pytestmark = pytest.mark.skipif(not HOOK.exists(), reason="data guard hook is local-only (.claude/)")

MUST_BLOCK_UNKNOWN_TOOL = [
    # A Kaggle server, whatever its argument names turn out to be.
    ("mcp__kaggle__dataset_download", {"dataset": "me/dat-niftis", "path": "data/raw/niftis"}),
    ("mcp__kaggle__read_file", {"file": "data/raw/train_labels.csv"}),
    ("mcp__kaggle__kernel_output", {"path": "./data/processed/cache.npy"}),
    ("mcp__gdrive__download", {"target": "models/fold0.ckpt"}),
    # Argument name we have never seen: the scan is over the whole serialised input.
    ("mcp__anything__odd_arg", {"whatever": "reports/eda/raw/shapes.json"}),
    # Exfiltration by other routes.
    (
        "WebFetch",
        {"url": "https://example.com/x", "prompt": "compare against data/raw/train_labels.csv"},
    ),
    ("Task", {"prompt": "read the volumes under data/raw/niftis and report the mean"}),
    ("SomeToolInventedNextYear", {"file_path": "submission_src/assets/model.safetensors"}),
]

MUST_ALLOW_UNKNOWN_TOOL = [
    # Pushing training code to a remote GPU is the whole point of connecting such a server.
    ("mcp__kaggle__kernel_push", {"path": "scripts/03_train.py"}),
    ("mcp__kaggle__list_gpus", {}),
    ("mcp__kaggle__kernel_status", {"kernel": "me/datpd-baseline"}),
    ("WebFetch", {"url": "https://www.drivendata.org/competitions/311/", "prompt": "the rules"}),
    ("SomeToolInventedNextYear", {"file_path": "reports/experiments/log.md"}),
]


@pytest.mark.parametrize("tool,payload", MUST_BLOCK_UNKNOWN_TOOL)
def test_unknown_tool_touching_guarded_material_is_blocked(tool: str, payload: dict) -> None:
    assert run_hook(tool, payload), f"should have been BLOCKED: {tool} {payload}"


@pytest.mark.parametrize("tool,payload", MUST_ALLOW_UNKNOWN_TOOL)
def test_unknown_tool_not_touching_data_is_allowed(tool: str, payload: dict) -> None:
    """Failing closed must not mean refusing everything, or the guard gets switched off."""
    assert not run_hook(tool, payload), f"should have been ALLOWED: {tool} {payload}"


class TestWriteIsJudgedOnPathNotContent:
    """Authoring pipeline source stays possible; writing into `data/` does not.

    Source files legitimately contain strings like the NIfTI suffix. Blocking a Write because
    its *content* mentions one would stop Claude writing the loader while protecting nothing —
    that content originates from Claude, not from the dataset.
    """

    def test_source_file_mentioning_a_guarded_suffix_is_allowed(self) -> None:
        assert not run_hook(
            "Write",
            {"file_path": "src/datpd/data/io.py", "content": "SUFFIX = '.nii' + '.gz'\n"},
        )

    def test_writing_into_the_dataset_is_blocked(self) -> None:
        assert run_hook(
            "Write", {"file_path": "data/raw/train_labels.csv", "content": "uid,label\n"}
        )
