"""Tests for the submission packager.

Packaging failures are the most expensive kind in a code-execution competition: the feedback
loop runs through the platform and is hours long, and the failure modes are structural rather
than statistical - a missing module, an import that only exists locally, a constant that was
never substituted. All of those are cheap to catch here.

No competition data is involved: the packager only ever touches source files and, in these
tests, an empty `assets/`.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "05_package_submission.py"


@pytest.fixture(scope="module")
def packager():
    """Import the packaging script by path - its module name starts with a digit."""
    spec = importlib.util.spec_from_file_location("_packager", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _isolated_build(packager, monkeypatch, tmp_path):
    """Run every test against a disposable copy of `submission_src`, and build to a temp path.

    Two separate hazards, both learned from:

    **Output location.** A test run must not leave a zip at the real path. It would be a
    prior-only build carrying whatever placeholder constants these tests use, sitting exactly
    where a finished submission belongs — and the failure mode is uploading it by mistake and
    reading the resulting prior-level score as a modelling result.

    **Asset isolation.** These tests build in prior-only mode, which refuses to run when
    weights are present. Reading the real `submission_src/assets/` meant all seven tests
    silently skipped as soon as checkpoints were staged — that is, precisely when a submission
    is imminent and this coverage is most valuable. So they get their own empty assets
    directory and never consult the real one.
    """
    fake_src = tmp_path / "submission_src"
    (fake_src / "assets").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "submission_src" / "main.py", fake_src / "main.py")

    monkeypatch.setattr(packager, "SUBMISSION_SRC", fake_src)
    monkeypatch.setattr(packager, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(packager, "ZIP_PATH", tmp_path / "submission.zip")


class TestPriorOnlyBuild:
    """The day-1 build: a valid submission with no model, to prove the platform contract."""

    def test_produces_a_zip_with_everything_inference_needs(self, packager) -> None:
        assert packager.build({"FALLBACK_PROB": 0.37}, prior_only=True) == 0

        with zipfile.ZipFile(packager.ZIP_PATH) as zf:
            names = set(zf.namelist())

        # main.py must sit at the ROOT of the zip, not inside a directory.
        assert "main.py" in names
        for required in (
            "datpd_vendored/__init__.py",
            "datpd_vendored/data/io.py",
            "datpd_vendored/data/preprocess.py",
            "datpd_vendored/models/resnet3d.py",
        ):
            assert required in names, f"{required} missing from the zip"

    def test_a_build_with_no_weights_fails_unless_prior_only_is_explicit(self, packager) -> None:
        """Shipping a weightless zip by accident scores the prior and looks like a bad model."""
        assert packager.build({"FALLBACK_PROB": 0.37}, prior_only=False) == 1


class TestConstantBaking:
    """Nothing may be fitted, inferred or defaulted at inference (CLAUDE.md 1.1)."""

    def test_constants_are_substituted_into_main(self, packager) -> None:
        packager.build(
            {
                "FALLBACK_PROB": 0.37,
                "TEMPERATURE": 1.42,
                "TARGET_SHAPE": (128, 128, 128),
                "NORMALIZATION": "max",
            },
            prior_only=True,
        )
        text = (packager.BUILD_DIR / "main.py").read_text(encoding="utf-8")

        assert "FALLBACK_PROB = 0.37" in text
        assert "TEMPERATURE = 1.42" in text
        assert "TARGET_SHAPE = (128, 128, 128)" in text
        assert "NORMALIZATION = 'max'" in text

    def test_a_constant_that_does_not_exist_fails_loudly(self, packager) -> None:
        """A silently-skipped substitution ships the default and is invisible until scoring."""
        with pytest.raises(SystemExit):
            packager.build({"NO_SUCH_CONSTANT": 1.0}, prior_only=True)


class TestRuntimeCompatibility:
    def test_vendored_tree_imports_with_no_access_to_the_training_package(
        self, packager
    ) -> None:
        """The platform has `src/datpd` nowhere on its path. The zip must stand alone.

        Run in a subprocess whose working directory is the build dir and whose sys.path does
        not include this repo's `src/`, which is exactly the platform's situation.
        """
        # The subprocess inherits this interpreter, so the vendored modules can only import if
        # their third-party dependencies are installed here. On an environment without torch
        # this test would be asserting the environment, not the zip.
        pytest.importorskip("torch")
        pytest.importorskip("nibabel")

        packager.build({"FALLBACK_PROB": 0.37}, prior_only=True)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from datpd_vendored.data.preprocess import preprocess;"
                "from datpd_vendored.data.io import load_volume;"
                "from datpd_vendored.models.resnet3d import ResNet3D;"
                "print('ok')",
            ],
            cwd=packager.BUILD_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_no_import_outside_the_official_runtime(self, packager) -> None:
        """A missing import is a failed submission, not a warning (CLAUDE.md section 3)."""
        packager.build({"FALLBACK_PROB": 0.37}, prior_only=True)
        py_files = sorted(packager.BUILD_DIR.rglob("*.py"))
        assert py_files
        assert packager.check_imports(py_files) == []

    def test_independence_scan_passes_on_the_shipped_entry_point(self, packager) -> None:
        packager.build({"FALLBACK_PROB": 0.37}, prior_only=True)
        assert packager.check_independence(packager.BUILD_DIR / "main.py") == []
