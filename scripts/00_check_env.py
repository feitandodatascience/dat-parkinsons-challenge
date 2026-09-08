#!/usr/bin/env python3
"""Verify the local environment against the competition execution runtime.

Catches the class of problem that only shows up hours later as a failed submission: developing
on the wrong Python, or against a package that is not in the official image.

Usage:
    python scripts/00_check_env.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The runtime pins Python 3.12. Anything else risks building a solution that cannot be run.
REQUIRED_PYTHON = (3, 12)

# Packages the pipeline actually imports, mapped to their import names.
REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "nibabel", "yaml"]
OPTIONAL = ["torch", "torchio", "monai", "SimpleITK", "matplotlib", "timm", "loguru"]

GUARDED_DIRS = ["data", "reports/eda/raw", "models"]


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{f' - {detail}' if detail else ''}")
    return ok


def main() -> int:
    print("Python")
    version_ok = sys.version_info[:2] == REQUIRED_PYTHON
    check(
        f"version {sys.version_info.major}.{sys.version_info.minor}",
        version_ok,
        "" if version_ok else f"runtime requires {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}",
    )

    print("\nRequired packages")
    missing = [p for p in REQUIRED if importlib.util.find_spec(p) is None]
    for pkg in REQUIRED:
        check(pkg, importlib.util.find_spec(pkg) is not None)

    print("\nOptional packages")
    for pkg in OPTIONAL:
        check(pkg, importlib.util.find_spec(pkg) is not None, "" if importlib.util.find_spec(pkg) else "not installed")

    print("\nGPU")
    if importlib.util.find_spec("torch"):
        import torch

        available = torch.cuda.is_available()
        check(
            "CUDA",
            available,
            torch.cuda.get_device_name(0) if available else "no GPU - training will be very slow",
        )
    else:
        check("CUDA", False, "torch not installed")

    print("\nGuardrails")
    settings = REPO_ROOT / ".claude" / "settings.json"
    hook = REPO_ROOT / ".claude" / "hooks" / "guard_data.py"
    gitignore = REPO_ROOT / ".gitignore"
    ok = True
    ok &= check("permission deny rules present", settings.exists())
    ok &= check("data guard hook present", hook.exists())
    ok &= check(
        "data/ is git-ignored",
        gitignore.exists() and "data/" in gitignore.read_text(encoding="utf-8"),
    )

    print("\nData layout (existence only - contents are never read here)")
    layout_ok = True
    for rel in ["data/raw/niftis", "data/raw/train_labels.csv"]:
        path = REPO_ROOT / rel
        if path.exists():
            detail = f"{len(list(path.glob('*.nii.gz')))} volumes" if path.is_dir() else "present"
            check(rel, True, detail)
        else:
            layout_ok = False
            check(rel, False, "not at the expected path")

    if not layout_ok:
        # The download page serves files with a random suffix appended (train_labels_XXXXXXX.csv
        # and so on). Every config, script and doc in this repo refers to the canonical names,
        # so normalise once here rather than threading the download names through all of them.
        # Only entry NAMES are inspected, never contents - and never inside niftis/, whose
        # filenames are per-examination uids.
        raw = REPO_ROOT / "data" / "raw"
        strays = sorted(p.name for p in raw.iterdir()) if raw.is_dir() else []
        if strays:
            print("\n  data/raw currently holds:")
            for name in strays:
                print(f"    {name}")
            print("\n  Rename (or extract) these to the canonical layout:")
            print("    data/raw/niftis/            <- the volumes")
            print("    data/raw/train_labels.csv   <- the labels")
            print("\n  The smoke-test archive does NOT belong here - it goes in the runtime")
            print("  repo checkout, which is what `just test-submission` reads.")

    print()
    if missing or not version_ok:
        print("Environment is not ready. Fix the FAIL rows above.")
        return 1
    print("Environment looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
