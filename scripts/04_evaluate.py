#!/usr/bin/env python3
"""Score a run's out-of-fold predictions and choose a calibrator.

Writes an aggregate-only report to reports/experiments/<run>.md, which is safe for Claude to
read (metrics, no uids).

Usage:
    python scripts/04_evaluate.py --run baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.evaluate import evaluate_run  # noqa: E402
from datpd.train import assemble_oof  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    run_dir = REPO_ROOT / args.models_dir / args.run
    oof_path = run_dir / "oof.csv"

    if not oof_path.exists():
        # `03_train.py` writes oof.csv only after its whole fold loop finishes, but banks each
        # fold as it completes. A run killed partway therefore leaves usable predictions and no
        # oof.csv; assemble them rather than making the user retrain hours of work for a score.
        fold_files = sorted(run_dir.glob("oof_fold*.csv"))
        if not fold_files:
            raise SystemExit(
                f"No oof.csv and no completed folds in {run_dir}.\n"
                "Train first: python scripts/03_train.py --config conf/cpu_baseline.yaml"
            )
        folds = [p.stem.replace("oof_fold", "") for p in fold_files]
        print(f"No oof.csv found; assembling from completed fold(s): {', '.join(folds)}")
        assemble_oof(run_dir).to_csv(oof_path, index=False)
        print(
            "\nWARNING: this is a PARTIAL out-of-fold set. Treat the calibration temperature\n"
            "and clip below as provisional — refit them once every fold has finished, or you\n"
            "will ship constants tuned on a fraction of the data.\n"
        )

    evaluate_run(run_dir, REPO_ROOT / "reports" / "experiments")
    print("\nRemember to add a row to reports/experiments/log.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
