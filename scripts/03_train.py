#!/usr/bin/env python3
"""Train the model with cross-validation.

Usage:
    python scripts/03_train.py --config conf/baseline.yaml
    python scripts/03_train.py --config conf/grouped_cv.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.config import load_config  # noqa: E402
from datpd.train import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/baseline.yaml")
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="override which folds to train, e.g. --folds 0 for a quick check")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    if args.folds is not None:
        cfg.train.folds = tuple(args.folds)

    print(f"Run: {cfg.run_name}  (config: {args.config})")
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
