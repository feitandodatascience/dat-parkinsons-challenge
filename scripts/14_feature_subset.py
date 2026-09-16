#!/usr/bin/env python3
"""Write a feature table restricted to an earlier version's columns, for a paired ablation.

Why this is needed
------------------
`src/datpd/features.py` grew from 61 columns to 157 on 2026-09-16, and the blend improved from
0.3038 to 0.2972 on the same CNN predictions and the same folds. That difference is the value of
the feature work — but it is a comparison between two *runs of the same procedure over different
inputs*, not a paired test, so it has no standard error. This project's own rule is to read sigma
rather than the point estimate, and an unpaired number cannot be read that way.

The extraction pass costs 97 minutes, so regenerating the old table is not the route. It is also
not necessary: the new features were added *beside* the originals rather than replacing them, so
every one of the 61 original columns still exists, with identical values, inside the 157-column
table. Selecting them reconstructs the old input exactly.

`scripts/09_blend.py --features <the file this writes>` then scores the old feature set through
the current code, and the two runs differ in nothing but the columns offered to the booster.

The column list is read from the OLD module rather than typed out here, because a hand-copied
list of 61 names that is wrong in one place produces a plausible number rather than an error:

    git show <commit>:src/datpd/features.py > old_features.py
    python scripts/14_feature_subset.py --from-module old_features.py

Usage:
    python scripts/14_feature_subset.py --from-module <path> [--out <csv>]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def names_from_module(path: Path) -> list[str]:
    """The feature order an earlier `features.py` produced, by running it.

    Derived from the module rather than from a written-down list, for the same reason
    `feature_names()` derives its order from a zero volume: two copies of an ordering drift, and
    the drift is invisible until a score fails to reproduce.
    """
    spec = importlib.util.spec_from_file_location("features_old", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import a module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.extract_features(np.zeros((96, 96, 96), dtype=np.float32)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-module", required=True,
                        help="path to the earlier features.py whose column set to select")
    parser.add_argument("--features", default="data/interim/sbr_features.csv")
    parser.add_argument("--out", default="data/interim/sbr_features_v1.csv")
    args = parser.parse_args()

    wanted = names_from_module(Path(args.from_module))
    table = pd.read_csv(REPO_ROOT / args.features)

    missing = [c for c in wanted if c not in table.columns]
    if missing:
        # A missing column means the new module renamed or dropped something rather than only
        # adding, which would make the ablation measure two changes at once instead of one.
        raise SystemExit(
            f"{len(missing)} of {len(wanted)} earlier columns are absent from the current table, "
            "so the current feature set is not a superset of the earlier one and this ablation "
            f"would not be like-for-like. First few: {missing[:5]}"
        )

    out_path = REPO_ROOT / args.out
    table[["uid", *wanted]].to_csv(out_path, index=False)
    print(f"{len(table)} rows, {len(wanted)} of {table.shape[1] - 1} feature columns kept")
    print(f"Wrote {args.out} - contains uids")
    print(f"\nNext: python scripts/09_blend.py --run baseline_tta --features {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
