"""Cross-validation folds, built so the CV number can be trusted.

The dataset comes from ten French hospitals but ships no centre column. Centre still leaks into
the images through acquisition geometry - voxel size and reconstruction matrix are properties of
the scanner and the local protocol, not of the patient. `acquisition_signature` turns those into
a fingerprint that stands in for centre.

Why it matters (docs/02 Delta 7): with a plain stratified split, scans from one centre land in
both train and validation. The model can then key on a centre-specific artefact - reconstruction
smoothness, field of view, count level - and score well on validation without having learned
anything about dopamine transporter binding. The leaderboard test set will contain centres, or at
least centre proportions, it has not seen, and the score collapses.

So build both splits and track both numbers:

    fold_strat  - stratified on the label. Lower variance, mildly optimistic.
    fold_group  - stratified AND grouped by signature. Higher variance, honest.

The *gap* between them is the diagnostic. A large gap means centre effects are real and
`fold_group` is the number that predicts the leaderboard.

WHAT THE SIGNATURE MUST NOT INCLUDE
-----------------------------------
The first version of this fingerprint used the full array shape and full-precision spacing. On
the real data (n=1362) that produced **134 groups, most of them singletons**, because two things
in the header vary per *patient* rather than per *centre*:

  - the **slice count**, which follows how much of the head was covered. One 3.9 mm scanner
    alone appeared as z = 33, 35, 36, ... 50: twenty "centres" that are one centre.
  - **full-precision spacing**, which carries reconstruction-zoom jitter (1.503, 1.509, 1.523,
    1.530 ... all from the same scanner family).

Near-unique groups make grouped CV *look* applied while protecting nothing - the folds end up
equivalent to stratified ones. Grouping on the stable part of the header instead (in-plane matrix
class, spacing to 0.1 mm) collapses those 134 into roughly a dozen real acquisition families.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

# Spacing is quantised to 0.1 mm. Coarse enough to absorb per-scan reconstruction jitter and
# header float noise, fine enough to keep genuinely different protocols apart (2.3 / 2.4 / 2.5 /
# 3.3 / 3.9 / 4.4 mm all survive as distinct families on the real data).
SPACING_DECIMALS = 1

# In-plane matrix sizes that identify a reconstruction protocol. Anything else is bucketed as
# "var": on the real data a single 2.46 mm scanner emits 118..178 in-plane depending on how the
# reconstruction was cropped, which is patient-level variation, not centre-level.
STANDARD_MATRICES = (64, 128, 256)


def acquisition_signature(shape: tuple[int, ...], spacing_mm: tuple[float, ...]) -> str:
    """A stable fingerprint of (reconstruction matrix class, voxel size): a proxy for centre.

    Deliberately ignores the slice count and quantises spacing - see the module docstring for
    what happens when it does not. Stable under header float noise and distinct between genuinely
    different protocols; both are asserted in `tests/test_pipeline_synthetic.py`.
    """
    in_plane = int(shape[0])
    matrix = str(in_plane) if in_plane in STANDARD_MATRICES else "var"
    spacing = "x".join(f"{float(s):.{SPACING_DECIMALS}f}" for s in spacing_mm)
    return f"{matrix}@{spacing}mm"


def _stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    folds = np.full(len(y), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[val_idx] = fold
    return folds


def _grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Stratified-and-grouped folds, with a fallback when stratification cannot be honoured.

    `StratifiedGroupKFold` keeps every group whole while trying to balance the label, which is
    what is wanted. It can still leave a fold empty when a few groups dominate; `GroupKFold`
    balances fold *sizes* instead and is the better failure mode, so fall back to it rather than
    silently returning a fold nobody validates on.

    Expect **uneven fold sizes here, and do not treat that as a bug.** On the real data the
    largest acquisition family is about a third of the dataset, so it lands whole in one fold.
    Unequal folds are the honest consequence of not splitting a centre; equal folds would mean
    the leakage is back.
    """
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(
            f"{n_groups} acquisition groups cannot fill {n_splits} grouped folds. "
            "Use fewer folds, or merge near-identical signatures."
        )

    folds = np.full(len(y), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups=groups)):
        folds[val_idx] = fold

    if len(np.unique(folds)) == n_splits and (folds >= 0).all():
        return folds

    folds = np.full(len(y), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(y, y, groups=groups)):
        folds[val_idx] = fold
    return folds


def group_size_report(df: pd.DataFrame, group_col: str = "acquisition_group") -> pd.DataFrame:
    """Per-group counts and positive rates, for checking whether centre confounds the label.

    Returns a frame; the caller decides what is safe to display. **Rows for tiny groups disclose
    individual labels** - a group of one with a 100% positive rate is that patient's diagnosis -
    so anything promoted to `reports/eda/shared/` must aggregate the small tail (CLAUDE.md 0.3).
    """
    grouped = df.groupby(group_col)["is_pathologic"]
    return pd.DataFrame({"n": grouped.size(), "positive_rate": grouped.mean()}).sort_values(
        "n", ascending=False
    )


def add_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    label_col: str = "is_pathologic",
    group_col: str = "acquisition_group",
) -> pd.DataFrame:
    """Attach `fold_strat` and `fold_group` columns. Does not mutate the input frame.

    Both are produced in one pass so every experiment can be scored under either regime without
    regenerating splits, which would make two runs incomparable.
    """
    out = df.copy().reset_index(drop=True)
    y = out[label_col].to_numpy()

    out["fold_strat"] = _stratified_folds(y, n_splits, seed)

    if group_col not in out.columns:
        raise KeyError(
            f"{group_col!r} missing; compute it with acquisition_signature() before splitting"
        )
    out["fold_group"] = _grouped_folds(y, out[group_col].to_numpy(), n_splits, seed)

    return out
