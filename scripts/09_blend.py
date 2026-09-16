#!/usr/bin/env python3
"""Fit a gradient-boosted model on the SBR features and blend it with the CNN.

The pipeline
------------
1. Cross-validate a gradient-boosted tree model over `data/interim/sbr_features.csv`, using the
   SAME fold assignments the CNN used. That matters: it makes the tree model's out-of-fold
   predictions row-for-row comparable with the CNN's, so the two can be combined without either
   having seen the other's validation rows.
2. Combine the components in logit space with a logistic stack, fitted OUT OF SAMPLE.
3. Compare against the same stack over the CNN alone, so recalibration is not counted as a gain.

Everything here is honest-by-construction or it is worthless
------------------------------------------------------------
On 2026-09-08 this project came within an hour of shipping an isotonic calibrator that had been
scored on its own training data (see reports/experiments/log.md). Blending offers the same trap
one level up: choose the mixing weight to minimise pooled OOF log loss, then quote that log
loss, and the number is optimistic by construction — it is the minimum of a curve fitted to the
data it is quoted on. So the stack is fitted on k-1 folds and scored on the held-out one.

There is a second, subtler trap that cost a rewrite of this script. A convex blend
`w*z_cnn + (1-w)*z_gbm` does not only mix the two models — it also *shrinks* the CNN's logits,
which is temperature scaling by another name. Against an overconfident CNN, `w < 1` lowers log
loss even when the second model is pure noise, so the blend appears to work when nothing has
been added. The fix is to fit a coefficient per component with an intercept, and to compare
against a stack over the CNN alone rather than against the raw CNN. Both sides then get the
same recalibration and the difference is attributable to the features.

**Hyperparameter tuning is the same trap a third level up.** Choosing the booster's
hyperparameters on the folds you then report is exactly "the minimum of a curve fitted to the
data it is quoted on", with the curve now indexed by a grid instead of by a weight. `--tune`
therefore selects hyperparameters INSIDE each outer fold, using only that fold's training rows,
and the report states which nesting it used and whether the quoted number is optimistic.

**Choosing between tree variants is the trap a fourth level up**, and no amount of nesting
removes it, because the choice is made by a human reading the table. The defence is a rule
fixed in advance: candidates are ordered by increasing complexity and a more complex one is
accepted only if it beats the currently accepted one by more than `--margin` with a bootstrap
interval that excludes zero. "Whichever number is lowest" is not the rule.

One more way this can lie to you
--------------------------------
The tree model is cross-validated on the CNN's fold assignment, so its out-of-fold predictions
carry a trace of each fold's *training* base rate — which, for arbitrary folds, is
anti-correlated with that fold's own base rate. A tree model fitted on pure noise can therefore
pick up a small genuine-looking signal, and the stack will happily give it a negative
coefficient and report a gain from nothing. `fold_strat` is stratified, which equalises the
base rates and closes this route; `tests/test_sbr_blend.py` reproduces the artifact with
unstratified folds. **Do not run this with a fold column that is not stratified on the label.**

N components
------------
`--extra-runs` adds further prediction sets (any run directory under `models/` holding an
`oof.csv` — a second TTA pass, another architecture, an ensemble) as extra stack columns. The
design is unchanged: one coefficient per component plus an intercept, scored against a stack
over the CNN alone. What decides whether a new column earns its place is the pairwise logit
correlation: this project's durable heuristic is that the ensemble gain scales with (1 - rho),
so all pairwise correlations are printed alongside the fitted coefficients.

**A stack over more than two components cannot be expressed in the three constants
`submission_src/main.py` bakes** (`STACK_INTERCEPT`, `STACK_COEF_CNN`, `STACK_COEF_GBM`). When
extra components are supplied, the shipped two-component `stack.json` is still written from the
CNN + trees fit, and the N-component fit is written beside it as `stack_ncomp.json`, flagged as
not shippable. Extending the inference path is a separate change to a file this script does not
own.

Shipping this, if it pays
-------------------------
`main.py` computes the same features per scan (they are per-sample by construction, so the
independence rule holds) and carries the fitted boosters. The boosters go out in lightgbm's text
format, which loads across library versions, rather than as a scikit-learn pickle whose
compatibility with the runtime's version is not guaranteed.

Usage:
    python scripts/09_blend.py --run baseline_tta
    python scripts/09_blend.py --run baseline_tta --tune
    python scripts/09_blend.py --run baseline_tta --gbm-bag 5
    python scripts/09_blend.py --run baseline_tta --extra-runs baseline_tta2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datpd.utils.metrics import clip_probs, competition_logloss, summarize  # noqa: E402

# lightgbm, not scikit-learn's HistGradientBoosting, even though the two implement much the
# same algorithm and sklearn was already installed. The reason is shipping: lightgbm's booster
# serialises to a plain text format that loads across versions, whereas a scikit-learn estimator
# ships as a pickle whose compatibility with whatever version the runtime happens to have is not
# guaranteed. Measuring with one model and shipping another would also mean the number that
# justified the change was never the number that ships.
#
# Deliberately small trees and strong regularisation. n=1362 with a few dozen features is a
# regime where a boosted model will happily memorise; the goal here is a model that DISAGREES
# usefully with the CNN, not one that squeezes the last drop out of the features.
GBM_PARAMS = dict(
    max_depth=3,
    num_leaves=7,
    n_estimators=300,
    learning_rate=0.05,
    min_child_samples=30,
    reg_lambda=1.0,
    subsample=1.0,
    colsample_bytree=0.8,
    verbose=-1,
)

# A short, curated search rather than a cartesian product. At n~1360 the inner-fold estimate of
# log loss has a standard error around 0.02, so a grid of hundreds of configurations would
# mostly be selecting noise — the winner of a large grid is the luckiest entry, not the best
# one. Twelve entries that differ in *kind* (capacity, shrinkage, bagging, feature sampling)
# span the space that matters here and keep the selection noise floor manageable.
#
# Nothing here is expressed in terms of the number of features: `colsample_bytree` is a
# fraction and `min_child_samples` counts rows. The feature module is still growing, so a
# hard-coded feature count would silently go stale.
TUNE_GRID: tuple[tuple[str, dict], ...] = (
    ("current-default", {}),
    ("stumps-slow-long", dict(max_depth=1, num_leaves=2, n_estimators=800, learning_rate=0.05)),
    ("shallow-long", dict(max_depth=2, num_leaves=3, n_estimators=600, learning_rate=0.05)),
    ("default-slow-long", dict(n_estimators=800, learning_rate=0.02)),
    ("default-fast-short", dict(n_estimators=150, learning_rate=0.10)),
    ("deeper", dict(max_depth=4, num_leaves=15, n_estimators=300)),
    ("deeper-regularised", dict(max_depth=4, num_leaves=15, n_estimators=600,
                                learning_rate=0.03, min_child_samples=40, reg_lambda=10.0)),
    ("leaves31-lgb-default", dict(max_depth=-1, num_leaves=31, n_estimators=200,
                                  min_child_samples=20)),
    ("bagged-rows", dict(n_estimators=400, subsample=0.7, subsample_freq=1,
                         colsample_bytree=0.5)),
    ("sparse-columns", dict(n_estimators=400, colsample_bytree=0.3)),
    ("low-min-child", dict(n_estimators=400, min_child_samples=10)),
    ("high-min-child", dict(n_estimators=400, min_child_samples=60, reg_lambda=5.0)),
)
GRID_BY_NAME = dict(TUNE_GRID)

# Seed offsets for `--gbm-bag`. FIXED CONSTANTS, never drawn from a clock: a bagged result that
# cannot be reproduced is not a result. Extend the tuple to allow a larger S.
BAG_SEED_OFFSETS: tuple[int, ...] = (0, 1009, 2017, 3023, 4027, 5039, 6047, 7057, 8069, 9091)

# Swept on the final stacked predictions and reported. An equivalent sweep for the CNN alone is
# in reports/experiments/baseline.md, where everything from 1e-6 to 5e-3 tied to four decimals.
CLIP_GRID: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2)


class GbmOof(NamedTuple):
    """Out-of-fold predictions, the fold models that produced them, and what they were tuned to.

    A NamedTuple rather than a bare tuple because `models` is now a list PER FOLD (seed bagging
    puts several boosters in each) and the tuning record only exists under `--tune`; callers
    that want only the predictions should not have to know any position but `.oof`.
    """

    oof: np.ndarray
    models: list[list[lgb.LGBMClassifier]]
    chosen: list[str]

    @property
    def n_boosters(self) -> int:
        return sum(len(fold) for fold in self.models)


class Paired(NamedTuple):
    """The decision statistics for one configuration against another."""

    mean: float
    se: float
    lo: float
    hi: float
    improved: float
    degenerate: bool

    @property
    def sigma_text(self) -> str:
        return "n/a" if self.degenerate else f"{abs(self.mean) / self.se:.1f}σ"

    def resolved(self) -> bool:
        """True when the bootstrap interval excludes zero."""
        return not self.degenerate and not (self.lo <= 0.0 <= self.hi)

    def beats(self, margin: float) -> bool:
        """The acceptance rule, fixed in advance: clear the margin AND exclude zero."""
        return self.mean < -margin and self.resolved()


def to_logit(p: np.ndarray) -> np.ndarray:
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


def from_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def resolve_params(override: dict | None = None) -> dict:
    """The default booster parameters with one grid entry applied on top."""
    params = dict(GBM_PARAMS)
    params.update(override or {})
    return params


def select_params(
    x: pd.DataFrame, y: np.ndarray, seed: int, mode: str, inner_splits: int
) -> tuple[int, list[float]]:
    """Choose a grid entry using ONLY the rows handed in. Returns (index, per-entry loss).

    This function never sees the outer held-out fold. That is the whole point: it is called
    from inside the outer loop with that fold's training rows, so the hyperparameter choice is
    part of the fitting procedure being cross-validated rather than a decision made on the
    numbers the script goes on to report.

    `mode="nested"` scores each entry by an inner k-fold over those rows; `mode="inner"` uses a
    single inner holdout, which is k times cheaper and correspondingly noisier in its choice.
    Both are leak-free — they differ only in how well they pick, never in whether the reported
    score is honest.
    """
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(x, y))
    if mode == "inner":
        splits = splits[:1]

    losses: list[float] = []
    for _, override in TUNE_GRID:
        fold_losses = []
        for train, test in splits:
            model = lgb.LGBMClassifier(random_state=seed, **resolve_params(override))
            model.fit(x.iloc[train], y[train])
            fold_losses.append(
                competition_logloss(y[test], model.predict_proba(x.iloc[test])[:, 1])
            )
        losses.append(float(np.mean(fold_losses)))
    return int(np.argmin(losses)), losses


def fit_gbm_oof(
    features: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    seed: int,
    tune: bool = False,
    tune_mode: str = "nested",
    inner_splits: int = 3,
    n_bag: int = 1,
    preselected: list[str] | None = None,
) -> GbmOof:
    """Out-of-fold predictions from the tree model, on the CNN's own fold assignment.

    Returns the fitted fold models as well, because those are what ship. Inference averages them
    in logit space, mirroring exactly what `main.py` already does with the CNN's fold
    checkpoints — and keeping the shipped ensemble the same shape as the one whose out-of-fold
    predictions the stack coefficients were fitted on.

    `n_bag > 1` fits several boosters per fold under different FIXED seeds and averages them in
    logit space **within the fold**, so no information crosses a fold boundary. Boosted trees at
    this n are a high-variance estimator and log loss pays directly for variance reduction; the
    members differ through lightgbm's per-tree feature sampling (`colsample_bytree < 1`) and row
    bagging where a configuration enables it. Averaging in logit space, not probability space,
    matches how folds and TTA views are already combined and stops one saturated member from
    dragging the mean.

    Under `--tune` each fold model is tuned on its own training rows, so the fold models may
    carry different parameter sets. That is deliberate: the shipped ensemble is then bit-for-bit
    the procedure whose out-of-fold score is quoted. Refitting one "best" parameter set on all
    the data afterwards would ship a model that was never measured. `preselected` replays an
    earlier run's per-fold choices so a second variant (bagged, say) can reuse them instead of
    paying for the search twice.
    """
    if n_bag > len(BAG_SEED_OFFSETS):
        raise SystemExit(
            f"--gbm-bag {n_bag} exceeds the {len(BAG_SEED_OFFSETS)} fixed seed offsets; "
            "add more constants to BAG_SEED_OFFSETS rather than generating them."
        )

    oof = np.full(len(y), np.nan)
    models: list[list[lgb.LGBMClassifier]] = []
    chosen: list[str] = []
    for position, fold in enumerate(sorted(np.unique(folds))):
        train = folds != fold
        x_train = features[train].reset_index(drop=True)
        y_train = y[train]

        if preselected is not None:
            name = preselected[position]
            override = GRID_BY_NAME[name]
        elif tune:
            index, _ = select_params(x_train, y_train, seed, tune_mode, inner_splits)
            name, override = TUNE_GRID[index]
        else:
            name, override = "current-default", {}

        params = resolve_params(override)
        members = []
        member_logits = []
        for offset in BAG_SEED_OFFSETS[:n_bag]:
            model = lgb.LGBMClassifier(random_state=seed + offset, **params)
            model.fit(x_train, y_train)
            members.append(model)
            member_logits.append(to_logit(model.predict_proba(features[~train])[:, 1]))

        oof[~train] = from_logit(np.mean(member_logits, axis=0))
        models.append(members)
        chosen.append(name)
    assert not np.isnan(oof).any(), "every row must be predicted exactly once"
    return GbmOof(oof, models, chosen)


def bagging_is_a_no_op(chosen: list[str]) -> bool:
    """True when every chosen configuration is deterministic given its data.

    lightgbm only consults the seed when it has a random choice to make. With
    `colsample_bytree = 1` and no row bagging the members of a bag would be identical boosters,
    and averaging them would be an expensive way to compute one of them. Worth saying out loud
    rather than reporting a difference of exactly zero and leaving the reader to wonder.
    """
    for name in chosen:
        params = resolve_params(GRID_BY_NAME.get(name, {}))
        samples_rows = params.get("subsample", 1.0) < 1.0 and params.get("subsample_freq", 0) > 0
        if params.get("colsample_bytree", 1.0) < 1.0 or samples_rows:
            return False
    return True


def fit_stack(logits: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit a logistic regression on component logits — standard stacking.

    Why not a convex weight `w * z_cnn + (1-w) * z_gbm`, which is the obvious thing:
    **it conflates mixing with calibration.** Lowering `w` both moves weight onto the second
    model AND shrinks the first model's logits toward zero. If the CNN is overconfident — it is;
    that is why temperature scaling helped — then `w < 1` reduces log loss through shrinkage
    alone, and a *pure noise* second model scores an apparent gain. That failure is reproduced
    in tests/test_sbr_blend.py.

    Fitting a coefficient per component with an intercept separates the two effects: the
    coefficient on the CNN absorbs the calibration, so the coefficients on the other components
    measure only what they add. It also makes the comparison a clean nested-model test — two
    parameters for the CNN alone against N+1 for the full set — with both sides cross-validated.
    """
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(logits, y)
    return model


def stack_predict(model: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    return model.predict_proba(logits)[:, 1]


def cv_score(logits: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Cross-validated held-out predictions from a stack over these columns.

    Used for every variant — CNN alone (one column), CNN + trees (two), CNN + trees + extras
    (N) — so the numbers differ only by the information available, never by how they were
    scored.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.full(len(y), np.nan)
    for train, test in splitter.split(logits, y):
        out[test] = stack_predict(fit_stack(logits[train], y[train]), logits[test])
    assert not np.isnan(out).any()
    return out


def per_sample_logloss(y: np.ndarray, p: np.ndarray, eps: float) -> np.ndarray:
    p = clip_probs(p, eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_stats(
    y: np.ndarray,
    p_reference: np.ndarray,
    p_variant: np.ndarray,
    eps: float,
    n_boot: int,
    seed: int,
) -> Paired:
    """Per-scan paired comparison, the same shape scripts/11_compare_runs.py reports.

    Paired because both sides predict the *same scans* from the *same folds*. An unpaired
    standard error at this n is around 0.016 and would call every change here unresolvable;
    the standard error of the per-scan difference is an order of magnitude smaller.

    No temperature scaling on either side, unlike 11_compare_runs.py: every configuration
    compared here is already the output of a cross-validated logistic stack, which carries its
    own calibration. Applying a temperature on top would be applying one twice.
    """
    diff = per_sample_logloss(y, p_variant, eps) - per_sample_logloss(y, p_reference, eps)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    if se == 0.0:
        return Paired(mean, 0.0, 0.0, 0.0, 0.0, degenerate=True)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return Paired(
        mean=mean,
        se=se,
        lo=float(np.percentile(means, 2.5)),
        hi=float(np.percentile(means, 97.5)),
        improved=float((diff < 0).mean()),
        degenerate=False,
    )


def correlation_rows(names: list[str], logits: np.ndarray) -> list[str]:
    """Every pairwise logit correlation, as markdown rows.

    (1 - rho) is how this project decides whether a component earns its place: the trees bought
    -0.0175 at rho = 0.80, and a second CNN run bought -0.0022 at rho = 0.974, which the
    heuristic predicted to within 0.0002 before it was measured.
    """
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho = float(np.corrcoef(logits[:, i], logits[:, j])[0, 1])
            rows.append(f"| {names[i]} | {names[j]} | {rho:.3f} | {1.0 - rho:.3f} |")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="baseline", help="run dir holding the CNN's oof.csv")
    parser.add_argument(
        "--extra-runs",
        nargs="*",
        default=[],
        help="further run dirs under models/ whose oof.csv becomes an extra stack column "
             "(e.g. a second, differently-augmented TTA pass). Each adds one coefficient.",
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--features", default="data/interim/sbr_features.csv")
    parser.add_argument("--folds-csv", default="data/interim/folds.csv")
    parser.add_argument("--fold-column", default="fold_strat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--clip-eps", type=float, default=1e-6,
                        help="clip used for every reported score; also the sweep's reference")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="bootstrap resamples for each paired comparison")
    parser.add_argument(
        "--ship",
        choices=("default", "bagged", "tuned", "tuned+bagged"),
        default=None,
        help="OVERRIDE the acceptance rule and ship this tree configuration regardless. For a "
             "free final submission only; the report states that the rule was overridden.",
    )
    parser.add_argument(
        "--out-run",
        default=None,
        help="name for the output run dir and report (default: <run>_blend). Use it to build a "
             "variant without overwriting artifacts a shipped build came from.",
    )
    parser.add_argument("--margin", type=float, default=0.002,
                        help="gain below which a change is not worth the complexity it adds")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="search TUNE_GRID for the booster's hyperparameters, INSIDE each outer fold",
    )
    parser.add_argument(
        "--tune-mode",
        choices=("nested", "inner"),
        default="nested",
        help="nested: inner k-fold over each outer fold's training rows (honest, slower). "
             "inner: a single inner holdout of those rows (honest, noisier choice, k times "
             "cheaper). Neither ever touches the outer held-out fold.",
    )
    parser.add_argument("--tune-inner-splits", type=int, default=3)
    parser.add_argument(
        "--gbm-bag",
        type=int,
        default=1,
        help="boosters per fold, fitted under fixed distinct seeds and averaged in logit space "
             "INSIDE the fold. 1 (default) is the current behaviour. S > 1 ships S times as "
             "many booster files.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    oof_path = REPO_ROOT / args.models_dir / args.run / "oof.csv"
    features_path = REPO_ROOT / args.features
    for path, hint in (
        (oof_path, "python scripts/03_train.py --config conf/baseline.yaml"),
        (features_path, "python scripts/08_sbr_features.py --config conf/baseline.yaml"),
    ):
        if not path.exists():
            raise SystemExit(f"Missing {path}.\nRun: {hint}")

    oof = pd.read_csv(oof_path)
    feats = pd.read_csv(features_path)
    folds = pd.read_csv(REPO_ROOT / args.folds_csv)[["uid", args.fold_column]]

    df = oof.merge(feats, on="uid", how="inner").merge(folds, on="uid", how="left")
    if df[args.fold_column].isna().any():
        raise SystemExit("some rows have no fold assignment; regenerate folds")
    if len(df) < len(oof):
        print(f"NOTE: {len(oof) - len(df)} CNN rows have no features and were dropped.")

    # Extra prediction sets, merged on uid. An inner merge keeps the comparison paired: every
    # configuration below is scored on exactly the same scans, or the differences would not be
    # differences.
    extra_columns: list[str] = []
    for i, run in enumerate(args.extra_runs):
        path = REPO_ROOT / args.models_dir / run / "oof.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Has run '{run}' been scored?")
        column = f"extra{i}_pred"
        extra = pd.read_csv(path)[["uid", "is_pathologic", "oof_pred"]].rename(
            columns={"oof_pred": column, "is_pathologic": "label_check"}
        )
        before = len(df)
        df = df.merge(extra, on="uid", how="inner")
        if len(df) < before:
            print(f"NOTE: run '{run}' covers {before - len(df)} fewer scans; the rest were dropped.")
        if not (df["label_check"] == df["is_pathologic"]).all():
            raise SystemExit(f"run '{run}' disagrees about the labels; check the fold assignment.")
        df = df.drop(columns=["label_check"])
        extra_columns.append(column)

    feature_columns = [c for c in feats.columns if c != "uid"]
    x = df[feature_columns]
    y = df["is_pathologic"].to_numpy()
    fold_ids = df[args.fold_column].to_numpy()

    print(f"n = {len(df)}, features = {len(feature_columns)}, folds = {args.fold_column}")

    # --- Tree candidates ----------------------------------------------------------------------
    # Ordered by increasing complexity; the acceptance chain below walks them in this order.
    # `default` is the shipped configuration and the reference every paired difference is
    # measured against.
    trees: dict[str, GbmOof] = {"default": fit_gbm_oof(x, y, fold_ids, args.seed)}
    labels = {"default": "trees (default)"}

    if args.gbm_bag > 1:
        trees["bagged"] = fit_gbm_oof(x, y, fold_ids, args.seed, n_bag=args.gbm_bag)
        labels["bagged"] = f"trees (bagged x{args.gbm_bag})"
        if bagging_is_a_no_op(trees["bagged"].chosen):
            print(
                "NOTE: every chosen configuration samples neither rows nor columns, so the bag "
                "members are identical boosters and bagging cannot change anything."
            )

    if args.tune:
        print(
            f"Tuning {len(TUNE_GRID)} configurations per outer fold, mode = {args.tune_mode}. "
            "This fits a lot of boosters; expect minutes, not seconds."
        )
        started = time.time()
        trees["tuned"] = fit_gbm_oof(
            x, y, fold_ids, args.seed,
            tune=True, tune_mode=args.tune_mode, inner_splits=args.tune_inner_splits,
        )
        labels["tuned"] = "trees (tuned)"
        print(f"Tuning finished in {time.time() - started:.0f}s.")
        if args.gbm_bag > 1:
            # Replay the per-fold choices rather than searching again: the selection saw only
            # training rows either way, and repeating it would cost the same minutes to reach
            # the same answer.
            trees["tuned+bagged"] = fit_gbm_oof(
                x, y, fold_ids, args.seed,
                n_bag=args.gbm_bag, preselected=trees["tuned"].chosen,
            )
            labels["tuned+bagged"] = f"trees (tuned, bagged x{args.gbm_bag})"

    # --- Components ----------------------------------------------------------------------------
    p_cnn = df["oof_pred"].to_numpy()
    component_names = [f"CNN ({args.run})"]
    component_probs = [p_cnn]
    tree_col: dict[str, int] = {}
    for key, fitted in trees.items():
        tree_col[key] = len(component_probs)
        component_names.append(f"SBR {labels[key]}")
        component_probs.append(fitted.oof)
    extra_cols = []
    for run, column in zip(args.extra_runs, extra_columns):
        extra_cols.append(len(component_probs))
        component_names.append(f"CNN ({run})")
        component_probs.append(df[column].to_numpy())

    z = np.column_stack([to_logit(p) for p in component_probs])
    cnn_col = 0

    # --- Variants ------------------------------------------------------------------------------
    # Every variant is the same estimator on the same folds; they differ only in which columns
    # are available. The CNN-alone row is the floor: it gets the same recalibration as the rest,
    # so anything above it is attributable to information rather than to temperature scaling.
    variants: list[tuple[str, list[int]]] = []
    predictions: dict[str, np.ndarray] = {}

    def score(name: str, cols: list[int]) -> np.ndarray:
        if name not in predictions:
            predictions[name] = cv_score(z[:, cols], y, args.n_splits, args.seed)
            variants.append((name, cols))
        return predictions[name]

    score("CNN alone", [cnn_col])
    for key in trees:
        score(f"CNN + {labels[key]}", [cnn_col, tree_col[key]])
    if extra_cols:
        score("CNN + extras, no trees", [cnn_col, *extra_cols])
        score("CNN + trees (default) + extras", [cnn_col, tree_col["default"], *extra_cols])

    reference_name = "CNN + trees (default)"

    # --- The acceptance chain, fixed in advance ------------------------------------------------
    # Walk the candidates from simplest to most complex and promote one only when it beats the
    # incumbent by more than the margin with an interval excluding zero. Comparing every
    # candidate against the default and taking the best would be a multiple-comparison problem
    # dressed up as a result.
    accepted = "default"
    chain: list[tuple[str, str, Paired, bool]] = []
    for key in ("bagged", "tuned", "tuned+bagged"):
        if key not in trees:
            continue
        challenger = paired_stats(
            y,
            predictions[f"CNN + {labels[accepted]}"],
            predictions[f"CNN + {labels[key]}"],
            args.clip_eps, args.n_boot, args.seed,
        )
        promote = challenger.beats(args.margin)
        chain.append((key, accepted, challenger, promote))
        if promote:
            accepted = key

    # An explicit override, for the one situation the rule was not written for: a submission slot
    # that is worth nothing if unused. The rule protects a scarce slot from noise; it does not say
    # a free slot should stay empty. Overriding is allowed, but never silently.
    rule_choice = accepted
    if args.ship is not None:
        if args.ship not in trees:
            raise SystemExit(f"--ship {args.ship}: that configuration was not fitted on this run")
        accepted = args.ship

    ship_name = f"CNN + {labels[accepted]}"
    ship_cols = [cnn_col, tree_col[accepted]]
    ship_gbm = trees[accepted]
    if extra_cols and accepted != "default":
        score(f"CNN + {labels[accepted]} + extras", [cnn_col, tree_col[accepted], *extra_cols])

    losses = {
        name: competition_logloss(y, predictions[name], args.clip_eps) for name, _ in variants
    }
    stats = {
        name: paired_stats(
            y, predictions[reference_name], predictions[name],
            args.clip_eps, args.n_boot, args.seed,
        )
        for name, _ in variants
    }

    ship_pred = predictions[ship_name]

    # --- Clip sweep, on the configuration that would ship --------------------------------------
    clip_losses = {eps: competition_logloss(y, ship_pred, eps) for eps in CLIP_GRID}
    best_eps = min(clip_losses, key=lambda e: clip_losses[e])

    # --- Artifacts -----------------------------------------------------------------------------
    out_name = args.out_run or f"{args.run}_blend"
    blend_dir = REPO_ROOT / args.models_dir / out_name
    blend_dir.mkdir(parents=True, exist_ok=True)

    # Bank the held-out predictions of the shipped configuration as a run of its own, so the
    # improvement can also be put through scripts/11_compare_runs.py against any other run.
    pd.DataFrame(
        {"uid": df["uid"], "is_pathologic": y, "oof_pred": ship_pred, "fold": fold_ids}
    ).to_csv(blend_dir / "oof.csv", index=False)

    # Stale boosters from a previous, differently-shaped run would be picked up by the packager's
    # `gbm_fold*.txt` glob and silently averaged in alongside this run's.
    for stale in blend_dir.glob("gbm_fold*.txt"):
        stale.unlink()
    n_bag_shipped = len(ship_gbm.models[0])
    for f, members in enumerate(ship_gbm.models):
        for s, model in enumerate(members):
            name = f"gbm_fold{f}.txt" if n_bag_shipped == 1 else f"gbm_fold{f}_bag{s}.txt"
            model.booster_.save_model(str(blend_dir / name))

    ship_stack = fit_stack(z[:, ship_cols], y)
    stack_meta = {
        "stack_format": "two-component-v1",
        "intercept": float(ship_stack.intercept_[0]),
        "coef_cnn": float(ship_stack.coef_[0][0]),
        "coef_gbm": float(ship_stack.coef_[0][1]),
        # Stored so inference can assert the column order it builds matches the one the trees
        # were fitted on. A silently permuted feature vector is the failure mode here: it raises
        # nothing, and produces predictions that are wrong in a way no check would catch. The
        # list is read off the feature file, never hard-coded, so a growing feature module needs
        # no change here.
        "feature_names": feature_columns,
        "n_features": len(feature_columns),
        "n_folds": len(ship_gbm.models),
        "n_bag": n_bag_shipped,
        "n_boosters": ship_gbm.n_boosters,
        "source_run": args.run,
        "tree_variant": accepted,
        "tuned": accepted.startswith("tuned"),
        "tune_mode": args.tune_mode if accepted.startswith("tuned") else None,
        "gbm_params_per_fold": ship_gbm.chosen,
    }
    (blend_dir / "stack.json").write_text(json.dumps(stack_meta, indent=2), encoding="utf-8")

    # The N-component fit, when there is one. Written to a DIFFERENT file: `main.py` bakes three
    # constants and would need a fourth to carry this, so silently writing N coefficients into
    # `stack.json` would either be ignored or mis-read. Kept beside the shipped file so the
    # numbers are available to whoever extends the inference path.
    ncomp_path = None
    if extra_cols:
        full_cols = [cnn_col, tree_col[accepted], *extra_cols]
        full_stack = fit_stack(z[:, full_cols], y)
        ncomp_path = blend_dir / "stack_ncomp.json"
        ncomp_path.write_text(
            json.dumps(
                {
                    "stack_format": "n-component-v1",
                    "shippable_by_current_main": False,
                    "note": (
                        "submission_src/main.py bakes STACK_INTERCEPT, STACK_COEF_CNN and "
                        f"STACK_COEF_GBM only. This fit has {len(full_cols)} coefficients plus "
                        "an intercept and cannot be expressed in those three."
                    ),
                    "variant": f"{ship_name} + extras",
                    "intercept": float(full_stack.intercept_[0]),
                    "components": [
                        {"name": component_names[c], "coef": float(w)}
                        for c, w in zip(full_cols, full_stack.coef_[0])
                    ],
                    "feature_names": feature_columns,
                    "n_features": len(feature_columns),
                    "n_folds": len(ship_gbm.models),
                    "source_run": args.run,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    in_sample = competition_logloss(
        y, stack_predict(ship_stack, z[:, ship_cols]), args.clip_eps
    )
    coef_cnn, coef_gbm = (float(c) for c in ship_stack.coef_[0])
    share_gbm = abs(coef_gbm) / max(abs(coef_cnn) + abs(coef_gbm), 1e-9)
    cnn_cal_loss = losses["CNN alone"]
    honest_cal = losses[ship_name]

    # --- Report ---------------------------------------------------------------------------------
    summaries = [summarize(y, p) for p in component_probs]
    lines = [
        f"# Blend: CNN ({args.run}) + SBR gradient boosting",
        "",
        f"- n = {len(df)}, features = {len(feature_columns)}, fold column = `{args.fold_column}`",
        f"- components available to the stack: {len(component_names)}",
        f"- scores clipped at {args.clip_eps:g}; bootstrap over {args.n_boot:,} resamples",
        "",
        "## Components, out of fold",
        "",
        "| model | log loss | AUROC | ECE |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {name} | {s['logloss']:.4f} | {s['auroc']:.4f} | {s['ece']:.4f} |"
        for name, s in zip(component_names, summaries)
    ]
    lines += [
        "",
        "### Pairwise logit correlations",
        "",
        "| a | b | rho | 1 - rho |",
        "|---|---|---:|---:|",
        *correlation_rows(component_names, z),
        "",
        "A low correlation is the point. Two models that agree everywhere cannot help each",
        "other; a component is worth something precisely to the extent it disagrees. The",
        "governing measurement: the trees bought -0.0175 at rho = 0.80, and a second CNN run",
        "bought -0.0022 at rho = 0.974 — a ratio that (1 - rho) predicted in advance. Read the",
        "last column before reading the coefficients.",
        "",
        "## Variants",
        "",
        "Every row is a logistic stack — one coefficient per component plus an intercept —",
        f"scored by the same {args.n_splits}-fold cross-validation. Rows differ only in which",
        "columns were offered, so a difference measures information added, not a recalibration",
        "the CNN would have got on its own.",
        "",
        f"The paired difference is against **{reference_name}**, the shipped configuration.",
        "Negative favours the variant. Both sides are already stack-calibrated out of fold, so",
        "unlike `11_compare_runs.py` no temperature is applied here — that would apply one twice.",
        "",
        "The score column is the **cross-validated log loss** of that stack; it is held out at",
        "every level, so it is not the minimum of a curve fitted to the rows it is quoted on.",
        "",
        "| stack over | params | CV log loss | Δ vs shipped | σ | 95% CI | scans improved |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for name, cols in variants:
        s = stats[name]
        if s.degenerate:
            row_stats = "— | — | — (reference) | —"
        else:
            row_stats = (
                f"{s.mean:+.5f} | {s.sigma_text} | [{s.lo:+.5f}, {s.hi:+.5f}] | {s.improved:.1%}"
            )
        marker = " **<- ships**" if name == ship_name else ""
        lines.append(
            f"| {name}{marker} | {len(cols) + 1} | {losses[name]:.4f} | {row_stats} |"
        )

    lines += [
        "",
        f"- **gain from adding the tree model: {honest_cal - cnn_cal_loss:+.4f}** "
        "(negative is an improvement)",
        f"- same stack scored in-sample, for reference: {in_sample:.4f} "
        f"(optimistic by {honest_cal - in_sample:+.4f})",
        "",
        f"- fitted coefficients: CNN {coef_cnn:+.3f}, trees {coef_gbm:+.3f} "
        f"(trees carry {share_gbm:.0%} of the total weight)",
        f"- intercept {float(ship_stack.intercept_[0]):+.3f}",
        "",
        "A CNN coefficient below 1.0 is the temperature scaling reappearing inside the stack —",
        "expected, and the reason the weight is fitted this way rather than as a convex blend.",
        "A coefficient near zero means that component adds nothing the CNN did not already",
        "have, whatever it scores on its own.",
        "",
        "**A paired difference smaller than about 0.01 is not resolved by this OOF set**, even",
        "when the interval excludes zero: the observed CV-to-leaderboard gap has run from",
        "+0.0024 to -0.0088. Read σ and the interval, never the point estimate alone.",
        "",
    ]

    # --- Which tree configuration ships, and why -----------------------------------------------
    lines += [
        "## Which tree configuration ships",
        "",
        "Fixed in advance, not chosen by reading the table: candidates are ordered by increasing",
        f"complexity and promoted only on a mean better than -{args.margin} **and** a bootstrap",
        "interval excluding zero. Each comparison is against the incumbent at that point, not",
        "against the default, so a chain of near-ties cannot accumulate into a promotion.",
        "",
    ]
    if not chain:
        lines += [
            "- no alternative candidates were fitted; the default configuration ships.",
            "",
        ]
    else:
        lines += [
            "| candidate | against | Δ | σ | 95% CI | improved | promoted |",
            "|---|---|---:|---:|---|---:|---|",
        ]
        for key, incumbent, s, promote in chain:
            cells = (
                "— | — | — | —"
                if s.degenerate
                else f"{s.mean:+.5f} | {s.sigma_text} | [{s.lo:+.5f}, {s.hi:+.5f}] | "
                     f"{s.improved:.1%}"
            )
            lines.append(
                f"| {labels[key]} | {labels[incumbent]} | {cells} | "
                f"{'**yes**' if promote else 'no'} |"
            )
        lines += ["", f"- shipping: **{labels[accepted]}**", ""]

    # --- Seed bagging ----------------------------------------------------------------------------
    lines += ["## Seed bagging", ""]
    if args.gbm_bag <= 1:
        lines += [
            "Not run (`--gbm-bag 1`, the default). One booster per fold, exactly as before.",
            "",
            "Boosted trees at this n are a high-variance estimator and log loss pays directly for",
            "variance reduction — the entire measured value of TTA in this project (-0.0022) was",
            "variance reduction rather than better ranking. `--gbm-bag S` fits S boosters per",
            "fold under fixed distinct seeds and averages them in logit space inside the fold.",
            "",
        ]
    else:
        lines += [
            f"- S = {args.gbm_bag} boosters per fold, seeds fixed at "
            f"`seed + {BAG_SEED_OFFSETS[:args.gbm_bag]}` — constants, never clock-derived, so the",
            "  result reproduces.",
            "- averaged in **logit** space, inside the fold. Nothing crosses a fold boundary, so",
            "  the out-of-fold predictions stay honest and the independence rule is untouched.",
            "- the members differ through lightgbm's per-tree feature sampling"
            f" (`colsample_bytree` < 1) and, where a configuration enables it, row bagging.",
            "",
            f"**Shipped asset count: {len(ship_gbm.models)} folds x {n_bag_shipped} = "
            f"{ship_gbm.n_boosters} booster files**, named `gbm_fold<f>_bag<s>.txt` when S > 1 "
            "and",
            "`gbm_fold<f>.txt` when S = 1. `main.py` globs `gbm_fold*.txt` and averages every",
            "booster it finds in logit space, so the flat mean over all of them equals the mean",
            "of the per-fold means as long as S is the same in every fold — which it is. No",
            "inference change is needed to load them, but whoever owns the packaging step should",
            "know the file count grew.",
            "",
        ]

    # Honesty protocol for the tuning path. Stated in the report, not just in the code, because
    # the report is what gets read six months later.
    lines += ["## Hyperparameter tuning", ""]
    if "tuned" not in trees:
        lines += [
            "Not run. The booster used the hand-set defaults in `GBM_PARAMS`; no hyperparameter",
            "was chosen on this data, so nothing in the table above is optimistic on that count.",
            "Re-run with `--tune` to search.",
            "",
        ]
    else:
        nesting = (
            f"inner {args.tune_inner_splits}-fold cross-validation within each outer fold's "
            "training rows"
            if args.tune_mode == "nested"
            else f"a single 1/{args.tune_inner_splits} holdout of each outer fold's training rows"
        )
        counts = pd.Series(trees["tuned"].chosen).value_counts()
        lines += [
            f"- grid: {len(TUNE_GRID)} configurations",
            f"- selection: **{args.tune_mode}** — {nesting}",
            "- the held-out fold is never seen by the selection, so **the CV log loss reported",
            "  above is NOT optimistic on account of the tuning.** The choice is part of the",
            "  procedure being cross-validated, exactly as the stack's own fit is.",
            "",
            "Configurations selected across the outer folds:",
            "",
            "| configuration | folds that chose it |",
            "|---|---:|",
            *[f"| `{name}` | {int(count)} |" for name, count in counts.items()],
            "",
            "Folds may disagree, and that is not a bug — it is the honest version. The shipped",
            "boosters each carry the parameters chosen on their own training rows, so the shipped",
            "ensemble is bit-for-bit the procedure that was measured. Refitting one 'best'",
            "configuration on all the data would ship something that was never scored.",
            "",
            "One residual optimism that no nesting can remove: the **decision** to prefer a",
            "tuned candidate is taken by looking at held-out numbers, so the margin by which it",
            "wins is itself selected. The acceptance chain above is the mitigation — a rule fixed",
            "in advance rather than 'whichever number is lower' — but a promoted candidate's",
            "quoted gain should still be read as the optimistic end of its interval.",
            "",
        ]
        if args.tune_mode == "inner":
            lines += [
                "`--tune-mode inner` was used, which is cheaper and picks more noisily. The score",
                "stays honest; the *choice* is worse. Confirm anything it promotes with",
                "`--tune-mode nested` before shipping it.",
                "",
            ]

    lines += [
        "## Clip epsilon",
        "",
        "Swept on the held-out predictions of the configuration that would ship. The equivalent",
        "sweep for the CNN alone is in `reports/experiments/baseline.md`.",
        "",
        "| clip eps | CV log loss |",
        "|---:|---:|",
    ]
    lines += [
        f"| {eps:g} | {loss:.4f}{' **<- best**' if eps == best_eps else ''} |"
        for eps, loss in clip_losses.items()
    ]
    lines += [
        "",
        f"- best on this OOF set: **{best_eps:g}** ({clip_losses[best_eps]:.4f}), against",
        f"  {clip_losses[args.clip_eps]:.4f} at the shipped {args.clip_eps:g} — a difference of",
        f"  {clip_losses[best_eps] - clip_losses[args.clip_eps]:+.4f}.",
        "- **this minimum is chosen in-sample.** Any apparent gain from moving off the shipped",
        "  epsilon is optimistic by exactly the amount the curve was fitted to this data, so a",
        "  tie to four decimals is a reason to keep the incumbent, not to switch.",
        "- a tight clip is insurance, not accuracy: it caps what one confidently-wrong scan can",
        "  cost. On a test set drawn differently from this one, the flat region is the safe place",
        "  to sit.",
        "",
    ]

    if extra_cols:
        lines += [
            "## Extra components and the shipped format",
            "",
            "`submission_src/main.py` bakes exactly three constants — `STACK_INTERCEPT`,",
            "`STACK_COEF_CNN`, `STACK_COEF_GBM` — so **a stack over more than two components",
            "cannot be expressed in what ships today.** `stack.json` therefore holds the",
            "two-component fit, and the N-component fit is written beside it as",
            "`stack_ncomp.json`, flagged as not shippable. Carrying it would need a fourth",
            "constant and a second CNN pass inside `main.py`: a change to the inference path,",
            "with its own timing and logging checks, not a change to this script.",
            "",
        ]

    if honest_cal < cnn_cal_loss - args.margin:
        lines += [
            "## Verdict: worth shipping",
            "",
            "Artifacts are written next to the predictions that justify them, in",
            f"`{args.models_dir}/{out_name}/`: `gbm_fold*.txt` (lightgbm boosters, text",
            "format, version-portable) and `stack.json` (the three coefficients plus the feature",
            "order). To build a submission carrying them:",
            "",
            "```",
            f"python scripts/05_package_submission.py --config conf/baseline.yaml"
            f" --blend-from {out_name}"
            f" --fallback {y.mean():.4f} --clip-eps {args.clip_eps:g}",
            "```",
            "",
            "`--temperature` is neither needed nor accepted alongside `--blend-from`: the stack's",
            "own CNN coefficient absorbs the calibration, so passing a temperature as well would",
            "apply it twice.",
            "",
        ]
    else:
        lines += [
            "## Verdict: not worth shipping",
            "",
            f"The blend does not clear a {args.margin} margin over the recalibrated CNN, which is",
            "inside the noise of this OOF set. Log it as a negative result and move on — the",
            "inference complexity it would add to `main.py` is not free, and day 14 is close.",
            "",
        ]

    lines += [
        "## Verify it",
        "",
        "The held-out predictions of the shipped configuration are banked as a run of their own,",
        "so the improvement can be given a standard error instead of being read off the pooled",
        "numbers above:",
        "",
        "```bash",
        f"python scripts/11_compare_runs.py --run-a {args.run} --run-b {out_name}",
        "```",
        "",
    ]

    if accepted != rule_choice:
        lines.insert(
            0,
            f"> **ACCEPTANCE RULE OVERRIDDEN.** The rule chose `{rule_choice}`; `--ship {accepted}` "
            "shipped instead. Read the variants table as the evidence, not as an endorsement.\n",
        )
    report_path = REPO_ROOT / "reports" / "experiments" / f"{out_name}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    if ncomp_path is not None:
        print(f"Wrote {ncomp_path.name} beside stack.json - NOT loadable by the current main.py")
    print(f"Wrote {report_path.relative_to(REPO_ROOT)} - aggregates only, safe for Claude")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
