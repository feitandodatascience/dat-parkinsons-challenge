#!/usr/bin/env python3
"""Fit a gradient-boosted model on the SBR features and blend it with the CNN.

The pipeline
------------
1. Cross-validate a gradient-boosted tree model over `data/interim/sbr_features.csv`, using the
   SAME fold assignments the CNN used. That matters: it makes the tree model's out-of-fold
   predictions row-for-row comparable with the CNN's, so the two can be combined without either
   having seen the other's validation rows.
2. Combine the two in logit space with a logistic stack, fitted OUT OF SAMPLE.
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

One more way this can lie to you
--------------------------------
The tree model is cross-validated on the CNN's fold assignment, so its out-of-fold predictions
carry a trace of each fold's *training* base rate — which, for arbitrary folds, is
anti-correlated with that fold's own base rate. A tree model fitted on pure noise can therefore
pick up a small genuine-looking signal, and the stack will happily give it a negative
coefficient and report a gain from nothing. `fold_strat` is stratified, which equalises the
base rates and closes this route; `tests/test_sbr_blend.py` reproduces the artifact with
unstratified folds. **Do not run this with a fold column that is not stratified on the label.**

Shipping this, if it pays
-------------------------
`main.py` would need to compute the same features per scan (they are per-sample by
construction, so the independence rule holds) and carry the fitted model. Note that this script
uses scikit-learn's HistGradientBoostingClassifier because it is already installed; for the
actual submission prefer `lightgbm`, which is in the runtime and whose text booster format
survives version differences that a scikit-learn pickle would not.

Usage:
    python scripts/09_blend.py --run baseline
    python scripts/09_blend.py --run baseline_tta          # blend against the TTA predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
# Deliberately small trees and strong regularisation. n=1362 with ~60 features is a regime where
# a boosted model will happily memorise; the goal here is a model that DISAGREES usefully with
# the CNN, not one that squeezes the last drop out of the features.
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


def to_logit(p: np.ndarray) -> np.ndarray:
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


def fit_gbm_oof(
    features: pd.DataFrame, y: np.ndarray, folds: np.ndarray, seed: int
) -> tuple[np.ndarray, list[lgb.LGBMClassifier]]:
    """Out-of-fold predictions from the tree model, on the CNN's own fold assignment.

    Returns the fitted fold models as well, because those are what ship. Inference averages all
    five in logit space, mirroring exactly what `main.py` already does with the CNN's fold
    checkpoints — and keeping the shipped ensemble the same shape as the one whose out-of-fold
    predictions the stack coefficients were fitted on.
    """
    oof = np.full(len(y), np.nan)
    models = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        model = lgb.LGBMClassifier(random_state=seed, **GBM_PARAMS)
        model.fit(features[train], y[train])
        oof[~train] = model.predict_proba(features[~train])[:, 1]
        models.append(model)
    assert not np.isnan(oof).any(), "every row must be predicted exactly once"
    return oof, models


def fit_stack(logits: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit a logistic regression on component logits — standard stacking.

    Why not a convex weight `w * z_cnn + (1-w) * z_gbm`, which is the obvious thing:
    **it conflates mixing with calibration.** Lowering `w` both moves weight onto the second
    model AND shrinks the first model's logits toward zero. If the CNN is overconfident — it is;
    that is why temperature scaling helped — then `w < 1` reduces log loss through shrinkage
    alone, and a *pure noise* second model scores an apparent gain. That failure is reproduced
    in tests/test_sbr_blend.py.

    Fitting a coefficient per component with an intercept separates the two effects: the
    coefficient on the CNN absorbs the calibration, so the coefficient on the tree model
    measures only what it adds. It also makes the comparison a clean nested-model test — two
    parameters for the CNN alone against three for the pair — with both sides cross-validated.
    """
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(logits, y)
    return model


def stack_predict(model: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    return model.predict_proba(logits)[:, 1]


def cv_score(logits: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Cross-validated held-out predictions from a stack over these columns.

    Used for both sides of the comparison — CNN alone (one column) and CNN + trees (two) — so
    the two numbers differ only by the information available, never by how they were scored.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.full(len(y), np.nan)
    for train, test in splitter.split(logits, y):
        out[test] = stack_predict(fit_stack(logits[train], y[train]), logits[test])
    assert not np.isnan(out).any()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="baseline", help="run dir holding the CNN's oof.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--features", default="data/interim/sbr_features.csv")
    parser.add_argument("--folds-csv", default="data/interim/folds.csv")
    parser.add_argument("--fold-column", default="fold_strat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

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

    feature_columns = [c for c in feats.columns if c != "uid"]
    x = df[feature_columns]
    y = df["is_pathologic"].to_numpy()
    fold_ids = df[args.fold_column].to_numpy()

    print(f"n = {len(df)}, features = {len(feature_columns)}, folds = {args.fold_column}")

    p_cnn = df["oof_pred"].to_numpy()
    p_gbm, gbm_models = fit_gbm_oof(x, y, fold_ids, args.seed)

    cnn = summarize(y, p_cnn)
    gbm = summarize(y, p_gbm)

    z_cnn, z_gbm = to_logit(p_cnn), to_logit(p_gbm)
    both = np.column_stack([z_cnn, z_gbm])
    cnn_only = z_cnn.reshape(-1, 1)

    # The comparison: a stack over the CNN alone, against a stack over both. Same estimator,
    # same folds, same scoring — the only difference is whether the tree model's column is
    # available. Any gain is therefore attributable to it and not to recalibration.
    p_cnn_stacked = cv_score(cnn_only, y, args.n_splits, args.seed)
    p_blend = cv_score(both, y, args.n_splits, args.seed)
    cnn_cal_loss = competition_logloss(y, p_cnn_stacked)
    honest_cal = competition_logloss(y, p_blend)

    # Bank the held-out blended predictions as a run of their own, so the improvement can be put
    # through the paired test in scripts/11_compare_runs.py rather than judged on the pooled
    # numbers below. A difference this size needs a standard error attached to it, and only the
    # per-scan pairing supplies one that is small enough to be informative.
    blend_dir = REPO_ROOT / args.models_dir / f"{args.run}_blend"
    blend_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"uid": df["uid"], "is_pathologic": y, "oof_pred": p_blend, "fold": fold_ids}
    ).to_csv(blend_dir / "oof.csv", index=False)

    # --- Artifacts for the submission ------------------------------------------------------
    # Everything inference needs, written next to the predictions that justify shipping it.
    # The boosters go out as lightgbm's text format, which loads across library versions; the
    # stack is three floats and the feature order.
    final_stack_for_ship = fit_stack(both, y)
    for i, model in enumerate(gbm_models):
        model.booster_.save_model(str(blend_dir / f"gbm_fold{i}.txt"))

    stack_meta = {
        "intercept": float(final_stack_for_ship.intercept_[0]),
        "coef_cnn": float(final_stack_for_ship.coef_[0][0]),
        "coef_gbm": float(final_stack_for_ship.coef_[0][1]),
        # Stored so inference can assert the column order it builds matches the one the trees
        # were fitted on. A silently permuted feature vector is the failure mode here: it raises
        # nothing, and produces predictions that are wrong in a way no check would catch.
        "feature_names": feature_columns,
        "n_folds": len(gbm_models),
        "source_run": args.run,
    }
    (blend_dir / "stack.json").write_text(json.dumps(stack_meta, indent=2), encoding="utf-8")

    final_stack = fit_stack(both, y)
    in_sample = competition_logloss(y, stack_predict(final_stack, both))
    coef_cnn, coef_gbm = (float(c) for c in final_stack.coef_[0])
    share_gbm = abs(coef_gbm) / max(abs(coef_cnn) + abs(coef_gbm), 1e-9)

    lines = [
        f"# Blend: CNN ({args.run}) + SBR gradient boosting",
        "",
        f"- n = {len(df)}, features = {len(feature_columns)}, fold column = `{args.fold_column}`",
        "",
        "## Components, out of fold",
        "",
        "| model | log loss | AUROC | ECE |",
        "|---|---:|---:|---:|",
        f"| CNN | {cnn['logloss']:.4f} | {cnn['auroc']:.4f} | {cnn['ece']:.4f} |",
        f"| SBR + trees | {gbm['logloss']:.4f} | {gbm['auroc']:.4f} | {gbm['ece']:.4f} |",
        "",
        f"- correlation of the two logit vectors: **{np.corrcoef(z_cnn, z_gbm)[0, 1]:.3f}**",
        "",
        "A low correlation is the point. Two models that agree everywhere cannot help each",
        "other; the blend is worth something precisely to the extent they disagree.",
        "",
        "## Blend",
        "",
        "Both rows below are a logistic stack scored by the same 5-fold cross-validation. They",
        "differ only in whether the tree model's logit is offered as a second column, so the",
        "difference measures what that model adds — not a recalibration the CNN would have got",
        "on its own.",
        "",
        "| stack over | parameters | cross-validated log loss |",
        "|---|---:|---:|",
        f"| CNN logit alone | 2 | {cnn_cal_loss:.4f} |",
        f"| CNN + tree logits | 3 | **{honest_cal:.4f}** |",
        "",
        f"- **gain from adding the tree model: {honest_cal - cnn_cal_loss:+.4f}** "
        "(negative is an improvement)",
        f"- same stack scored in-sample, for reference: {in_sample:.4f} "
        f"(optimistic by {honest_cal - in_sample:+.4f})",
        "",
        f"- fitted coefficients: CNN {coef_cnn:+.3f}, trees {coef_gbm:+.3f} "
        f"(trees carry {share_gbm:.0%} of the total weight)",
        f"- intercept {float(final_stack.intercept_[0]):+.3f}",
        "",
        "A CNN coefficient below 1.0 is the temperature scaling reappearing inside the stack —",
        "expected, and the reason the weight is fitted this way rather than as a convex blend.",
        "A tree coefficient near zero means the features add nothing the CNN did not already",
        "have, whatever the tree model scores on its own.",
        "",
    ]

    if honest_cal < cnn_cal_loss - 0.002:
        lines += [
            "## Verdict: worth shipping",
            "",
            "Artifacts are written next to the predictions that justify them, in",
            f"`{args.models_dir}/{args.run}_blend/`: `gbm_fold*.txt` (lightgbm boosters, text",
            "format, version-portable) and `stack.json` (the three coefficients plus the feature",
            "order). To build a submission carrying them:",
            "",
            "```",
            f"python scripts/05_package_submission.py --config conf/baseline.yaml"
            f" --blend-from {args.run}_blend"
            f" --fallback {y.mean():.4f} --clip-eps 1e-06",
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
            "The blend does not clear a 0.002 margin over the recalibrated CNN, which is inside",
            "the noise of this OOF set. Log it as a negative result and move on — the inference",
            "complexity it would add to `main.py` is not free, and day 14 is close.",
            "",
        ]

    lines += [
        "## Verify it",
        "",
        "The held-out blended predictions are banked as a run of their own, so the improvement",
        "can be given a standard error instead of being read off the pooled numbers above:",
        "",
        "```bash",
        f"python scripts/11_compare_runs.py --run-a {args.run} --run-b {args.run}_blend",
        "```",
        "",
    ]

    report_path = REPO_ROOT / "reports" / "experiments" / f"{args.run}_blend.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Wrote {report_path.relative_to(REPO_ROOT)} - aggregates only, safe for Claude")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
