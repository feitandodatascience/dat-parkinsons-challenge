# Solution — DaT Parkinson's Challenge

DrivenData competition #311, French Society of Nuclear Medicine (SFMN) / Health Data Hub / GaelO.
Binary classification of DaT SPECT scans as normal or abnormal, scored by **log loss**.

**Status: draft, 2026-09-11.** Written before the freeze deliberately — the exercise of stating
each decision and the evidence for it is what makes the freeze safe, and doing it afterwards
would be a write-up rather than a review.

| | |
|---|---|
| Best submission | **0.2973** log loss, AUROC 0.9396 (rank #98/897 at submission) |
| Out-of-fold estimate | 0.3061 |
| Prior baseline (predict the training rate) | 0.6884 |
| Inference runtime | 15 min of a 3 h budget, on the platform's A100 |
| Training hardware | CPU only (Intel Iris Xe, no CUDA) — the binding constraint on everything below |

---

## 1. The solution in one paragraph

A 5-fold ensemble of 3D residual CNNs over resampled DaT SPECT volumes, averaged in logit space
across folds and across nine laterality-preserving test-time-augmentation views, then **stacked
with a LightGBM model over 61 handcrafted SBR-style geometric features** via a three-parameter
logistic regression. The stack's CNN coefficient absorbs the calibration that a temperature
scaler would otherwise apply, so no separate temperature is used in the shipped path.
Predictions are clipped to [1e-6, 1-1e-6].

The single most valuable component is not the CNN. It is the **blend** — worth −0.0175 against
−0.0022 for TTA — and it pays precisely because the tree model *disagrees* with the CNN.

---

## 2. Pipeline

### 2.1 Preprocessing

| step | choice | why |
|---|---|---|
| Orientation | `as_closest_canonical` | All 1362 training volumes are stored RAS, so this is a no-op *on this data*. Kept because the test set is not guaranteed to match. |
| Resample | 2.0 mm isotropic | Native spacing varies (2.46 mm, 3.895 mm, …). A common grid is required before a convolutional model sees the data. |
| Crop | **centre of mass**, not geometric centre | The decisive preprocessing choice. Field of view spans **176–630 mm per axis**; in the widest third of acquisitions the head occupies ~1% of the array. A geometric centre crop clips the striatum on one side, destroying the asymmetry the task depends on. |
| Output shape | 96³ | The largest that fit the CPU budget. See §5.1. |
| Intensity | `reference_region` | Normalises to an occipital reference region **within the same scan**. Per-sample by construction, which is also what the competition's independence rule requires (§6.1). |

`center_of_mass_crop` uses a fraction-of-quantile foreground test rather than a percentile. At
this sparsity a percentile selects air, which silently degrades COM cropping back to centre
cropping — a failure that raises nothing and shows up only as a worse score.

### 2.2 Model

A 4-stage 3D residual CNN, channel widths 16/32/64/128, dropout 0.3, adaptive global pooling to
a single logit. Specifics in `src/datpd/models/resnet3d.py`.

Training: 5-fold stratified CV, 30 epochs, AdamW (lr 3e-4, weight decay 1e-4), cosine annealing
with `T_max = epochs`, batch size 8, AMP, gradient clipping at 1.0, early stopping on validation
log loss with patience 8. Augmentation: rotations ±10°, translations ±5 mm, scale 0.9–1.1,
intensity jitter 0.1, Gaussian noise σ 0.01. **No left-right flip** — see §3.2.

Model selection is on **validation log loss**, the competition metric, and the best epoch's
weights are restored rather than the last.

### 2.3 Test-time augmentation

Nine deterministic views: small rotations, translations and scale jitter drawn from the same
distribution as training augmentation, all **laterality-preserving**. Averaged in logit space.

Worth −0.0022 on OOF. That is within the CV↔leaderboard spread (§4.2), so it was shipped as a
rider on the blend rather than on its own evidence — its marginal cost is near zero, since
`05_package_submission.py --tta` is bit-identical to the no-TTA build when off, and the shipped
`_make_view` is pinned numerically against the function the measurement was made with
(`tests/test_tta.py`).

### 2.4 The SBR feature model and the stack

61 features computed from the same COM-cropped canonical volume the CNN sees, in four groups:
intensity (global and within a striatal box), supra-threshold extent and shape descriptors,
laterality indices per axis, and the caudate-to-putamen gradient within each hemisphere.
`src/datpd/features.py` is the single definition, and it is **vendored into the submission**, so
inference cannot drift from what the trees were fitted on.

Five LightGBM boosters (one per fold), combined by a logistic stack over two inputs — the CNN
logit and the tree logit:

```
z = 0.0236 + 0.6164 * cnn_logit + 0.3779 * gbm_logit
```

**Why it works, and it is not what it looks like.** The tree model *alone* scores 0.3919 against
the CNN's 0.3186 — substantially worse. It improves the ensemble anyway because its logits
correlate only **0.797** with the CNN's. Diversity paid, not quality. That single number is the
most useful thing learned in this project, and it is what closed the case for adding a second CNN
architecture (§5.2).

**LightGBM rather than scikit-learn, decided before measuring rather than after.** A scikit-learn
estimator ships as a pickle whose compatibility with the runtime's version is not guaranteed;
LightGBM's booster is text and loads across versions. Measuring with one library and shipping
another would mean the number that justified the change was never the number submitted. Re-measured
on LightGBM, the gain held (−0.01562 against scikit-learn's −0.01458).

### 2.5 Calibration

Temperature scaling, fitted on out-of-fold predictions. **In the shipped blend path there is no
separate temperature**: `STACK_COEF_CNN = 0.6164` was fitted on logits that were never
temperature-scaled, so it already carries the calibration (equivalent to T ≈ 1.62). Applying
`TEMPERATURE` as well would soften twice and give back most of the gain. The packager refuses
`--temperature` alongside `--blend-from`, and a test asserts the blend path never references it.

---

## 3. Decisions, and the evidence for each

### 3.1 Centre-of-mass cropping over a fixed centre crop

**Evidence:** EDA measured field of view spanning 176–630 mm per axis — a 3.6× range. Adopted
before the first training run, on that measurement rather than on a score.

A later diagnostic appeared to show the widest acquisition family (`256@2.5mm`, 34% of the data)
cropping badly: log loss 0.3691 against 0.2339 for `128@2.4mm`. **That lead was closed as a
confound.** The metric used (`box_max / global_max`) is depressed by disease prevalence, because
a pathologic striatum is dimmer and its peak leaves the box without any crop having failed. After
stratifying by label the effect disappeared. Recorded because it nearly bought a retraining run.

### 3.2 No left-right flip augmentation

**Evidence: domain, not measurement.** DaT abnormality is frequently asymmetric and the
more-affected side carries clinical meaning (`docs/01_domain_primer_dat_scans.md` §4). Averaging a
scan with its mirror destroys exactly the signal the task depends on.

Deliberately left unmeasured. Confirming an expected negative costs a full CPU training run
(~1 R), and with a fixed compute budget that is a poor use of a night. The reasoning is recorded
rather than the number.

### 3.3 Stratified CV, not grouped

**Evidence: the leaderboard, directly.** Run 1 scored OOF 0.3186 → LB 0.3210, a gap of +0.0024.

Grouped CV exists as a *proxy* for "will the leaderboard track my CV". The leaderboard answered
that question directly, which makes the proxy redundant for ranking purposes. Grouped CV measures
generalisation to an unseen eleventh hospital — a real question about the solution's worth, but
not one the competition scores, and optimising against it would have been actively pessimistic
about leaderboard performance.

The fold structure was fixed first, though, and that mattered: the original acquisition
fingerprint (full shape + full-precision spacing) produced **134 groups over 1362 scans**, mostly
singletons, because slice count and reconstruction zoom vary per patient. Grouped CV was
therefore doing nothing while appearing to be applied. `acquisition_signature` now uses in-plane
matrix class + spacing rounded to 0.1 mm, collapsing these into ~21 real families.

### 3.4 Log loss discipline throughout

AUROC is reported everywhere as context and never as a decision metric. The rule applied
consistently: **a change that improves AUROC while worsening log loss is not an improvement until
recalibrated.**

This also settled an anomaly. Submission 2 beat its predicted ~0.305 with 0.2973 — an
over-performance of about −0.008. Because **LB AUROC (0.9396) also beat OOF AUROC (0.9375)**, and
AUROC is calibration-invariant, the surplus is a marginally easier test draw rather than the
method transferring better than measured. The method is still valued at the −0.0156 that was
measured. *A gain larger than predicted is an unexplained residual, not a confirmation.*

---

## 4. Methodological errors found and fixed

Recorded because each was invisible in the metric it corrupted, and the shapes generalise.

### 4.1 Calibration was being selected in-sample

`choose_calibrator` fitted each candidate on the OOF predictions and scored it on **those same
predictions**. Temperature scaling has one parameter and barely benefits; isotonic regression has
of order *n* knots and can drive in-sample log loss almost arbitrarily low. Isotonic therefore won
every comparison by construction. The tell was an ECE of exactly **0.0000**.

Re-scored with 5-fold CV *inside* the OOF set:

| method | in-sample (wrong) | cross-validated (honest) |
|---|---:|---:|
| isotonic | 0.3079 | **0.3520** |
| temperature | 0.3186 | **0.3217** |
| none | 0.3271 | 0.3271 |

**Isotonic is worse than no calibration at all.** Shipping it would have cost real leaderboard
points while the report claimed a gain.

A second bug rode along: `evaluate.py` emitted `--temperature 1.0000` whenever the winner had no
`.temperature` attribute, so following its own instructions would have shipped an *uncalibrated*
model.

**General form: the more flexible method wins any comparison scored on its own training data.**

### 4.2 The CV↔leaderboard gap was estimated from one point

After run 1 the gap was quoted as **±0.0024**, and Phase 3 decisions were made against it. One
observation cannot measure a spread — it has nothing to differ from. With two:

| | OOF | LB | gap |
|---|---:|---:|---:|
| Run 1 | 0.3186 | 0.3210 | +0.0024 |
| Submission 2 | 0.3061 | 0.2973 | −0.0088 |

The honest bound is **±0.01**, four times looser, and on opposite sides of zero. Phase 3's
decisions survive (they rested on −0.0156 and −0.0175, well clear), but the threshold for
everything after moved: **an OOF margin under ~0.01 is unresolved, not won.**

### 4.3 A convex blend weight conflates mixing with calibration

`w * cnn + (1-w) * gbm` shrinks the CNN's logits as `w` falls, which is temperature scaling by
another name. Against an overconfident CNN, a **pure noise** second model then scores an apparent
gain. `09_blend.py` fits a logistic stack instead and compares against the same stack over the CNN
alone, so both sides carry identical recalibration and the difference is attributable to the
features.

### 4.4 The tree model's fold column must be stratified

On unstratified folds a tree model fitted on noise learns each fold's training base rate, which is
anti-correlated with its own validation rate, and manufactures a gain from nothing.

### 4.5 Comparisons must be paired

The unpaired standard error at n=1362 is ~0.022, which would call a real 0.002 change
unresolvable. `11_compare_runs.py` does a per-scan paired comparison with a standard error on the
difference itself — 4× smaller — plus a bootstrap CI and a sign test.

### 4.6 Submission logs must not describe the test set

Run 1's submission returned a compliance warning: three lines replaced by `WARNING: line
filtered`, all three containing `len(submission)`. A line reporting `Loaded 5 fold models`
survived, which is what made the cause diagnosable — the surviving line acted as a control.

**The forbidden content is information about the test data, above all its row count** — as a
total, a progress denominator, a row index, or a failure count. Progress is now reported as
elapsed time and fallbacks as whether-not-how-many. Submission 2 returned **zero filtered lines**.

A line-by-line audit on 2026-09-11 found one survivor: the per-failure warning was gated on
`failures <= 5`, so **how many times it appeared** was a lower bound on how many test scans
failed. `check_logging` cannot see that — it scans for `len(...)` and row indices, and
multiplicity is neither — and the branch had never executed on the platform, since submission 2
needed no fallbacks. Now gated on `failures == 1`.

### 4.7 A diagnostic must survive the failure it diagnoses

The per-epoch training curve existed only as stdout, so once a terminal closed the run could no
longer answer questions about itself. Fixed by banking `history_fold*.json` every epoch — and the
fix was then made useless by reading the report out of `metrics.json`, which is written only after
*all* folds finish. An interrupted run had every curve on disk and printed no table at all. The
report now reads the history files directly.

---

## 5. Rejected, with reasons

### 5.1 Higher resolution (128³ @ 1.5 mm) — the largest win not taken

2.37× the voxels of 96³, priced at **~2.4 R** where R is one full 5-fold run. On the available
CPU that is roughly five days against a budget of about two. Expected gain −0.02 to −0.05, which
would very likely have made it the single largest improvement in the project.

Renting GPU compute was cleared with the organizers (see `docs/07_cloud_compute_compliance.md` for
the ruling and its three conditions) and would have reduced this to ~4 hours. It was not taken up
in time. `conf/highres.yaml` is written and ready.

**This is the honest gap in the solution.** It was a compute constraint, not a judgement.

### 5.2 A second CNN architecture, ensembled

Rejected on the correlation argument in §2.4 rather than by measurement. The blend paid −0.0175 at
ρ = 0.797. Two 3D CNNs on identical data and preprocessing correlate ~0.90–0.95, so the
variance-reduction benefit scales to roughly **−0.004 to −0.009** — inside the ±0.01 band of §4.2,
meaning the result could not have been confirmed on the leaderboard even if positive.

Shipping cost is asymmetric and reinforced the decision: `main.py` globs `fold*.pt` and averages
whatever it finds, so a second *seed* is free to ship, while a second *architecture* requires
changes to the model construction, the vendor list, the packager's pre-flight loader, and its
tests — new inference-path code close to a deadline.

If compute had allowed one architecture, the better choice would have been the **2.5D route**
(slice-wise 2D encoders with ImageNet pretraining), not another 3D CNN — because its correlation
with the 3D CNN would sit nearer the tree model's 0.797, which is where ensemble value lives. It
also carries no PPMI provenance problem (§6.2).

### 5.3 Completing the cosine anneal

Early stopping (patience 8) with `T_max = 30` meant a fold stopping near epoch 20 selected weights
at ~65% of the peak learning rate — the low-rate endgame never ran. `conf/baseline_anneal.yaml`
shortened the schedule to 20 epochs with early stopping off, so the anneal completes.

**Measured null: +0.00765 at 1.2σ**, CI [−0.0048, +0.0200], 45.5% of scans improved. Not shipped.

The schedule table explains why. Folds 1 and 4 peaked at epochs 17 and 15, at 5% and 15% of peak
rate — deep in the anneal, exactly as predicted. But shortening `T_max` buys a completed anneal at
the price of a lower rate at *every* epoch (1.27e-4 against 2.05e-4 at epoch 11), and the extra
learning-rate heat in the longer schedule was doing real work. The effects cancel.

A first look at this on **one** fold suggested the opposite mechanism — "the model peaks at epoch
11–12 regardless of schedule". Five folds contradicted it. The verdict survived; the reasoning did
not. Reading a mechanism off a 20% sample was overreach.

### 5.4 Label smoothing

Ranked for its calibration benefit, but this pipeline already fits a post-hoc temperature on OOF
predictions, which addresses the same overconfidence more directly and for free. Buying the
overlap for a full training run is poor value.

### 5.5 Pretrained 3D backbones

Most public DaT checkpoints are PPMI-derived. See §6.2.

---

## 6. Compliance

### 6.1 Per-sample independence

Each test examination is predicted from its own voxels and the shipped weights alone. Audited
line by line on 2026-09-11:

- `predict_one(models, uid)` takes a uid and the models; it holds no cross-sample state.
- Intensity normalisation is `reference_region` — computed **within** the scan. No test-set
  statistic is fitted, pooled, or referenced.
- `model.eval()` is set at load, so normalisation layers never use batch statistics.
- `torch.no_grad()` throughout; nothing is fitted at inference.
- TTA views and SBR features are derived from the single scan under prediction.
- `FALLBACK_PROB` is the *training* positive rate, baked in as a literal.
- The booster cache holds shipped weights, not data.
- No pseudo-labelling, no test-time retraining, no transduction.

### 6.2 External data and pretrained weights

**None used.** Every weight shipped was trained from scratch on the competition training set.

**PPMI is ruled out** — the organizers confirmed its DUA does not cover competition use, so PPMI
data *and PPMI-derived weights* make a solution prize-ineligible. Since much published DaT work is
PPMI-based, any public checkpoint would need its provenance verified first. See
`docs/EXTERNAL_ASSETS.md`.

### 6.3 Data handling

The competition rules prohibit making the data available to parties who have not agreed to them,
explicitly including third-party AI services. This repository was developed with an AI coding
assistant under a three-layer guard — deny rules, a `PreToolUse` hook, and a behavioural
specification in `CLAUDE.md` §0 — such that the assistant wrote code and **never read the data**.
Aggregate reports under `reports/` are the only channel from data to model-facing decisions.

The organizers' guidance on this ("some coding assistants read local files by default, so
competitors should check what a given tool has access to") is exactly what that arrangement
addresses. `docs/07_cloud_compute_compliance.md` records the ruling in full.

### 6.4 Licence

MIT (`LICENSE`, `pyproject.toml`). Winners are required to open-source under MIT; this repository
already is.

---

## 7. Reproduction

Python `>=3.12,<3.13`, matching the competition runtime. `pip install -e '.[dev]'`.

```bash
python scripts/00_check_env.py                              # verify against the runtime
python scripts/02_make_folds.py                             # stratified + grouped fold columns
python scripts/03_train.py --config conf/baseline.yaml      # 5-fold CV  (~1 night on CPU)
python scripts/04_evaluate.py --run baseline                # OOF metrics, calibration choice
python scripts/07_tta.py --config conf/baseline.yaml --run baseline
python scripts/08_sbr_features.py --config conf/baseline.yaml
python scripts/09_blend.py --run baseline_tta
python scripts/11_compare_runs.py --run-a baseline --run-b baseline_tta_blend
python scripts/05_package_submission.py --config conf/baseline.yaml --tta \
    --blend-from baseline_tta_blend --fallback 0.5485 --clip-eps 1e-06
```

`seed: 42` throughout, set via `datpd.utils.seed.set_seed`. `pytest -q` runs the full suite on
**synthetic volumes only** and requires no competition data.

Every experiment is one config plus one row in `reports/experiments/log.md`, including the
negative results. Per-submission platform logs are in `reports/submissions/`.

---

## 8. What I would do differently

1. **Rent the GPU on day one.** Every rejected item in §5 was rejected on compute, not on
   evidence. The compute constraint shaped the solution more than any modelling insight did.
2. **Bank diagnostics from the first run.** §4.7 cost a session spent unable to answer a one-line
   question about a run that had already finished.
3. **Hold something out of every comparison, from the start.** §4.1 nearly shipped a calibrator
   that was worse than none.
4. **Estimate spread before trusting a gap.** §4.2 — the ±0.0024 that guided a week's decisions
   was a single point, and a single point has no spread.
5. **Look for diversity, not capacity.** The tree model is much worse than the CNN and is the
   largest single gain in the solution. That generalises further than anything else here.
