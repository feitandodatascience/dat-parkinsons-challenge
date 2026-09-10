# Experiment log

One row per experiment. Fill it in **as you go** — reconstructing this at the end is how people
lose two days and repeat experiments they already ran.

Rules for this file:

- **Log failures too.** A negative result you can see on day 14 stops you retrying it at 2am.
- **Log loss is the decision metric.** AUROC is context. If AUROC improves and log loss worsens,
  the change is not (yet) an improvement — recalibrate before judging it.
- Record the **grouped** CV number alongside the stratified one. When they diverge, trust
  grouped and write down the gap.
- Note the git commit so any row can be reproduced.

| # | date | config | commit | change | OOF logloss (strat) | OOF logloss (grouped) | AUROC | ECE | LB | keep? | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-09-05 | — | — | **prior baseline** (predict 0.5485 for everything) | **0.6884** | — | 0.500 | — | — | — | the floor; anything worse than this is not a model. n=1362, 747 pos / 615 neg |
| 0b | 2026-09-05 | — | | **prior-only submission** (`--prior-only --fallback 0.5485`), no weights | — | — | 0.500 | — | **0.6687** (smoke) | — | wire test passed: platform contract proven end to end. Score implies smoke-set positive rate ≈ 0.65 vs train 0.5485 — see note below before believing it |
| 1 | 2026-09-08 | `baseline.yaml` | | first end-to-end run: 96³ @ 2.0 mm, `crop: com`, ResNet3D, 5-fold stratified | **0.3186** | not run | 0.9302 | 0.0220 | **0.3210** | yes | temperature = 1.2614, clip 1e-06. **CV↔LB gap = +0.0024.** Beats the prior by 0.3698 |
| 2 | 2026-09-08 | — | | **per-family error decomposition** (`06_score_by_group.py`), no training | — | — | — | — | — | n/a | measurement, not a model. Leave-one-family-out temperature transfer penalty **+0.0006**; per-centre oracle prior correction worth at most **0.004**. Closes row 0b |
| 3 | 2026-09-08 | `baseline.yaml` | | **TTA**, 9 laterality-preserving views on the run-1 checkpoints | **0.3164** | — | 0.9306 | 0.0269 | not submitted | marginal | gain **−0.0022**, temperature falls to 1.2457. AUROC +0.0004, so the gain is variance reduction, not ranking. Same order as the CV↔LB gap |
| 4 | 2026-09-08 | `baseline.yaml` | | **SBR features + gradient boosting, stacked onto the CNN** | **0.3048** | — | — | — | not submitted | **yes** | gain **−0.0175**, cross-validated. Trees alone 0.3919 / AUROC 0.9046; logit correlation 0.797. Largest measured gain in the project |
| 5 | 2026-09-09 | `baseline.yaml` | | **TTA + SBR blend, packaged** (lightgbm, 5 boosters, logistic stack) | **0.3061** | — | 0.9375 | — | **0.2973** | **yes — shipped** | paired vs baseline: **−0.01562, 3.1σ**, CI [−0.0255, −0.0057], 63.4% of scans improved. Stack: intercept 0.0236, CNN 0.6164, trees 0.3779. No temperature — the stack carries it. **LB AUROC 0.9396, rank #98/897, 15 min of a 3 h budget, zero filtered log lines.** Full log: `reports/submissions/002_2026-09-09_tta_blend.md` |

**Run 1 packaged 2026-09-08**: `submission.zip`, 37.9 MB, 5 fold checkpoints ensembled in logit
space. All pre-flight checks passed, including the new one that loads every shipped checkpoint
into the architecture `main.py` constructs. Baked constants verified against `baseline.yaml`:
96³ @ 2.0 mm, `com`, `reference_region`, T=1.2614, clip 1e-06, fallback 0.5485.

**Expect ~0.32 on the leaderboard.** How it actually lands is itself a measurement:

- **near 0.32** — CV transfers; trust it for Phase 3 decisions.
- **modestly worse** — consistent with the prior-shift hypothesis (row 0b). The full test set is
  a large enough sample to settle what the smoke set could not.
- **much worse** — suspect train/inference preprocessing skew first, despite the vendoring and
  the baked constants. Check `TARGET_SHAPE` in the shipped `main.py` against
  `models/baseline/metrics.json`; adaptive pooling means a shape mismatch raises no error.

Row 0b is worth filling in even though its score is uninteresting. It is the only row where CV
and leaderboard are known to be *the same quantity*, so the difference between them measures the
platform, not the model. Every later CV↔LB gap is read against it.

## Running notes

### Open questions

- [x] **How many distinct acquisition signatures are there, and are the groups big enough for
      5-fold grouped CV?** — Answered 2026-09-05, and the first answer was a trap. The original
      fingerprint (full shape + full-precision spacing) gave **134 groups over 1362 scans**, most
      of them singletons, because slice count and reconstruction zoom vary *per patient*. Grouped
      CV was therefore doing nothing while appearing to be applied. Fixed in
      `datpd.data.splits.acquisition_signature`: in-plane matrix class + spacing to 0.1 mm,
      which collapses those into roughly a dozen real families. The largest is ~34% of the data,
      so **expect uneven grouped folds** — that is correct, not a bug.
- [x] **Does positive rate vary a lot by signature?** — Yes, moderately. Across the large
      families the positive rate spans roughly 45% to 59% against an overall 54.8%. Centre is a
      real confound: enough that grouped CV matters, not so extreme that stratified CV is
      useless. Track the gap between `fold_strat` and `fold_group` from run 1 onward.
- [x] **Is a fixed centre crop safe?** — No. Field of view spans **176–630 mm** per axis; in the
      widest acquisitions (a third of the data) the head is ~1% of the array. `crop: com` is now
      the default in `baseline.yaml`, and `center_of_mass_crop` uses a fraction-of-quantile
      foreground test rather than a percentile, because a percentile selects air at that sparsity
      and silently degrades COM cropping back to centre cropping.
- [x] **Does orientation handling matter?** — Not here. All 1362 volumes are stored `RAS`, so
      `as_closest_canonical` is a no-op. One less source of variance; keep the reorientation in
      place anyway, since the test set is not guaranteed to match.
- [ ] Does `flip_lr` help or hurt? (Prior: hurts — asymmetry is diagnostic. Test it.)
- [ ] Optimal prediction clip on OOF?
- [ ] Is the CV↔leaderboard gap stable enough to trust CV for decisions?
- [ ] **Did run 1's early stopping cut the cosine anneal short, and what did that cost?**
      Unanswerable for run 1 — see below. Instrumented 2026-09-10; the next run answers it.

### The run could not answer questions about itself (found and fixed 2026-09-10)

Prompted by "should we retrain with more epochs — it stopped at 20 with the score still going
up?" The honest answer was that **nobody could tell**, and that was the real problem.

**What was checkable.** Selection and early stopping both run on validation log loss
(`train.py`, `if metrics["logloss"] < best_loss`). So a fold that broke at 20 with
`early_stopping_patience: 8` had its best epoch at 12 and got *worse* for eight consecutive
epochs after it. Log loss cannot have been "still going up". The per-epoch line prints
`val_logloss` and `auroc` side by side, and AUROC rising while log loss degrades is this file's
opening rule — a model getting better at ranking while getting more overconfident.

**What was not checkable, and is the more interesting half.** `_make_scheduler` sets
`T_max = cfg.train.epochs = 30`. At `lr: 0.0003`, epoch 12 is still at **1.96e-4 — 65% of peak**.
If run 1 stopped around 20, **the anneal never finished**, and the shipped weights were selected
well before the low-rate endgame cosine exists to provide. That is a real cost and the instinct
behind the question was sound.

**Raising `epochs` is the wrong lever, and would make it worse.** `T_max` is tied to `epochs`, so
a longer schedule decays *more slowly*: at `epochs: 50`, epoch 12 sits at ~2.6e-4. The levers are
a shorter schedule or more patience.

**None of this could be confirmed, because the curve existed only as stdout** and the terminal
was long closed. `metrics.json` kept only each fold's best epoch. So:

- `train_fold` now banks `history_fold<k>.json` every epoch — epoch, **lr**, train loss,
  val log loss, AUROC, seconds — and carries it through the resume checkpoint.
- `metrics.json` gains a `schedule` block per fold: best epoch, last epoch, whether it stopped
  early, and **`lr_at_best_over_peak`**, which is the single number this question needs.
- `04_evaluate.py` renders it as a **Training schedule** table in `reports/experiments/<run>.md`,
  so it is visible without reading anything under `models/`.
- `conf/highres.yaml` sets `early_stopping_patience: 999`. On a GPU a fold is minutes, so early
  stopping buys nothing and costs the end of the schedule. `best_state` restoration means
  training past the best epoch cannot make the shipped weights worse.

**Incidental: `warmup_epochs` is dead.** Declared in `config.py`, set to 2 in every config, read
by nothing — `_make_scheduler` returns a plain `CosineAnnealingLR`. Every run so far started at
full learning rate from step one. It costs no score, but the configs describe a run that did not
happen. Left as-is under the freeze; worth either implementing or deleting after the deadline.

**The general lesson, which outlives this question:** an experiment that does not persist its own
diagnostics can only be argued about, not settled. The cost was not the missing warmup — it was
spending a session unable to answer a one-line question about a run that had already finished.

### The anneal hypothesis looks wrong, on one fold (2026-09-10)

`baseline_anneal` (96³, `epochs: 20`, early stopping off) was started and **did not finish**:
fold 0 completed all 20 epochs, **fold 1 died four epochs in**, folds 2–4 never ran. So its
report is n=273 — one fold's validation set — and **its 0.3203 is not comparable to run 1's
0.3186** over 1362. SE at n=273 is around ±0.04. Nothing about better-or-worse is readable.

What fold 0 *does* say, and it is the point of the experiment:

| | best epoch | of | lr at best / peak |
|---|---:|---:|---:|
| run 1 (`T_max=30`, patience 8) | ~12 (inferred) | 30 | ~0.65 |
| `baseline_anneal` (`T_max=20`, no early stop) | **11** | 20 | **0.42** |

**The anneal ran to zero and the low-rate endgame made things worse.** Epochs 12–19 — exactly the
phase the shortened schedule was built to deliver — all scored worse than epoch 11. The model
peaks at epoch 11–12 *regardless of the learning-rate schedule*, which means the binding
constraint is **overfitting, not an unfinished anneal**, and `best_state` selection was already
handling it. Run 1's early stopping at ~20/30 was doing the right thing.

One fold is weak evidence and this is not yet closed. But the prior has moved from "the anneal
cost us something" to "it did not", and the remaining ~76 fold-epochs would most likely buy a
confirmed null. Worth finishing only if no GPU appears; 128³ at −0.02 to −0.05 dominates it.

**A defect in yesterday's instrumentation, found by this run.** `_schedule_section` read the
`schedule` block out of `metrics.json` — which `run_cv` writes only *after* the whole fold loop.
An interrupted run therefore had every `history_fold*.json` safely on disk and no table in the
report: the reporting went blind in precisely the case the per-epoch banking existed for. Now it
reads the history files directly, computes the peak rate from `history[0]["lr"]` (recorded before
the first `scheduler.step()`, so it *is* `cfg.train.lr`), and marks the schedule length `?` when
the config was never written. `tests/test_train_smoke.py` pins the partial-run case.

Lesson, and it is the same one twice in two days: **the diagnostic has to survive the failure it
is meant to diagnose.** Banking the curve every epoch was right; putting the report behind a file
that only exists on success undid it.

### Fold structure (2026-09-05, after the signature fix)

21 acquisition groups over n=1362. Stratified folds are 272-273 each at 54.8-54.9% positive.
Grouped folds are deliberately uneven:

| fold | n (group) | pos rate |
|---|---:|---:|
| 0 | 460 | 59.3% |
| 1 | 325 | 58.2% |
| 2 | 115 | 51.3% |
| 3 | 313 | 50.2% |
| 4 | 149 | 46.3% |

Two things to carry forward:

**Grouped log loss is penalised twice, and only one of the penalties is a model defect.** With
validation priors from 46.3% to 59.3%, each grouped fold has a train/val prior mismatch, so the
model over- or under-predicts systematically before discrimination is even considered. AUROC is
invariant to that. **Read grouped AUROC and grouped log loss separately**: AUROC holding while
log loss degrades means calibration, not architecture.

**Per-fold grouped scores are not comparable to each other** — fold 2 trains on 1247 and
validates on 115; fold 0 trains on 902 and validates on 460. Only pooled OOF means anything.
Per-fold numbers are for spotting a fold that fell over.

### Possible prior shift between train and test (VOID — see the correction below)

> **Correction, 2026-09-08: the entire section below rests on a false premise.** The platform's
> smoke test runs against **fake data** — "a small set of fake data", excluded from both the
> submission limit and the leaderboard. Row 0b's 0.6687 was therefore scored on synthetic
> labels, and **no** statement about the real test set's positive rate can be inferred from it.
> The arithmetic below is correct and the conclusion it reaches is meaningless.
>
> This is a better reason to close row 0b than the one recorded further down, which argued the
> observation was sampling noise. It was not noise — it was a measurement of something else
> entirely. Keep the section as written, because the mistake is instructive: a number that
> *could* be computed was computed, and the question of what it measured got asked afterwards.
>
> The real prior-shift evidence is run 1's CV↔LB gap of **+0.0024**, which is on real data and
> settles it: no meaningful shift.

### The original argument (2026-09-05)

The prior-only submission scored **0.6687** on the smoke set. For a constant prediction `p`
against a test set with positive rate `q`, `logloss = 0.7952 - 0.1947·q` at `p = 0.5485`, so
that score implies **q ≈ 0.65** — against a training positive rate of **0.5485**.

Whether that is real depends entirely on the smoke set size, which `main.py` logs as
`Loaded submission format: N rows`:

| N | 0.65 vs 0.5485 is |
|---|---|
| ~30 | 1.1 SE — noise |
| ~100 | 2.0 SE — marginal |
| ~300 | 3.5 SE — real |

**Do not tune `FALLBACK_PROB` to this.** It is a single observation on a sample seen once, and
fitting a constant to leaderboard feedback is probing rather than modelling.

Evidence it is probably noise: the highest positive rate of any large acquisition family is
59.3% (see fold structure above), so **no mix of these centres produces a 65% test set**. A
q that high would need the small high-prevalence tail to be heavily over-represented, or the
smoke sample to be small enough for 0.65 to be an ordinary draw from 0.5485.

It does matter if real: a model calibrated on 55%-positive OOF data systematically under-predicts
on a 65%-positive test set, and log loss punishes that directly. The response would be a prior
correction applied in logit space at inference (a constant offset, shipped as a literal — still
per-sample independent, so it stays inside the rules), *not* refitting anything on test. Revisit
once the full test score is available, which is a much larger sample than the smoke set.

### CV is trustworthy, and that changes the plan (2026-09-08)

**OOF 0.3186 → leaderboard 0.3210. A gap of +0.0024, under 1% relative.**

Three questions close on this single number.

**1. Is CV trustworthy?** Yes. Phase 2's stated exit criterion was being able to say "my CV is
trustworthy to within X and the leaderboard tracks it". X = 0.0024. Every Phase 3 decision can
now be made on OOF log loss without waiting for a submission slot.

**2. Is there prior shift between train and test?** No. The smoke score implied a test positive
rate near 0.65 against a training rate of 0.5485; a shift that size would have cost far more
than 0.0024. That observation said nothing about the test set: the smoke test runs on fake data. **Row 0b is closed.**
      *(Superseded 2026-09-08: it was not noise. The smoke test runs on **fake data**, so row
      0b's score measured nothing about the real test set at all. Row 0b stays closed, for a
      stronger reason. See the correction above the prior-shift section.)*

**3. Is the grouped-CV run still worth scarce CPU?** Probably not, and this reverses the earlier
recommendation. Grouped CV exists as a *proxy* for "will the leaderboard track my CV". The
leaderboard has now answered that directly, to within 0.0024. Stratified CV is evidently the
right estimator for *this* test set, which suggests the test split shares the training set's
centre composition rather than holding out whole hospitals — in which case grouped CV would be
actively pessimistic about leaderboard performance and would mislead if optimised against.

It remains the right measurement for a different question — "would this generalise to an
eleventh hospital" — which is a real question about the solution's worth, just not one the
competition scores. **Run it if compute becomes cheap; do not spend a CPU night on it.**

### Submission logs must not describe the test set (2026-09-08)

Run 1's submission came back with a compliance warning: three log lines replaced by
`WARNING: line filtered`, and a note that non-compliance may result in disqualification.

The filtered lines were exactly the three containing `len(submission)`. The line reporting
`Loaded 5 fold models on cuda` survived, so numbers as such are fine — **the test set's row
count is not**, in any form: a total, a progress denominator, a row index, or a failure count.

`submission_src/main.py` now reports progress as elapsed time and fallbacks as whether-not-how-
many, and `05_package_submission.py` scans logger calls at package time (`check_logging`), with
`tests/test_logging_compliance.py` pinning both the scanner and the shipped file.

Two things worth carrying forward. First, the warning is content-based, so "log less" is the
wrong lesson — the right one is "log nothing derived from the test data". Second, this was only
diagnosable because the surviving line acted as a control; had every line been filtered there
would have been nothing to reason from.

### Calibration was being selected in-sample (found and fixed 2026-09-08)

Run 1's first evaluation chose **isotonic** at an OOF log loss of 0.3079. That number was wrong,
and the way it was wrong is worth remembering.

`choose_calibrator` fitted each candidate on the OOF predictions and scored it **on those same
predictions**. Temperature scaling has one parameter and barely benefits from that; isotonic
regression has of order *n* knots and can drive in-sample log loss almost arbitrarily low. So
isotonic won every comparison by construction. The tell was an ECE of exactly **0.0000** — not a
well-calibrated model, a curve memorising its own training data.

Re-scored with 5-fold cross-validation *inside* the OOF set:

| method | in-sample (wrong) | cross-validated (honest) |
|---|---:|---:|
| isotonic | 0.3079 | **0.3520** |
| temperature | 0.3186 | **0.3217** |
| none | 0.3271 | 0.3271 |

**Isotonic is worse than applying no calibration at all.** Shipping it would have cost real
leaderboard points while the report claimed a gain.

A second bug rode along with it: `evaluate.py` emitted `--temperature 1.0000` whenever the
winner had no `.temperature` attribute, so following its own instructions would have shipped an
**uncalibrated** model. The shipping command is now always built from the temperature scaler,
with an explicit note when that differs from the winner.

General lesson, applicable to every comparison from here: **the more flexible method wins any
comparison scored on its own training data.** Always hold something out.

### Phase 3, first three experiments (2026-09-08)

Rows 2-4. All three reuse run 1's checkpoints, so together they cost under an hour of CPU
against the ~2 days a retraining run would take (docs/05). Full reports sit beside this file.

**Row 2 closed two open questions.** The leave-one-family-out temperature transfer penalty is
**+0.0006** — the calibration constant is a property of the model, not of the centre mix it was
fitted on, so the shipped T is safe on unseen centres. And per-family *oracle* logit offsets,
using each family's own labels, reach only 0.3142 against 0.3186 for the plain global
temperature: a ceiling of about **0.004** on any per-centre prior correction, with cheating.
**Row 0b is closed.**

**One lead is still open.** The largest acquisition family, `256@2.5mm` (n=460, 34% of the
data), is the weakest of the large families: log loss 0.3691 and AUROC 0.9176, against 0.2339
and 0.9560 for `128@2.4mm`. Spacing does not separate those two; **field of view does** — 640 mm
against 307 mm — and EDA already recorded that the head occupies ~1% of the array in the widest
acquisitions, where centre-of-mass cropping has least margin. `scripts/10_features_by_family.py`
tests this in seconds via `box_max / global_max`, which is ~1 when the crop caught the striatum.
Not yet run. A crop fix would need a retraining run, so confirm the pattern before spending it.

**Row 3 (TTA) is a marginal keep at best.** −0.0022 is the same order as the CV↔LB gap of
+0.0024, so it may not be visible on the leaderboard at all. AUROC barely moved, so it is buying
variance reduction rather than better ranking. The shipping path is built and tested
(`05_package_submission.py --tta`, off by default and bit-identical when off), so the marginal
cost of including it is near zero — but it is a moving part in `main.py` five days out.

**Row 4 (the blend) is the real result.** Stacking is worth **−0.0175**, eight times TTA, and
the evidence holds up: the two logit vectors correlate 0.797 so the tree model genuinely
disagrees with the CNN; in-sample 0.3000 against cross-validated 0.3048 is only 0.0048 of
optimism on three parameters; and the comparison is a nested-model test with the same estimator
and folds on both sides, so the gain is not recalibration the CNN would have had anyway.

Two traps were found and fixed while building it, both recorded in `scripts/09_blend.py`. A
convex blend weight conflates mixing with calibration — lowering it shrinks the CNN's logits,
which is temperature scaling by another name, and against an overconfident CNN a *pure noise*
second model then scores an apparent gain. And the fold column must be stratified: on
unstratified folds a tree model fitted on noise learns each fold's training base rate, which is
anti-correlated with its own, and manufactures a gain from nothing.

**Still to do before either ships.** Put both through `scripts/11_compare_runs.py`, which does a
*paired* per-scan comparison — the unpaired standard error at n=1362 is ~0.016 and would call a
0.002 change unresolvable, while the paired one is of the difference itself. Then, for the
blend: refit the tree model on `lightgbm` rather than scikit-learn, whose pickle format is
fragile against whatever version the runtime has, and vendor it with the stack coefficients.
`src/datpd/features.py` is already the single definition of the 61 features and is already
vendored, so inference cannot drift from what the model was fitted on.

### The blend and TTA together, packaged 2026-09-09

`submission.zip`, 38.3 MB: 5 CNN fold checkpoints, 5 lightgbm boosters, `stack.json`, and
`datpd_vendored/features.py`. Baked constants verified against the source of truth after the
build — `USE_BLEND` true, 9 TTA views, 61 features in an order matching the vendored code,
`source_run: baseline_tta`.

**The gains are not additive.** TTA alone was −0.00215 and the blend alone −0.01458, which would
sum to −0.0167; together they measure **−0.01562**. Both partly work through variance reduction,
so they overlap. Worth remembering before assuming two independent-looking improvements will
stack.

**lightgbm rather than scikit-learn**, decided before measuring rather than after. A
scikit-learn estimator ships as a pickle whose compatibility with the runtime's version is not
guaranteed; lightgbm's booster is text and loads across versions. Measuring with one model and
shipping another would have meant the number that justified the change was never the number
submitted. Re-measured on lightgbm the gain held (−0.01562 against sklearn's −0.01458).

**The stack replaces temperature scaling, it does not compose with it.** `STACK_COEF_CNN` is
0.6164 — fitted on logits that were never temperature-scaled, so it already carries the
calibration, equivalent to a temperature of about 1.62. Applying `TEMPERATURE` as well would
soften twice and hand back most of the gain. The packager refuses `--temperature` alongside
`--blend-from`, and a test asserts the blend code path never references it.

**Two silent failure modes are guarded**, both of which raise nothing and produce plausible
wrong probabilities: a feature vector assembled in a different column order than the trees were
fitted on (`stack.json` records the fit-time order, the loader asserts against it), and a blend
switched on without its assets staged.

Expect **~0.305** on the leaderboard, from 0.3210 less the 0.0156, read against the +0.0024
CV↔LB gap established by run 1.

### It came back at 0.2973 — better than predicted, and the reason is not the method (2026-09-09)

**LB 0.2973, AUROC 0.9396, rank #98/897.** Prediction was ~0.305. Runtime 15 min of 180, no
fallbacks, and — the other thing this submission was testing — **zero filtered log lines and no
compliance warning**, so the §3.1 logging rewrite is validated on the platform. Full log in
`reports/submissions/002_2026-09-09_tta_blend.md`.

The temptation is to conclude the blend is worth more than measured. It is not, and the evidence
against that is in the same result.

| | OOF | LB | gap |
|---|---:|---:|---:|
| Run 1 | 0.3186 | 0.3210 | **+0.0024** |
| Submission 2 | 0.3061 | 0.2973 | **−0.0088** |

The realised gain is 0.3210 − 0.2973 = **−0.0237**, against a measured −0.01562 with CI
[−0.0255, −0.0057]. So the leaderboard gain sits near the *top* of the CI while the OOF estimate
sat in the middle — an over-performance of about −0.008.

**AUROC settles where that came from.** OOF AUROC 0.9375 → LB AUROC 0.9396. If the surplus were
the blend calibrating better on the test set than on OOF, log loss would improve and AUROC would
not move; log loss is calibration-sensitive and AUROC is calibration-invariant. **Both improved.**
That is a test set that is a marginally easier draw for this model, not a method that transfers
better than it measured.

**Two consequences, and the second is the important one.**

First: keep valuing TTA + blend at the −0.0156 that was measured. Do not re-plan around −0.0237.

Second, and this revises a claim this file has been leaning on hard: **the CV↔LB gap is not
±0.0024.** It is now two observations, +0.0024 and −0.0088, on opposite sides of zero. A single
point measured nothing about the spread — it could not, having no second point to differ from.
The honest bound is roughly **±0.01**.

That is still small enough for the purpose it was put to: Phase 3 decisions were made on gains of
−0.0156 and −0.0175, comfortably clear of ±0.01, and they held up. But it is four times looser
than the figure quoted in "CV is trustworthy, and that changes the plan", and it changes the
threshold going forward. **Any future decision resting on an OOF margin under ~0.01 is unresolved,
not won** — including a 128³ run, which needs to clear that bar on OOF before it is worth
repackaging for. TTA alone, at −0.0022, was always inside this band; it shipped as a rider on the
blend rather than on its own evidence, which was the right call for a different reason (near-zero
marginal cost) than the one recorded at the time.

The general form of the mistake avoided here: **a gain larger than predicted is not confirmation,
it is an unexplained residual.** It gets the same scrutiny as a shortfall would.

### Decisions made

_(record each decision and the evidence for it — this becomes docs/SOLUTION.md at the end,
which winners are required to submit)_
