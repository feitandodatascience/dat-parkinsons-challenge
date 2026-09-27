# 05 — Phase 3 under a CPU budget

**Written 2026-09-08. Deadline 2026-09-16; intended final model submitted Mon 2026-09-14.**

The Phase 3 ranking in [03_plan.md](../03_plan.md) was written assuming a GPU would be rented before
day 5. It has not been, so that ranking is now wrong — not because the ideas changed, but because
the denominator did. This file re-ranks the same list by **expected log-loss gain per CPU-hour**,
which is the only currency left.

The headline change: the plan's #1 (higher resolution) is unreachable, and the plan's #8
(handcrafted SBR features) is the best remaining ratio by a wide margin. Both for the same
reason — on CPU, *anything that requires retraining the 3D CNN costs roughly the whole remaining
budget*, and anything that reuses the existing checkpoints is nearly free.

---

## 1. The unit of account

Let **R = wall-clock hours for one full 5-fold CV run of `conf/baseline.yaml`** (96³ @ 2.0 mm,
30 epochs, this machine). Run 1 is exactly that, so R is already known — from the run's start and
end times, not from an estimate.

> **R = 16.6 hours.** Measured 2026-09-15, not estimated: `conf/timing_probe.yaml` trains fold 0
> for exactly one epoch at run 1's settings, against the warm cache, and reports **399 s**.
> Five folds x 30 epochs = 150 fold-epochs x 399 s = 16.6 h.
>
> **This blank was load-bearing, and the placeholder was wrong by 3x.** [03_plan.md](../03_plan.md)
> guessed ~48 h; [07_cloud_compute_compliance.md](07_cloud_compute_compliance.md) then back-derived
> "2.4 R ~ 115 CPU-hours" from that guess, and from there "a retrain is unaffordable on CPU"
> propagated into every Phase 3 decision and into the next-session brief. It was never checked
> against a clock, because the one run that could have answered it did not persist its own
> timings — the same failure recorded in the experiment log under "the run could not answer
> questions about itself". A one-epoch probe costs 7 minutes and would have settled it on day one.
>
> What it changes, on 2026-09-15 with ~31 h left:
>
> | | at R = 48 h (assumed) | at R = 16.6 h (measured) |
> |---|---|---|
> | Retrain at 96 cubed | unaffordable | **16.6 h — fits** |
> | 128 cubed @ 1.5 mm (2.37x voxels) | 115 h, absurd | 39.4 h — **still does not fit**, but now closed on a measurement |
> | Second seed for ensembling | unaffordable | affordable, and still not worth it: rho = 0.974 predicts -0.002 |
>
> So the headline conclusion of this file survives — higher resolution is out of reach — but for
> a reason three times weaker than the one written down, and the retrain it ruled out was
> affordable all along.

Costs below are derived from training work, counted as *scan-epochs* (samples pushed through
forward+backward). Across any 5-fold CV the total is 4n = 5448 scan-epochs per epoch of
schedule, whichever fold column is used — so **grouped CV costs the same as stratified CV**. The
savings come only from doing fewer folds, fewer epochs, or smaller volumes.

At the deadline there are about **6 calendar days and therefore ~5 usable CPU-days** before the
Mon 14 Sep lock-down. If R ≈ 48 h, that is **~2.5 R total**. Spend it deliberately.

---

## 2. Ranked shortlist

| # | change | cost | expected Δ log loss | ratio | verdict |
|---|---|---:|---|---|---|
| 1 | Per-family error decomposition + LOGO temperature | **0** | 0 to −0.02 | ∞ | **do first, today** |
| 2 | Test-time augmentation on existing checkpoints | **~0.02 R** | −0.005 to −0.02 | very high | **do** |
| 3 | SBR-style features → GBM → logit blend | **~0.03 R** | −0.01 to −0.03 | very high | **do** |
| 4 | Second seed, ensembled | 1.0 R | −0.005 to −0.015 | low | only if a night is spare |
| 5 | Label smoothing 0.05 | 1.0 R | ~0 | low | **skip** |
| 6 | `flip_lr` ablation | 1.0 R | likely +ve (worse) | low | **skip** |
| 7 | Grouped CV | 0.36–1.0 R | answers a question the LB already answered | low | **deprioritised** |
| 8 | 128³ @ 1.5 mm | **2.4 R** on CPU, ~4 h on an A100 | −0.02 to −0.05 | high, *if* a GPU is rented | **reopened** — see docs/07 |
| 9 | Pretrained 3D backbone (DenseNet121) | >2 R | unknown | unreachable | **rule out** |
| 10 | 2.5D timm model | >2 R | unknown | unreachable | **rule out** |

> **Revised 2026-09-08, after the leaderboard came back.** Run 1 scored **0.3210** against an
> OOF of 0.3186 — a gap of +0.0024. That single number moved grouped CV from #4 to #7. Grouped
> CV is a *proxy* for "will the leaderboard track my CV"; the leaderboard has now answered
> directly, and stratified CV is evidently the right estimator for this test set. Running it
> would measure generalisation to an unseen eleventh hospital — a real question about the
> solution's worth, but not one the competition scores, and not worth a CPU night at this
> point in the calendar. The reasoning is in `reports/experiments/log.md`.
>
> Items 1–3 get *more* valuable from the same number, not less: a trustworthy CV means each of
> them can be accepted or rejected on OOF alone, with no submission slot spent finding out.

### 1. Per-family error decomposition — free

`python scripts/06_score_by_group.py --run baseline`

Runs in seconds against the OOF predictions run 1 already produced. Two outputs.

**Per-family AUROC.** The question worth asking now that the leaderboard has validated CV: is
discrimination uniform across acquisition families, or is one family carrying most of the error?
A family with visibly worse AUROC is not a calibration problem — it points at preprocessing
failing on that geometry, most plausibly the COM crop on the widest fields of view, where the
head occupies ~1% of the array. That is a fixable bug with real log loss attached, and it costs
nothing to look for.

**Leave-one-family-out temperature transfer.** Fit the calibration on every family except one,
apply it to the held-out one, pool. The shipped submission carries a single scalar temperature,
so this checks that the constant is a property of the model rather than of the centre mix it was
fitted on. Less urgent than it was before the leaderboard came back — the test set evidently
resembles the training composition — but it is free, and it is the cheap half of the question
grouped CV would have answered expensively.

The report's last section also bounds the row-0b prior-shift question from the other side, by
giving each family an oracle logit offset and reporting the ceiling that per-centre prior
correction could ever reach.

### 2. Test-time augmentation — ~0.02 R

```bash
python scripts/07_tta.py --config conf/baseline.yaml --run baseline --limit 40   # smoke first
python scripts/07_tta.py --config conf/baseline.yaml --run baseline
python scripts/04_evaluate.py --run baseline_tta
```

Forward passes only, no retraining. A full OOF pass with k=8 augmentations is ~11k forward
passes against the ~500k forward-equivalents that R buys, hence the 2%.

Use small rotations, translations and scale jitter — the same distribution as training
augmentation. **Do not use `flip_lr`**: left-right asymmetry is diagnostic in DaT (docs/01 §4),
and averaging a scan with its mirror deletes exactly the signal the task depends on.

Rules check: TTA augments a single scan and averages that scan's own predictions, so it stays
per-sample independent (CLAUDE.md §1.1). Average in **logit** space, consistent with how the
fold ensemble is already combined, then **refit the temperature** — TTA averaging softens
predictions on its own, so the existing T=1.2614 will be wrong for it.

**Can it be shipped inside the 3-hour inference budget?** Yes, comfortably. Measured on this
machine (`ResNet3D(16,32,64,128)` at 96³, CPU):

| | per scan |
|---|---:|
| one forward pass | 112 ms |
| nine views, batched | 437 ms (**3.9×**, not 9× — the batch amortises) |
| 5 fold models × 1 view | 0.56 s |
| 5 fold models × 9 views | 2.19 s |

Against the ~2.5 s/scan observed during the TTA smoke run, the model work is 0.44 s and
**preprocessing is the other ~83%** — and TTA does not touch preprocessing. On the platform's
A100 the forward passes are faster still while preprocessing stays CPU-bound, so nine views add
a few percent to total runtime, not 9×. At ~2 s/scan the 3-hour budget covers roughly 5,000
scans single-threaded, with 24 vCPUs available to parallelise preprocessing if it ever gets
tight. Inference cost is not the reason to reject TTA; only the OOF result is.

### 3. SBR-style features + GBM blend — ~0.03 R

The plan ranks this last. On CPU it should be near the top, because it is the only modelling
item that buys *model diversity* without a CNN retrain: one preprocessing pass over 1362 scans
(cached already), a few dozen geometric features, and a LightGBM 5-fold CV that finishes in
seconds.

Features worth extracting, all computed from the existing COM-cropped canonical volume:
striatal-box uptake relative to the occipital reference region, left/right asymmetry index,
putamen-to-caudate gradient along the anterior-posterior axis, and the shape descriptors that
distinguish a comma from a dot (eccentricity, extent of the supra-threshold region).

```bash
python scripts/08_sbr_features.py --config conf/baseline.yaml --limit 50   # smoke first
python scripts/08_sbr_features.py --config conf/baseline.yaml            # ~30 min
python scripts/09_blend.py --run baseline                                 # seconds
```

Three traps, two of which were found by writing this and are now guarded in the scripts:

- **The blend weight must be chosen out of sample**, or it is the isotonic mistake one level up.
- **A convex weight `w*cnn + (1-w)*gbm` conflates mixing with calibration.** Lowering `w` also
  shrinks the CNN's logits, which is temperature scaling by another name — so against an
  overconfident CNN, a *pure noise* second model scores an apparent gain. `09_blend.py` fits a
  logistic stack instead and compares against the same stack over the CNN alone, so both sides
  carry the same recalibration and the difference is attributable to the features.
- **The fold column must be stratified on the label.** A tree model cross-validated on
  unstratified folds picks up each fold's training base rate, which is anti-correlated with its
  own, and manufactures a small gain from nothing. `fold_strat` is safe; do not substitute
  `fold_group` here.

One caveat on shipping: the script uses scikit-learn's `HistGradientBoostingClassifier` because
it is already installed locally. If the blend pays, refit with `lightgbm` before shipping — its
text booster format survives a version difference that a scikit-learn pickle would not.

### 7. Grouped CV — deprioritised, but here is how to run it cheaply

No longer the priority (see the revision note above), but if a night comes free the
infrastructure to get it cheaply already exists. `train.py` banks `oof_fold<k>.csv` the moment each fold finishes, and
`scripts/04_evaluate.py` assembles a partial OOF from whatever completed. So do not commit to
five folds up front — **run the folds largest-validation-set first and stop when the CPU is
needed elsewhere**:

```bash
python scripts/03_train.py --config conf/grouped_cv.yaml --folds 0 1 3 4 2
python scripts/04_evaluate.py --run grouped_cv     # score whatever has finished
```

Fold order 0, 1, 3, 4, 2 is by validation-set size (460, 325, 313, 149, 115), so each hour spent
buys the most OOF coverage available at that moment:

| stopping after | OOF coverage | cumulative cost |
|---|---:|---:|
| fold 0 | 460 / 1362 (34%) | 0.17 R |
| folds 0, 1 | 785 (58%) | 0.36 R |
| folds 0, 1, 3 | 1098 (81%) | 0.55 R |
| all five | 1362 (100%) | 1.00 R |

**Stopping at 58% coverage is enough to answer the question.** The standard error on a log loss
estimated from 785 samples is around ±0.02 — far smaller than the strat-vs-grouped gap worth
reacting to. Folds 0 and 1 are also the two highest-prior folds (59.3%, 58.2%), so an early stop
is biased *pessimistic*, which is the right direction for a risk measurement.

Read the result the way `log.md` prescribes: **grouped AUROC and grouped log loss separately.**
AUROC holding while log loss degrades is a calibration problem with a cheap fix (ship a softer
temperature). Both degrading is a representation problem, and at this point in the calendar
there is no time to fix that — the response would be to ship a more conservative temperature and
accept it.

### 5–6. Why label smoothing and `flip_lr` come off the list

Not because they are bad ideas — because each costs a full R to evaluate and neither is likely
to repay it *here*.

Label smoothing is ranked 5th in the plan for its calibration benefit, but this pipeline already
fits a post-hoc temperature on OOF predictions, which addresses the same overconfidence more
directly and for free. The two largely overlap; buying the overlap for 1.0 R is poor value.

`flip_lr` has a stated prior that it hurts, and a well-argued domain reason (docs/01 §4). Paying
1.0 R to confirm a negative result, with two days left, is the definition of a bad ratio. It
stays an open question in the log — that is the correct place for it.

### 8–10. What "unreachable" means

128³ @ 1.5 mm is 2.37× the voxels of 96³, so ~2.4 R ≈ 5 days of the ~2.5 R available. It is very
likely the single largest available win and it cannot be had on this hardware. If GPU access
appears before Sat 12 Sep, **this is the one thing to spend it on** — ahead of everything else in
this file, including item 4.

> **Reopened 2026-09-09.** The organizers have answered the cloud-compute question: renting GPU
> compute is permitted, subject to three conditions (no third-party retention, private to the
> competitor, every copy deletable). [07_cloud_compute_compliance.md](07_cloud_compute_compliance.md)
> records the ruling, the providers it eliminates, and the operational shape that satisfies it.
> `conf/highres.yaml` is the config; `train.py` needs no change.
>
> Two corrections to the framing above, both from docs/07. **Compute stops being the constraint**
> — 2.4 R is ~4 hours on an A100, not five days. And **the refit chain becomes the constraint**:
> the shipped temperature and stack coefficients are properties of the 96³ geometry and are void
> at 128³, so accepting this item means re-running all of Phase 3 and repackaging before Sun 13.
> Judge it on OOF against the current build's 0.3061 — the +0.0024 CV↔LB gap means no submission
> slot need be spent to decide, which matters because spending the 09-12 slot forecloses 09-14.

The pretrained-backbone route carries a second problem beyond compute: most public DaT
checkpoints are PPMI-derived, and PPMI makes a solution prize-ineligible
([EXTERNAL_ASSETS.md](../EXTERNAL_ASSETS.md)). Any candidate needs its training provenance checked
before it is worth benchmarking.

---

## 3. The suggested order

Items 1–3 together cost under **0.06 R** — a couple of hours, not a night. That is the whole
argument for doing them first: they are decided before any expensive option would have finished
its first fold, and a trustworthy CV (gap +0.0024) means each can be accepted or rejected on OOF
alone.

1. **Item 1** (seconds). Read the per-family AUROC before anything else — if one acquisition
   family is carrying the error, fixing that outranks everything below.
2. **Item 2**, smoke first with `--limit 40`, then the full pass. Score with `04_evaluate.py`,
   which refits the temperature; TTA softens logits, so the existing T=1.2614 will not apply.
3. **Item 3**, smoke first with `--limit 50`. Feature extraction is the slow half (~30 min);
   the blend itself is seconds and can be re-run freely.
4. Log all three in `reports/experiments/log.md`, **including the ones that do not pay**. A
   negative result recorded today is what stops it being retried on the 14th.
5. Only then consider item 4 (a second seed) with whatever nights remain.
6. **Sun 13**: freeze. Repackage, Docker smoke test, audit `main.py` against §1.1 and the
   logging rule.
7. **Mon 14**: submit. Two days of buffer remain, deliberately.

If a GPU appears before Sat 12 Sep, stop and spend it on item 8 (128³ @ 1.5 mm) ahead of
everything here. It is very likely the largest single win available and it is the only one this
hardware cannot reach.

The thing most likely to go wrong is not modelling. It is arriving at Monday with an improvement
that was measured but never packaged.
