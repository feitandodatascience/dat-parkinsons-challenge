# 05 — Phase 3 under a CPU budget

**Written 2026-09-08. Deadline 2026-09-16; intended final model submitted Mon 2026-09-14.**

The Phase 3 ranking in [03_plan.md](03_plan.md) was written assuming a GPU would be rented before
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

> **R = ______ hours** ← fill this in; every row below is priced in it.
> [03_plan.md](03_plan.md) guessed ~48 h. If the preprocessing cache was cold for run 1 and is
> warm now, the true figure for a *repeat* run at the same geometry is lower, possibly much
> lower — `conf/grouped_cv.yaml` uses byte-identical preprocessing settings, so it reuses the
> cache in `data/processed/cache` and pays no resampling cost at all.

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
| 4 | Grouped CV, folds ordered by size, stop when needed | 0.36–1.0 R | measurement, not gain | — | **do, in background** |
| 5 | Second seed, ensembled | 1.0 R | −0.005 to −0.015 | low | only if 4 finishes early |
| 6 | Label smoothing 0.05 | 1.0 R | ~0 | low | **skip** |
| 7 | `flip_lr` ablation | 1.0 R | likely +ve (worse) | low | **skip** |
| 8 | 128³ @ 1.5 mm | **2.4 R** | −0.02 to −0.05 | unreachable | **rule out** |
| 9 | Pretrained 3D backbone (DenseNet121) | >2 R | unknown | unreachable | **rule out** |
| 10 | 2.5D timm model | >2 R | unknown | unreachable | **rule out** |

### 1. Per-family error decomposition — free

`python scripts/06_score_by_group.py --run baseline`

Runs in seconds against the OOF predictions run 1 already produced. It answers Phase 2's second
task ("is one centre carrying all the error?") and, more importantly, produces the
**leave-one-family-out temperature transfer penalty**: fit the calibration on every acquisition
family except one, apply it to the held-out one, pool.

That number is the *calibration half* of the stratified-to-grouped gap, and it is the half that
actually transfers to the leaderboard, because the submission ships exactly one scalar
temperature that has to work on centres it has never seen. It cannot measure the other half —
whether the network's features are centre-specific — because these folds are stratified and the
model saw every family in training. **It is a lower bound on the grouped gap, not a substitute
for item 4.**

It also bounds the row-0b prior-shift question: the report's last section gives every family its
own oracle logit offset and reports the pooled result. If that headroom is small, per-centre
prior shift is second-order and the question can be closed.

### 2. Test-time augmentation — ~0.02 R

Forward passes only, no retraining. A full OOF pass with k=8 augmentations is ~11k forward
passes against the ~500k forward-equivalents that R buys, hence the 2%.

Use small rotations, translations and scale jitter — the same distribution as training
augmentation. **Do not use `flip_lr`**: left-right asymmetry is diagnostic in DaT (docs/01 §4),
and averaging a scan with its mirror deletes exactly the signal the task depends on.

Rules check: TTA augments a single scan and averages that scan's own predictions, so it stays
per-sample independent (CLAUDE.md §1.1). Average in **logit** space, consistent with how the
fold ensemble is already combined, then **refit the temperature** — TTA averaging softens
predictions on its own, so the existing T=1.2614 will be wrong for it.

### 3. SBR-style features + GBM blend — ~0.03 R

The plan ranks this last. On CPU it should be near the top, because it is the only modelling
item that buys *model diversity* without a CNN retrain: one preprocessing pass over 1362 scans
(cached already), a few dozen geometric features, and a LightGBM 5-fold CV that finishes in
seconds.

Features worth extracting, all computed from the existing COM-cropped canonical volume:
striatal-box uptake relative to the occipital reference region, left/right asymmetry index,
putamen-to-caudate gradient along the anterior-posterior axis, and the shape descriptors that
distinguish a comma from a dot (eccentricity, extent of the supra-threshold region).

Two traps:

- **Blend weight must be chosen out-of-sample.** Fitting the CNN/GBM mixing weight on the OOF set
  and then reporting the OOF score of that blend is the same in-sample error that made isotonic
  regression "win" on 2026-09-08. Use the cross-validation-within-OOF pattern from
  `choose_calibrator`.
- Recalibrate the blend. An ensemble's calibration is not its members' calibration.

### 4. Grouped CV — 0.36 R to 1.0 R, and it is interruptible

This is the number this phase exists to get, and the infrastructure to get it cheaply already
exists. `train.py` banks `oof_fold<k>.csv` the moment each fold finishes, and
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

### 6–7. Why label smoothing and `flip_lr` come off the list

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

The pretrained-backbone route carries a second problem beyond compute: most public DaT
checkpoints are PPMI-derived, and PPMI makes a solution prize-ineligible
([EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md)). Any candidate needs its training provenance checked
before it is worth benchmarking.

---

## 3. The suggested order

1. **Now**: item 1 (free). Read the transfer penalty before deciding how much item 4 is worth.
2. **Tonight**: start item 4 in the background with the fold ordering above. Let it run.
3. **Wed 9 – Thu 10**: items 2 and 3 in the foreground, accepting that they contend with item 4
   for cores. Both are short enough that the contention costs less than the serialisation would.
4. **Fri 11 – Sat 12**: stop item 4 wherever it is. Score it. Decide the final shipped
   temperature from the grouped result and the LOGO penalty together — pick the more
   conservative of the two.
5. **Sun 13**: freeze. Repackage, Docker smoke test, audit `main.py` against §1.1 and the
   logging rule.
6. **Mon 14**: submit. Two days of buffer remain, deliberately.

The thing most likely to go wrong is not modelling. It is spending Saturday on item 8 because a
GPU appeared, and arriving at Monday with weights that were never packaged.
