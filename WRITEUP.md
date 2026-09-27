# DaT Parkinson's Challenge — write-up

DrivenData competition #311 (SFMN / Health Data Hub / GaelO). The task was to classify 3D DaT SPECT
brain scans from ten French hospitals as normal or abnormal. Scoring was by **log loss**, and
submissions ran as code on the platform: one A100, no network, a 3-hour limit. The competition
closed on 2026-09-16.

The full technical record is [docs/SOLUTION.md](docs/SOLUTION.md), and every experiment, including
the negative results, is in [reports/experiments/log.md](reports/experiments/log.md). This page is
the short version.

## Results

**Final rank: #159 / 1009** (top 16%) on the private leaderboard.

| # | Submission | OOF | Public LB | Private LB |
|---|---|---:|---:|---:|
| 1 | Baseline: 5-fold 3D ResNet + temperature scaling | 0.3186 | 0.3210 | 0.3767 |
| 2 | + 9-view TTA, stacked with LightGBM on 61 SBR-style features | 0.3061 | 0.2973 | 0.3573 |
| 3 | Candidate A: features repaired and extended to 157, booster tuned | 0.2929 | 0.2935 | 0.3476 |
| 4 | **Candidate B**: + a second 5-fold CNN run with richer augmentation (10 CNNs) | 0.2841 | **0.2855** | 0.3448 |
| 5 | Candidate C: candidate B with the tuned booster | 0.2821 | 0.2871 | **0.3447** |

For reference, predicting the training positive rate for every scan scores 0.6884.

Across the project, private log loss fell from **0.3767 to 0.3447 (−0.032, −8.5%)**. Every
submission improved on the private leaderboard over the one before, so no step that looked like
progress in cross-validation turned out to be a regression.

## What the private leaderboard showed

**The private set was harder, and roughly equally harder for every build.** Private minus public
was +0.054 to +0.060 for all five submissions. Because the shift was nearly constant, the ranking
between builds held up even though the absolute scores did not.

**Out-of-fold scores predicted the public board and not the private one.** On the public board the
out-of-fold → leaderboard gap stayed within about ±0.01, and I made my decisions against that band.
On the private board the gap was +0.05 to +0.06. Cross-validation ordered the builds correctly but
set their absolute level too low. A split this size probably differs in centre mix or prevalence.
I can't check which, because doing so would mean deriving statistics from the test set.

**The gains got smaller the later they came.** Private step gains were −0.019 (TTA + stack), then
−0.010 (feature repair), −0.003 (second CNN) and −0.0001 (tuned booster). The last step was
unresolved in cross-validation (−0.0020 at 1.3σ). The two leaderboards then disagreed on its sign:
+0.0016 on public, −0.0001 on private. That is what a null looks like, and it supports the rule I
adopted mid-project: an out-of-fold margin under ~0.01 is unresolved, not won.

## What worked

- **Centre-of-mass cropping.** Field of view ranged from 176 to 630 mm per axis. A fixed centre
  crop clipped the striatum in wide acquisitions and destroyed the left-right asymmetry the task
  depends on.
- **Diversity over quality.** On its own, the LightGBM model over handcrafted striatal features
  was much worse than the CNN (0.39 vs 0.32). Its logits correlated only 0.80 with the CNN's, and
  stacking it was the largest single gain. The richer-augmentation CNN was also *worse* alone
  than the baseline CNN, yet it improved the ensemble, for the same reason.
- **A logistic stack instead of a convex blend.** The stack's CNN coefficient absorbs the
  calibration, so no separate temperature is needed. A convex weight would have mixed
  calibration in with model mixing and credited a noise model with a gain.
- **No left-right flip augmentation.** Asymmetry is clinically meaningful, so mirroring a scan
  destroys signal rather than regularising.
- **Paired comparisons.** Per-scan paired differences with a bootstrap CI have about a quarter of
  the unpaired standard error. That is what made gains of 0.002–0.01 decidable, or honestly
  undecidable.

## What went wrong, or was left on the table

- **Higher resolution (128³ @ 1.5 mm) was never trained.** All training ran on a laptop CPU, and
  one full 5-fold run cost 16.6 h. I estimated that cost at ~48 h for most of the project, without
  measuring it, and the overestimate ruled out work that was in fact affordable. This was the
  largest expected gain not taken.
- **The calibrator was first selected in-sample**, which made isotonic regression look best. Under
  cross-validation it was worse than no calibration at all. Caught before shipping.
- **Three SBR features divided by means that cross zero.** The aggregate report showing it existed
  for a week before anyone read it. Fixing them, together with the booster tuning that only paid
  once they were fixed, was worth −0.010 on private.
- **The first submission's log included the test-set row count**, and the platform filtered those
  lines with a compliance warning. Logs now report elapsed time and whether a fallback fired, never
  counts. `05_package_submission.py` enforces this at package time.

## What I would do differently

1. Rent a GPU on day one. Nearly every rejected idea was rejected on compute, not on evidence.
2. Measure the cost of a training run before planning in units of it.
3. Treat the CV↔LB gap as a distribution, not a number. Two public points suggested ±0.01, but
   the private board moved every build by +0.06.
4. Look for models whose errors differ from the ones you already have, not only for better models.

## How it was built

The code was written with an AI coding assistant under a hard constraint: competition rules forbid
sharing the data with third-party AI services. Three layers enforced that the assistant wrote code
and never read the data: permission deny rules, a `PreToolUse` hook with its own test suite, and
the behavioural rules in `CLAUDE.md` §0. Only aggregate, user-vetted summaries crossed
from the data to the assistant. No external data or pretrained weights were used. PPMI in
particular was excluded by the organizers.

Reproduction steps are in [docs/SOLUTION.md §7](docs/SOLUTION.md#7-reproduction). The test suite
(`pytest -q`) runs on synthetic volumes only.
