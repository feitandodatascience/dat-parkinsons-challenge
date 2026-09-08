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
| 1 | 2026-09-08 | `baseline.yaml` | | first end-to-end run: 96³ @ 2.0 mm, `crop: com`, ResNet3D, 5-fold stratified | **0.3186** | not yet run | 0.9302 | 0.0220 | _packaged, awaiting score_ | yes | temperature = 1.2614, clip 1e-06. Beats the prior by 0.3698. See the calibration note below — the first evaluation of this run was wrong |

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

### Possible prior shift between train and test (open, 2026-09-05)

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

### Decisions made

_(record each decision and the evidence for it — this becomes docs/SOLUTION.md at the end,
which winners are required to submit)_
