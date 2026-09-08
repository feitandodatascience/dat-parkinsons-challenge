# 03 — Working plan: 11 days, learning while competing

**Today: 2026-09-05. Deadline: 2026-09-16, 23:59 UTC.** Eleven days, not the fifteen this plan
was originally written for. That is tight but workable — 801 people joined and most will not
submit anything. A completed, calibrated, honestly-validated submission puts you ahead of the
majority.

The plan is built on one principle: **get a valid submission on the leaderboard on day 1**,
then improve it. A great model that isn't packaged by the 16th scores nothing.

Learning is interleaved, not front-loaded. You learn the domain fact at the moment it changes a
decision you are making — that is when it sticks.

## What the four lost days cost, and where they come from

The compression is taken almost entirely out of Phase 3 (model improvement), because that is
the only phase with a long tail of optional work. Phases 1, 2 and 4 are load-bearing: skipping
validation gives you a number you cannot trust, and skipping the lock-down phase is how a
finished model fails to reach the leaderboard.

| phase | original | revised | dates |
|---|---|---|---|
| 0 — setup | day 1 am | day 1 am | Sat 5 Sep |
| 1 — see the data, submit something | days 1–2 | days 1–2 | Sat 5 – Sun 6 Sep |
| 2 — make validation trustworthy | days 3–5 | days 3–4 | Mon 7 – Tue 8 Sep |
| 3 — improve the model | days 6–11 | days 5–9 | Wed 9 – Sun 13 Sep |
| 4 — lock down and submit | days 12–15 | days 10–12 | Mon 14 – Wed 16 Sep |

**The one date that is not negotiable: submit your intended final model by Mon 14 Sep.** Days 15
and 16 are buffer for a failed submission, not for modelling. Submission limits are per day, so
a zip that fails at 23:00 on the 16th cannot be fixed.

## Status as of 2026-09-05

The repository scaffolding was written but `src/datpd/data/` and `src/datpd/models/` were
missing entirely, so every script failed at import. Those modules now exist and the full
synthetic suite passes. Concretely, this is now ready to run:

- `datpd.data.io` — canonical-RAS loading, header-only inspection, uid→path index
- `datpd.data.preprocess` — resample → crop → normalise, all scale-invariant
- `datpd.data.splits` — acquisition signature, stratified + grouped folds
- `datpd.data.dataset` — cached Dataset with 3D augmentation
- `datpd.models.resnet3d` / `factory` — the GroupNorm CNN from docs/02

`scripts/05_package_submission.py` also gained `--prior-only` and `--config`. See Phase 1.

**Data downloaded 2026-09-05** into `data/raw/`: the full NIfTI set, the training labels, and the
smoke-test archive. They arrive under suffixed names and need normalising before anything runs —
see Phase 0.

**Environment ready 2026-09-05.** `.venv` on Python 3.12.3 with `torch 2.12.1+cpu` (the runtime's
torch version), plus torchio, monai, simpleitk, timm, loguru and the dev tools. `00_check_env.py`
passes every row except CUDA; `pytest -q` is 98 passed, 0 skipped. Phase 0 is done apart from
Docker and the runtime-repo checkout.

Use `.venv\Scripts\python.exe` for everything — the system Python 3.14 is still first on PATH and
has none of this installed.

---

## Phase 0 — Setup (day 1, morning, ~2 hours)

| | Task |
|---|---|
| ☐ | **Install Python 3.12.** You have 3.14; the runtime is 3.12-only and some pinned wheels won't exist for 3.14. `py -3.12 -m venv .venv` then `.venv\Scripts\activate` |
| ☐ | `pip install -e ".[dev]"` |
| ☐ | `pytest -q` — the whole suite must pass on synthetic data, with no competition data present. It covers preprocessing, splits, the model, calibration, the guard hook and submission packaging. |
| ☐ | Accept the competition rules and **download the data yourself** into `data/raw/` |
| ☐ | **Normalise the download names.** They arrive suffixed (`niftis_XXXXXXX`, `train_labels_XXXXXXX`, `smoke_test_data_XXXXXXX.tar`). Rename to `data/raw/niftis/` and `data/raw/train_labels.csv`; move the smoke-test tar to the runtime repo checkout. Re-run `00_check_env.py` until both rows pass. |
| ☐ | Install **Docker Desktop** — needed to test submissions locally |
| ☐ | Clone the [runtime repo](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime) as a sibling directory, `just pull` (needs ~21 GB free) |
| ☐ | Confirm GPU: `python -c "import torch; print(torch.cuda.is_available())"` |

**Resolved 2026-09-05: this machine has no CUDA GPU** — Intel Iris Xe integrated graphics only.
`torch 2.12.1+cpu` is installed, which matches the runtime's torch version and is correct for
development, tests, EDA and packaging. It is not a training environment.

Rough order of magnitude on CPU, at n=1362 with ~1090 training scans per fold:

| config | per epoch | 30 epochs x 5 folds |
|---|---|---|
| 96³ @ 2.0 mm (`baseline.yaml`) | ~15-20 min | ~2 days |
| 64³ @ 3.0 mm (reduced) | ~5 min | ~14 h |

So a single reduced-resolution baseline can run overnight locally, but the Phase 3 loop — one
change, one CV run, one decision — cannot. **Get GPU time before Phase 3 (Wed 9 Sep).** Colab
Pro or a rented L4/A10 for a few days costs less than one wasted day of the eleven.

> **Check the rules before uploading anything.** Renting compute you control is normal
> competition practice, but the data-sharing rule (docs/00) governs where patient volumes may be
> copied. Confirm what is permitted for *your* dataset before pushing it to any cloud storage.
> This is a decision for you, not for Claude, and it is not covered by the §0 guards — those
> stop Claude reading the data, not you moving it.

**Reading while things download (~30 min):** [docs/01](01_domain_primer_dat_scans.md) §§1–4.
Enough to know what a comma-shaped striatum is and why it becomes a dot.

---

## Phase 1 — See the data, end-to-end submission (days 1–2, Sat 5 – Sun 6 Sep)

The goal of these two days is a number on the leaderboard. Any number.

**Day 1 splits the "first submission" in two, and the split is the point.** A prior-only zip
carries no model at all — it predicts one constant for every row — so nothing about it can fail
for a modelling reason. Whatever breaks is the *platform contract*: a path, an import, the
output location, the zip layout. Finding that on day 1 with a two-minute build, rather than on
day 2 tangled up with a model that also might be wrong, is worth the twenty minutes it costs.
Its score is meaningless and that is fine; it is a wire test, not an entry.

**Day 1 — prove the pipeline, no model involved**

| | Task |
|---|---|
| ☐ | `python scripts/01_inspect_data.py` |
| ☐ | Read `reports/eda/raw/SHARED_DRAFT.md` **yourself**. Confirm no per-sample values. |
| ☐ | Copy it to `reports/eda/shared/dataset_profile.md` → now Claude can help |
| ☐ | Note the **positive rate** from that summary — it is the `--fallback` value below, and the prior-only log loss it implies is row zero of the experiment log |
| ☐ | `python scripts/05_package_submission.py --prior-only --fallback <pos rate>` |
| ☐ | Test in Docker: `just pack-submission && just test-submission` |
| ☐ | **Submit it.** The score should land within noise of the prior baseline. If it does not, your understanding of the submission format is wrong — find out now. |
| ☐ | Open a few scans in a viewer (ITK-SNAP, 3D Slicer, or `nibabel` + matplotlib in a notebook). **Look at them.** Find the comma. Find one that is a dot. |

**Day 2 — the first real model through the same, now-proven, pipeline**

| | Task |
|---|---|
| ☐ | `python scripts/02_make_folds.py` |
| ☐ | Sanity check before the long run: `python scripts/03_train.py --config conf/baseline.yaml --folds 0` |
| ☐ | Train the baseline: `python scripts/03_train.py --config conf/baseline.yaml` |
| ☐ | `python scripts/04_evaluate.py --run baseline` → OOF log loss, AUROC, ECE. Compare to the prior baseline. |
| ☐ | Copy `models/baseline/fold*.pt` into `submission_src/assets/` |
| ☐ | `python scripts/05_package_submission.py --config conf/baseline.yaml --temperature <fitted> --fallback <pos rate> --clip-eps <best>` |
| ☐ | Test in Docker, then **submit**. |

> `--config` is not optional for a real build. It bakes the preprocessing and architecture the
> weights were trained with into `main.py`. Without it, inference silently uses the module
> defaults — which produces no error, no warning, and a leaderboard score unrelated to your CV.

> Looking at the images yourself is not optional and it is not a formality. It is the one part
> of this project Claude fundamentally cannot do for you, and it is where your intuition about
> what the model must detect comes from. Budget an hour.

**Learning checkpoints (as they come up):**
- When you see the shape/spacing table → [docs/02](02_cv_for_3d_medical_imaging.md) Delta 1.
- When you first see OOF log loss → [docs/00](00_competition_brief.md) "The metric", and read
  [calibrate.py](../src/datpd/calibrate.py)'s module docstring. Compute the prior baseline and
  write it as row zero of the experiment log.
- If the smoke test fails → the error is almost certainly structural (imports, paths, output
  location). That is exactly why we do this on day 2 rather than day 15.

**Exit criterion: a score on the leaderboard.** Do not proceed until you have one.

---

## Phase 2 — Make the validation trustworthy (days 3–4, Mon 7 – Tue 8 Sep)

A CV number you don't trust is worse than no number, because you will optimise against it.

| | Task |
|---|---|
| ☐ | Run `conf/grouped_cv.yaml`. Compare `fold_strat` vs `fold_group` log loss. |
| ☐ | Score per acquisition signature. Is one centre carrying all the error? |
| ☐ | Compare your CV log loss to the leaderboard score. Note the gap and its sign. |
| ☐ | Fit temperature and isotonic calibration on OOF. Record all three (`choose_calibrator`). |
| ☐ | Tune the prediction clip on OOF. |
| ☐ | Ablate normalisation: `reference_region` vs `zscore_brain` vs `max`. |
| ☐ | Ablate crop: `center` vs `com`. |

By the end you should be able to say: *"my CV is trustworthy to within X, and the leaderboard
tracks it."* Every later decision leans on that.

**Learning:** [docs/01](01_domain_primer_dat_scans.md) §5 (SBR — why `reference_region` is the
principled default) and §6 (why ambiguity is irreducible and calibration is the response).
Then [docs/02](02_cv_for_3d_medical_imaging.md) Deltas 6 and 7.

---

## Phase 3 — Improve the model (days 5–9, Wed 9 – Sun 13 Sep)

Now, and only now, chase score. One change at a time, each judged on OOF log loss, each a row
in [reports/experiments/log.md](../reports/experiments/log.md).

**This is the phase the four lost days come out of, so treat the list below as ranked, not as a
checklist.** Items 1–2 and 5–6 are cheap and reliably pay; items 3–4 and 7–8 are where days
disappear. Get through the cheap ones, then take on at most one expensive one, and only if
Phase 2 left you with a CV you actually trust. Stopping at item 6 with a well-calibrated
ensemble is a better outcome than a half-finished 2.5D model on the 14th.

Ordered by expected value per unit of effort:

1. **Resolution.** 96³ @ 2 mm → 128³ @ 1.5 mm. Frequently the single biggest win in 3D medical
   imaging, because the structures are small relative to the field of view.
2. **Augmentation sweep.** Rotation/translation/scale magnitude. Then a controlled test of
   `flip_lr` — you now know why it is contentious (docs/01 §4).
3. **Pretrained 3D backbones.** `monai.networks.nets.DenseNet121(spatial_dims=3)` and friends.
   Check weight licences; log them in [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md).
4. **A 2.5D model.** Slices through a `timm` ImageNet backbone with attention pooling. Different
   enough from the 3D CNN to ensemble well (docs/02 Delta 2).
5. **Label smoothing** (0.05) — cheap, usually helps calibration directly.
6. **Ensemble.** Average fold models in **logit** space; then average across architectures.
   Recalibrate the ensemble — an ensemble's calibration differs from its members'.
7. **Centre-of-mass cropping / registration to a template.** If EDA shows brains are not
   consistently centred.
8. **Explicit SBR-style features** (putamen/caudate ratio, asymmetry index) fed to a small
   gradient-boosted model, blended with the CNN. Interpretable, and diverse in a useful way.

**Hard rule for this phase:** if a change does not improve OOF log loss, revert it. Keep the
log honest, including failures — the failures are what stop you retrying the same idea on day
14 at 2am.

**Learning:** by now you are reading papers with real questions. Search PubMed for "deep
learning DaT SPECT". Read one or two and ask specifically: what preprocessing? single or
multicentre? how validated? Multicentre generalisation is the hard part and most published
work sidesteps it.

---

## Phase 4 — Lock down and submit (days 10–12, Mon 14 – Wed 16 Sep)

Stop adding ideas. This phase is about not losing what you have.

| | Task |
|---|---|
| ☐ | Freeze the architecture. Retrain final folds with a fixed seed. |
| ☐ | Refit calibration on final OOF predictions. Bake the temperature into `main.py`. |
| ☐ | Set `FALLBACK_PROB` to the training positive rate. |
| ☐ | Full local Docker test on the demo data. Check timing — extrapolate to the full test set and confirm you are inside 3 hours with margin. |
| ☐ | Re-read the independence rules ([docs/00](00_competition_brief.md)). Audit `main.py` line by line. |
| ☐ | Submit the intended final model on **Mon 14 Sep**, leaving two clear days. Submission limits are per day; leave room to fix a failure. |
| ☐ | Write `docs/SOLUTION.md` — required of winners, and it makes you re-examine your own choices. |
| ☐ | Confirm the repo is MIT-licensed and reproducible. |

**Tue 15 and Wed 16 Sep are buffer, deliberately.** Something will go wrong. Common failures: a checkpoint that
doesn't fit in the zip, an import that only exists locally, an inference run that exceeds the
time limit at full test-set scale.

---

## The learning track, standalone

If you want a spine independent of the competition clock:

**Domain (~3 hours total)**
1. [docs/01](01_domain_primer_dat_scans.md) end to end — 40 min.
2. Radiopaedia "DaT scan" — 20 min, mostly for the pictures.
3. The EANM/SNMMI DaT SPECT practice guideline, skimmed — 45 min. Focus on acquisition and
   interpretation criteria.
4. One PubMed paper on DaT SPECT deep learning — 45 min.
5. PPMI's dataset documentation — 30 min. Context for nearly all published work. Read it as
   literature only: the PPMI data is **not** usable in this competition
   ([EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md)).

**Computer vision for 3D medical imaging (~6 hours)**
1. [docs/02](02_cv_for_3d_medical_imaging.md) end to end — 45 min.
2. The MONAI tutorials repo, `3d_classification` notebooks — 2 h. The closest public analogue
   of this task.
3. TorchIO documentation on transforms — 1 h. The 3D augmentation vocabulary.
4. On calibration: read [calibrate.py](../src/datpd/calibrate.py) and its tests, then search for
   *"On Calibration of Modern Neural Networks"* (Guo et al., 2017) — the temperature-scaling
   paper. Short and directly applicable.
5. Any nnU-Net paper/documentation — 1 h. It's segmentation, not classification, but its
   preprocessing philosophy (resample to a common spacing, normalise per case, let the data
   decide) is exactly what applies here.

**The habit worth building:** every time you hit a term you don't know — *putamen*, *SBR*,
*OSEM*, *SWEDD* — add a row to the cheat-sheet in [docs/01](01_domain_primer_dat_scans.md) §7.
Two weeks of that is a real working vocabulary.

---

## How to run sessions with Claude Code

See [docs/04](04_working_with_claude_code.md) for the detail. The short version:

- **Claude cannot see your data. Ever.** That is enforced, not merely requested. Work as:
  *Claude writes the script → you run it → you promote aggregates → Claude reads those.*
- **Start each session by saying what phase you're in** and pasting the last few rows of
  [reports/experiments/log.md](../reports/experiments/log.md). That is the context that makes
  advice specific instead of generic.
- **Ask for the reasoning, not just the code.** "Why GroupNorm here?" is the highest-value
  question you can ask while learning, and it costs nothing.
- **Have Claude review your `main.py` against the independence rules** before every submission.
  It is a mechanical check, exactly the sort that is easy to get wrong at 2am on day 15.
