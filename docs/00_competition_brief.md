# 00 — Competition brief

Everything binding about the DaT Parkinson's Challenge, in one place. Read this before writing
any code. Source pages are linked at the bottom.

---

## The ask, in one sentence

Given a 3D DaT SPECT brain scan, output the **probability** that it is pathological (abnormal).

## Facts

| | |
|---|---|
| Host | DrivenData, competition #311 |
| Organizers | French Society of Nuclear Medicine (SFMN), Health Data Hub, GaelO |
| Funding | France 2030 plan |
| Prize pool | €25,000 — €12,500 / €7,500 / €5,000 (paid in USD at the 2026-07-22 rate) |
| **Deadline** | **2026-09-16, 23:59 UTC** |
| Participants | ~801 at the time of writing |
| Target column | `is_pathologic` — `0.0` normal, `1.0` abnormal |
| **Metric** | **Log loss.** Lower is better. |
| Secondary | AUROC, displayed for reference, **does not affect ranking** |
| Submission type | Code execution (you ship code, not predictions) |

Winners are invited to a ceremony in France and may be approached about an optional follow-up
collaboration on predicting **age and sex** from DaT scans — a hint that those signals are
present in the images, which is worth remembering when you think about confounders.

---

## The data

- **Format:** compressed NIfTI, `.nii.gz`, one 3D reconstructed volume per file.
- **Voxels:** 16-bit unsigned integers (raw SPECT counts, uncalibrated across scanners).
- **Filenames:** `<uid>.nii.gz`, where `uid` joins to the labels.
- **Labels:** `train_labels.csv` with columns `uid`, `is_pathologic`.
- **Provenance:** ten French hospital centres. One of the largest multicentre DaT collections
  assembled for ML. Labelled by **11 nuclear medicine specialists**.

### The defining property: heterogeneity

Volumes differ in **both** array shape and physical voxel size:

| Example shape | Example spacing |
|---|---|
| 142 x 142 x 112 | 2.46 mm isotropic |
| 128 x 128 x 128 | 3.895 mm isotropic |

This is not noise to be ignored — it is the central engineering problem. A 128-voxel-wide
volume at 3.895 mm covers ~499 mm; a 142-voxel-wide volume at 2.46 mm covers ~349 mm. Feed
both to a CNN unresampled and the same convolution kernel is looking at anatomy at different
physical scales, and the network wastes capacity learning to undo the mismatch.

**Resample everything to a common millimetre grid.** See
[src/datpd/data/preprocess.py](../src/datpd/data/preprocess.py).

Shape and spacing also act as a **scanner fingerprint** — a usable proxy for hospital centre,
which is otherwise not given. Use it for grouped cross-validation
([src/datpd/data/splits.py](../src/datpd/data/splits.py)).

---

## The metric: log loss

```
L(y, p) = -( y·log(p) + (1-y)·log(1-p) )
```

averaged over samples. Three consequences that should shape every decision:

1. **You are scored on calibration, not just ranking.** A model with excellent AUROC and
   overconfident probabilities can lose to a worse-ranking, well-calibrated one. Fit a
   calibrator on out-of-fold predictions — see [src/datpd/calibrate.py](../src/datpd/calibrate.py).
2. **Confident mistakes are catastrophic and unbounded.** A prediction of `p=1e-6` on a
   positive contributes ~13.8 to the sum, versus ~0.1 for a decent prediction. Clip your
   outputs. Tune the clip on OOF data.
3. **Know your floor.** Predicting the class prior for everything gives a specific log loss.
   Compute it first (`datpd.utils.metrics.baseline_logloss`) and record it as experiment zero.
   Anything that does not beat it is not a model.

---

## Submission: this is a code-execution competition

You do **not** upload predictions. You upload code that DrivenData runs against a test set you
never see.

### The zip

```
submission.zip
├── main.py            (required, at root)
├── my_module/...      (any supporting code)
└── assets/...         (model weights — must be included, there is no network)
```

### The execution environment

| | |
|---|---|
| Python | **3.12 only** |
| Hardware | 1x NVIDIA A100 80GB, 24 vCPU, 220 GB RAM |
| Time limit | **3 hours** full run, **6 minutes** smoke test |
| Network | **None** |
| Logging | ≤ 300 lines, ≤ 300 chars/line |

> **This A100 is for scoring your submission, not for training on.** It runs `main.py` against
> the test set, and "no retraining at inference" is a competition rule (§1.1) — so it cannot be
> repurposed as a training box, even though the hardware would be ideal.
>
> **DrivenData provides no training compute.** Checked against the official overview page on
> 2026-09-05: there is no hosted notebook, lab, JupyterHub, cloud credit or sponsored GPU. The
> rules state participants are *"solely responsible for all equipment ... necessary to ... develop
> and upload any Submission, and any telephone, data, hosting or other service fees associated
> with such access."*
>
> That phrase "hosting ... service fees" implies the organizers anticipate participants using
> hosted services, which is weak support for renting cloud GPU being acceptable. It is **not**
> explicit permission to upload the Data to a third-party provider — that is governed by the
> data-handling rule below, and should be confirmed with DrivenData before any upload.

### Filesystem at runtime

```
/code_execution/
├── data/                     (read-only)
│   ├── niftis/<uid>.nii.gz
│   └── submission_format.csv
└── src/                      (your unzipped files; working directory)
```

`main.py` must write `submission.csv` to the working directory:

```
uid,is_pathologic
xaji0y6d,0.04
pbhsahxt,0.12
```

### Dependencies

Only what is already in the image. The notable ones: `torch==2.12.1+cu129`, `torchvision`,
`torchio==1.2.1`, `monai`, `nibabel`, `simpleitk`, `antspyx`, `pytorch-lightning`, `timm`,
`transformers`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `polars`, `xgboost`, `lightgbm`,
`opencv-python-headless`, `einops`, `matplotlib`, `loguru`, `tensorflow`. Full list in
[CLAUDE.md §3](../CLAUDE.md). A missing import is a failed submission.

New dependencies require a PR to the runtime repo — with the deadline where it is, assume that
is not an option.

### Local testing

Clone <https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime>, then:

```bash
just pull              # download the official image (needs ~21 GB free)
just pack-submission   # zip submission_src/
just test-submission   # run it in the container, exactly as the platform will
```

Then submit a **smoke test** (6 min, small sample) before the full run. Do this early; the
first submission of any competition usually fails on something structural, not on the model.

---

## Rules that constrain your solution

These are not guidelines. Violating them invalidates the submission.

- **Per-sample independence.** "Process each test data sample independently during inference
  without the use of information from other cases in the test set." This forbids: normalising
  with statistics computed over the test set, pseudo-labelling on test, adaptive BatchNorm,
  any TTA or ensembling that pools across samples. Use per-sample normalisation and
  batch-independent norm layers (GroupNorm), and always `model.eval()`.
- **No retraining at inference.** Weights ship in the zip.
- **No manual annotation of test data.**
- **External data and pretrained weights are allowed** if you hold all rights and licences
  without needing third-party approval, and can prove it on request. Must be vendored into the
  zip. Log provenance in [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md).
  **PPMI is specifically excluded** — its Data Use Agreement does not cover competition use, so
  PPMI data and PPMI-derived weights make a solution ineligible for prizes. A DUA that gates
  access usually also restricts purpose; "publicly available" is not "licensed for this".
- **No private code sharing** outside your team. Public sharing is permitted and is
  automatically MIT-licensed.
- **One account per person.** Multiple accounts means disqualification.
- **Winners must open-source under MIT**, with documentation from DrivenData's template and a
  description of the resources needed to build and run the solution. Keep the repo
  reproducible as you go; retrofitting this at the end is miserable.

### Data handling (why CLAUDE.md §0 exists)

The rules permit "reading and learning from the Data, analyzing the Data, modifying the Data
and generally preparing your Submission", and prohibit transmitting, duplicating, publishing,
redistributing or otherwise making the Data available to any party not participating —
**explicitly including third-party AI services**. You must also make no attempt to identify
individuals.

Claude Code is a third-party AI service. That is the whole reason for the guardrails in
[CLAUDE.md](../CLAUDE.md) §0, the deny rules in `.claude/settings.json`, and the hook in
`.claude/hooks/guard_data.py`. Claude writes the code; you run it; Claude reads only the
aggregates you promote.

---

## Sources

- [Overview](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/987/)
- [About the data](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/988/)
- [Problem description](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/990/)
- [Code submission format](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/)
- [Official rules](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/rules/)
- [Runtime repository](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime)
