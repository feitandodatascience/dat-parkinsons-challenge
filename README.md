# DaT Parkinson's Challenge

Solution repository for the [DrivenData DaT Parkinson's Challenge](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)
— binary classification of dopamine transporter (DaT) SPECT brain scans as normal or abnormal,
across a multicentre French dataset from ten hospitals.

**Metric:** log loss · **Closed:** 2026-09-16 · **Submission:** code execution

## Result

**Final rank: #159 / 1009** (top 16%) on the private leaderboard.

| | Public LB | Private LB |
|---|---:|---:|
| First model (3D ResNet baseline) | 0.3210 | 0.3767 |
| Best public — candidate B (10 CNNs + TTA, stacked with LightGBM on SBR features) | **0.2855** | 0.3448 |
| Best private — candidate C (candidate B with a tuned booster) | 0.2871 | **0.3447** |

**Read [WRITEUP.md](WRITEUP.md)** for the short story: all five submissions, what the private
leaderboard showed, and what I would change. The full technical record is
[docs/SOLUTION.md](docs/SOLUTION.md).

---

## ⚠️ Data handling policy

**Competition data must never be shared with third-party AI services, including Claude Code.**
This is a DrivenData rule; breaking it means disqualification.

This repo enforces it in four independent layers — permission deny rules, a `PreToolUse` hook,
`.gitignore`, and the behavioural rules in `CLAUDE.md` §0. The hook is covered by
47 tests in [tests/test_guard_hook.py](tests/test_guard_hook.py).

`CLAUDE.md` and `.claude/` (deny rules and hook) are kept locally and are not published here;
[docs/04](docs/04_working_with_claude_code.md) describes them, and the hook tests skip without them.

The workflow is: **Claude writes the analysis script → you run it → you promote aggregate,
non-identifying summaries to `reports/eda/shared/` → Claude reads those.** Details in
[docs/04](docs/04_working_with_claude_code.md).

Nothing under `data/` is ever committed.

---

## Start here

| Doc | What it gives you |
|---|---|
| [WRITEUP.md](WRITEUP.md) | The short version: results, lessons, what I'd do differently |
| [docs/SOLUTION.md](docs/SOLUTION.md) | The full solution: pipeline, every decision with its evidence, reproduction |
| [docs/01 — Domain primer](docs/01_domain_primer_dat_scans.md) | What a DaT scan is, from zero. **Read this first** if the domain is new. |
| [docs/02 — MNIST → 3D medical imaging](docs/02_cv_for_3d_medical_imaging.md) | The seven things that change when you leave 2D |
| [docs/00 — Competition brief](docs/00_competition_brief.md) | Rules, metric, submission format, runtime constraints |
| [docs/03 — Working plan](docs/03_plan.md) | Day-by-day plan to the deadline, with learning interleaved |
| [docs/04 — Working with Claude Code](docs/04_working_with_claude_code.md) | Session patterns under the data constraint |

Each doc also has a Vietnamese translation alongside it, with a `_vn` suffix — e.g.
[docs/01 — Nhập môn lĩnh vực](docs/01_domain_primer_dat_scans_vn.md). The English files are the
source of truth; the translations are kept in step by hand.

**Prefer to read offline?** `python scripts/build_docs_pdf.py` renders all ten (five English,
five Vietnamese) to `docs/pdf/` with a table of contents and working cross-references. Needs
`pip install pypandoc-binary` plus a LaTeX engine (TinyTeX is enough). The Markdown files remain
the source of truth — regenerate the PDFs after editing them.

```bash
python scripts/build_docs_pdf.py                # everything
python scripts/build_docs_pdf.py --lang vn      # just the translations
python scripts/build_docs_pdf.py --check-glyphs # verify no character is silently dropped
```

Run `--check-glyphs` after adding an unusual character: the PDF engine drops a glyph the font
lacks *silently*, so a tick or an arrow becomes an invisible gap. One known limitation: in the
Vietnamese PDFs the tone marks display correctly but do not survive copy/paste or in-PDF search
(no font on a stock Windows install round-trips them). Search the Markdown instead.

---

## Setup

The competition runtime is **Python 3.12 only**. Develop on 3.12 or you may build something
that cannot be submitted.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pytest -q                       # synthetic data only — no competition data needed
```

Download the competition data yourself from the competition data page into `data/raw/`:

```
data/raw/
├── niftis/<uid>.nii.gz
└── train_labels.csv
```

**The download page appends a random suffix to every file** — you will get names like
`niftis_XXXXXXX`, `train_labels_XXXXXXX` and `smoke_test_data_XXXXXXX.tar`. Normalise them once
to the canonical names above; every config, script and doc in this repo refers to those, and
`scripts/00_check_env.py` will tell you if they are not in place.

```bash
# from data/raw/ — extract if the download is an archive, otherwise just rename
tar -xf niftis_XXXXXXX -C . && mv niftis_extracted niftis   # if it is a tar
mv niftis_XXXXXXX niftis                                    # if it is already a directory
mv train_labels_XXXXXXX train_labels.csv
```

`smoke_test_data_*.tar` does **not** belong in `data/raw/`. It is the local Docker test fixture
and goes in the [runtime repo](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime)
checkout, which is what `just test-submission` reads.

For local submission testing you also need Docker and the
[official runtime repo](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime).

## Pipeline

```bash
python scripts/00_check_env.py                        # verify env vs. the competition runtime
python scripts/01_inspect_data.py                     # -> reports/eda/raw/  (you promote it)
python scripts/02_make_folds.py                       # stratified + grouped CV folds
python scripts/03_train.py --config conf/baseline.yaml
python scripts/04_evaluate.py --run baseline          # OOF log loss, AUROC, ECE, calibration
cp models/baseline/fold*.pt submission_src/assets/
python scripts/05_package_submission.py --config conf/baseline.yaml \
    --temperature <fitted> --fallback <pos rate> --clip-eps <best>
```

`--config` is required for a real build: it bakes the preprocessing and architecture the shipped
weights were trained with into `main.py`. Omitting it leaves inference on the module defaults,
which raises no error and produces a leaderboard score unrelated to your CV.

**Before any of that, on day 1**, ship a submission with no model at all:

```bash
python scripts/05_package_submission.py --prior-only --fallback <pos rate>
```

It predicts one constant for every row, so nothing about it can fail for a modelling reason —
whatever breaks is the platform contract (paths, imports, output location, zip layout). Its score
is meaningless; it is a wire test. Finding a format problem here costs minutes, finding it on day
2 tangled up with an untrained model costs hours.

Then test the zip in the runtime container (`just pack-submission && just test-submission`)
before submitting.

## Layout

```
WRITEUP.md             short results write-up
docs/                  guides 00–04 (+ _vn translations), SOLUTION.md, EXTERNAL_ASSETS.md
docs/project-notes/    frozen working notes: compute budget, GPU runbook, compliance ruling
conf/                  YAML experiment configs
src/datpd/             the package: io, preprocess, splits, models, metrics, calibration
scripts/               CLI entry points you run
submission_src/        exactly what gets zipped for the platform
tests/                 synthetic-data tests + guardrail tests
reports/eda/raw/       QUARANTINE — Claude-blocked, git-ignored
reports/eda/shared/    user-vetted aggregates — Claude may read
reports/experiments/   one row per experiment
reports/submissions/   platform logs and scores, one file per submission
data/                  git-ignored, Claude-blocked
```

## License

MIT — see [LICENSE](LICENSE). Winning solutions must be MIT-licensed under the competition
rules, so this is set up that way from the start.
