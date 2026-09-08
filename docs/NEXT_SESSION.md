# Next session — starting prompt

Paste the block below into a fresh Claude Code session. It is written to re-establish the
context that actually changes decisions, and nothing else.

---

```
Phase 2 of docs/03_plan.md — making the validation trustworthy. 8 days to the deadline
(2026-09-16); final model goes in Mon 2026-09-14, not later.

Where things stand:
- Run 1 (conf/baseline.yaml, 96³ @ 2.0mm, crop: com, ResNet3D, 5-fold stratified) scored
  OOF log loss 0.3186 / AUROC 0.9302, against a prior baseline of 0.6884. Packaged and
  submitted. Leaderboard score: <FILL IN, or say "not back yet">.
- Read reports/experiments/log.md first — rows 0, 0b, 1 and the notes under them. The
  calibration note and the fold-structure note are the two that matter.
- Environment: .venv on Python 3.12.3, torch 2.12.1+cpu. NO CUDA GPU on this machine
  (Intel Iris Xe). Always invoke as .venv\Scripts\python.exe.
- Cloud GPU: <FILL IN — did DrivenData answer the forum question about uploading data to
  rented compute?>

What I want this session:
1. The grouped-CV number. conf/grouped_cv.yaml has not been run. Given a 46–59% positive
   rate spread across acquisition families, the strat-vs-group gap is the most informative
   measurement left — it tells me whether 0.3186 survives unseen centres.
2. Then the Phase 3 shortlist, ranked by expected log-loss gain per hour of CPU.

Constraints to respect, from CLAUDE.md §0: you never read the competition data. I run the
scripts; you read reports/eda/shared/ and reports/experiments/ only. If you need something
about the data, write me a script.
```

---

## Notes for whoever picks this up

**Fill in the two placeholders before pasting.** The leaderboard score for run 1 and the
DrivenData answer on cloud GPU are the two facts that change what to do next, and neither is
recoverable from the repository.

**The grouped-CV run is the priority, not a model change.** Run 1's 0.3186 is a stratified
number, and the fold summary shows positive rate varying from 46.3% to 59.3% across acquisition
families. Until the grouped number exists, there is no basis for believing the leaderboard will
track CV — and every Phase 3 decision leans on that.

```powershell
.venv\Scripts\python.exe scripts\03_train.py --config conf/grouped_cv.yaml
```

Note `grouped_cv.yaml` trains at 96³ like `baseline.yaml`, so budget the same wall clock. Fold
sizes will be uneven (115 to 460); that is correct, not a bug.

**Two things that are easy to get wrong here, both already paid for once:**

- Compare grouped **AUROC and log loss separately.** Each grouped fold has a train/val prior
  mismatch, so its log loss is penalised for a reason unrelated to whether the model generalises.
  AUROC is invariant to that. AUROC holding while log loss degrades means calibration, not
  architecture.
- **Anything scored on its own training data favours the more flexible option.** That is how
  isotonic calibration "beat" temperature scaling by 0.01 while actually being 0.03 worse.

**Delete `models/<run>/oof_fold*.csv` before retraining a run after any config change**, or the
assembled OOF will silently mix two models' predictions.

## Standing risks

| risk | status |
|---|---|
| No local GPU; a 96³ fold is hours on CPU | open — cloud GPU decision is on the critical path |
| Prior shift between train and test (smoke set implied ~0.65 vs train 0.5485) | open — the full test score settles it |
| Grouped CV never measured | open — this session's job |
| Machine sleeps mid-run | mitigated — per-epoch resume checkpoints, but set Windows Update active hours |
