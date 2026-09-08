# Next session — starting prompt

Paste the block below into a fresh Claude Code session. It is written to re-establish the
context that actually changes decisions, and nothing else.

---

```
Phase 3 of docs/03_plan.md — improving the model. 8 days to the deadline (2026-09-16);
final model goes in Mon 2026-09-14, not later.

Where things stand:
- Run 1 (conf/baseline.yaml, 96³ @ 2.0mm, crop: com, ResNet3D, 5-fold stratified):
  OOF log loss 0.3186, AUROC 0.9302, prior baseline 0.6884.
  LEADERBOARD 0.3210 — a CV↔LB gap of +0.0024, so CV is trustworthy and I can make
  decisions on OOF alone without burning submission slots.
- Read reports/experiments/log.md first — row 1 and the note "CV is trustworthy, and that
  changes the plan". Grouped CV is deprioritised there; don't reflexively suggest it.
- Environment: .venv on Python 3.12.3, torch 2.12.1+cpu. NO CUDA GPU on this machine
  (Intel Iris Xe), so a 96³ 5-fold run is a whole night. Always invoke as
  .venv\Scripts\python.exe.
- Cloud GPU: <FILL IN — did DrivenData answer the forum question about uploading data to
  rented compute? This gates everything expensive.>

What I want this session:
1. A ranked Phase 3 shortlist by expected log-loss gain PER HOUR OF CPU, given ~5 usable
   compute nights left. Be specific about what I can afford, not what would be ideal.
2. Start the top item.

Constraints, from CLAUDE.md §0: you never read the competition data. I run the scripts; you
read reports/eda/shared/ and reports/experiments/ only. If you need something about the
data, write me a script.
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
