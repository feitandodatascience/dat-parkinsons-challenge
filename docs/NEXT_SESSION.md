# Next session — starting prompt

Paste the block below into a fresh Claude Code session. It is written to re-establish the
context that actually changes decisions, and nothing else.

---

```
Phase 3 tail / Phase 4 of docs/03_plan.md — lock down and submit. 7 days to the deadline
(2026-09-16); intended final model goes in Mon 2026-09-14, not later.

Where things stand:
- Run 1 (conf/baseline.yaml, 96³ @ 2.0mm, crop: com, ResNet3D, 5-fold stratified):
  OOF 0.3186, LEADERBOARD 0.3210. CV↔LB gap +0.0024, so CV is trustworthy and Phase 3
  decisions were made on OOF alone.
- Rows 2-5 of reports/experiments/log.md are this project's Phase 3. Read that file first,
  especially "CV is trustworthy, and that changes the plan" and the note under row 5.
- Submission 2 is BUILT but its result is <FILL IN>: submission.zip carrying TTA (9 views)
  + the SBR/lightgbm blend. Paired OOF gain vs baseline: −0.01562 at 3.1σ, CI
  [−0.0255, −0.0057]. Predicted leaderboard ~0.305.
  Leaderboard: <FILL IN, or "not back yet">.

- SUBMISSION BUDGET IS THE BINDING CONSTRAINT: 3 per rolling 7 days. Used: 0b (09-05),
  run 1 (09-08), TTA+blend (09-09). Next slot frees ~09-12. Smoke tests are free and
  unlimited but run on FAKE data and are excluded from the leaderboard — use them to
  validate a code path, never to estimate a score.

- Environment: .venv on Python 3.12.3, torch 2.12.1+cpu, lightgbm 4.7.0. NO CUDA GPU
  (Intel Iris Xe), so a 96³ 5-fold run is a whole night. Run everything from
  D:\work\dat-parkinsons-challenge — the runtime repo is a sibling checkout and has
  no .venv, which is worth remembering before pasting a command into the wrong prompt.
- Cloud GPU: <FILL IN — did DrivenData answer about uploading data to rented compute?
  This gates the only large win left, 128³ @ 1.5mm.>

What I want this session:
1. <FILL IN once the leaderboard is back — see "how to read it" below.>
2. Phase 4: freeze, then write docs/SOLUTION.md, which winners must submit.

How to read the leaderboard when it lands:
- ~0.305 — CV still transfers. Bank it and move to Phase 4 lock-down.
- ~0.319 — the blend did not transfer but TTA did. Suspect the feature path at inference;
  first check stack.json's feature order against datpd.features.feature_names().
- worse than 0.3210 — something in the blend path is wrong. Revert to the run-1 build
  (same command minus --tta and --blend-from, with --temperature 1.2614) and spend a slot
  re-establishing the known-good score rather than debugging blind.

Constraints, from CLAUDE.md §0: you never read the competition data. I run the scripts; you
read reports/eda/shared/ and reports/experiments/ only. If you need something about the
data, write me a script.
```

---

## Housekeeping carried into the next session

- **`git gc` is still unrun.** `.git` holds ~620 MB of unreachable objects from the history
  rewrite that removed the competition archives. `git reflog expire --expire=now --all &&
  git gc --prune=now` reclaims it. **Before running it**, confirm
  `tests/smoke_test_data_XXXXXXX.tar.gz` has reached the runtime checkout — after the gc the
  working-tree copy is the only one left.
- **Both branches are pushed** and `main` is fast-forwarded to the work. Nothing tracked is
  data-shaped; the largest file in the repo is `reports/experiments/log.md` at 20 KB.

## What is deliberately not being pursued, and why

Recorded so it does not get re-proposed at 2am:

- **Grouped CV** — deprioritised, not forgotten. It is a proxy for "will the leaderboard
  track my CV", and the leaderboard answered that directly to within 0.0024. It measures
  generalisation to an unseen eleventh hospital, which is a real question about the
  solution's worth but not one the competition scores.
- **The cropping lead** — closed. The wide-field family that looked weakest crops fine; the
  metric that suggested otherwise was confounded with disease prevalence, because a
  pathologic striatum is dimmer and its peak leaves the box without any crop having failed.
- **128³ @ 1.5 mm** — the largest available win and unreachable on CPU at ~2.4× a full run.
  If a GPU appears before Sat 12 Sep, this is what to spend it on, ahead of everything else.
- **`flip_lr`** — still open in the log, and should stay open. A full run to confirm an
  expected negative is a poor use of the remaining nights.

## Phase 4 checklist (docs/03_plan.md, days 10-12)

The one date that is not negotiable: **submit the intended final model by Mon 14 Sep**, and
note that with 3 submissions per 7 days the slot has to be planned for, not assumed.

- [ ] Freeze the architecture. No new ideas.
- [ ] Full local Docker test on the runtime checkout; extrapolate timing to the full test
      set and confirm margin inside 3 hours. TTA adds ~415 ms/scan of CPU view generation
      that the A100 does not accelerate.
- [ ] Re-read the independence rules and audit `main.py` line by line.
- [ ] Write `docs/SOLUTION.md` — required of winners, and it forces a re-examination of
      every choice.
- [ ] Confirm the repo is MIT-licensed and reproducible.
