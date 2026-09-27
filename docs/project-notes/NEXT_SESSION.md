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
- Submission 2 SHIPPED and is the current best: TTA (9 views) + SBR/lightgbm blend.
  OOF 0.3061 → **LEADERBOARD 0.2973**, AUROC 0.9396, rank #98/897. Ran in 15 min of the
  3 h budget, no fallbacks, zero filtered log lines (the §3.1 logging fix is validated).
  Full log: reports/submissions/002_2026-09-09_tta_blend.md.
- **The CV↔LB gap is ±0.01, not ±0.0024.** Two observations now: +0.0024 (run 1) and
  −0.0088 (submission 2), opposite sides of zero. The 0.2973 beat its ~0.305 prediction,
  but LB AUROC also beat OOF AUROC, so the surplus is an easier test draw, not the method
  transferring better. Keep valuing TTA+blend at the −0.0156 measured. Any future OOF
  margin under ~0.01 is unresolved, not won.

- SUBMISSION BUDGET IS THE BINDING CONSTRAINT: 3 per rolling 7 days. Used: 0b (09-05),
  run 1 (09-08), TTA+blend (09-09). Next slot frees ~09-12 — but **spending 09-12
  forecloses 09-14**: on 09-14 the trailing window would hold 09-08, 09-09 and 09-12,
  which is three. Verify against the platform's own counter, then pick ONE. Smoke tests
  are free and unlimited but run on FAKE data and are excluded from the leaderboard — use
  them to validate a code path, never to estimate a score.

- Environment: .venv on Python 3.12.3, torch 2.12.1+cpu, lightgbm 4.7.0. NO CUDA GPU
  (Intel Iris Xe), so a 96³ 5-fold run is a whole night. Run everything from
  the repo root — the runtime repo is a sibling checkout and has
  no .venv, which is worth remembering before pasting a command into the wrong prompt.
- Cloud GPU: ANSWERED — permitted, conditionally. The organizers gave a three-part test
  (no third-party retention / private to me / every copy deletable), not a yes. Ruled out:
  Vast.ai and RunPod Community (stranger's hardware), Kaggle (licence grant), Colab
  (Google's terms + Gemini). Viable: Lambda Labs on-demand, RunPod Secure, hyperscalers.
  Full analysis, operational rules and the teardown record: docs/07_cloud_compute_compliance.md.
  conf/highres.yaml is the 128³ @ 1.5mm config; train.py needs no change (it already
  auto-selects CUDA and gates AMP on it).

What I want this session:
1. Decide the 128³ question, then execute or close it. It is now a schedule bet, not a
   compute one: ~4 h on an A100, but the refit chain behind it (temperature, TTA, SBR
   features, stack coefficients — ALL of them are 96³ properties and void at 128³) is the
   whole of Phase 3 re-run. Bar to clear: OOF beating 0.3061 by more than ~0.01.
   Hard stop: not training by Sat 12 Sep → abandon.
   **The runbook is written: docs/08_gpu_run_runbook.md.** Step 0 to teardown, with
   scripts/cloud/bootstrap_gpu_box.sh for the instance and conf/highres{,_smoke}.yaml
   for the run. If we are executing, start at its step 0 rather than re-deriving anything.
2. Phase 4: freeze, then write docs/SOLUTION.md, which winners must submit.

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
- [x] ~~Full local Docker test; extrapolate timing and confirm margin inside 3 hours.~~
      **Answered by submission 2 directly, and better than the local test would have:
      15 min of 180 on the real test set and the real A100, no fallbacks.** A 12× margin.
      Re-run the local Docker test only if the shipped build changes (e.g. 128³).
- [ ] Re-read the independence rules and audit `main.py` line by line.
- [ ] Write `docs/SOLUTION.md` — required of winners, and it forces a re-examination of
      every choice.
- [ ] Confirm the repo is MIT-licensed and reproducible.
