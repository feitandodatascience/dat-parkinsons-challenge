# Submission logs

One file per submission, named `NNN_YYYY-MM-DD_<short-name>.md`. Each holds the platform's run
log verbatim plus the score it came back with.

Why keep these rather than only the row in `../experiments/log.md`:

- **The log is the only view of the platform's own behaviour.** Run 1's compliance warning was
  diagnosable only because the surviving log lines acted as a control against the filtered ones.
  That reasoning is impossible to reconstruct from a score alone.
- **Runtime is measured here and nowhere else.** Local Docker timing is an extrapolation; this is
  the real test set on the real A100.
- **Submissions are scarce** — 3 per rolling 7 days. Anything spent one on is worth keeping.

`../experiments/log.md` stays the decision record; this directory is the evidence behind it.

## Compliance note

These logs are safe for Claude to read **because `main.py` is built so that they are**: it logs no
data values and nothing describing the test set, including its row count (CLAUDE.md §3.1). If a
future log ever contains a value or a count, it does not belong in this directory — fix `main.py`
first.

## Index

Private scores were published after the competition closed on 2026-09-16.

| # | date | platform id | submission | public LB | private LB | public AUROC |
|---|---|---|---|---:|---:|---:|
| 0b | 2026-09-05 | — | prior-only wire test | 0.6687 (smoke, fake data) | — | 0.500 |
| 1 | 2026-09-08 | 321492 | run 1, ResNet3D 5-fold + temperature | 0.3210 | 0.3767 | — |
| 2 | 2026-09-09 | 321564 | [TTA (9 views) + SBR/lightgbm blend](002_2026-09-09_tta_blend.md) | 0.2973 | 0.3573 | 0.9396 |
| 3 | 2026-09-16 | 322664 | candidate A: 157 repaired features, tuned booster (log row 9) | 0.2935 | 0.3476 | — |
| 4 | 2026-09-16 | 322850 | **candidate B**: + aug_rich CNN run, 10 CNNs (log row 10) | **0.2855** | 0.3448 | — |
| 5 | 2026-09-16 | 322889 | candidate C: candidate B + tuned booster (log row 12) | 0.2871 | **0.3447** | — |

Platform logs were kept verbatim for submission 2 only. Submissions 3–5 are recorded by their
scores and their rows in `../experiments/log.md`.
