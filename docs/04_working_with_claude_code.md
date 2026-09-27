# 04 — Working with Claude Code on this project

The constraint that shapes everything: **Claude must never see the competition data.** This
document is the practical guide to being productive inside that constraint.

---

## Why the constraint exists

DrivenData's rules prohibit making the Data available to any party not participating in the
competition, and name third-party AI services explicitly. Claude Code is one. Data reaching
Claude's context is a rules violation, with disqualification as the consequence — and the data
is patient imaging from ten French hospitals, so the ethical case matches the legal one.

## How it is enforced

Three independent layers, so that no single mistake is sufficient:

| Layer | File | Catches |
|---|---|---|
| Permission deny rules | `.claude/settings.json` | `Read`/`Grep`/`Glob` on `data/`, `reports/eda/raw/`, `models/`, and by extension (`*.nii.gz`, `*.pt`, ...) |
| PreToolUse hook | `.claude/hooks/guard_data.py` | Bash bypasses a path rule can't see: `cat`, `head`, `strings`, `xxd`, `zcat`, `python -c "print(df.head())"` |
| Behavioural rule | `CLAUDE.md` §0 | What tooling can't see — you pasting data into chat, a subagent being asked to peek |
| Git ignore | `.gitignore` | Committing data to a repo that will become public under MIT |

The hook is tested. `tests/test_guard_hook.py` asserts 47 cases — every bypass listed above is
in there. Run `pytest tests/test_guard_hook.py -q` after touching the hook.

If Claude gets blocked, that is the system working. Don't disable it, and don't work around it
by pasting the contents into chat — that defeats all three layers at once.

---

## The workflow that replaces "let me look at your data"

```
   ┌──────────┐  writes   ┌──────────────────┐
   │  Claude  │──────────▶│ scripts/*.py     │   code only — no data seen
   └──────────┘           └──────────────────┘
                                   │ you run it
                                   ▼
                          ┌──────────────────┐
                          │ reports/eda/raw/ │   QUARANTINE — Claude blocked
                          └──────────────────┘
                                   │ you read + vet
                                   ▼
                          ┌────────────────────┐
   ┌──────────┐  reads    │reports/eda/shared/ │   aggregates only
   │  Claude  │◀──────────└────────────────────┘
   └──────────┘
```

**You are the airlock.** The vetting step is a human judgement and cannot be automated —
that's why `01_inspect_data.py` writes its draft into `raw/` and makes you move it, rather than
writing straight to `shared/`.

### What may go in `shared/`

✅ Counts, class balance, positive rate
✅ Distributions over the whole dataset — quantiles, min/max, histograms
✅ Distinct shape/spacing values and their frequencies
✅ Per-group aggregates where groups are large (e.g. positive rate per acquisition signature)
✅ Metric values, loss curves, timing

❌ Any `uid`
❌ Any per-sample row, value, or voxel
❌ Images or renders of scans
❌ A group so small that a statistic identifies an individual

Test to apply: *could this be inverted to recover one patient's scan or label?* If yes, or if
you're unsure, it stays in `raw/`.

---

## Session patterns that work

### Starting a session

Give Claude the state it can't discover for itself:

> I'm in Phase 2. Baseline OOF log loss 0.42 (stratified), 0.51 (grouped), AUROC 0.89,
> prior baseline 0.61. Leaderboard 0.49. Here are the last 5 rows of the experiment log: [...]
> I want to understand why grouped CV is so much worse.

That is a question Claude can answer well. "Look at my data and tell me what to do" is one it
cannot answer at all.

### The four things Claude is genuinely good at here

1. **Writing and refactoring the pipeline code.** Preprocessing, training loops, config
   plumbing, packaging. Data-independent, and the bulk of the work.
2. **Auditing `main.py` against the competition rules** before each submission. Mechanical,
   high-stakes, easy for a tired human to get wrong. Ask explicitly:
   *"audit submission_src/main.py against CLAUDE.md §1.1 and §3."*
3. **Explaining the domain and the method.** "Why GroupNorm?", "what does putamen/caudate ratio
   mean and how would I compute it?", "why is my AUROC up but log loss worse?" — ask these
   constantly. This is the learning half of the project and it costs nothing.
4. **Interpreting aggregate results.** Given `shared/` summaries and metric logs, Claude can
   reason about centre effects, calibration, and what to try next.

### The three things to never ask

1. "Read `data/raw/train_labels.csv` and tell me the balance." → run
   `scripts/01_inspect_data.py` instead.
2. "Here's a traceback" — with a printed array or dataframe row in it. **Redact values before
   pasting.** Shapes and dtypes are fine; voxels and labels are not.
3. "Just check one scan for me." The rule has no threshold. One row is a violation.

### Debugging without showing data

Add assertions and shape-only logging instead of prints:

```python
# safe: structure only
logger.info(f"batch={x.shape} dtype={x.dtype} device={x.device}")
assert x.shape[1:] == (1, 96, 96, 96), f"unexpected shape {x.shape}"
assert torch.isfinite(x).all(), "non-finite values after preprocessing"

# NOT safe: values
logger.info(f"batch values: {x[0, 0, 48]}")   # ← never
print(df.head())                              # ← never
```

Then paste the assertion failure. `AssertionError: unexpected shape torch.Size([8, 1, 96, 96,
112])` tells Claude everything needed, and reveals nothing about a patient.

---

## Useful prompts, ready to paste

```
Audit submission_src/main.py against CLAUDE.md §1.1 (per-sample independence) and §3
(runtime imports). List anything that would fail on the platform.
```

```
Here is reports/eda/shared/dataset_profile.md. Given the signature distribution, is
grouped CV going to be viable, or are some groups too small? What n per fold do I get?
```

```
My OOF: logloss 0.44, AUROC 0.91, ECE 0.08. Prior baseline 0.61. Explain what the ECE
implies and what I should do before trying a bigger model.
```

```
Write a script that measures <X> about the dataset and writes only aggregates to
reports/eda/raw/. I'll run it and promote the summary.
```

```
Teach me the domain background behind <term I hit today>, and tell me whether it should
change anything in my preprocessing.
```

---

## Keeping the guardrails honest

- After editing `.claude/hooks/guard_data.py`, run `pytest tests/test_guard_hook.py -q`.
- If you add a new data location, add it to **all four** layers: `GUARDED_DIRS` in the hook,
  `permissions.deny` in settings, `.gitignore`, and CLAUDE.md §0.
- If Claude ever produces output that appears to contain data values, stop, and check which
  layer failed. Add a test case for it.
- `CLAUDE.md` and `.claude/` are kept local and are not published in the public repository.
  This document describes what they contain.
