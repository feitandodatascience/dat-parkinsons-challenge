# 08 — Runbook: the 128³ run on rented GPU

**Written 2026-09-09.** Execution steps for the decision taken in
[07_cloud_compute_compliance.md](07_cloud_compute_compliance.md). Read docs/07 first — it holds
the ruling, the provider analysis, and the conditions this runbook exists to satisfy.

Follow it in order. The ordering is not cosmetic: **step 7 destroys the only copy of the data on
the box**, and every step that needs the data comes before it.

---

## The decision this run exists to serve

| | |
|---|---|
| Incumbent | TTA + SBR blend, **OOF 0.3061 → LB 0.2973**, already banked on the leaderboard |
| Challenger | `conf/highres.yaml`, 128³ @ 1.5 mm |
| Bar to clear | **OOF beating 0.3061 by more than ~0.01** (the CV↔LB spread, `log.md`) |
| Expected | −0.02 to −0.05 (docs/05 item 8) — a real win clears the bar comfortably |
| Abort line | **not training by Sat 12 Sep → close it and go to Phase 4** |
| Cost | ~$1–3/hr for ~6 h. Tens of dollars. Check current pricing; it moves. |

**Nothing is at risk except the rental fee and the calendar.** 0.2973 is already on the
leaderboard and needs no further submission to keep. If this run does not clear the bar, do
nothing, record the negative in `log.md`, and ship what you have.

---

## Step 0 — Before you rent anything

- [ ] **Confirm the submission-slot arithmetic** against the platform's own counter. docs/07 §5
      reads the rule as: spending 09-12 forecloses 09-14. If that holds, **do not spend 09-12**.
- [ ] **Confirm DrivenData scores your best submission** rather than one you nominate. If it is
      nominated, find out where and when that happens.
- [ ] Commit and push. The box clones from GitHub; anything uncommitted stays home.

## Step 1 — Rent it

**Lambda Labs on-demand, 1× A100 40GB.** Fall back to **RunPod Secure Cloud** (never *Community*)
if Lambda has no A100 capacity — which is common. Reasoning and the full ladder: docs/07 §2.

**Not a hyperscaler**, unless GPU quota is already approved on an account you hold. Azure/AWS/GCP
have the best contract language and the worst calendar: quota starts at zero and the increase
request routinely takes days, which is the whole abort line.

- **1× A100 40GB is enough.** The model is a small ResNet3D at batch 8; an H100 will not pay for
  itself here. What matters more is **RAM and vCPU count**, because the cold cache build is
  CPU-bound: aim for ≥ 100 GB RAM and ≥ 16 vCPU.
- **Prefer an EU region if the provider offers one** at no delay. Not a rules requirement, just
  cheap prudence for French patient imaging. Never worth a quota wait.
- [ ] **No persistent filesystem.** Lambda offers one; do not attach it. Same for any provider's
      network volume, snapshot, or image. Instance-local only.
- [ ] Note the instance id — it goes in the teardown record, docs/07 §7.

## Step 2 — Bootstrap

```bash
ssh <user>@<host>
git clone <your repo url> dat-parkinsons-challenge && cd dat-parkinsons-challenge
bash scripts/cloud/bootstrap_gpu_box.sh
```

The script checks the GPU, mounts a tmpfs at `/mnt/ram` sized to 60% of RAM, **symlinks `data/`
and `models/` into it**, installs Python 3.12 and torch 2.12.1+cu129 to match the competition
runtime, and runs `pytest`.

**Why the tmpfs matters.** Nothing data-shaped is ever written to the instance's disk. docs/07
condition 3 — every copy deletable at the end — stops being a policy question about the provider
and becomes a property of the machine: terminate it, or just lose power, and the data is gone.
Checkpoints live there too, since CLAUDE.md §0 treats weights as data-derived.

**It will not run if `data/` already exists on disk with contents in it.** That guard is
deliberate; investigate rather than deleting.

## Step 3 — Upload

From your **local** machine:

```bash
scp data/raw/<archive>      <user>@<host>:/mnt/ram/data/raw/
scp data/interim/folds.csv  <user>@<host>:/mnt/ram/data/interim/
```

Then unpack on the box into `data/raw/niftis/` as `conf/highres.yaml` expects.

- [ ] **Upload `folds.csv`. Do not regenerate it.** `11_compare_runs.py` does a paired per-scan
      comparison, and that is only valid if both runs used identical fold assignments. A
      regenerated split would silently invalidate the one number this whole trip is for.
- [ ] SSH transport is encrypted, so nothing extra is needed in flight. If you park the archive
      anywhere other than the instance — don't — encrypt it first.

## Step 4 — Smoke test, which also builds the cache

```bash
python scripts/03_train.py --config conf/highres_smoke.yaml
```

Two epochs, fold 0 only. Fold 0's train and val splits together cover all 1362 scans, so **this
one command warms the entire preprocessing cache** — the slow, CPU-bound half of a cold
high-resolution run. `highres_smoke.yaml`'s geometry block is byte-identical to `highres.yaml`,
so the cache key matches and the real run pays no resampling cost.

- [ ] **Epoch 2 must be dramatically faster than epoch 1.** If it is not, the cache is not being
      hit. Stop and find out why — otherwise the real run pays the resampling cost five times.
- [ ] Check the new `lr` column in the per-epoch line. It should start at `3.00e-04` and fall.
- [ ] Note epoch 2's wall clock. Multiply by `30 epochs × 5 folds` for the real run's estimate.
      If that lands beyond ~10 hours, reconsider before committing.
- [ ] This is a smoke test. **Never log a row from it.**

## Step 5 — The real run

```bash
nohup python scripts/03_train.py --config conf/highres.yaml > train_highres.log 2>&1 &
tail -f train_highres.log
```

Use `nohup` (or `tmux`) — an SSH drop must not kill a paid run. Expect ~3–6 h.

Produces `models/highres/`: `fold{0..4}.pt`, `oof_fold*.csv`, `oof.csv`, `metrics.json`, and
`history_fold*.json` — the per-epoch curve, which run 1 did not keep.

**Note `early_stopping_patience: 999` in `conf/highres.yaml`** — off by design. Early stopping
saves CPU nights, and on an A100 a fold is minutes. What it costs is the end of the cosine
schedule: `T_max = epochs`, so a fold that stops at 20 of 30 selects weights trained at a rate
well above the annealed one. `train_fold` already restores the best-validation-log-loss epoch,
so training past that point cannot make the shipped weights worse — it only costs GPU minutes.

- [ ] After the run, read the **Training schedule** table that `04_evaluate.py` now writes into
      `reports/experiments/highres.md`. `lr at best / peak` is the number to look at: near 1
      means the anneal barely started and the schedule is worth revisiting; near 0 means it
      finished and this question is closed.

## Step 6 — Everything else that needs the data, **while the box is still up**

This is the step people forget, and forgetting it means renting the box twice.

```bash
# TTA — 9 views x 5 folds x 1362 scans. Minutes on the GPU, hours on CPU.
python scripts/07_tta.py --config conf/highres.yaml --run highres --limit 40   # smoke first
python scripts/07_tta.py --config conf/highres.yaml --run highres

# SBR features at the NEW geometry. Preprocessing-bound.
python scripts/08_sbr_features.py --config conf/highres.yaml \
    --out data/interim/sbr_features_highres.csv
```

- [ ] **`--out` is not optional.** `08_sbr_features.py` defaults to
      `data/interim/sbr_features.csv`, which is your 96³ feature set. Overwriting it destroys
      the incumbent's provenance and there is no copy on this box.
- [ ] The features are computed from the COM-cropped canonical volume, so their **values change
      at 1.5 mm**. The 96³ features are not reusable. This is the step that makes a resolution
      change expensive.

## Step 7 — Pull back, then terminate

```bash
# From your LOCAL machine
scp -r <user>@<host>:/mnt/ram/models/highres           models/
scp -r <user>@<host>:/mnt/ram/models/highres_tta       models/
scp    <user>@<host>:/mnt/ram/data/interim/sbr_features_highres.csv  data/interim/
scp    <user>@<host>:/mnt/ram/dat-parkinsons-challenge/train_highres.log  reports/experiments/
```

- [ ] Verify locally: 5 `.pt` files in `models/highres/`, `oof.csv` in both run dirs,
      `sbr_features_highres.csv` non-empty.
- [ ] **Then terminate the instance. Terminate, do not stop** — a stopped instance keeps its
      disk. Confirm in the provider console that it is gone, not paused.
- [ ] **Fill in the teardown record in docs/07 §7.** Provider, instance id, dates. If this
      solution places, that table is what a compliance review asks for.

## Step 8 — The refit chain, locally

Every constant currently shipped is a property of the 96³ geometry and is **void** at 128³.

```bash
python scripts/04_evaluate.py --run highres          # new temperature; 1.2614 does NOT carry over
python scripts/09_blend.py --run highres_tta \
    --features data/interim/sbr_features_highres.csv # new stack coefficients
python scripts/11_compare_runs.py --run-a baseline_tta_blend --run-b highres_tta_blend
```

**`11_compare_runs.py` is the decision.** Paired, per-scan, with a standard error on the
difference itself. Read its gain against the ~0.01 bar, not against the pooled numbers.

- The old stack coefficients (intercept 0.0236, CNN 0.6164, trees 0.3779) are **void**. They were
  fitted on 96³ logits.
- The stack replaces temperature scaling; do not compose them. The packager already refuses
  `--temperature` alongside `--blend-from`.

## Step 9 — Only if it cleared the bar

```bash
python scripts/05_package_submission.py --config conf/highres.yaml --tta \
    --blend-from highres_tta_blend --fallback 0.5485 --clip-eps 1e-06
```

- [ ] **Verify `TARGET_SHAPE` in the packaged `main.py` reads 128, and the spacing 1.5.** This is
      the highest-risk item in the whole exercise: adaptive pooling means a 128³ model fed 96³
      volumes **raises no error and returns plausible wrong probabilities**. `log.md` flagged it
      on run 1; a resolution change makes it live again.
- [ ] Check `stack.json`'s `source_run` is `highres_tta` and its `feature_names` match
      `datpd.features.feature_names()`.
- [ ] Full local Docker test on the runtime checkout. Submission 2 closed this question for the
      96³ build (15 min of 180); a new geometry reopens it. Expect ~35 min — 2.37× the voxels —
      so the margin should stay large, but confirm rather than assume.
- [ ] Log the row in `reports/experiments/log.md` **and** start a
      `reports/submissions/003_*.md` file when it comes back.

## Step 10 — If it did not clear the bar

Also a result, and cheaper than the alternative.

- [ ] Record the negative in `log.md` with the measured gain and the bar it failed. docs/05's
      closing line is that the thing most likely to go wrong is arriving at Monday with an
      improvement that was measured but never packaged — the corollary is that a measured
      negative, written down, is worth what it cost.
- [ ] Ship the existing 0.2973 build. It is already on the leaderboard; nothing further is needed
      to keep it.
- [ ] Go to the Phase 4 checklist in `NEXT_SESSION.md`.

---

## The four ways this goes wrong

1. **`folds.csv` regenerated instead of uploaded** — the paired comparison silently stops being
   paired, and the one number the trip was for means nothing.
2. **`08_sbr_features.py` run without `--out`** — clobbers the incumbent's 96³ features, on a box
   where no backup exists.
3. **Terminating before step 6** — the data is gone and the box must be rented again.
4. **Shipping a 128³ model with a 96³ `TARGET_SHAPE` baked in** — no error, no warning,
   plausible wrong probabilities, and a wasted submission slot.
