# 07 — Renting GPU compute without breaking the data rules

**Written 2026-09-09.** Settles the question left open as `<FILL IN>` in `NEXT_SESSION.md`, which
gated the only large modelling win remaining (128³ @ 1.5 mm, docs/05 item 8).

---

## 1. The ruling

A competitor asked the organizers, on the official forum, two questions: whether an AI coding
assistant may be used to write code provided it never reads the data, and whether the data may be
uploaded to a private cloud compute instance used purely as infrastructure.

The organizers did not answer yes or no. They restated the governing sentence —

> "Never share, copy, or publish the data. For example, you may not use tools like Codex and
> ChatGPT that store or retain uploaded data, though you may download model weights and run
> models locally"

— and then gave the **test** to apply:

> "So the things to watch for are whether the challenge data is being retained by a third party,
> whether it's accessible to anyone who hasn't agreed to the competition rules, and whether every
> copy of it can be deleted at the end of the challenge. Some coding assistants read local files
> by default, so competitors should check what a given tool has access to. If you're using cloud
> compute, any workspace, notebook, or storage holding the data should be private to you, with the
> provider's terms not granting them rights to retain your uploaded content."

**Renting GPU compute is therefore permitted, conditionally.** Three conditions, all of which must
hold:

1. **No third-party retention.** The provider's terms must not grant them rights over uploaded
   content.
2. **Private.** No one who has not agreed to the competition rules may have access — including the
   owner of the physical machine.
3. **Fully deletable.** Every copy must be removable at the end of the challenge.

Note also the first half of the answer, which validates this repository's whole working model:
"some coding assistants read local files by default, so competitors should check what a given tool
has access to." That is precisely what `CLAUDE.md` §0 and `.claude/hooks/guard_data.py` exist to
guarantee. The arrangement here — Claude writes code and never reads `data/` — is the arrangement
the organizers described as the thing to check for.

## 2. Providers, against the three conditions

| Option | Verdict | Why |
|---|---|---|
| **Vast.ai**, **RunPod Community Cloud** | **Ruled out** | GPUs are hosted on third parties' own hardware. The machine's owner has physical access to the disk. Fails condition 2 outright, and condition 3 is unverifiable. Not defensible for patient imaging under any price argument. |
| **Kaggle Notebooks** | **Ruled out** | Kaggle's terms take a licence in uploaded content (condition 1). Separately, uploading one organizer's competition data to a competing competition platform is indefensible on its face, whatever the licence says. |
| **Google Colab** (+ Drive) | **Avoid** | Google's consumer terms take a broad hosting licence; backup and retention are not under the user's control (conditions 1 and 3). Colab now embeds Gemini, which puts it in the category the rule names explicitly. |
| **Lambda Labs on-demand**, **RunPod Secure Cloud**, **AWS / GCP / Azure** | **Viable** | Provider-owned datacentres, enterprise customer-content clauses that disclaim rights in customer data, and instance teardown under the customer's control. |

### Pick one: Lambda Labs on-demand, 1× A100 40GB

Not a menu — a ladder, because on-demand A100s are frequently sold out:

1. **Lambda Labs on-demand.** SSH in ten minutes, instance-local NVMe, no persistent volume
   unless one is deliberately attached, terms adequate for this.
2. **RunPod Secure Cloud.** Explicitly *Secure*, never *Community*.
3. **A hyperscaler only if GPU quota is already approved.** See below.

**The hyperscalers have the best contract language and are still the wrong choice here.** Their
customer agreements state in terms that the provider acquires no rights in customer content,
which is the cleanest possible answer to condition 1. But condition 1 is not the binding
constraint — **the calendar is**, and on the calendar Azure/AWS/GCP are the worst of the viable
options:

- **GPU quota starts at zero.** A new subscription cannot launch a GPU VM until a quota increase
  is granted, which routinely takes days. That alone can consume the Sat 12 Sep abort line
  before a single epoch has run.
- **Durable storage is the default.** The tmpfs design in §3 exists to keep data off persistent
  disk; the hyperscalers' defaults push the other way, so there is more to get right under time
  pressure.
- **More surface area** — resource groups, VNets, security groups, storage accounts — all of it
  first-time setup during lock-down week.
- **More expensive per GPU-hour** than the specialist providers.

**Do not rent an H100.** The model is a small ResNet3D at batch 8 and will not saturate an A100.
The wall clock is set by the cold cache build, which is CPU-bound: prioritise **≥ 100 GB RAM and
≥ 16 vCPU** over GPU class.

**Region: prefer the EU where the choice is free.** This is French hospital patient imaging. It is
*not* a rules requirement — the organizers distribute the data to competitors worldwide, so they
are plainly not restricting geography — but an EU region costs nothing and is easier to defend.
Lambda's region choice is limited and mostly US; RunPod has EU capacity. Treat this as a
tiebreaker between otherwise equal options, **not** as a reason to accept a multi-day quota wait
at a hyperscaler.

## 3. The operational shape matters as much as the vendor

Adopt this regardless of which viable provider is chosen. It converts condition 3 from "audit the
provider's retention policy" into "the instance no longer exists":

- **Ephemeral instance storage only.** No persistent volume, no attached block storage that
  outlives the instance, no S3/GCS/Azure Blob bucket, no provider-side snapshot or image.
- **Encrypt before it leaves this machine.** `age` or `gpg` the archive locally, `scp` the
  ciphertext, decrypt on the instance. Anything the provider incidentally captures — a host
  snapshot, a transit log — is then ciphertext.
- **Pull back model artifacts and metrics only.** Checkpoints, `oof_*.csv`, `metrics.json`. Never
  copy volumes back.
- **Terminate, do not stop.** A stopped instance retains its disk.
- **Record the teardown.** Date, provider, instance id, in this file. If the solution wins, the
  reproduction documentation should be able to state that every copy was destroyed.

One further point specific to this repo: metrics that come back from a cloud run reach Claude the
same way local ones do, so `CLAUDE.md` §0 applies unchanged. Aggregates into
`reports/experiments/`; nothing per-scan.

## 4. What this unlocks, and what it does not

`conf/highres.yaml` is the config. GPU support needs no code change — `train.py` selects CUDA
automatically and AMP is already gated on it.

**Compute is not the constraint.** docs/05 prices 128³ @ 1.5 mm at 2.4 R ≈ 115 CPU-hours. On an
A100 with AMP that is roughly **3–6 hours**, plus ~15 minutes to build the cold preprocessing cache
across 24 vCPUs. Half a day, not five days.

**The constraint is the refit chain behind it.** Every calibration constant currently shipped is a
property of the 96³ run and is void at a new geometry:

```
03_train.py --config conf/highres.yaml    new checkpoints
04_evaluate.py --run highres              new temperature — T=1.2614 does not carry over
07_tta.py --run highres                   TTA softening differs at the new geometry
08_sbr_features.py --config highres       ~30 min; features come from the COM-cropped
                                          canonical volume, so their values change
09_blend.py --run highres_tta             new stack coefficients — 0.6164 / 0.3779 are void
11_compare_runs.py                        paired comparison against the current 0.3061 build
05_package_submission.py                  re-verify every baked constant
```

That is the whole of Phase 3 re-run. It is cheap in CPU (docs/05 prices items 2–3 at 0.06 R) and
expensive in opportunities to make a mistake under deadline pressure.

**The highest-risk failure is silent.** `main.py` bakes `TARGET_SHAPE`, and adaptive pooling means
a 128³ model fed 96³ volumes raises no error and returns plausible wrong probabilities. `log.md`
already flagged this as the first thing to check if run 1 came back much worse. A resolution change
makes it live again.

## 5. The submission-slot arithmetic, which is the actual binding constraint

`NEXT_SESSION.md` states both "next slot frees ~09-12" and "submit the intended final model Mon
09-14". On the conservative reading of 3 submissions per rolling 7 days, **those are mutually
exclusive**. Submissions used: 09-05, 09-08, 09-09.

| Submitting on | Trailing 7-day window holds | Slots free |
|---|---|---|
| 09-12 | 09-08, 09-09 | 1 |
| 09-14, *if* 09-12 was spent | 09-08, 09-09, 09-12 | **0** |
| 09-14, if 09-12 was *not* spent | 09-08, 09-09 | 1 |

**Spending the 09-12 slot forecloses the 09-14 submission.** Verify against the platform's own
counter rather than this table, but plan for it.

This does not block the GPU run. The CV↔LB gap is now bounded at roughly **±0.01** by two
observations (+0.0024 on run 1, −0.0088 on submission 2 — see `log.md`), which is loose enough to
matter but tight enough that **a high-resolution run can be accepted or rejected on OOF alone,
with no submission spent finding out.** The GPU buys a measurement, not a submission. The two do
not compete.

## 6. Recommendation

Rent it, run 128³, and treat the result as a measurement to be judged against the current build's
OOF of 0.3061 — not as a submission to be made.

- If the highres OOF beats 0.3061 by more than **~0.01** — the CV↔LB spread established by the two
  submissions to date, not the ±0.0024 a single point appeared to give — it becomes the intended
  final model, and the whole refit chain in §4 has to complete and be packaged before Sun 13.
  docs/05 expects −0.02 to −0.05, so a real win should clear this comfortably; a marginal one
  should be read as unresolved and declined on schedule risk alone.
- If it does not, the current TTA + blend build is the final model and nothing is lost but the
  rental fee. **Record the negative in `log.md`** — docs/05's closing line is that the thing most
  likely to go wrong is arriving at Monday with an improvement that was measured but never
  packaged.
- Either way, **keep the 09-14 slot free**, and do not spend 09-12 unless the highres build is
  packaged, Docker-tested, and measurably better.

**The bar is now higher than when this file was drafted.** Submission 2 came back at **0.2973**
(AUROC 0.9396, rank #98/897). A 128³ run has to beat a build that is already working, already
packaged, already validated on the platform, and already inside its runtime budget by 12×. The
downside of chasing it is not the rental fee — it is arriving at Sunday with a half-refit chain
and having to ship the old zip anyway.

One thing the 0.2973 run does make easier: **inference runtime is no longer a question.** It
completed in 15 minutes of a 3-hour budget on the real test set. Even if 128³ scaled the entire
run linearly with voxel count (2.37×, and it will not — some of that 15 min is fixed cost), the
result is ~35 min. Resolution is not going to blow the time limit.

The 128³ deadline stands as docs/05 wrote it: **if it is not training by Sat 12 Sep, abandon it**
and spend the remaining days on the Phase 4 lock-down checklist.

## 7. Teardown record

Fill in at the end of the challenge. Required by condition 3, and worth having in writing if a
prize triggers a compliance review.

| provider | instance id | data uploaded | terminated | verified deleted by |
|---|---|---|---|---|
| | | | | |
