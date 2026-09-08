# 02 — From MNIST to 3D medical imaging

You have trained a CNN on MNIST, with ReduceLROnPlateau and the usual Keras/PyTorch loop. That
transfers further than you'd think. This document is about the parts that **don't** transfer,
written as a set of deltas rather than a from-scratch tutorial.

---

## The seven deltas

| | MNIST | This competition |
|---|---|---|
| 1. Input | 28x28x1, uniform | ~142x142x112 **or** 128x128x128, variable **physical** scale |
| 2. Op | `Conv2d` | `Conv3d` — compute and memory scale with the cube |
| 3. Norm layer | `BatchNorm` | `GroupNorm` (batch is too small, and BN risks a rules violation) |
| 4. Data volume | 60,000 clean samples | Low thousands, noisy, expensive |
| 5. Label quality | Unambiguous | ~1 in 5 genuinely ambiguous, expert disagreement is real |
| 6. Metric | Accuracy | **Log loss** — calibration is scored |
| 7. Split | Random is fine | **Grouped** — ten centres, leakage is silent |

Work through them in order; each one is a place where MNIST instincts actively mislead.

---

## Delta 1 — Variable shape *and* variable physical scale

The genuinely new idea: in medical imaging a voxel has a **size in millimetres**, and it
differs between scans.

```
Volume A: 142 x 142 x 112 @ 2.46 mm   ->  349 x 349 x 276 mm of head
Volume B: 128 x 128 x 128 @ 3.895 mm  ->  499 x 499 x 499 mm of head
```

If you resize both to 96³ with a naive `zoom`, the striatum ends up a different number of
voxels wide in each — so a 3x3x3 kernel spans different amounts of anatomy, and the network
burns capacity undoing that.

**Fix, in this order:**

1. **Reorient** to canonical RAS (`nibabel.as_closest_canonical`) so axes mean the same thing.
2. **Resample** to a fixed spacing, e.g. 2.0 mm isotropic. Now one voxel = one physical size.
3. **Crop or pad** to a fixed array shape, e.g. 96³. Now one tensor shape.

Steps 2 and 3 are different operations and conflating them is the classic beginner error.
Implemented in [preprocess.py](../src/datpd/data/preprocess.py); the test
`test_full_pipeline_yields_fixed_shape` asserts heterogeneous inputs give one output shape.

**Cache the result.** Resampling thousands of volumes every epoch will dominate your runtime.
Preprocess once to `data/processed/cache/`, then training reads fixed-size arrays.

---

## Delta 2 — 3D convolutions and the memory wall

A `Conv3d` with kernel 3 has 27 weights per channel pair versus 9 for `Conv2d`. Worse, the
activation tensors are cubic:

```
96 x 96 x 96 x 32 channels x 4 bytes  =  113 MB   for ONE sample, ONE layer
```

Practical consequences:

- **Batch sizes are small** (4–16), not 128.
- **Downsample early.** The stem in [resnet3d.py](../src/datpd/models/resnet3d.py) uses
  `stride=2` immediately; halving each dimension cuts activation memory 8x.
- **Channel counts stay modest** — (16, 32, 64, 128), not (64, 128, 256, 512).
- **Use AMP** (`torch.autocast` + `GradScaler`). Roughly halves memory, ~1.5–2x faster on A100.
- **Gradient checkpointing** if you need a bigger model than fits.

### The 2.5D alternative

Instead of a 3D CNN, take 2D slices (axial, or a few around the striatum), feed them to a
**pretrained 2D backbone** (ResNet/EfficientNet via `timm` — available in the runtime), and
aggregate per-scan (mean, max, or attention pooling — this is multiple-instance learning).

Why consider it:
- **ImageNet pretraining is available in 2D and mostly is not in 3D.** With low-thousands of
  training samples, that transfer can matter more than the extra dimension.
- Far cheaper, so you iterate faster.

Why not:
- Discards through-plane context, which encodes the posterior-anterior gradient (docs/01 §4).

Given the deadline, a fair plan: 3D CNN as the primary line, 2.5D as the diversity member of a
final ensemble. Ensembling architecturally different models reliably helps log loss.

---

## Delta 3 — GroupNorm, not BatchNorm

BatchNorm normalises using statistics **across the batch**. Two problems here:

1. **Statistical.** At batch size 4–8 those statistics are noisy, and the running means used at
   inference are poor estimates. This is a well-known failure mode in 3D medical imaging.
2. **Legal.** If a BatchNorm layer is ever in `train()` mode at inference, a sample's prediction
   depends on the other samples in its batch. That directly violates the competition's
   per-sample independence rule ([00_competition_brief](00_competition_brief.md)).

**GroupNorm** normalises within channel groups of a single sample. It is batch-independent by
construction, which makes independence structurally true rather than a promise you have to
remember to keep. `test_eval_mode_is_batch_size_independent` in the test suite verifies it.

---

## Delta 4 — Small data

You cannot brute-force a few thousand 3D volumes the way you can 60,000 MNIST digits.

**Augmentation that respects the anatomy** (see [AugConfig](../src/datpd/config.py)):

| Augmentation | Verdict |
|---|---|
| Small rotations (±10°) | Yes — head positioning genuinely varies |
| Translation (±5 mm) | Yes — centring varies |
| Scaling (0.9–1.1) | Yes — head size varies |
| Intensity jitter / Gaussian noise | Yes — mimics count-statistics variation |
| **Left-right flip** | **Default off.** Asymmetry is diagnostic (docs/01 §4). Test it as an experiment. |
| Elastic deformation | Cautiously — can distort the very shape you are classifying |
| Heavy cropping | No — you may cut off the structure of interest |
| MixUp | Worth one experiment; it also tends to improve calibration |

`torchio` is in the runtime and is purpose-built for 3D medical augmentation. `monai.transforms`
likewise. Prefer them over hand-rolling.

**Regularisation:** dropout before the head, weight decay, early stopping on validation
**log loss** (not accuracy, and not validation loss if you use a different training loss).

---

## Delta 5 — Ambiguous labels

MNIST labels are certain. Here, 11 specialists hand-labelled scans and ~20% of cases are hard.
Some labels are effectively coin flips.

- **Do not chase 100% training accuracy.** You will be memorising label noise.
- **Label smoothing** (target 0.05/0.95 instead of 0/1) is a natural fit and directly helps
  calibration.
- **Never treat a confidently-misclassified training sample as a bug to fix.** It may simply be
  a hard case.

---

## Delta 6 — Log loss changes your whole loop

This is the biggest practical shift. Concretely:

**Train with `BCEWithLogitsLoss`**, not sigmoid-then-BCE. It fuses the operations and is
numerically stable — the model outputs a logit, which is why
[resnet3d.py](../src/datpd/models/resnet3d.py) returns logits.

**Early-stop on validation log loss.** Not accuracy, not AUROC.

**Always calibrate.** Standard flow:

```
train K folds  ->  collect out-of-fold predictions  ->  fit temperature on OOF
               ->  bake the scalar into main.py     ->  submit
```

`TemperatureScaler` in [calibrate.py](../src/datpd/calibrate.py) is one parameter, takes
seconds, and typically buys a real log-loss improvement for free. Fit it on **OOF predictions
only** — fitting on training predictions gives a temperature near 1 and does nothing.

**Clip your outputs.** One prediction of 1e-6 on the wrong side costs ~13.8. Tune the clip on
OOF (`DEFAULT_CLIP` in [metrics.py](../src/datpd/utils/metrics.py)).

**Report both metrics, every time.** AUROC up + log loss up = a calibration problem, not a
modelling failure. Recalibrate before you conclude anything.

### Your MNIST habits, translated

Your MNIST notebook used `ReduceLROnPlateau`. It works here too — just point it at validation
**log loss**. For runs of ~30 epochs, cosine annealing with a short warmup is usually the
stronger default and needs no patience tuning; both are supported in
[config.py](../src/datpd/config.py). Keep `ReduceLROnPlateau` if you want fewer moving parts —
it is a reasonable choice, not a mistake.

---

## Delta 7 — Grouped cross-validation

Random splits are fine for MNIST. Here, ten centres means a random split can put scans from the
same scanner on both sides, and the model can exploit reconstruction artefacts. CV looks great,
the leaderboard doesn't.

Centre isn't given, but `(shape, spacing)` is close to a scanner fingerprint. Compute both
splits ([splits.py](../src/datpd/data/splits.py)):

- `fold_strat` — stratified. Lower variance, mildly optimistic.
- `fold_group` — grouped by acquisition signature. Pessimistic, more honest.

**Track both.** A large gap means centre effects are real; then trust the grouped number and
prioritise robustness (stronger augmentation, per-scan normalisation) over squeezing CV.

---

## A concrete first architecture

Start with [`ResNet3D`](../src/datpd/models/resnet3d.py):

```
input  (B, 1, 96, 96, 96)
  stem       Conv3d k5 s2 -> GroupNorm -> SiLU        (B, 16, 48, 48, 48)
  stage 1    ConvBlock s2, ConvBlock                  (B, 32, 24, 24, 24)
  stage 2    ConvBlock s2, ConvBlock                  (B, 64, 12, 12, 12)
  stage 3    ConvBlock s2, ConvBlock                  (B, 128, 6, 6, 6)
  pool       AdaptiveAvgPool3d(1)                     (B, 128)
  head       Dropout -> Linear                        (B,) logits
```

Roughly 1–2 M parameters. Small on purpose: with a few thousand samples, a bigger model mostly
buys overfitting. Get this working end to end, submit it, *then* scale up.

**When you want more:** `monai.networks.nets` ships DenseNet121 / ResNet / SEResNet in 3D, some
with pretrained weights from `MedicalNet`-style corpora. MONAI is in the runtime. Check the
licence of any pretrained weights and record it in [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md).

---

## Debugging a 3D pipeline (in this order)

1. **Overfit 10 samples.** If loss doesn't reach ~0, the bug is in data or the loop, not the
   model. Fastest signal you can get.
2. **Check shapes and spacings after preprocessing** — assert, don't print. Shapes and dtypes
   are safe to log; voxel values are not (CLAUDE.md §0).
3. **Verify the training and inference preprocessing are identical.** The #1 cause of "CV is
   great, leaderboard is terrible". `05_package_submission.py` vendors the same modules
   specifically to prevent divergence.
4. **Check orientation.** If some volumes are flipped, the model sees two mirror populations.
   `as_closest_canonical` handles it; confirm the orientation codes are uniform in your EDA
   summary.
5. **Compare against the prior baseline.** If you can't beat `baseline_logloss`, stop tuning and
   go back to step 1.
