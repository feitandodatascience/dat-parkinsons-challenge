# SBR features by acquisition family

Testing one hypothesis: is the worst-performing acquisition family worst because its
**crops** are worse, rather than its images? See the module docstring of
`scripts/10_features_by_family.py` for the reasoning.

- n = 1362, families reported at n >= 30

## Crop quality

`peak in box` is the brightest voxel inside the central striatal box divided by the
brightest voxel anywhere in the volume. After a correct centre-of-mass crop the
striatum is the brightest thing present and sits in the box, so this should be at or
near **1.00**. Values well below 1 mean the peak uptake is outside the box — the crop
missed. The 5th percentile matters more than the median: a crop that fails on one scan
in twenty still costs real log loss.

| family | n | peak in box (median) | peak in box (p05) | striatal contrast (median) |
|---|---:|---:|---:|---:|
| 256@2.5x2.5x2.5mm | 460 | 1.000 | 0.568 | -2.187 |
| 128@3.9x3.9x3.9mm | 325 | 0.941 | 0.469 | -3.893 |
| 128@2.3x2.3x2.3mm | 207 | 1.000 | 0.653 | -11.866 |
| 128@2.4x2.4x2.4mm | 149 | 1.000 | 1.000 | -1.481 |
| small families (pooled) | 125 | 1.000 | 0.609 | -1.840 |
| var@2.5x2.5x2.5mm | 66 | 1.000 | 0.504 | -3.428 |
| 128@1.5x1.5x1.5mm | 30 | 1.000 | 0.655 | -1.055 |

- spread in median peak-in-box across families: **0.059**
- lowest: **128@3.9x3.9x3.9mm** (n=325) at 0.941

**Verdict: worth investigating.** Families differ materially in how often the crop
captures peak uptake. Before acting, check whether the low-scoring families are the
wide-field ones — a 256 matrix at 2.5 mm is a 640 mm cube, against 307 mm for a 128
matrix at 2.4 mm — because that is the mechanism this predicts. A crop fix requires
retraining, so confirm the pattern first.

## Other features

| family | n | striatal contrast | supra-threshold fraction | lobe elongation (low side) | absolute asymmetry |
|---|---:|---:|---:|---:|---:|
| 256@2.5x2.5x2.5mm | 460 | -2.187 | 0.027 | 0.734 | 0.045 |
| 128@3.9x3.9x3.9mm | 325 | -3.893 | 0.026 | 0.763 | 0.029 |
| 128@2.3x2.3x2.3mm | 207 | -11.866 | 0.029 | 0.774 | 0.039 |
| 128@2.4x2.4x2.4mm | 149 | -1.481 | 0.052 | 0.713 | 0.027 |
| small families (pooled) | 125 | -1.840 | 0.037 | 0.718 | 0.039 |
| var@2.5x2.5x2.5mm | 66 | -3.428 | 0.032 | 0.701 | 0.044 |
| 128@1.5x1.5x1.5mm | 30 | -1.055 | 0.034 | 0.700 | 0.061 |

Medians over each family. These are the model's raw inputs, so a family whose
distribution sits far from the others is a family the network sees differently — which
is worth knowing whether or not the crop turns out to be the cause.

