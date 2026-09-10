# Evaluation: baseline_anneal

- n = 273, positive rate = 0.549
- **prior baseline log loss = 0.6882** (the floor to beat)

## Training schedule

| fold | best epoch | last epoch | of | stopped early | lr at best / peak |
|---:|---:|---:|---:|---|---:|
| 0 | 11 | 19 | ? | ? | 0.42 |
| 1 | 3 | 3 | ? | ? | 0.95 |

- 2 folds trained.

## Uncalibrated
- log loss **0.3314**
- AUROC 0.9295
- ECE 0.0656

## Calibration

| method | OOF log loss |
|---|---:|
| temperature | 0.3226 **<- chosen** |
| none | 0.3314 |
| isotonic | 0.3938 |

- chosen: **temperature** (temperature = 1.2897)
- ECE after calibration: 0.0511 (was 0.0656)
- AUROC after calibration: 0.9295 (calibration is monotonic, so this should be unchanged)

## Prediction clipping

| clip eps | OOF log loss |
|---|---:|
| 1e-06 | 0.3203 **<- best** |
| 0.0001 | 0.3203 |
| 0.001 | 0.3203 |
| 0.005 | 0.3203 |
| 0.01 | 0.3203 |
| 0.02 | 0.3208 |
| 0.05 | 0.3293 |

## Summary

- final OOF log loss: **0.3203**
- improvement from calibration + clipping: **0.0111**
- improvement over prior baseline: **0.3680**

## To ship this

```
python scripts/05_package_submission.py --config conf/<the config these weights were trained with> --temperature 1.2897 --fallback 0.5495 --clip-eps 1e-06
```

Copy the fold checkpoints into `submission_src/assets/` first.
