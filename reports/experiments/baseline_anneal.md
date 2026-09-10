# Evaluation: baseline_anneal

- n = 1362, positive rate = 0.548
- **prior baseline log loss = 0.6884** (the floor to beat)

## Training schedule

| fold | best epoch | last epoch | of | stopped early | lr at best / peak |
|---:|---:|---:|---:|---|---:|
| 0 | 11 | 19 | 20 | no | 0.42 |
| 1 | 17 | 19 | 20 | no | 0.05 |
| 2 | 11 | 19 | 20 | no | 0.42 |
| 3 | 10 | 19 | 20 | no | 0.50 |
| 4 | 15 | 19 | 20 | no | 0.15 |

- 5 folds trained.

## Uncalibrated
- log loss **0.3336**
- AUROC 0.9269
- ECE 0.0374

## Calibration

| method | OOF log loss |
|---|---:|
| temperature | 0.3294 **<- chosen** |
| none | 0.3336 |
| isotonic | 0.3710 |

- chosen: **temperature** (temperature = 1.2189)
- ECE after calibration: 0.0194 (was 0.0374)
- AUROC after calibration: 0.9269 (calibration is monotonic, so this should be unchanged)

## Prediction clipping

| clip eps | OOF log loss |
|---|---:|
| 1e-06 | 0.3274 **<- best** |
| 0.0001 | 0.3274 |
| 0.001 | 0.3274 |
| 0.005 | 0.3274 |
| 0.01 | 0.3275 |
| 0.02 | 0.3285 |
| 0.05 | 0.3355 |

## Summary

- final OOF log loss: **0.3274**
- improvement from calibration + clipping: **0.0062**
- improvement over prior baseline: **0.3610**

## To ship this

```
python scripts/05_package_submission.py --config conf/<the config these weights were trained with> --temperature 1.2189 --fallback 0.5485 --clip-eps 1e-06
```

Copy the fold checkpoints into `submission_src/assets/` first.
