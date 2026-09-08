# Evaluation: baseline

- n = 1362, positive rate = 0.548
- **prior baseline log loss = 0.6884** (the floor to beat)

## Uncalibrated
- log loss **0.3271**
- AUROC 0.9302
- ECE 0.0404

## Calibration

| method | OOF log loss |
|---|---:|
| temperature | 0.3217 **<- chosen** |
| none | 0.3271 |
| isotonic | 0.3520 |

- chosen: **temperature** (temperature = 1.2614)
- ECE after calibration: 0.0220 (was 0.0404)
- AUROC after calibration: 0.9302 (calibration is monotonic, so this should be unchanged)

## Prediction clipping

| clip eps | OOF log loss |
|---|---:|
| 1e-06 | 0.3186 **<- best** |
| 0.0001 | 0.3186 |
| 0.001 | 0.3186 |
| 0.005 | 0.3186 |
| 0.01 | 0.3188 |
| 0.02 | 0.3199 |
| 0.05 | 0.3263 |

## Summary

- final OOF log loss: **0.3186**
- improvement from calibration + clipping: **0.0085**
- improvement over prior baseline: **0.3698**

## To ship this

```bash
python scripts/05_package_submission.py \
    --config conf/<the config these weights were trained with> \
    --temperature 1.2614 \
    --fallback 0.5485 \
    --clip-eps 1e-06
```

Copy the fold checkpoints into `submission_src/assets/` first.
