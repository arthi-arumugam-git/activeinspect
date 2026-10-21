# Evaluation results

Held-out scenes: 30 seeds × 7 conditions × 3 defect types = 630 parts (2/3 defective).

| System | Defect recall | Escaped defects | False rejects | Escalated to human | Accuracy on auto-decisions | Captures/part | ms/part |
|---|---|---|---|---|---|---|---|
| single_shot | 9.8% | 379 | 0 | 0 | 39.8% | 1.0 | 8.1 |
| single_shot_clahe | 9.0% | 382 | 0 | 0 | 39.4% | 1.0 | 6.7 |
| agent | 90.7% | 8 | 0 | 61 | 98.6% | 7.15 | 46.1 |

## Recall by condition

| Condition | single_shot | single_shot_clahe | agent | agent escalations |
|---|---|---|---|---|
| nominal | 68% | 63% | 97% | 1 |
| dim | 0% | 0% | 97% | 0 |
| defocus | 0% | 0% | 97% | 1 |
| glare | 0% | 0% | 98% | 0 |
| dim+defocus | 0% | 0% | 97% | 0 |
| glare+defocus | 0% | 0% | 98% | 0 |
| very_dark | 0% | 0% | 52% | 59 |
