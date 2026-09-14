# False-positive / False-negative Analysis

Operating point: confidence >= 0.25, same-class IoU >= 0.50.
These counts are threshold-dependent and are not interchangeable with AP/mAP.

| Class | True positives | False positives | False negatives |
|---|---:|---:|---:|
| mouse_bite | 303 | 20 | 29 |
| spur | 314 | 27 | 34 |
| missing_hole | 377 | 21 | 2 |
| short | 351 | 20 | 10 |
| open_circuit | 325 | 24 | 20 |
| spurious_copper | 360 | 65 | 28 |

See `error_examples.csv` for up to 200 reviewable examples.
