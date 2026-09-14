# Final Model Evaluation

**Model:** YOLO11n (`best.pt`)

**Dataset:** cleaned, held-out `dataset_final/test` split

**Evaluation date:** 2026-09-14
**Test set:** 1,063 images, 2,153 annotated defect instances

## Overall test metrics

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.7470 | 0.9156 | 0.7656 | 0.3670 |

## Per-class test metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| mouse_bite | 0.7802 | 0.8817 | 0.7520 | 0.3523 |
| spur | 0.7771 | 0.8746 | 0.7585 | 0.3556 |
| missing_hole | 0.7196 | 0.9894 | 0.7739 | 0.3946 |
| short | 0.7316 | 0.9630 | 0.7772 | 0.3836 |
| open_circuit | 0.7641 | 0.9167 | 0.7985 | 0.3530 |
| spurious_copper | 0.7093 | 0.8681 | 0.7336 | 0.3631 |

## Error analysis

At the app's default operating point (confidence >= 0.25; same-class IoU >=
0.50), `spurious_copper` has the most false positives (65), while
`mouse_bite` has the most false negatives (29). `missing_hole` has the highest
recall (0.9894). See the reproducible outputs in
`reports/model_evaluation/`, especially `confusion_matrix.png`,
`fp_fn_analysis.md`, and `error_examples.csv`.

## Interpretation

Test mAP50-95 is lower than test mAP50, which indicates that the detector finds
many defects but has less precise localization at strict IoU thresholds. This
is a known challenge for small PCB defects.

The checkpoint's embedded validation metrics remain recorded in
`model/model_info.txt`; they are not substituted for these held-out test
results.
