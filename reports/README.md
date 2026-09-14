# Evaluation outputs

The final test evaluation has been generated from the cleaned `dataset_final/`
test split. Its results are in `model_evaluation/`:

- `evaluation_results.md` and `per_class_metrics.csv`
- `confusion_matrix.png` and `confusion_matrix_normalized.png`
- `fp_fn_analysis.md`, `fp_fn_summary.csv`, and `error_examples.csv`
- PR/precision/recall/F1 curves and representative prediction images

To reproduce it:

```powershell
python scripts/evaluate_model.py --data dataset_final/data.yaml
```

It regenerates `reports/model_evaluation/` with verified per-class metrics,
confusion matrices, curves, and thresholded FP/FN evidence from the final
held-out test split.
