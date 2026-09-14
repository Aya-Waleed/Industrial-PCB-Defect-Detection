# Final Submission Checklist

## Already verified in this repository

- [x] `model/best.pt` is present and loads.
- [x] Class names in the checkpoint and `model/class_names.txt` match.
- [x] Streamlit application starts locally.
- [x] Integration smoke test passes (`7/7`).
- [x] Training configuration was extracted from checkpoint metadata.
- [x] Dataset preparation report and reproducible audit/cleaning scripts are included.

## Requires Aya's actual dataset

- [x] Place the verified `dataset_final/` folder beside this README, including
  `train/`, `val/`, `test/`, and `data.yaml`; or provide an approved download
  link plus reproducible preparation steps.
- [x] Run `python scripts/evaluate_model.py --data dataset_final/data.yaml`.
- [x] Commit/attach the generated `reports/model_evaluation/` folder containing
  per-class metrics, confusion matrices, and review evidence for FP/FN.

## Requires real images / human review

- [ ] Complete the OOD protocol in `reports/OOD_TESTING.md` using at least 10
  genuinely external PCB images.
- [ ] Add 1–2 screenshots of the running app with an authorized PCB image and
  a detection result. Do not use a screenshot of an error screen as evidence.

## Before presenting

- [ ] Re-run `python -m pytest tests/ -v` after `pip install -r requirements.txt`.
- [ ] Re-run `python scripts/benchmark_inference.py` on the actual demo machine
  and report that machine's result.
- [ ] Keep all reported metrics labelled **validation** or **test** correctly.
