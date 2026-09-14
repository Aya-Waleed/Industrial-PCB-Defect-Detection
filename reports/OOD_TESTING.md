# Out-of-Distribution (OOD) Test Protocol

This check is intentionally pending: the repository contains no real PCB
photos outside the training dataset. Do not fabricate OOD results.

## Goal

Test whether `model/best.pt` behaves sensibly on PCB images from a different
source, camera, lighting condition, board design, or manufacturing line than
the training data.

## Procedure

1. Collect at least 10 images not present in `dataset_final/`. Record their
   source and confirm that using them is permitted.
2. Include clean boards and boards with known defects where possible. Keep the
   original files unchanged in a private `ood_samples/` folder; do not commit
   restricted industrial images to a public repository.
3. Run the Streamlit app or the command below for each image:

   ```powershell
   python -c "from ultralytics import YOLO; YOLO('model/best.pt').predict(source='ood_samples', imgsz=320, conf=0.25, save=True)"
   ```

4. Manually inspect every detection. Record false positives, missed known
   defects, uncertain cases, and the image conditions (glare, blur, rotation,
   low contrast, unfamiliar board layout).
5. Add representative annotated images to `reports/ood_results/` only if they
   may be shared, then complete the table below.

## Results template

| Image/source | Condition | Known defect(s) | Prediction(s) | Outcome | Notes |
|---|---|---|---|---|---|
| _pending_ | | | | | |

## Reporting rule

This is a qualitative robustness check, not a replacement for the held-out
test-set metrics. State the sample count and selection method alongside any
conclusion.
