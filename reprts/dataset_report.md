# Industrial PCB Defect Detection System
## Dataset Preparation & Quality Report

**Role:** Data Preparation & Dataset  
**Dataset:** PCB Defect dataset  
**Format:** YOLO Object Detection  
**Prepared dataset:** `dataset_final/`

---

## 1. Purpose

This report documents the complete dataset preparation process for the Industrial PCB Defect Detection System.

The project is an **object detection** task. The model must identify the type of PCB defect and its location using bounding-box annotations.

The project brief identifies six PCB defect categories. The selected dataset uses the following six class names:

1. `mouse_bite`
2. `spur`
3. `missing_hole`
4. `short`
5. `open_circuit`
6. `spurious_copper`

> **Terminology note:** The project brief uses the term **Pin Hole**, while the selected dataset uses **missing_hole**. The original dataset terminology and class IDs are preserved rather than renaming the class without evidence.

---

## 2. Dataset Source

The selected dataset is the **PCB Defect dataset** published on Kaggle by **Norbert Elter**.

The dataset is provided for PCB defect detection using YOLO annotations.

The original dataset contains:

- **10,668 images**
- **10,668 corresponding label files**
- **6 defect classes**
- Separate `train`, `val`, and `test` splits

The original split was:

| Split | Images |
|---|---:|
| Train | 8,534 |
| Validation | 1,066 |
| Test | 1,068 |
| **Total** | **10,668** |

---

## 3. Original Dataset Structure

The downloaded dataset was organized as:

```text
pcb-defect-dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

The `data.yaml` file was inspected to verify the class mapping.

---

## 4. Verified Class Mapping

The class mapping from `data.yaml` is:

| Class ID | Class Name |
|---:|---|
| 0 | `mouse_bite` |
| 1 | `spur` |
| 2 | `missing_hole` |
| 3 | `short` |
| 4 | `open_circuit` |
| 5 | `spurious_copper` |

This mapping is preserved in the cleaned dataset.

---

## 5. Annotation Format

The dataset uses the **YOLO object-detection annotation format**.

Each annotation contains five values:

```text
class_id x_center y_center width height
```

Example:

```text
2 0.4350 0.7008 0.0700 0.0550
```

Where:

- `2` = class ID (`missing_hole`)
- `x_center` = normalized bounding-box center X
- `y_center` = normalized bounding-box center Y
- `width` = normalized bounding-box width
- `height` = normalized bounding-box height

The bounding-box coordinates are normalized to the range `[0, 1]`.

Multiple annotations can exist in a single image.

---

## 6. Dataset Inspection

The dataset was inspected before preparation.

The following were verified:

- Image directories
- Label directories
- `data.yaml`
- Class IDs
- Class names
- YOLO annotation structure
- Image/label relationships
- Number of annotations
- Class distribution

Two image/label filename conventions were identified.

### Convention A

```text
Image:
light_01_missing_hole_02_3_600.jpg

Label:
light_01_missing_hole_02_3_256.txt
```

### Convention B

```text
Image:
l_light_01_missing_hole_01_1_600.jpg

Label:
l_light_01_missing_hole_01_1_600.txt
```

This was important during dataset auditing because comparing image and label stems directly would incorrectly report valid pairs as missing.

The audit script was therefore updated to recognize both conventions.

---

## 7. Duplicate Analysis

An exact duplicate analysis was performed using image file hashes.

The purpose was to identify identical images appearing across different dataset splits because exact duplicates between training and evaluation data can cause data leakage.

The analysis found:

- **9 cross-split exact duplicate groups**
- **6 validation copies** identified for removal
- **5 test copies** identified for removal

The training copies were preserved.

### Cleaning policy

The following rule was used:

> Preserve the training version and remove exact duplicate copies from validation or test when the same image already exists in the training split.

This produced a cleaner separation between training and evaluation data.

The original dataset was not modified.

A separate cleaned dataset was created:

```text
dataset_final/
```

---

## 8. Perceptual Duplicate Analysis

Perceptual similarity checks were also investigated.

However, broad perceptual hashing was **not used as the final duplicate-removal criterion**.

PCB images contain large amounts of shared board structure. As a result, perceptual similarity can classify visually similar but genuinely different defect images as duplicates.

Therefore:

- Exact image-hash matching was used for confirmed duplicate removal.
- Perceptual similarity results were not treated as confirmed duplicates.
- No additional images were removed based only on perceptual similarity.

This avoids unnecessarily deleting valid training examples.

---

## 9. Final Clean Dataset Structure

The final dataset is:

```text
dataset_final/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
└── duplicate_cleaning_report.txt
```

The original dataset remains separately available:

```text
pcb-defect-dataset/
```

The original source dataset was not modified.

---

## 10. Final Dataset Statistics

After duplicate cleaning, the final dataset contains:

| Split | Images | Label Files | Annotations |
|---|---:|---:|---:|
| Train | 8,534 | 8,534 | 17,342 |
| Validation | 1,060 | 1,060 | 2,154 |
| Test | 1,063 | 1,063 | 2,153 |
| **Total** | **10,657** | **10,657** | **21,649** |

### Final split percentages

| Split | Percentage |
|---|---:|
| Train | 80.08% |
| Validation | 9.95% |
| Test | 9.97% |
| **Total** | **100%** |

The split was inherited from the dataset's original train/validation/test organization. The validation and test counts changed slightly only because exact cross-split duplicates were removed.

---

## 11. Final Class Distribution

The final annotation distribution is:

| Class ID | Class | Annotations | Percentage |
|---:|---|---:|---:|
| 0 | `mouse_bite` | 3,684 | 17.02% |
| 1 | `spur` | 3,636 | 16.80% |
| 2 | `missing_hole` | 3,612 | 16.68% |
| 3 | `short` | 3,497 | 16.15% |
| 4 | `open_circuit` | 3,544 | 16.37% |
| 5 | `spurious_copper` | 3,676 | 16.98% |
| | **Total** | **21,649** | **100%** |

The class distribution is well balanced. The smallest class represents 16.15% of annotations and the largest represents 17.02%.

No severe class imbalance was identified.

---

## 12. Data Integrity Audit

A final automated audit was performed on `dataset_final`.

The audit checked:

- Image count
- Label-file count
- Annotation count
- Image/label matching
- Empty label files
- YOLO annotation structure
- Numeric annotation values
- Class IDs
- Normalized bounding-box coordinates
- Corrupted/unreadable images

### Final audit result

| Check | Train | Val | Test |
|---|---:|---:|---:|
| Missing labels | 0 | 0 | 0 |
| Missing images | 0 | 0 | 0 |
| Empty label files | 0 | 0 | 0 |
| Invalid annotations | 0 | 0 | 0 |
| Invalid class IDs | 0 | 0 | 0 |
| Corrupted images | 0 | 0 | 0 |

**Final status: AUDIT PASSED - DATASET IS VALID**

---

## 13. Problems Identified and Actions Taken

### Problem 1 — Exact duplicates across splits

**Finding:**  
Nine groups of identical images were found across train/validation/test splits.

**Action:**  
Duplicate validation and test copies were removed while preserving the training copies.

**Result:**  
The final evaluation splits no longer contain the confirmed exact cross-split duplicates identified during the cleaning step.

---

### Problem 2 — Different image/label filename conventions

**Finding:**  
Some images use `_600` while their corresponding labels use `_256`.

**Action:**  
The dataset cleaning and audit scripts were updated to support both naming conventions.

**Result:**  
The final audit correctly matches all images with their corresponding labels.

---

### Problem 3 — Perceptual similarity is too broad for this dataset

**Finding:**  
Perceptual-hash methods produced many apparent matches because PCB images share substantial visual structure.

**Action:**  
Perceptual similarity was not used to delete data.

**Result:**  
Only confirmed exact duplicates were removed.

---

## 14. Cleaning Scripts

The following scripts were created during the preparation process:

```text
scripts/
├── analyze_duplicates.py
├── check_duplicates.py
├── check_visual_duplicates.py
├── clean_exact_duplicates.py
├── dataset_audit.py
└── dataset_summary.py
```

### Main responsibilities

- `dataset_summary.py` — summarizes dataset counts and distributions.
- `dataset_audit.py` — validates images, labels, annotations, class IDs, coordinates, and image integrity.
- `check_duplicates.py` — identifies exact duplicate images.
- `analyze_duplicates.py` — analyzes duplicate groups and their labels.
- `clean_exact_duplicates.py` — creates the cleaned dataset while preserving the original.
- `check_visual_duplicates.py` — exploratory perceptual similarity analysis; its results were not used as the final deletion criterion.

---

## 15. Reproducibility

The preparation workflow is reproducible using the scripts stored in the `scripts/` directory.

The raw source dataset is preserved separately from the cleaned dataset.

Recommended workflow:

```text
Raw Dataset
    ↓
Dataset Inspection
    ↓
Annotation Verification
    ↓
Class Distribution Analysis
    ↓
Exact Duplicate Detection
    ↓
Duplicate Cleaning
    ↓
Final Integrity Audit
    ↓
Training-Ready Dataset
```

---

## 16. Training Handoff

The model-training stage should use:

```text
dataset_final/
```

and its:

```text
data.yaml
```

The training pipeline should use:

```text
train → model training
val   → validation during training/model selection
test  → final held-out evaluation
```

The training team should preserve the class mapping:

```text
0 = mouse_bite
1 = spur
2 = missing_hole
3 = short
4 = open_circuit
5 = spurious_copper
```

No class IDs should be reassigned without updating the corresponding annotations and `data.yaml`.

---

## 17. Final Handoff Checklist

- [x] Dataset downloaded
- [x] Dataset structure inspected
- [x] `data.yaml` verified
- [x] Class IDs verified
- [x] YOLO annotation format verified
- [x] Image counts verified
- [x] Annotation counts verified
- [x] Class distribution analyzed
- [x] Exact cross-split duplicates checked
- [x] Confirmed duplicate copies removed
- [x] Original dataset preserved
- [x] Image/label filename conventions handled
- [x] Empty labels checked
- [x] Invalid annotations checked
- [x] Invalid class IDs checked
- [x] Bounding-box coordinates checked
- [x] Corrupted images checked
- [x] Final audit passed
- [x] Training-ready dataset created
- [x] Preparation process documented

---

## 18. Final Conclusion

The dataset preparation stage is complete.

The final training-ready dataset contains:

**10,657 images, 10,657 label files, 21,649 valid annotations, and 6 balanced defect classes.**

All final integrity checks passed with zero missing labels, zero missing images, zero empty label files, zero invalid annotations, zero invalid class IDs, and zero corrupted images.

The cleaned dataset is ready to be handed over to the model-training stage.

**Final status: READY FOR MODEL TRAINING**
