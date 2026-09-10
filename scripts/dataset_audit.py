from pathlib import Path
from collections import Counter
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Project structure:
#
# project/
# ├── pcb-defect-dataset/       <- original/raw dataset
# ├── dataset_final/            <- cleaned dataset
# │   ├── train/
# │   │   ├── images/
# │   │   └── labels/
# │   ├── val/
# │   │   ├── images/
# │   │   └── labels/
# │   ├── test/
# │   │   ├── images/
# │   │   └── labels/
# │   └── data.yaml
# └── scripts/
#     └── dataset_audit.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# IMPORTANT:
# Audit the cleaned dataset, NOT the original dataset.
DATASET_ROOT = PROJECT_ROOT / "dataset_final"


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}


# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# LABEL MATCHING
# ============================================================

def get_expected_label_stem(image_file, labels_dir):
    """
    Find the label stem corresponding to an image.

    This dataset uses two filename conventions:

    Convention 1:
        image: light_01_missing_hole_01_1_600.jpg
        label: light_01_missing_hole_01_1_256.txt

    Convention 2:
        image: l_light_01_missing_hole_01_1_600.jpg
        label: l_light_01_missing_hole_01_1_600.txt

    The function checks the _256 convention first,
    then checks the exact image stem.
    """

    stem = image_file.stem

    # Convention 1:
    # xxx_600.jpg -> xxx_256.txt
    if stem.endswith("_600"):
        candidate = stem[:-4] + "_256"

        if (labels_dir / f"{candidate}.txt").exists():
            return candidate

    # Convention 2:
    # xxx_600.jpg -> xxx_600.txt
    if (labels_dir / f"{stem}.txt").exists():
        return stem

    # No matching label found
    return None


def get_expected_image_stem(label_file, images_dir):
    """
    Find the image stem corresponding to a label.

    Handles both:

    image: xxx_600.jpg
    label: xxx_256.txt

    and:

    image: xxx_600.jpg
    label: xxx_600.txt
    """

    stem = label_file.stem

    # Convention 1:
    # xxx_256.txt -> xxx_600.jpg
    if stem.endswith("_256"):
        candidate = stem[:-4] + "_600"

        # Check supported image extensions
        for extension in IMAGE_EXTENSIONS:
            if (images_dir / f"{candidate}{extension}").exists():
                return candidate

    # Convention 2:
    # xxx_600.txt -> xxx_600.jpg
    for extension in IMAGE_EXTENSIONS:
        if (images_dir / f"{stem}{extension}").exists():
            return stem

    # No matching image found
    return None


# ============================================================
# AUDIT FUNCTION
# ============================================================

def audit_split(split_name):

    split_path = DATASET_ROOT / split_name
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"

    # --------------------------------------------------------
    # Check split folders
    # --------------------------------------------------------

    if not images_dir.exists():
        print(f"\nERROR: Images folder not found:")
        print(images_dir)
        return {
            "images": 0,
            "labels": 0,
            "annotations": 0,
            "class_counts": Counter(),
            "missing_labels": 0,
            "missing_images": 0,
            "empty_labels": 0,
            "invalid_annotations": 0,
            "invalid_class_ids": 0,
            "corrupted_images": 0,
        }

    if not labels_dir.exists():
        print(f"\nERROR: Labels folder not found:")
        print(labels_dir)
        return {
            "images": 0,
            "labels": 0,
            "annotations": 0,
            "class_counts": Counter(),
            "missing_labels": 0,
            "missing_images": 0,
            "empty_labels": 0,
            "invalid_annotations": 0,
            "invalid_class_ids": 0,
            "corrupted_images": 0,
        }

    # --------------------------------------------------------
    # Get images and labels
    # --------------------------------------------------------

    images = [
        p for p in images_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = [
        p for p in labels_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    ]

    # --------------------------------------------------------
    # IMAGE ↔ LABEL MATCHING
    #
    # IMPORTANT:
    # Do NOT simply compare image stems with label stems.
    #
    # The dataset contains:
    #
    # xxx_600.jpg ↔ xxx_256.txt
    #
    # as well as:
    #
    # xxx_600.jpg ↔ xxx_600.txt
    # --------------------------------------------------------

    actual_label_stems = {
        p.stem
        for p in labels
    }

    actual_image_stems = {
        p.stem
        for p in images
    }

    # Find labels expected by images
    missing_labels = []

    for image_file in images:

        expected_label = get_expected_label_stem(
            image_file,
            labels_dir
        )

        if expected_label is None:
            missing_labels.append(image_file.stem)

    # Find images expected by labels
    missing_images = []

    for label_file in labels:

        expected_image = get_expected_image_stem(
            label_file,
            images_dir
        )

        if expected_image is None:
            missing_images.append(label_file.stem)

    # --------------------------------------------------------
    # Class statistics
    # --------------------------------------------------------

    class_counts = Counter()

    total_annotations = 0

    empty_labels = []
    invalid_annotations = []
    invalid_class_ids = []

    # --------------------------------------------------------
    # Check every label
    # --------------------------------------------------------

    for label_file in labels:

        try:
            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                lines = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

        except Exception as e:

            invalid_annotations.append(
                f"{label_file.name}: "
                f"could not read file ({e})"
            )

            continue

        # ----------------------------------------------------
        # Empty label
        # ----------------------------------------------------

        if not lines:

            empty_labels.append(
                label_file.name
            )

            continue

        # ----------------------------------------------------
        # Check every annotation
        # ----------------------------------------------------

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            parts = line.split()

            # YOLO annotation must contain exactly:
            #
            # class_id
            # x_center
            # y_center
            # width
            # height

            if len(parts) != 5:

                invalid_annotations.append(
                    f"{label_file.name} "
                    f"line {line_number}: "
                    f"expected 5 values, "
                    f"got {len(parts)}"
                )

                continue

            # ------------------------------------------------
            # Convert values
            # ------------------------------------------------

            try:

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                invalid_annotations.append(
                    f"{label_file.name} "
                    f"line {line_number}: "
                    f"non-numeric value"
                )

                continue

            # ------------------------------------------------
            # Check class ID
            # ------------------------------------------------

            if class_id not in CLASS_NAMES:

                invalid_class_ids.append(
                    f"{label_file.name} "
                    f"line {line_number}: "
                    f"class ID {class_id}"
                )

                continue

            # ------------------------------------------------
            # Check YOLO normalized coordinates
            # ------------------------------------------------

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):

                invalid_annotations.append(
                    f"{label_file.name} "
                    f"line {line_number}: "
                    f"invalid coordinates "
                    f"{x_center} "
                    f"{y_center} "
                    f"{width} "
                    f"{height}"
                )

                continue

            # Valid annotation
            class_counts[class_id] += 1
            total_annotations += 1

    # --------------------------------------------------------
    # Check image files
    # --------------------------------------------------------

    corrupted_images = []

    for image_file in images:

        try:

            with Image.open(image_file) as img:
                img.verify()

        except Exception:

            corrupted_images.append(
                image_file.name
            )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"DATASET SPLIT: {split_name.upper()}")
    print("=" * 60)

    print(
        f"Images:              {len(images)}"
    )

    print(
        f"Label files:         {len(labels)}"
    )

    print(
        f"Annotations:         {total_annotations}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"  {class_id}: "
            f"{class_name:<20} "
            f"{class_counts[class_id]}"
        )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    print("\nIntegrity checks:")

    print(
        f"Missing labels:      "
        f"{len(missing_labels)}"
    )

    print(
        f"Missing images:      "
        f"{len(missing_images)}"
    )

    print(
        f"Empty label files:   "
        f"{len(empty_labels)}"
    )

    print(
        f"Invalid annotations: "
        f"{len(invalid_annotations)}"
    )

    print(
        f"Invalid class IDs:   "
        f"{len(invalid_class_ids)}"
    )

    print(
        f"Corrupted images:    "
        f"{len(corrupted_images)}"
    )

    # --------------------------------------------------------
    # Show problematic files
    # --------------------------------------------------------

    if missing_labels:

        print("\nMissing labels:")

        for item in sorted(
            missing_labels
        )[:50]:

            print(" ", item)

        if len(missing_labels) > 50:

            print(
                f"  ... and "
                f"{len(missing_labels) - 50} more"
            )

    if missing_images:

        print("\nMissing images:")

        for item in sorted(
            missing_images
        )[:50]:

            print(" ", item)

        if len(missing_images) > 50:

            print(
                f"  ... and "
                f"{len(missing_images) - 50} more"
            )

    if empty_labels:

        print("\nEmpty labels:")

        for item in empty_labels[:20]:

            print(" ", item)

    if invalid_annotations:

        print("\nInvalid annotations:")

        for item in invalid_annotations[:20]:

            print(" ", item)

    if invalid_class_ids:

        print("\nInvalid class IDs:")

        for item in invalid_class_ids[:20]:

            print(" ", item)

    if corrupted_images:

        print("\nCorrupted images:")

        for item in corrupted_images[:20]:

            print(" ", item)

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "images": len(images),
        "labels": len(labels),
        "annotations": total_annotations,
        "class_counts": class_counts,
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "empty_labels": len(empty_labels),
        "invalid_annotations": len(invalid_annotations),
        "invalid_class_ids": len(invalid_class_ids),
        "corrupted_images": len(corrupted_images),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nPCB DEFECT DATASET AUDIT")
    print("=" * 60)

    print(
        f"Dataset path: {DATASET_ROOT}"
    )

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        print(
            "\nERROR: Dataset folder "
            "was not found!"
        )

        print(
            "Expected:"
        )

        print(DATASET_ROOT)

        return

    # --------------------------------------------------------
    # Audit all splits
    # --------------------------------------------------------

    results = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        results[split] = audit_split(
            split
        )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    total_images = sum(
        r["images"]
        for r in results.values()
    )

    total_labels = sum(
        r["labels"]
        for r in results.values()
    )

    total_annotations = sum(
        r["annotations"]
        for r in results.values()
    )

    total_class_counts = Counter()

    for r in results.values():

        total_class_counts.update(
            r["class_counts"]
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print("\n\n" + "=" * 60)
    print("OVERALL DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Total images:        "
        f"{total_images}"
    )

    print(
        f"Total label files:   "
        f"{total_labels}"
    )

    print(
        f"Total annotations:   "
        f"{total_annotations}"
    )

    print(
        f"Number of classes:   "
        f"{len(CLASS_NAMES)}"
    )

    # --------------------------------------------------------
    # Overall class distribution
    # --------------------------------------------------------

    print("\nOverall class distribution:")

    for class_id, class_name in CLASS_NAMES.items():

        count = total_class_counts[
            class_id
        ]

        percentage = (
            count /
            total_annotations *
            100
            if total_annotations > 0
            else 0
        )

        print(
            f"  {class_name:<20} "
            f"{count:>6} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Final audit status
    # --------------------------------------------------------

    total_missing_labels = sum(
        r["missing_labels"]
        for r in results.values()
    )

    total_missing_images = sum(
        r["missing_images"]
        for r in results.values()
    )

    total_empty_labels = sum(
        r["empty_labels"]
        for r in results.values()
    )

    total_invalid_annotations = sum(
        r["invalid_annotations"]
        for r in results.values()
    )

    total_invalid_class_ids = sum(
        r["invalid_class_ids"]
        for r in results.values()
    )

    total_corrupted_images = sum(
        r["corrupted_images"]
        for r in results.values()
    )

    all_checks_passed = (
        total_missing_labels == 0
        and total_missing_images == 0
        and total_empty_labels == 0
        and total_invalid_annotations == 0
        and total_invalid_class_ids == 0
        and total_corrupted_images == 0
    )

    print("\n" + "=" * 60)

    if all_checks_passed:

        print(
            "AUDIT PASSED - "
            "DATASET IS VALID"
        )

    else:

        print(
            "AUDIT FAILED - "
            "ISSUES DETECTED"
        )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()