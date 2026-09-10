from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "pcb-defect-dataset"

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


for split in ["train", "val", "test"]:

    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"

    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = [
        p for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]

    class_counts = Counter()
    total_annotations = 0

    for label_file in labels:

        with open(label_file, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                if class_id in CLASS_NAMES:
                    class_counts[class_id] += 1
                    total_annotations += 1

    print()
    print("=" * 50)
    print(split.upper())
    print("=" * 50)

    print(f"Images:       {len(images)}")
    print(f"Labels:       {len(labels)}")
    print(f"Annotations:  {total_annotations}")

    print("\nClass distribution:")

    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"{class_id}: "
            f"{class_name:<20} "
            f"{class_counts[class_id]}"
        )

print()
print("=" * 50)
print("SUMMARY COMPLETE")
print("=" * 50)