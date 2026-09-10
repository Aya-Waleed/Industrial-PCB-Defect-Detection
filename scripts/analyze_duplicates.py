from pathlib import Path
from collections import defaultdict
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "pcb-defect-dataset"

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_images(split):
    image_dir = DATASET_ROOT / split / "images"

    return [
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_hash(path):
    hasher = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def find_label(split, image_name):

    labels_dir = DATASET_ROOT / split / "labels"

    stem = Path(image_name).stem

    # First: exact same stem
    candidates = [
        labels_dir / f"{stem}.txt"
    ]

    # Then try changing 600 -> 256
    if stem.endswith("_600"):
        candidates.append(
            labels_dir / f"{stem[:-4]}_256.txt"
        )

    # Then try removing common image prefixes
    possible_stems = [
        stem,
        stem.replace("rotation_90_", ""),
        stem.replace("rotation_270_", ""),
        stem.replace("l_", ""),
    ]

    expanded = []

    for s in possible_stems:

        expanded.append(s)

        if s.endswith("_600"):
            expanded.append(
                s[:-4] + "_256"
            )

    # Remove duplicates while preserving order
    seen = set()

    for s in expanded:

        if s in seen:
            continue

        seen.add(s)

        candidate = labels_dir / f"{s}.txt"

        if candidate.exists():
            return candidate

    return None


def read_label(label_path):

    if label_path is None:
        return "LABEL NOT FOUND"

    with open(label_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():

    hashes = defaultdict(list)

    # --------------------------------------------------------
    # Hash all images
    # --------------------------------------------------------

    for split in SPLITS:

        images = get_images(split)

        print(f"Reading {split}: {len(images)} images")

        for image in images:

            image_hash = get_hash(image)

            hashes[image_hash].append(
                (split, image.name)
            )

    # --------------------------------------------------------
    # Find cross-split duplicates
    # --------------------------------------------------------

    duplicate_groups = []

    for image_hash, items in hashes.items():

        splits_found = {item[0] for item in items}

        if len(splits_found) > 1:

            duplicate_groups.append(
                (image_hash, items)
            )

    print("\n" + "=" * 70)
    print("CROSS-SPLIT DUPLICATE ANALYSIS")
    print("=" * 70)

    print(
        f"\nCross-split duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    # --------------------------------------------------------
    # Analyze groups
    # --------------------------------------------------------

    for number, (image_hash, items) in enumerate(
        duplicate_groups,
        start=1
    ):

        print("\n" + "-" * 70)
        print(f"DUPLICATE GROUP {number}")
        print(f"Hash: {image_hash}")

        labels = []

        for split, image_name in items:

            label_path = find_label(
                split,
                image_name
            )

            label = read_label(label_path)

            print(f"\n[{split}]")
            print(f"Image: {image_name}")

            if label_path:
                print(
                    f"Label file: {label_path.name}"
                )
            else:
                print("Label file: NOT FOUND")

            print("Label:")
            print(label)

            labels.append(label)

        # ----------------------------------------------------
        # Compare labels
        # ----------------------------------------------------

        if "LABEL NOT FOUND" in labels:

            print(
                "\nRESULT: LABEL MATCH STILL NEEDS INSPECTION"
            )

        elif len(set(labels)) == 1:

            print(
                "\nRESULT: SAME IMAGE + IDENTICAL LABEL"
            )

        else:

            print(
                "\nRESULT: SAME IMAGE + DIFFERENT LABEL"
            )

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()