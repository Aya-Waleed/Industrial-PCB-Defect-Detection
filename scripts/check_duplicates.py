from pathlib import Path
from PIL import Image
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "pcb-defect-dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(split):
    folder = DATASET_ROOT / split / "images"

    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def file_hash(path):
    hasher = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def main():

    splits = ["train", "val", "test"]

    hashes = {}

    for split in splits:

        hashes[split] = {}

        images = get_images(split)

        print(f"Checking {split}: {len(images)} images")

        for image in images:
            h = file_hash(image)
            hashes[split].setdefault(h, []).append(image.name)

    print("\n" + "=" * 60)
    print("DUPLICATE CHECK")
    print("=" * 60)

    found = False

    comparisons = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for split_a, split_b in comparisons:

        common_hashes = (
            set(hashes[split_a])
            & set(hashes[split_b])
        )

        print(
            f"\n{split_a.upper()} ↔ {split_b.upper()}: "
            f"{len(common_hashes)} exact duplicates"
        )

        if common_hashes:
            found = True

            for h in list(common_hashes)[:10]:

                print("\nHash:", h)

                print(
                    " ",
                    split_a,
                    "->",
                    hashes[split_a][h]
                )

                print(
                    " ",
                    split_b,
                    "->",
                    hashes[split_b][h]
                )

    print("\n" + "=" * 60)

    if found:
        print("WARNING: Exact duplicates were found.")
    else:
        print("No exact cross-split duplicates found.")

    print("=" * 60)


if __name__ == "__main__":
    main()