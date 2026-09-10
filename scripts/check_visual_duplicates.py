from pathlib import Path
from PIL import Image
import imagehash
import numpy as np

# =========================
# SETTINGS
# =========================

DATASET_DIR = Path("pcb-defect-dataset")

SPLITS = ["train", "val", "test"]

# Very strict pHash threshold
PHASH_THRESHOLD = 2

# Pixel similarity threshold
# Lower = more similar
MAE_THRESHOLD = 0.015

IMAGE_SIZE = (128, 128)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================
# IMAGE HASH + PIXELS
# =========================

def process_image(image_path):

    with Image.open(image_path) as img:

        img = img.convert("RGB")

        # Perceptual hash
        phash = imagehash.phash(img)

        # Resize for pixel comparison
        small = img.resize(IMAGE_SIZE)

        pixels = np.asarray(small).astype(np.float32) / 255.0

    return phash, pixels


# =========================
# LOAD SPLIT
# =========================

def load_split(split):

    image_dir = DATASET_DIR / split / "images"

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    results = []

    print(f"\nProcessing {split}: {len(images)} images")

    for i, image_path in enumerate(images, 1):

        try:

            phash, pixels = process_image(image_path)

            results.append(
                {
                    "name": image_path.name,
                    "hash": phash,
                    "pixels": pixels
                }
            )

        except Exception as e:

            print(f"ERROR: {image_path.name} -> {e}")

        if i % 500 == 0:
            print(f"  Processed {i}/{len(images)}")

    print(f"Finished {split}: {len(results)} valid images")

    return results


# =========================
# PIXEL DIFFERENCE
# =========================

def mean_absolute_error(img1, img2):

    return np.mean(np.abs(img1 - img2))


# =========================
# COMPARE SPLITS
# =========================

def compare_splits(split_a, data_a, split_b, data_b):

    print("\n" + "=" * 70)
    print(f"{split_a.upper()} <-> {split_b.upper()}")
    print("=" * 70)

    candidates = 0
    strong_matches = []

    for item_a in data_a:

        for item_b in data_b:

            # First filter: perceptual hash
            hash_distance = item_a["hash"] - item_b["hash"]

            if hash_distance <= PHASH_THRESHOLD:

                candidates += 1

                # Second filter: actual pixel difference
                mae = mean_absolute_error(
                    item_a["pixels"],
                    item_b["pixels"]
                )

                if mae <= MAE_THRESHOLD:

                    strong_matches.append(
                        (
                            hash_distance,
                            mae,
                            item_a["name"],
                            item_b["name"]
                        )
                    )

    strong_matches.sort(
        key=lambda x: (x[1], x[0])
    )

    print(f"pHash candidates: {candidates}")
    print(
        f"Strong visual matches: "
        f"{len(strong_matches)}"
    )

    if strong_matches:

        print("\nStrongest matches:\n")

        for hash_distance, mae, name_a, name_b in strong_matches[:50]:

            print(
                f"pHash={hash_distance} | "
                f"MAE={mae:.5f} | "
                f"{name_a} <--> {name_b}"
            )

        if len(strong_matches) > 50:

            print(
                f"\n... and "
                f"{len(strong_matches) - 50} more."
            )

    else:

        print(
            "No strong visual duplicates detected."
        )

    return strong_matches


# =========================
# MAIN
# =========================

def main():

    print("=" * 70)
    print("PCB DATASET - STRICT VISUAL DUPLICATE CHECK")
    print("=" * 70)

    all_data = {}

    for split in SPLITS:

        all_data[split] = load_split(split)

    train_val = compare_splits(
        "train",
        all_data["train"],
        "val",
        all_data["val"]
    )

    train_test = compare_splits(
        "train",
        all_data["train"],
        "test",
        all_data["test"]
    )

    val_test = compare_splits(
        "val",
        all_data["val"],
        "test",
        all_data["test"]
    )

    print("\n\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"TRAIN <-> VAL:   {len(train_val)} strong matches"
    )

    print(
        f"TRAIN <-> TEST:  {len(train_test)} strong matches"
    )

    print(
        f"VAL <-> TEST:    {len(val_test)} strong matches"
    )

    total = (
        len(train_val)
        + len(train_test)
        + len(val_test)
    )

    print("-" * 70)
    print(
        f"TOTAL STRONG MATCHES: {total}"
    )

    print("\nSettings:")
    print(f"pHash threshold: {PHASH_THRESHOLD}")
    print(f"MAE threshold:   {MAE_THRESHOLD}")

    print("\nIMPORTANT:")
    print(
        "These are candidate visual duplicates only."
    )
    print(
        "Do NOT delete files based on this report."
    )


if __name__ == "__main__":
    main()