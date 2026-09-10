from pathlib import Path
import hashlib
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path("pcb-defect-dataset")
OUTPUT = Path("dataset_final")

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# MD5 HASH
# ============================================================

def md5(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# ============================================================
# GET IMAGES
# ============================================================

def get_images(split):

    image_dir = SOURCE / split / "images"

    return [
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ============================================================
# FIND CORRESPONDING LABEL
# ============================================================

def get_label_path(image_path, labels_dir):

    stem = image_path.stem

    # --------------------------------------------------------
    # Convention 1:
    #
    # image: light_..._600.jpg
    # label: light_..._256.txt
    # --------------------------------------------------------

    if stem.endswith("_600"):

        label_256 = labels_dir / (
            stem[:-4] + "_256.txt"
        )

        if label_256.exists():
            return label_256

    # --------------------------------------------------------
    # Convention 2:
    #
    # image: l_light_..._600.jpg
    # label: l_light_..._600.txt
    #
    # Also works as a general exact-stem fallback.
    # --------------------------------------------------------

    label_exact = labels_dir / f"{stem}.txt"

    if label_exact.exists():
        return label_exact

    # --------------------------------------------------------
    # No matching label found
    # --------------------------------------------------------

    return None


# ============================================================
# FIND EXACT CROSS-SPLIT DUPLICATES
# ============================================================

def find_duplicates():

    hashes = {
        "train": {},
        "val": {},
        "test": {}
    }

    print("=" * 70)
    print("SCANNING FOR EXACT CROSS-SPLIT DUPLICATES")
    print("=" * 70)

    for split in SPLITS:

        images = get_images(split)

        print(f"\n{split.upper()}: {len(images)} images")

        for i, image in enumerate(images, 1):

            file_hash = md5(image)

            hashes[split].setdefault(
                file_hash,
                []
            ).append(image)

            if i % 1000 == 0:
                print(
                    f"  Processed {i}/{len(images)}"
                )

    # Files to remove from evaluation splits
    remove = {
        "val": [],
        "test": []
    }

    duplicate_groups = []

    # --------------------------------------------------------
    # TRAIN <-> VAL
    # --------------------------------------------------------

    common_train_val = (
        set(hashes["train"])
        & set(hashes["val"])
    )

    for file_hash in common_train_val:

        train_files = hashes["train"][file_hash]
        val_files = hashes["val"][file_hash]

        duplicate_groups.append(
            (
                "train",
                "val",
                file_hash,
                train_files,
                val_files
            )
        )

        remove["val"].extend(val_files)

    # --------------------------------------------------------
    # TRAIN <-> TEST
    # --------------------------------------------------------

    common_train_test = (
        set(hashes["train"])
        & set(hashes["test"])
    )

    for file_hash in common_train_test:

        train_files = hashes["train"][file_hash]
        test_files = hashes["test"][file_hash]

        duplicate_groups.append(
            (
                "train",
                "test",
                file_hash,
                train_files,
                test_files
            )
        )

        remove["test"].extend(test_files)

    # --------------------------------------------------------
    # VAL <-> TEST
    # --------------------------------------------------------

    common_val_test = (
        set(hashes["val"])
        & set(hashes["test"])
    )

    for file_hash in common_val_test:

        val_files = hashes["val"][file_hash]
        test_files = hashes["test"][file_hash]

        duplicate_groups.append(
            (
                "val",
                "test",
                file_hash,
                val_files,
                test_files
            )
        )

        # Keep validation copy, remove test copy
        remove["test"].extend(test_files)

    remove["val"] = list(set(remove["val"]))
    remove["test"] = list(set(remove["test"]))

    return duplicate_groups, remove


# ============================================================
# CREATE CLEAN DATASET
# ============================================================

def copy_clean_dataset(remove):

    print("\n")
    print("=" * 70)
    print("CREATING CLEAN DATASET")
    print("=" * 70)

    # Delete old cleaned dataset only
    if OUTPUT.exists():

        print(
            f"\nRemoving previous {OUTPUT}..."
        )

        shutil.rmtree(OUTPUT)

    OUTPUT.mkdir(parents=True)

    for split in SPLITS:

        source_images = SOURCE / split / "images"
        source_labels = SOURCE / split / "labels"

        output_images = OUTPUT / split / "images"
        output_labels = OUTPUT / split / "labels"

        output_images.mkdir(parents=True)
        output_labels.mkdir(parents=True)

        removed_names = {
            p.name
            for p in remove.get(split, [])
        }

        images = get_images(split)

        copied_images = 0
        copied_labels = 0
        removed = 0
        missing_labels = []

        for image in images:

            # Skip confirmed duplicate
            if image.name in removed_names:

                removed += 1
                continue

            # Copy image
            shutil.copy2(
                image,
                output_images / image.name
            )

            copied_images += 1

            # Find corresponding label
            label = get_label_path(
                image,
                source_labels
            )

            if label is not None:

                shutil.copy2(
                    label,
                    output_labels / label.name
                )

                copied_labels += 1

            else:

                missing_labels.append(
                    image.name
                )

        print(
            f"{split.upper()}: "
            f"images copied={copied_images}, "
            f"labels copied={copied_labels}, "
            f"removed={removed}, "
            f"missing labels={len(missing_labels)}"
        )

        if missing_labels:

            print("\nMissing labels:")

            for name in missing_labels[:20]:

                print(" ", name)

    # Copy YAML
    yaml_file = SOURCE / "data.yaml"

    if yaml_file.exists():

        shutil.copy2(
            yaml_file,
            OUTPUT / "data.yaml"
        )


# ============================================================
# WRITE REPORT
# ============================================================

def write_report(
    duplicate_groups,
    remove
):

    report = OUTPUT / "duplicate_cleaning_report.txt"

    with open(
        report,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "PCB DATASET - EXACT DUPLICATE CLEANING REPORT\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Duplicate groups found: "
            f"{len(duplicate_groups)}\n"
        )

        f.write(
            f"Validation images removed: "
            f"{len(remove['val'])}\n"
        )

        f.write(
            f"Test images removed: "
            f"{len(remove['test'])}\n\n"
        )

        for i, group in enumerate(
            duplicate_groups,
            1
        ):

            split_a, split_b, file_hash, files_a, files_b = group

            f.write(
                f"GROUP {i}\n"
            )

            f.write(
                f"Hash: {file_hash}\n"
            )

            f.write(
                f"{split_a.upper()}:\n"
            )

            for file in files_a:

                f.write(
                    f"  {file.name}\n"
                )

            f.write(
                f"{split_b.upper()}:\n"
            )

            for file in files_b:

                f.write(
                    f"  {file.name}\n"
                )

            f.write("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    if not SOURCE.exists():

        print(
            "ERROR: Original dataset was not found:"
        )

        print(
            SOURCE.resolve()
        )

        return

    duplicate_groups, remove = (
        find_duplicates()
    )

    print("\n")
    print("=" * 70)
    print("DUPLICATE SUMMARY")
    print("=" * 70)

    print(
        f"Duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Validation copies to remove: "
        f"{len(remove['val'])}"
    )

    print(
        f"Test copies to remove: "
        f"{len(remove['test'])}"
    )

    print(
        "\nTRAIN IS ALWAYS PRESERVED."
    )

    copy_clean_dataset(remove)

    write_report(
        duplicate_groups,
        remove
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nClean dataset created at:"
    )

    print(
        OUTPUT.resolve()
    )

    print(
        "\nOriginal dataset was NOT modified."
    )


if __name__ == "__main__":
    main()