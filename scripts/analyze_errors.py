"""Create an auditable FP/FN analysis from a YOLO test set and COCO predictions.

The analysis uses an explicit operating point (confidence >= 0.25, IoU >=
0.50), which is intentionally separate from AP/mAP's threshold sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
NAMES = ["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def label_path(image: Path, labels_dir: Path) -> Path:
    direct = labels_dir / f"{image.stem}.txt"
    if direct.is_file():
        return direct
    alternate = labels_dir / f"{image.stem.removesuffix('_600')}_256.txt"
    return alternate


def load_ground_truth(images_dir: Path) -> dict[str, list[dict]]:
    ground_truth: dict[str, list[dict]] = {}
    labels_dir = images_dir.parent / "labels"
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        boxes = []
        path = label_path(image_path, labels_dir)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = map(float, parts)
                x1, y1 = (xc - w / 2) * width, (yc - h / 2) * height
                x2, y2 = (xc + w / 2) * width, (yc + h / 2) * height
                boxes.append({"class_id": int(cls), "box": (x1, y1, x2, y2)})
        ground_truth[image_path.name] = boxes
    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse FP/FN at a fixed detection threshold.")
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports" / "model_evaluation" / "predictions.json")
    parser.add_argument("--images", type=Path, default=ROOT / "dataset_final" / "test" / "images")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "model_evaluation")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    gt_by_image = load_ground_truth(args.images)
    predictions_by_image: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        if prediction["score"] >= args.confidence:
            # Ultralytics JSON uses COCO-style IDs (1..6); the dataset uses 0..5.
            prediction["class_id"] = int(prediction["category_id"]) - 1
            predictions_by_image[prediction["file_name"]].append(prediction)

    counts = {name: Counter() for name in NAMES}
    examples: list[dict] = []
    for filename, truths in gt_by_image.items():
        used = set()
        preds = sorted(predictions_by_image.get(filename, []), key=lambda row: row["score"], reverse=True)
        for pred in preds:
            x, y, w, h = pred["bbox"]
            pred_box = (x, y, x + w, y + h)
            matches = [
                (iou(pred_box, truth["box"]), index)
                for index, truth in enumerate(truths)
                if index not in used and truth["class_id"] == pred["class_id"]
            ]
            best_iou, best_index = max(matches, default=(0.0, -1))
            pred_name = NAMES[pred["class_id"]]
            if best_iou >= args.iou:
                used.add(best_index)
                counts[pred_name]["tp"] += 1
            else:
                counts[pred_name]["fp"] += 1
                examples.append({"error": "false_positive", "image": filename, "class": pred_name, "confidence": f"{pred['score']:.4f}", "iou": f"{best_iou:.4f}"})
        for index, truth in enumerate(truths):
            if index not in used:
                name = NAMES[truth["class_id"]]
                counts[name]["fn"] += 1
                examples.append({"error": "false_negative", "image": filename, "class": name, "confidence": "", "iou": ""})

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fp_fn_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "true_positive", "false_positive", "false_negative"])
        writer.writeheader()
        for name in NAMES:
            writer.writerow({"class": name, "true_positive": counts[name]["tp"], "false_positive": counts[name]["fp"], "false_negative": counts[name]["fn"]})
    with (args.output / "error_examples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["error", "image", "class", "confidence", "iou"])
        writer.writeheader()
        writer.writerows(examples[:200])

    lines = [
        "# False-positive / False-negative Analysis",
        "",
        f"Operating point: confidence >= {args.confidence:.2f}, same-class IoU >= {args.iou:.2f}.",
        "These counts are threshold-dependent and are not interchangeable with AP/mAP.",
        "",
        "| Class | True positives | False positives | False negatives |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(f"| {name} | {counts[name]['tp']} | {counts[name]['fp']} | {counts[name]['fn']} |" for name in NAMES)
    lines.extend(["", "See `error_examples.csv` for up to 200 reviewable examples."])
    (args.output / "fp_fn_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
