"""Reproducible training experiments for improving PCB detector localization.

Run from the repository root after placing the cleaned dataset locally:

    python scripts/train_model.py --data dataset_final/data.yaml --imgsz 640 --epochs 80

The baseline checkpoint was trained at 320 pixels. This script uses 640 pixels
by default because small PCB defects need more spatial detail. It never
overwrites ``model/best.pt``: promote a new checkpoint only after evaluating it
on the held-out test set with ``scripts/evaluate_model.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT)

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PCB defect detector.")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset_final" / "data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Pretrained base weights or a previous checkpoint.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=-1, help="-1 asks Ultralytics to auto-size the batch.")
    parser.add_argument("--device", default=0, help="GPU index; use cpu only for a slow smoke run.")
    parser.add_argument("--name", default="yolo11n_640_experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        pretrained=True,
        patience=15,
        seed=0,
        deterministic=True,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
