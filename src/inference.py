"""
Inference utilities for the PCB Defect Detection app.
Wraps the trained YOLO11n model (model/best.pt) for use in the Streamlit UI.
"""

import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "best.pt"
TEST_METRICS_PATH = Path(__file__).resolve().parent.parent / "model" / "test_metrics.txt"

# Some restricted Windows environments do not allow Ultralytics to create its
# default configuration folder under AppData. Keep its non-model settings next
# to this project instead. We assign (rather than ``setdefault``) because a
# stale inherited value can still point at an inaccessible AppData location.
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent.parent)

# Fixed color per class (RGB) so boxes are visually consistent across runs
CLASS_COLORS = {
    "mouse_bite": (231, 76, 60),
    "spur": (241, 196, 15),
    "missing_hole": (52, 152, 219),
    "short": (155, 89, 182),
    "open_circuit": (230, 126, 34),
    "spurious_copper": (26, 188, 156),
}


def load_model(model_path: Path = MODEL_PATH):
    """Load the YOLO model once (call this behind st.cache_resource in the app)."""
    from ultralytics import YOLO

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Make sure 'best.pt' is placed inside the 'model/' folder."
        )
    return YOLO(str(model_path))


def load_model_stats(path: Path = TEST_METRICS_PATH) -> dict:
    """Parse auditable held-out metrics for the Streamlit model card.

    The app deliberately shows test metrics, not the more optimistic validation
    figures embedded in the checkpoint. Pass an alternate file for diagnostics.
    """
    stats = {}
    if not Path(path).exists():
        return stats
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                stats[key.strip()] = val.strip()
    return stats


def run_inference(model, image: Image.Image, conf: float = 0.25, iou: float = 0.45, imgsz: int = 320):
    """
    Run detection on a PIL image.

    Returns:
        detections: list of dicts with keys
            {class_name, confidence, box (x1, y1, x2, y2)}
        annotated_image: PIL.Image with bounding boxes drawn
        elapsed_ms: inference time in milliseconds
    """
    image = image.convert("RGB")
    start = time.perf_counter()
    results = model.predict(source=np.array(image), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
            }
        )

    annotated_image = draw_boxes(image, detections)
    return detections, annotated_image, elapsed_ms


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes + labels on a copy of the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["class_name"], (0, 255, 0))
        label = f"{det['class_name']} {det['confidence']:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, max(0, y1 - text_h - 4), x1 + text_w + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - text_h - 4)), label, fill=(255, 255, 255), font=font)

    return annotated
