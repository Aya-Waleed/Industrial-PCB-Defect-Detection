"""Small helper utilities for the PCB Defect Detection app."""

from pathlib import Path

CLASS_NAMES_PATH = Path(__file__).resolve().parent.parent / "model" / "class_names.txt"


def load_class_names() -> dict:
    """Load {id: name} mapping from model/class_names.txt."""
    mapping = {}
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                idx, name = line.split(":", 1)
                mapping[int(idx.strip())] = name.strip()
    return mapping


def is_supported_image(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
