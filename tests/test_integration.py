"""
End-to-end integration smoke test.
Role: Application & Integration (Saif) — addresses the "integration testing"
gap flagged during project review: runs the full Data->Model->App chain
(model loading + inference + drawing) once before every submission/demo.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_integration.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import load_model, run_inference, load_model_stats, CLASS_COLORS
from src.utils import is_supported_image, load_class_names


def test_model_file_exists():
    model_path = Path(__file__).resolve().parent.parent / "model" / "best.pt"
    assert model_path.exists(), "model/best.pt is missing — required for the app to run"


def test_model_loads():
    model = load_model()
    assert model is not None
    assert len(model.names) == 6, f"Expected 6 classes, got {len(model.names)}"


def test_class_names_consistent():
    """class_names.txt must match the classes actually baked into best.pt."""
    mapping = load_class_names()
    model = load_model()
    assert mapping, "model/class_names.txt could not be parsed"
    for idx, name in mapping.items():
        assert model.names[idx] == name, (
            f"Mismatch at id {idx}: class_names.txt says '{name}', "
            f"model says '{model.names[idx]}'"
        )


def test_inference_runs_on_blank_image():
    """Smoke test: the full predict -> draw pipeline must not crash."""
    model = load_model()
    dummy = Image.fromarray(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))
    detections, annotated, elapsed_ms = run_inference(model, dummy)
    assert isinstance(detections, list)
    assert annotated.size == dummy.size
    assert elapsed_ms > 0


def test_file_type_validation():
    assert is_supported_image("board.jpg")
    assert is_supported_image("board.PNG")
    assert not is_supported_image("board.txt")
    assert not is_supported_image("board.pdf")


def test_class_colors_cover_all_classes():
    mapping = load_class_names()
    for name in mapping.values():
        assert name in CLASS_COLORS, f"No color defined for class '{name}' in src/inference.py"


def test_model_stats_readable():
    stats = load_model_stats()
    assert "Precision" in stats and "Recall" in stats, (
        "model/test_metrics.txt should be readable for the sidebar Model Card"
    )
    assert stats.get("Split") == "Test", "The app must display held-out test metrics"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
