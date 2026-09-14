"""
Inference speed benchmark for the trained PCB defect detection model.
Role: Application & Integration (Saif) — addresses the "inference speed"
gap flagged during project review.

Run with:
    python scripts/benchmark_inference.py [--n 30] [--imgsz 320]

Notes:
- Numbers depend heavily on the machine (CPU vs GPU). Re-run this on the
  deployment machine before quoting FPS numbers in the final report/demo.
- Uses a synthetic image (no real PCB photo needed) purely to time the
  forward pass + NMS; it does not measure detection accuracy.
"""

import argparse
import os
import statistics
import time
from pathlib import Path

# Keep Ultralytics settings inside the project on restricted Windows accounts.
# This must be set before importing ``ultralytics``.
ROOT = Path(__file__).resolve().parent.parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT)

import numpy as np
from ultralytics import YOLO

MODEL_PATH = ROOT / "model" / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of timed runs")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH} ...")
    model = YOLO(str(MODEL_PATH))

    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    print(f"Warming up ({args.warmup} runs) ...")
    for _ in range(args.warmup):
        model.predict(source=dummy, imgsz=args.imgsz, verbose=False)

    print(f"Timing {args.n} runs at imgsz={args.imgsz} ...")
    times_ms = []
    for _ in range(args.n):
        start = time.perf_counter()
        model.predict(source=dummy, imgsz=args.imgsz, verbose=False)
        times_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(times_ms)
    p50 = statistics.median(times_ms)
    p95 = sorted(times_ms)[int(0.95 * len(times_ms)) - 1]

    print("\n--- Results ---")
    print(f"Runs:        {args.n}")
    print(f"Image size:  {args.imgsz}x{args.imgsz}")
    print(f"Mean:        {mean_ms:.2f} ms  ({1000/mean_ms:.1f} FPS)")
    print(f"Median (p50):{p50:.2f} ms")
    print(f"p95:         {p95:.2f} ms")
    print("\nNote: this ran on the machine executing this script (see hardware")
    print("below). Re-run on the target deployment machine for real numbers.")

    import torch
    device = "CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"
    print(f"Device used: {device}")


if __name__ == "__main__":
    main()
