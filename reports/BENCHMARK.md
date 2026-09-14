# Inference Benchmark

**Date:** 2026-09-14

**Model:** `model/best.pt` (YOLO11n)

**Device:** CPU

**Input size:** 320 × 320
**Method:** 5 warm-up runs, then 30 synthetic-image predictions including
forward pass and NMS.

| Metric | Result |
|---|---:|
| Mean latency | 39.50 ms/image |
| Median latency | 38.37 ms/image |
| p95 latency | 48.38 ms/image |
| Mean throughput | 25.3 FPS |

This is a reproducibility reference only. Re-run
`python scripts/benchmark_inference.py` on the final demo/deployment machine
and replace these values if its hardware differs.
