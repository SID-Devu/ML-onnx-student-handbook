#!/usr/bin/env python3
"""
Exercise 5 — Baseline latency distribution

Loads an ONNX model, runs 5 warmup + 50 timed iterations on CPU,
then prints mean / median / std / p95 / p99 and per-run latencies.

Run:  python e5_timings.py /path/to/model.onnx
Requires: pip install onnxruntime numpy
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

WARMUP = 5
RUNS = 50


def resolve_shape(shape: list) -> list[int]:
    return [d if isinstance(d, int) and d > 0 else 1 for d in shape]


ORT_TYPE_MAP = {
    "tensor(float)":   np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)":  np.float64,
    "tensor(int32)":   np.int32,
    "tensor(int64)":   np.int64,
    "tensor(int8)":    np.int8,
    "tensor(uint8)":   np.uint8,
    "tensor(bool)":    np.bool_,
}


def build_feed(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    feed = {}
    for inp in session.get_inputs():
        shape = resolve_shape(inp.shape)
        dtype = ORT_TYPE_MAP.get(inp.type, np.float32)
        if np.issubdtype(dtype, np.floating):
            feed[inp.name] = np.random.randn(*shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            feed[inp.name] = np.zeros(shape, dtype=dtype)
        else:
            feed[inp.name] = np.zeros(shape, dtype=dtype)
    return feed


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python e5_timings.py <model.onnx>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"Not found: {model_path}")
        sys.exit(1)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path.as_posix(), sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    feed = build_feed(session)
    print(f"Model  : {model_path.name}")
    print(f"Warmup : {WARMUP}   Timed runs: {RUNS}\n")

    # --- Warmup ---
    for i in range(WARMUP):
        session.run(None, feed)
    print(f"Warmup done ({WARMUP} runs discarded).\n")

    # --- Timed runs ---
    latencies_ms: list[float] = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        session.run(None, feed)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    arr = np.array(latencies_ms)

    print("Per-run latencies (ms):")
    for idx, v in enumerate(latencies_ms):
        print(f"  run {idx+1:3d}: {v:8.3f} ms")

    print(f"\n{'='*40}")
    print(f"  Mean   : {arr.mean():.3f} ms")
    print(f"  Median : {np.median(arr):.3f} ms")
    print(f"  Std    : {arr.std():.3f} ms")
    print(f"  Min    : {arr.min():.3f} ms")
    print(f"  Max    : {arr.max():.3f} ms")
    print(f"  p95    : {np.percentile(arr, 95):.3f} ms")
    print(f"  p99    : {np.percentile(arr, 99):.3f} ms")
    print(f"{'='*40}")

    # --- Jitter analysis hint ---
    spread = arr.max() - arr.min()
    cv = arr.std() / arr.mean() * 100 if arr.mean() > 0 else 0
    print(f"\n  Range (max-min): {spread:.3f} ms")
    print(f"  CV (std/mean)  : {cv:.1f}%")
    if cv > 10:
        print("  -> High jitter. Common causes: thermal throttling, background")
        print("     processes, swap, filesystem cache cold/hot, CPU frequency scaling.")
    else:
        print("  -> Relatively stable.")

    print("\n[OK] Exercise 5 complete.")


if __name__ == "__main__":
    main()
