#!/usr/bin/env python3
"""
Exercise 3 — ORT run (CPU first)

Loads an ONNX model with CPUExecutionProvider, creates random input that
matches the first binding's shape (replacing dynamic dims with 1), runs
inference, and prints output shapes.

Run:  python e3_ort_cpu_run.py /path/to/model.onnx
Requires: pip install onnxruntime numpy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


# Mapping from ORT string type names to NumPy dtypes
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


def resolve_shape(shape: list) -> list[int]:
    """Replace any dynamic (string/None) dims with 1."""
    resolved = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            resolved.append(d)
        else:
            resolved.append(1)
    return resolved


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python e3_ort_cpu_run.py <model.onnx>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"File not found: {model_path}")
        sys.exit(1)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path.as_posix(),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    print(f"Model   : {model_path}")
    print(f"Providers used: {session.get_providers()}")

    # Build feed dict for ALL inputs
    feed = {}
    print("\nInputs:")
    for inp in session.get_inputs():
        shape_raw = inp.shape
        shape = resolve_shape(shape_raw)
        dtype = ORT_TYPE_MAP.get(inp.type, np.float32)
        print(f"  {inp.name}: declared={shape_raw} -> resolved={shape} dtype={dtype.__name__}")

        if np.issubdtype(dtype, np.floating):
            feed[inp.name] = np.random.randn(*shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            feed[inp.name] = np.zeros(shape, dtype=dtype)
        elif dtype == np.bool_:
            feed[inp.name] = np.ones(shape, dtype=np.bool_)
        else:
            feed[inp.name] = np.zeros(shape, dtype=dtype)

    # Run inference
    outputs = session.run(None, feed)

    print("\nOutputs:")
    for meta, arr in zip(session.get_outputs(), outputs):
        print(f"  {meta.name}: shape={arr.shape} dtype={arr.dtype}")

    print("\n[OK] Exercise 3 complete — CPU inference succeeded.")


if __name__ == "__main__":
    main()
