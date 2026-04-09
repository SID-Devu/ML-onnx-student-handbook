#!/usr/bin/env python3
"""
Exercise 7 — ROCm snapshot before and after a synthetic load

Captures `rocm-smi` output before/after a 60-second busy loop on GPU,
then compares clocks and temperatures.

Run:  python e7_rocm_snapshot.py
Requires: rocm-smi in PATH, onnxruntime with MIGraphXExecutionProvider or
          CPUExecutionProvider (falls back to CPU stress if no GPU EP).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

LOAD_SECONDS = 60


def capture_rocm_smi() -> str:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showtemp", "--showclocks", "--showmeminfo", "all"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout + result.stderr
    except FileNotFoundError:
        return "<rocm-smi not found — skip GPU snapshot>"
    except subprocess.TimeoutExpired:
        return "<rocm-smi timed out>"


def cpu_stress(seconds: float) -> None:
    """Fallback: stress CPU cores as a synthetic load."""
    import numpy as np
    end = time.monotonic() + seconds
    print(f"  Running CPU stress for {seconds}s (no GPU EP available)...")
    while time.monotonic() < end:
        a = np.random.randn(512, 512).astype(np.float32)
        _ = a @ a


def gpu_stress(seconds: float) -> None:
    """Run repeated ORT inferences as a synthetic GPU load."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        cpu_stress(seconds)
        return

    providers = ort.get_available_providers()
    if "MIGraphXExecutionProvider" not in providers:
        print("  MIGraphX EP not available, falling back to CPU stress.")
        cpu_stress(seconds)
        return

    # Build a tiny throwaway ONNX model (MatMul) to hammer the GPU
    import onnx
    from onnx import helper, TensorProto as TP

    A = helper.make_tensor_value_info("A", TP.FLOAT, [256, 256])
    B = helper.make_tensor_value_info("B", TP.FLOAT, [256, 256])
    C = helper.make_tensor_value_info("C", TP.FLOAT, [256, 256])
    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "stress", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    tmp = Path("/tmp/_e7_stress.onnx")
    onnx.save(model, tmp.as_posix())

    sess = ort.InferenceSession(
        tmp.as_posix(), providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"],
    )
    a = np.random.randn(256, 256).astype(np.float32)
    b = np.random.randn(256, 256).astype(np.float32)

    print(f"  Running GPU stress for {seconds}s ...")
    end = time.monotonic() + seconds
    iters = 0
    while time.monotonic() < end:
        sess.run(None, {"A": a, "B": b})
        iters += 1
    print(f"  Completed {iters} GPU iterations.")
    tmp.unlink(missing_ok=True)


def main() -> None:
    print("=== Exercise 7: ROCm snapshot ===\n")

    print("--- BEFORE load ---")
    before = capture_rocm_smi()
    print(before)

    gpu_stress(LOAD_SECONDS)

    print("\n--- AFTER load ---")
    after = capture_rocm_smi()
    print(after)

    # --- Analysis hints ---
    print("\n--- Analysis checklist ---")
    print("1. Did GPU temperature increase significantly (>10 C)?")
    print("   If yes, prolonged runs risk thermal throttling.")
    print("2. Did sclk (shader clock) stay at the max locked value?")
    print("   If it dropped, power or thermal limits may be active.")
    print("3. Did mclk (memory clock) stay stable?")
    print("4. Did VRAM/GTT usage change between snapshots?")
    print("   Our stress model is tiny; real models show much larger changes.")
    print()

    report = Path(__file__).with_name("e7_rocm_report.txt")
    report.write_text(
        "=== BEFORE ===\n" + before + "\n\n=== AFTER ===\n" + after + "\n"
    )
    print(f"Raw output saved to {report}")
    print("\n[OK] Exercise 7 complete.")


if __name__ == "__main__":
    main()
