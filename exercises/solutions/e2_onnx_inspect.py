#!/usr/bin/env python3
"""
Exercise 2 — Inspect ONNX without ORT

Loads an ONNX file with the `onnx` library only (no runtime needed).
Prints opset, node count, inputs, outputs, and checks for external data.

Run:  python e2_onnx_inspect.py /path/to/model.onnx
Requires: pip install onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnx import TensorProto


def dim_str(dim) -> str:
    """Return the human-readable form of one ONNX dimension."""
    if dim.HasField("dim_value"):
        return str(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param          # symbolic name like "batch" or "sequence"
    return "?"


def inspect(model_path: Path) -> None:
    model = onnx.load(model_path.as_posix(), load_external_data=False)

    opset = model.opset_import[0].version if model.opset_import else "unknown"
    nodes = model.graph.node
    inits = model.graph.initializer

    print(f"File          : {model_path}")
    print(f"Opset         : {opset}")
    print(f"Graph nodes   : {len(nodes)}")
    print(f"Initializers  : {len(inits)}")

    # --- Op type histogram (top 10) ---
    from collections import Counter
    op_counts = Counter(n.op_type for n in nodes)
    print("\nTop op types:")
    for op, count in op_counts.most_common(10):
        print(f"  {op:30s} {count}")

    # --- Inputs ---
    print("\nGraph inputs:")
    init_names = {i.name for i in inits}
    for inp in model.graph.input:
        # Skip initializers that also appear in graph.input (weights)
        if inp.name in init_names:
            continue
        t = inp.type.tensor_type
        dims = [dim_str(d) for d in t.shape.dim]
        dtype = TensorProto.DataType.Name(t.elem_type)
        has_dynamic = any(d.HasField("dim_param") for d in t.shape.dim)
        tag = " [DYNAMIC]" if has_dynamic else ""
        print(f"  {inp.name}: shape={dims} dtype={dtype}{tag}")

    # --- Outputs ---
    print("\nGraph outputs:")
    for out in model.graph.output:
        t = out.type.tensor_type
        dims = [dim_str(d) for d in t.shape.dim]
        dtype = TensorProto.DataType.Name(t.elem_type)
        print(f"  {out.name}: shape={dims} dtype={dtype}")

    # --- External data heuristic ---
    data_sidecar = model_path.with_suffix(model_path.suffix + ".data")
    # also check for <stem>.onnx.data when suffix is already .onnx
    alt_sidecar = model_path.parent / (model_path.stem + ".onnx.data")
    has_external = data_sidecar.exists() or alt_sidecar.exists()
    print(f"\nExternal data sidecar found: {has_external}")
    if has_external:
        found = data_sidecar if data_sidecar.exists() else alt_sidecar
        size_mb = found.stat().st_size / (1024 * 1024)
        print(f"  -> {found} ({size_mb:.1f} MB)")

    print("\n[OK] Exercise 2 complete.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python e2_onnx_inspect.py <model.onnx>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    inspect(path)


if __name__ == "__main__":
    main()
