#!/usr/bin/env python3
"""
Minimal ONNX inspection example for the student handbook.
Edit MODEL_PATH before running. Requires: pip install onnx
"""
from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto


def main() -> None:
    # TODO: student — set this to a real file on your machine
    model_path = Path("/home/sudhdevu/R1models/ML-onnx-student-handbook/scripts/examples/REPLACE_ME.onnx")
    if not model_path.exists():
        print(f"Edit MODEL_PATH. Missing file: {model_path}")
        return

    model = onnx.load(model_path.as_posix(), load_external_data=False)

    opset = model.opset_import[0].version if model.opset_import else "unknown"
    print(f"File: {model_path}")
    print(f"Opset: {opset}")
    print(f"Nodes: {len(model.graph.node)}")
    print(f"Initializers: {len(model.graph.initializer)}")

    def describe_dims(dims):
        out = []
        for d in dims:
            if d.HasField("dim_value"):
                out.append(int(d.dim_value))
            elif d.HasField("dim_param"):
                out.append(d.dim_param)
            else:
                out.append("?")
        return out

    print("Inputs:")
    for i in model.graph.input:
        t = i.type.tensor_type
        shape = describe_dims(t.shape.dim)
        print(f"  - {i.name}: shape={shape} elem_type={TensorProto.DataType.Name(t.elem_type)}")

    print("Outputs:")
    for o in model.graph.output:
        t = o.type.tensor_type
        shape = describe_dims(t.shape.dim)
        print(f"  - {o.name}: shape={shape} elem_type={TensorProto.DataType.Name(t.elem_type)}")


if __name__ == "__main__":
    main()
