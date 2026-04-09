#!/usr/bin/env python3
"""
List ONNX models under configured directories and report ORT input shapes.
Dynamic dimensions show up as strings in session.get_inputs()[].shape.

Requires: pip install onnxruntime onnx
Edit SEARCH_ROOTS to match your machine.
"""
from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Install onnxruntime: pip install onnxruntime")
        sys.exit(1)

    # TODO: student — point at folders that contain .onnx files
    search_roots = [
        Path("/home/sudhdevu/R1models"),
    ]

    exts = {".onnx"}
    seen = 0
    for root in search_roots:
        if not root.exists():
            print(f"Skip missing root: {root}")
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            # skip common huge sidecars if misnamed (heuristic)
            if path.name.endswith(".onnx.data"):
                continue
            seen += 1
            try:
                sess = ort.InferenceSession(
                    path.as_posix(), providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                print(f"SKIP {path}: {e}")
                continue
            for inp in sess.get_inputs():
                shape = inp.shape
                has_dynamic = any(isinstance(dim, str) for dim in shape)
                tag = "DYNAMIC" if has_dynamic else "static"
                print(f"{tag}: {path} :: {inp.name} shape={shape} type={inp.type}")

    if seen == 0:
        print("No .onnx files found. Edit SEARCH_ROOTS.")


if __name__ == "__main__":
    main()
