#!/usr/bin/env python3
"""
Exercise 4 — Dynamic shape audit

Walks model directories, loads each .onnx file via ORT CPUExecutionProvider,
and writes a report of every input that has at least one dynamic (string) dim.

Run:  python e4_dynamic_audit.py
Output: exercises/solutions/e4_dynamic_report.txt

Edit SEARCH_ROOTS below to match your machine.
Requires: pip install onnxruntime
"""
from __future__ import annotations

import sys
from pathlib import Path

SEARCH_ROOTS = [
    Path("/home/sudhdevu/R1models"),
]

REPORT_PATH = Path(__file__).with_name("e4_dynamic_report.txt")


def main() -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("pip install onnxruntime")
        sys.exit(1)

    lines: list[str] = ["# Dynamic-shape audit report", ""]
    total_models = 0
    dynamic_count = 0
    static_count = 0

    for root in SEARCH_ROOTS:
        if not root.exists():
            lines.append(f"SKIP missing root: {root}")
            continue
        for path in sorted(root.rglob("*.onnx")):
            if path.name.endswith(".onnx.data"):
                continue
            # Skip files inside the handbook itself (the template scripts)
            if "r1-ml-onnx-student-handbook" in path.parts:
                continue
            total_models += 1
            try:
                sess = ort.InferenceSession(
                    path.as_posix(), providers=["CPUExecutionProvider"]
                )
            except Exception as exc:
                lines.append(f"ERROR loading {path}: {exc}")
                continue

            for inp in sess.get_inputs():
                has_dynamic = any(isinstance(d, str) for d in inp.shape)
                if has_dynamic:
                    dynamic_count += 1
                    lines.append(
                        f"DYNAMIC  {path}  input={inp.name}  shape={inp.shape}  type={inp.type}"
                    )
                else:
                    static_count += 1
                    lines.append(
                        f"static   {path}  input={inp.name}  shape={inp.shape}"
                    )

    lines.insert(2, f"Models scanned : {total_models}")
    lines.insert(3, f"Dynamic inputs : {dynamic_count}")
    lines.insert(4, f"Static inputs  : {static_count}")
    lines.insert(5, "")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Report written to {REPORT_PATH}")
    print(f"  models={total_models}  dynamic_inputs={dynamic_count}  static_inputs={static_count}")
    print("\n[OK] Exercise 4 complete.")


if __name__ == "__main__":
    main()
