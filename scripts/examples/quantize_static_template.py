#!/usr/bin/env python3
"""
TEMPLATE — static INT8 quantization with ONNX Runtime.

Read modules/06 first. This is educational:
- Random calibration is OK only for plumbing tests, NOT for accuracy.
- MIGraphX EP support for INT8 graphs varies; validate on your ORT build.

Requires: pip install onnx onnxruntime
(Optional): onnxruntime-extensions — only if your quant path needs it

Run:
  python quantize_static_template.py --in model_fp32.onnx --out model_int8.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="model_in", required=True, type=Path)
    parser.add_argument("--out", dest="model_out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=16, help="Calibration samples (use real data in production)")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as e:
        raise SystemExit(f"Missing deps: {e}") from e

    class _Reader(CalibrationDataReader):
        def __init__(self, model_path: Path, n: int):
            sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
            self._name = sess.get_inputs()[0].name
            self._shape = sess.get_inputs()[0].shape
            self._n = n
            self._i = 0

        def get_next(self):
            if self._i >= self._n:
                return None
            # Replace any dynamic dims with 1 for a smoke test, or fix shapes properly.
            shape = []
            for d in self._shape:
                if isinstance(d, str):
                    shape.append(1)
                elif d is None:
                    shape.append(1)
                else:
                    shape.append(int(d))
            x = np.random.randn(*shape).astype(np.float32)
            self._i += 1
            return {self._name: x}

    quantize_static(
        model_input=args.model_in.as_posix(),
        model_output=args.model_out.as_posix(),
        calibration_data_reader=_Reader(args.model_in, args.samples),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"Wrote: {args.model_out}")


if __name__ == "__main__":
    main()
