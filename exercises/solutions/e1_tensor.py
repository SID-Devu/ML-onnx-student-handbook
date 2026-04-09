#!/usr/bin/env python3
"""
Exercise 1 — Python + NumPy warm-up

Creates a tensor, prints shape/dtype/strides, and demonstrates slicing.
Run:  python e1_tensor.py
"""
import numpy as np

x = np.random.randn(1, 3, 224, 224).astype(np.float32)

print("=== Exercise 1: Tensor basics ===")
print(f"Shape  : {x.shape}")          # (1, 3, 224, 224)
print(f"Dtype  : {x.dtype}")          # float32
print(f"Strides: {x.strides}")        # bytes per step along each axis

# Strides explanation:
#   strides = (C*H*W*4, H*W*4, W*4, 4)  for float32 (4 bytes each)
#   = (3*224*224*4, 224*224*4, 224*4, 4)
#   = (602112, 200704, 896, 4)

sliced = x[0, :, 0:1, 0:1]
print(f"\nSlice x[0, :, 0:1, 0:1]")
print(f"  shape : {sliced.shape}")     # (3, 1, 1) — batch dim dropped, kept 3 channels
print(f"  values: {sliced.flatten()}")

# ---- Self-check questions ----
# 1. Why is strides[-1] == 4?  Because float32 is 4 bytes wide and the last
#    axis is contiguous in memory (C-order / row-major).
# 2. What happens if you do x[0, :, 0, 0] instead of x[0, :, 0:1, 0:1]?
#    You get shape (3,) — integer indexing drops the dimension, slice keeps it.

print("\n[OK] Exercise 1 complete.")
