# Module 02 — NumPy: tensors, shapes, dtypes (full depth)

In inference scripts, **tensors are NumPy arrays**. Shape mistakes are the #1 runtime error. Every `build_*_inputs()` function in your benchmark suite creates NumPy arrays.

---

## 1. Creating arrays

```python
import numpy as np

# Common constructors
x = np.zeros((1, 3, 224, 224), dtype=np.float32)     # all zeros
x = np.ones((1, 3, 224, 224), dtype=np.float32)      # all ones
x = np.random.randn(1, 3, 640, 640).astype(np.float32)  # random normal

# From a list
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Empty (uninitialized — fast, but values are garbage)
x = np.empty((1, 3, 224, 224), dtype=np.float32)
```

---

## 2. Dtypes — models expect specific data types

| Dtype | Bytes | When used |
|-------|-------|-----------|
| `np.float32` | 4 | Default for most model inputs |
| `np.float16` | 2 | FP16 models (when `ort_migraphx_fp16: True`) |
| `np.int64` | 8 | Token IDs (LLMs), indices, attention masks |
| `np.int32` | 4 | Some integer inputs |
| `np.int8` | 1 | INT8 quantized models (though inputs often stay float32) |
| `np.uint8` | 1 | Raw image pixels before normalization |
| `np.bool_` | 1 | Attention masks (True/False) |

**Common mistake:** Python defaults to `float64`. Always cast explicitly:

```python
# BAD — float64, will cause ORT type mismatch
x = np.random.randn(1, 3, 224, 224)

# GOOD — float32
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
```

---

## 3. Reading shapes — the key insight

### Image tensor: `(1, 3, 640, 640)` in NCHW layout

```
Shape: (1,   3,   640,   640)
        |    |     |      |
        |    |     |      └── W = Width (pixels)
        |    |     └───────── H = Height (pixels)
        |    └─────────────── C = Channels (3 = RGB)
        └──────────────────── N = Batch size (1 = single image)
```

### Audio tensor: `(1, 48000)`

```
Shape: (1,     48000)
        |        |
        |        └── Number of audio samples (3 seconds × 16000 Hz sample rate)
        └─────────── Batch size
```

### Transformer hidden states: `(1, 128, 2048)`

```
Shape: (1,   128,   2048)
        |     |       |
        |     |       └── Hidden dimension (model's internal width)
        |     └────────── Sequence length (128 tokens)
        └──────────────── Batch size
```

### Your specific models' input shapes

| Model | Input shape | Meaning |
|-------|-------------|---------|
| YOLO v12m | `(1, 3, 640, 640)` | 1 image, RGB, 640×640 pixels |
| MobileNetV2 | `(1, 3, 224, 224)` | 1 image, RGB, 224×224 pixels |
| CLIP ViT-B/32 | `(1, 3, 224, 224)` | 1 image, RGB, 224×224 pixels |
| CLIP ViT-B/16 | `(1, 3, 224, 224)` | 1 image, RGB, 224×224 pixels |
| DeepSort OSNet | `(1, 3, 256, 128)` | 1 person crop, RGB, 256×128 pixels |
| EfficientNet | `(1, 3, 224, 224)` | 1 image, RGB, 224×224 pixels |
| Qwen3-1.7B | `(1, 128)` int64 | 1 sequence, 128 token IDs |
| Whisper | mel spectrogram tensor | Frequency × time representation of audio |
| XTTS | text + speaker conditioning | Text IDs + speaker embedding |

---

## 4. Layout: NCHW vs NHWC

**NCHW** — Channels first (PyTorch default, most ONNX exports):
```
(Batch, Channels, Height, Width)
(1,     3,        640,    640)
```

**NHWC** — Channels last (TensorFlow default):
```
(Batch, Height, Width, Channels)
(1,     640,    640,   3)
```

**RDNA 3.5 / MIGraphX** may prefer certain layouts for certain ops. MIGraphX handles layout transforms automatically, but your **input must match what the ONNX model declares**. Check with Netron or `session.get_inputs()`.

---

## 5. Reshape and transpose

```python
# Reshape — reinterpret data with different dimensions (element count must match)
x = np.zeros((1, 3, 224, 224), dtype=np.float32)
flat = x.reshape(1, -1)             # (1, 150528) — flatten spatial dims
print(flat.shape)

# Transpose — reorder dimensions
x_nchw = np.zeros((1, 3, 224, 224))
x_nhwc = x_nchw.transpose(0, 2, 3, 1)  # (1, 224, 224, 3)
print(x_nhwc.shape)
```

**When you need this:** converting between NCHW and NHWC, or flattening for classification heads.

---

## 6. Indexing and slicing

```python
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

batch0 = x[0]                    # shape (3, 224, 224) — drops batch dim
channel0 = x[0, 0]               # shape (224, 224) — single channel
patch = x[0, :, 32:64, 32:64]   # shape (3, 32, 32) — spatial crop
pixel = x[0, :, 100, 100]       # shape (3,) — RGB values at one pixel

# IMPORTANT: integer index drops the dimension, slice keeps it
x[0, :, 0, 0].shape      # (3,)      — integer indexing
x[0, :, 0:1, 0:1].shape  # (3, 1, 1) — slice indexing
```

---

## 7. Strides (what `x.strides` means)

```python
x = np.zeros((1, 3, 224, 224), dtype=np.float32)
print(x.strides)  # (602112, 200704, 896, 4)
```

Strides = bytes to jump to reach the next element along each axis:

```
Axis 0 (batch):   3 × 224 × 224 × 4 bytes = 602112
Axis 1 (channel): 224 × 224 × 4 bytes = 200704
Axis 2 (height):  224 × 4 bytes = 896
Axis 3 (width):   4 bytes (one float32)
```

**Why it matters:** Some native code paths require C-contiguous arrays. If strides are weird after transpose, use:

```python
x = np.ascontiguousarray(x)
```

---

## 8. Random inputs vs real inputs

| Scenario | Use random? | Why |
|----------|-------------|-----|
| Latency benchmarking | Yes | Performance doesn't depend on pixel values |
| INT8 calibration | **NO** | Random data gives meaningless scale/zero-point estimates |
| Accuracy validation | **NO** | Need real images to verify output correctness |
| Smoke test (does it run?) | Yes | Just checking shapes/types work |

For INT8 calibration (Module 06), you need 100-500 **representative** images from the actual deployment domain.

---

## 9. Common operations you'll see

```python
# Check properties
print(x.shape)      # (1, 3, 224, 224)
print(x.dtype)      # float32
print(x.size)       # 150528 (total elements)
print(x.nbytes)     # 602112 (total bytes)

# Type casting
x_fp16 = x.astype(np.float16)
ids = np.array([101, 2003, 102]).astype(np.int64)

# Concatenation
batch = np.concatenate([img1, img2], axis=0)  # stack along batch dim

# Comparison
np.allclose(out_fp32, out_fp16, atol=1e-3)  # check FP16 vs FP32 outputs match
```

---

## Module 02 checklist

- [ ] Given `(1, 3, 640, 640)`, explain each axis verbally
- [ ] Can create a correctly typed random tensor matching any model's input shape
- [ ] Know the difference between `x[0]` (drops dim) and `x[0:1]` (keeps dim)
- [ ] Can explain why `float64` default is wrong for ORT inputs
- [ ] Can convert NCHW → NHWC with transpose
- [ ] Know when random data is fine vs when you need real data

**Next:** `03-neural-nets-inference-vs-training.md`
