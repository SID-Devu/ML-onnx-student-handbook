# Module 06 — Optimizations: graph-level, quantization, fusion, static shapes (full depth with code)

This is the longest module because it covers every optimization step with full code examples matching your setup.

---

## 1. ORT graph optimizations (`ORT_ENABLE_ALL`)

Your `benchmark_cooldown.py` already sets this:

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

What `ORT_ENABLE_ALL` does:
- Fuses ops: Conv+BN+ReLU → single kernel, MatMul+Add → fused kernel
- Eliminates redundant nodes
- Folds constants (computes subgraphs with known values at load time)
- Layout transforms where beneficial

**Status in your setup: DONE.**

---

## 2. ONNX Simplifier (`onnxsim`) — Step-by-step

Many of your models already went through `onnxsim` during export. Check each one to see if further simplification helps.

### Install

```bash
pip install onnxsim
```

### Check current node count

```python
import onnx
model = onnx.load("yolo/yolov12m.onnx")
print(f"Nodes before: {len(model.graph.node)}")
print(f"Initializers: {len(model.graph.initializer)}")
```

### Simplify (CLI)

```bash
python -m onnxsim yolo/yolov12m.onnx yolo/yolov12m_simplified.onnx
```

### Compare before/after

```python
import onnx
original = onnx.load("yolo/yolov12m.onnx")
simplified = onnx.load("yolo/yolov12m_simplified.onnx")
print(f"Before: {len(original.graph.node)} ops")
print(f"After:  {len(simplified.graph.node)} ops")
print(f"Removed: {len(original.graph.node) - len(simplified.graph.node)} ops")
```

### For large models (>2GB) like Qwen3, OpenVLA

```python
import onnx
from onnxsim import simplify

model = onnx.load("qwen3/qwen3-1.7b.onnx", load_external_data=True)
simplified, check = simplify(model)
onnx.save(
    simplified,
    "qwen3/qwen3-1.7b_simplified.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="qwen3-1.7b_simplified.onnx.data"
)
print("Simplified with external data")
```

**Typical results:** 5-15% fewer ops, which speeds up MIGraphX compilation and sometimes runtime.

**Do this for every model in your zoo.**

---

## 3. Shape inference (ONNX)

```python
import onnx
from onnx import shape_inference

model = onnx.load("model.onnx")
model = shape_inference.infer_shapes(model)
onnx.save(model, "model_with_shapes.onnx")
```

Helps backends generate better compiled kernels because all tensor shapes are known statically.

---

## 4. Static shape pinning (critical for MIGraphX)

**Why:** MIGraphX compiles kernels per-shape. Dynamic shapes force generic fallback kernels. **This is one of the biggest wins for MIGraphX specifically.**

### Check which models currently use dynamic shapes

```python
import onnxruntime as ort
import os

model_dirs = [
    "yolo", "crossformer", "mobilenetv2", "clip_vit_b32",
    "clip_vit_b16", "deepsort", "qwen3", "xtts", "openvla"
]

for d in model_dirs:
    if not os.path.exists(d):
        continue
    onnx_files = [f for f in os.listdir(d) if f.endswith('.onnx')]
    for f in onnx_files:
        path = os.path.join(d, f)
        try:
            sess = ort.InferenceSession(path)
            for inp in sess.get_inputs():
                has_dynamic = any(isinstance(dim, str) for dim in inp.shape)
                if has_dynamic:
                    print(f"DYNAMIC: {path} -> {inp.name}: {inp.shape}")
        except:
            pass
```

### Fix: add `dim_overrides` in your ModelSpec

```python
ModelSpec(
    name="clip_vit_b32",
    onnx_path="clip_vit_b32/model.onnx",
    build_inputs=build_clip_inputs,
    dim_overrides={"batch_size": 1, "sequence_length": 77},  # pin to static
)
```

MIGraphX then compiles a single optimized kernel for that exact shape instead of a general-purpose one.

**Expected gain: 10-30% for models with dynamic dims.**

---

## 5. Precision levels — FP32, FP16, INT8

| Precision | Bytes per value | Relative speed | Accuracy | Your use |
|-----------|----------------|---------------|----------|----------|
| **FP32** (full) | 4 bytes | 1x (baseline) | Highest | Default for all models |
| **FP16** (half) | 2 bytes | ~2x faster | Slight loss OK for most models | Already enabled via `ort_migraphx_fp16: True` |
| **INT8** (quarter) | 1 byte | ~4x faster* | Needs calibration; risky for attention | Vision models (YOLO, MobileNet, EfficientNet) |

*Theoretical throughput; actual speedup depends on hardware support and memory bandwidth.

**FP16:** You're already using this for several models via the `ort_migraphx_fp16: True` flag in your ModelSpec. MIGraphX compiles FP16 kernels that run ~2x faster on RDNA 3.5 CUs and use half the memory.

**Status in your setup: DONE (per-model flag).**

---

## 6. INT8 quantization — step by step (biggest remaining win for vision models)

**Applies to:** YOLO, MobileNetV2, CLIP ViT-B/32, CLIP ViT-B/16, DeepSort OSNet, EfficientNet

**What you need:**
- A calibration dataset (100-500 representative images)
- ORT's quantization tools

### Install

```bash
pip install onnxruntime
# onnxruntime-extensions is only needed if your quantization path uses custom ops
```

### Full quantization script

```python
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

class ImageCalibrationReader(CalibrationDataReader):
    """Feeds sample images to calibrate INT8 ranges."""
    def __init__(self, model_path, num_samples=100):
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name
        raw_shape = session.get_inputs()[0].shape  # e.g. [1,3,640,640] or ['batch',3,'h','w']
        self.input_shape = [1 if isinstance(d, str) or d is None else int(d) for d in raw_shape]
        self.num_samples = num_samples
        self.index = 0

    def get_next(self):
        if self.index >= self.num_samples:
            return None
        data = {self.input_name: np.random.randn(*self.input_shape).astype(np.float32)}
        self.index += 1
        return data

# Quantize
model_fp32 = "/home/sudhdevu/R1models/yolo/yolov12m.onnx"
model_int8 = "/home/sudhdevu/R1models/yolo/yolov12m_int8.onnx"

quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=ImageCalibrationReader(model_fp32),
    quant_format=QuantFormat.QDQ,   # QDQ (QuantizeLinear/DequantizeLinear nodes) or QOperator
    per_channel=True,               # Better accuracy than per-tensor
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)
print(f"INT8 model saved to {model_int8}")
```

### Verify it works

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "yolo/yolov12m_int8.onnx",
    providers=["MIGraphXExecutionProvider"]
)
inp = np.random.randn(1, 3, 640, 640).astype(np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: inp})
print("INT8 inference OK, output shape:", out[0].shape)
```

### Input shapes for each vision model

| Model | Input shape | Notes |
|-------|-------------|-------|
| YOLO | `[1, 3, 640, 640]` | Object detection |
| MobileNetV2 | `[1, 3, 224, 224]` | Classification |
| CLIP ViT-B/32 | `[1, 3, 224, 224]` | Vision encoder |
| CLIP ViT-B/16 | `[1, 3, 224, 224]` | Vision encoder |
| DeepSort OSNet | `[1, 3, 256, 128]` | Re-ID crops |
| EfficientNet | `[1, 3, 224, 224]` | Classification |

### Per-channel vs per-tensor quantization

| Mode | What it means | Accuracy | Speed |
|------|--------------|----------|-------|
| **Per-tensor** | One scale/zero-point per entire tensor | Lower | Faster |
| **Per-channel** (for weights) | One scale/zero-point per output channel | Higher | Slightly slower |

Use `per_channel=True` for better accuracy.

### Mixed precision (advanced concept)

Keep sensitive layers in FP16, quantize compute-heavy conv/dense layers to INT8:
- Attention layers → FP16 (sensitive to quantization)
- Conv/Dense layers → INT8 (compute-heavy, tolerates quantization)

**Expected gain: 1.5-2x speedup for vision models.**

---

## 7. MIGraphX exhaustive kernel tuning — step by step

**What it does:** MIGraphX has multiple kernel implementations per op (assembly kernels, composable kernels, MIOpen). Exhaustive tuning tries all of them and picks the fastest.

### Enable in your ORT session

```python
migraphx_options = {
    "device_id": 0,
    "migraphx_fp16_enable": True,           # already doing this
    "migraphx_exhaustive_tune": True,        # NEW: try all kernel variants
    "migraphx_save_compiled_model": True,    # cache the tuned result
    "migraphx_load_compiled_model": True,    # load cache on next run
    "migraphx_model_cache_path": "/home/sudhdevu/R1models/.migraphx_cache/"
}

session = ort.InferenceSession(
    model_path,
    sess_options,
    providers=[("MIGraphXExecutionProvider", migraphx_options)]
)
```

### Create cache and run tuning pass

```bash
mkdir -p /home/sudhdevu/R1models/.migraphx_cache

# First run with tuning enabled (slow — minutes to hours per model)
python benchmark_cooldown.py --warmup 1 --runs 1 --cooldown 10 --ep migraphx \
    --models yolo crossformer mobilenetv2 clip_vit_b32

# After tuning, all subsequent runs use cached kernels automatically
```

**Important notes:**
- First run with exhaustive tuning is **much slower** (minutes to hours per model)
- Result is **cached** — subsequent runs use pre-tuned kernels
- **Typical improvement: 10-30% latency reduction**
- Cache invalidates when ORT/ROCm version changes

---

## 8. Fused attention for Transformer models — step by step

**Applies to:** Qwen3, CLIP, CrossFormer (any model with self-attention)

### Option A: Verify ORT's built-in fusion

ORT already tries to fuse attention patterns when `ORT_ENABLE_ALL` is set:

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "qwen3/qwen3_optimized.onnx"  # save optimized graph

session = ort.InferenceSession("qwen3/qwen3-1.7b.onnx", sess_options)
```

Then inspect what fused:

```python
import onnx
model = onnx.load("qwen3/qwen3_optimized.onnx")
op_types = [n.op_type for n in model.graph.node]
print("FusedAttention count:", op_types.count("Attention"))
print("FusedMultiHeadAttention count:", op_types.count("MultiHeadAttention"))
print("Unfused MatMul count:", op_types.count("MatMul"))
```

### Option B: ORT Transformer Optimizer (more aggressive)

```python
from onnxruntime.transformers import optimizer

optimized_model = optimizer.optimize_model(
    "qwen3/qwen3-1.7b.onnx",
    model_type="gpt2",           # ORT label for causal-decoder graphs (Qwen, Llama, GPT-like)
    num_heads=16,                 # from config.json
    hidden_size=2048,             # from config.json
    use_gpu=False,                # True targets CUDA; for ROCm/MIGraphX keep False
    opt_level=2                   # aggressive fusion
)
optimized_model.save_model_to_file("qwen3/qwen3-1.7b_fused.onnx")
```

### Model type mapping

| Model | `model_type` | `num_heads` | `hidden_size` |
|-------|-------------|-------------|---------------|
| Qwen3-1.7B | `"gpt2"` (ORT label for causal-decoder architectures) | 16 | 2048 |
| CLIP ViT-B/32 | `"bert"` | 12 | 768 |
| CLIP ViT-B/16 | `"bert"` | 12 | 768 |
| CrossFormer | `"bert"` | Check config.json | Check config.json |

**Expected gain: 10-25% for attention-heavy models.**

---

## 9. CPU-side thread pinning

For models with CPU-bound pre-processing (XTTS, Whisper):

### Shell

```bash
# Pin inference to cores 0-7
taskset -c 0-7 python benchmark_cooldown.py --warmup 3 --runs 3 --cooldown 120 --ep migraphx

# Or with NUMA binding
numactl --cpunodebind=0 --membind=0 python benchmark_cooldown.py --models yolo --ep migraphx
```

### Python (programmatic)

```python
import os
os.sched_setaffinity(0, set(range(8)))  # Pin to first 8 cores
```

**Expected gain: 5-10% less jitter.**

---

## 10. Manual operator fusion targets (advanced)

Beyond ORT's automatic fusions, manual rewriting can help:

- Fuse **LayerNorm** patterns into single ops
- Fuse **GELU** activation patterns
- Replace **Softmax + MatMul** attention with fused **MultiHeadAttention** if supported
- These need to match what MIGraphX can map to optimized ROCm kernels

---

## 11. Recommended execution order (do these in this order)

| Step | What | Time | Expected gain |
|------|------|------|---------------|
| 1 | `onnxsim` on all models | 5-10 min | 5-15% fewer ops |
| 2 | Static shape pinning check | 10 min | 10-30% for dynamic models |
| 3 | MIGraphX exhaustive tune | 1-3 hours (one-time) | 10-30% runtime |
| 4 | INT8 quantization (vision) | 30 min per model | 1.5-2x speedup |
| 5 | Fused attention (transformers) | 15 min per model | 10-25% for attention-heavy |
| 6 | Thread pinning | 2 min | 5-10% less jitter |

**Total potential improvement: 2-4x for vision models, 1.3-2x for LLMs/transformers.**

---

## 12. Model compression beyond quantization

INT8/FP16 aren't the only ways to make models smaller and faster.

### Pruning

Remove unimportant weights (set to zero). **Structured pruning** removes entire channels or attention heads.

| Type | What it removes | Result |
|------|----------------|--------|
| **Unstructured** | Individual weight values | Sparse matrix (needs hardware support for speedup) |
| **Structured** | Entire channels, filters, or heads | Smaller dense model (immediate speedup) |

**When useful:** model is too large for your GTT, and quantization alone isn't enough.

### Knowledge distillation

Train a smaller "student" model to mimic a larger "teacher" model's outputs.

- Teacher: large accurate model (e.g., EfficientNet-B7)
- Student: small fast model (e.g., MobileNetV2)
- Train student to match teacher's output distribution, not just ground truth labels

**Result:** student gets close to teacher's accuracy at a fraction of the size/speed.

### Architecture search (NAS)

Automatically find efficient architectures. EfficientNet was discovered this way — NAS found the best width/depth/resolution scaling ratios.

Not something you run yourself, but explains why EfficientNet and MobileNetV2 exist.

### Low-rank factorization (LoRA)

Approximate large weight matrices with the product of two smaller matrices.

```
Original: W (4096 × 4096) = 67M parameters
LoRA:     W ≈ A (4096 × 16) × B (16 × 4096) = 131K parameters
```

**Primary use:** efficient fine-tuning of LLMs (train only the small A and B matrices). Can also reduce deployment size via merging.

### Weight sharing

Multiple layers use the same weight tensor. Reduces model file size without changing architecture.

### When to consider each

| Technique | Your use case |
|-----------|--------------|
| **INT8/FP16** | First choice — already covered above |
| **Pruning** | When models barely don't fit in GTT |
| **Distillation** | When you need a faster model for real-time robotics |
| **LoRA** | When fine-tuning Qwen3 or LLaMA for your tasks |

---

## Module 06 checklist

- [ ] Can run onnxsim on a model and compare node counts before/after
- [ ] Can check which models have dynamic shapes and add dim_overrides
- [ ] Can explain INT8 calibration: what CalibrationDataReader does and why real data matters
- [ ] Can set MIGraphX exhaustive tuning provider options
- [ ] Can use ORT Transformer Optimizer with correct model_type/num_heads/hidden_size
- [ ] Can pin CPU threads with taskset or sched_setaffinity
- [ ] Know the recommended execution order and why shape pinning comes before quantization
- [ ] Can name 4 compression techniques beyond quantization and when each is useful

**Next:** `07-amd-apu-memory-gtt-unified.md`
