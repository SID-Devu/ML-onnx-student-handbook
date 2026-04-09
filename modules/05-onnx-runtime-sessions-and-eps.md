# Module 05 — ONNX Runtime: sessions, execution providers, runs (full depth)

**ONNX Runtime (ORT)** is the engine that loads ONNX graphs and executes them. This module covers every concept you interact with when running benchmarks.

---

## 1. The key pattern (memorize this)

This is the pattern you'll see in every inference script:

```python
import onnxruntime as ort
import numpy as np

# 1. Create session (loads + compiles model)
session = ort.InferenceSession(
    "yolo/yolov12m.onnx",
    providers=["MIGraphXExecutionProvider"]  # Use AMD GPU
)

# 2. Prepare input
input_name = session.get_inputs()[0].name   # "images"
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)

# 3. Run inference
outputs = session.run(None, {input_name: input_data})

# 4. Read output
print(outputs[0].shape)  # e.g. (1, 84, 8400) for YOLO detections
```

---

## 2. `InferenceSession` — what happens when you create one

Creating a session:

1. **Parses** the ONNX graph (reads protobuf)
2. **Applies graph optimizations** (based on SessionOptions — fusions, constant folding)
3. **Partitions** the graph across execution providers (some nodes on GPU, some on CPU if needed)
4. **Compiles / allocates** EP-specific resources (MIGraphX kernel compilation, memory allocation)

**First session creation is slow** (especially with MIGraphX — may take seconds to minutes for compilation). Subsequent `session.run()` calls are fast.

---

## 3. Execution Providers (EPs) — the backends

An EP is a backend implementation that runs operators:

| EP | What it uses | When to use |
|----|-------------|-------------|
| `CPUExecutionProvider` | CPU (always available) | Baseline, fallback, testing |
| `MIGraphXExecutionProvider` | AMD GPU via MIGraphX | Your primary target |
| `CUDAExecutionProvider` | NVIDIA GPU via CUDA | Not for your AMD setup |
| `TensorrtExecutionProvider` | NVIDIA TensorRT | Not for your AMD setup |
| `ROCMExecutionProvider` | AMD GPU via ROCm/MIOpen | Alternative AMD path |

**Provider order matters.** ORT tries providers in the order you pass them:

```python
providers = [
    "MIGraphXExecutionProvider",  # try GPU first
    "CPUExecutionProvider",       # fall back to CPU
]
```

If MIGraphX can't handle a node, CPU picks it up.

---

## 4. Getting input/output metadata

```python
session = ort.InferenceSession("model.onnx")

# Inputs
for inp in session.get_inputs():
    print(f"Name: {inp.name}")
    print(f"Shape: {inp.shape}")      # e.g. [1, 3, 640, 640] or ['batch', 3, 'h', 'w']
    print(f"Type: {inp.type}")        # e.g. "tensor(float)"

# Outputs
for out in session.get_outputs():
    print(f"Name: {out.name}")
    print(f"Shape: {out.shape}")
    print(f"Type: {out.type}")
```

**Dynamic dims show as strings** in `.shape`. Static dims show as integers.

---

## 5. `session.run()` — running inference

```python
# Run all outputs
outputs = session.run(None, {"images": input_array})

# Run specific outputs only
outputs = session.run(["output0"], {"images": input_array})
```

- First argument: output names (`None` = all outputs)
- Second argument: `{input_name: numpy_array}` feed dict

---

## 6. `SessionOptions` — graph optimization and configuration

```python
sess_options = ort.SessionOptions()

# Graph optimization level
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Save the optimized graph to inspect what fusions happened
sess_options.optimized_model_filepath = "model_optimized.onnx"

# Thread settings (for CPU EP)
sess_options.intra_op_num_threads = 4
sess_options.inter_op_num_threads = 1

# Logging
sess_options.log_severity_level = 1  # 0=verbose, 1=info, 2=warning, 3=error

session = ort.InferenceSession("model.onnx", sess_options, providers=[...])
```

**Graph optimization levels:**

| Level | What it does |
|-------|-------------|
| `ORT_DISABLE_ALL` | No optimization |
| `ORT_ENABLE_BASIC` | Basic optimizations (constant folding, redundant node elimination) |
| `ORT_ENABLE_EXTENDED` | + more complex fusions |
| `ORT_ENABLE_ALL` | All optimizations including layout transforms. **Your benchmark uses this.** |

---

## 7. Provider Options — EP-specific configuration

MIGraphX EP options are passed as a dict:

```python
migraphx_options = {
    "device_id": 0,
    "migraphx_fp16_enable": True,           # compile FP16 kernels
    "migraphx_exhaustive_tune": True,        # try all kernel variants (slow compile, faster run)
    "migraphx_save_compiled_model": True,    # cache compiled program
    "migraphx_load_compiled_model": True,    # load from cache
    "migraphx_model_cache_path": "/home/sudhdevu/R1models/.migraphx_cache/"
}

session = ort.InferenceSession(
    model_path,
    sess_options,
    providers=[("MIGraphXExecutionProvider", migraphx_options)]
)
```

**Note:** exact option key names depend on your ORT build version. Always verify against your build.

---

## 8. Warmup and caching

First runs can trigger:

- **Kernel compilation** (MIGraphX compiles ONNX ops into GPU kernels)
- **Memory planning** (allocating GPU buffers)
- **Cache creation** (compiled kernels saved to disk)
- **Page migration** (with unified memory, pages may migrate on first access)

So benchmarks use **warmup** iterations that are discarded from statistics:

```python
# Warmup — discard these timings
for i in range(warmup):
    session.run(None, feed)

# Timed runs — measure these
latencies = []
for i in range(runs):
    t0 = time.perf_counter()
    session.run(None, feed)
    dt = (time.perf_counter() - t0) * 1000
    latencies.append(dt)
```

---

## 9. Common failures you will see

| Error | Cause | Fix |
|-------|-------|-----|
| `MIGraphXExecutionProvider` not in `get_available_providers()` | ORT not built with MIGraphX EP | Use your custom ORT build |
| `InvalidArgument: Got invalid dimensions for input` | Shape mismatch | Check `get_inputs()[0].shape` and match it |
| `Type Error: unexpected input data type` | Wrong dtype (e.g. float64 instead of float32) | `.astype(np.float32)` |
| `Model loading failed` | Corrupt ONNX, missing external data | Run `onnx.checker.check_model()`, check `.data` file exists |
| `Unsupported op on MIGraphX` | Op not implemented in MIGraphX | Falls back to CPU; check ORT logs for which nodes |
| Session creation extremely slow | MIGraphX compiling all kernels from scratch | Enable compiled model caching; use warmup |

---

## 10. Checking which EP is actually running your model

```python
session = ort.InferenceSession("model.onnx", providers=[
    "MIGraphXExecutionProvider", "CPUExecutionProvider"
])

# Check what providers are actually in use
print(session.get_providers())
# ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
```

**Warning:** even if MIGraphX is listed, some **individual nodes** may run on CPU if MIGraphX doesn't support them. This is graph partitioning (Module 19). You think you're "all GPU" but hot nodes silently run on CPU.

---

## Module 05 checklist

- [ ] Can write the 4-step pattern (create session → get inputs → run → read output) from memory
- [ ] Can explain EP vs session
- [ ] Can list inputs/outputs and construct a valid feed dict
- [ ] Can set `ORT_ENABLE_ALL` in SessionOptions
- [ ] Can explain why first run differs from steady state (compilation + caching)
- [ ] Know to check `get_providers()` and understand partitioning risk

**Next:** `06-optimizations-fusion-quantization.md`
