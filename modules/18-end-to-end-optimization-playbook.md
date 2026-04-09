# Module 18 — End-to-end optimization playbook (full depth)

This ties together modules 04–08 into an operational recipe with your exact setup status.

---

## 1. What's already optimized in your setup

| Optimization | Status | Where configured |
|-------------|--------|-----------------|
| ORT graph optimization (`ENABLE_ALL`) | **Done** | `benchmark_cooldown.py` SessionOptions |
| FP16 for supported models | **Done** | Per-model `ort_migraphx_fp16: True` in ModelSpec |
| Static shape pinning (`dim_overrides`) | **Done for several models** | ModelSpec entries |
| GPU clocks locked to max (sclk=2900, mclk=1000, fclk=2000) | **Done** | `r1-gpu-perf.service` |
| CPU latency minimized (cstate=1, nomwait) | **Done** | Kernel boot parameters |
| GTT maximized (28 GB) | **Done** | `amdgpu.gttsize=28672` |
| Unified memory (`HSA_XNACK=1`) | **Done** | `/etc/environment` |
| Swap for large models (400 GB on NVMe) | **Done** | `/etc/fstab` |
| MGLRU enabled | **Done** | `r1-gpu-perf.service` |
| THP madvise | **Done** | Boot param + service |
| All sysctl vm.* tuned | **Done** | `r1-gpu-perf.service` |

---

## 2. Biggest remaining wins (prioritized)

| Priority | Optimization | Applies to | Expected gain |
|----------|-------------|-----------|---------------|
| 1 | INT8 quantization | Vision models (YOLO, MobileNetV2, CLIP, DeepSort, EfficientNet) | 1.5-2x latency reduction |
| 2 | MIGraphX exhaustive kernel tuning | All models on MIGraphX EP | 10-30% improvement |
| 3 | ONNX Simplifier pass | All models (check which haven't been simplified) | 5-15% fewer ops |
| 4 | Static shape pinning | Any model still using dynamic dims without dim_overrides | 10-30% for those models |
| 5 | Kernel fusion for transformers | Qwen3, CLIP, CrossFormer | 10-25% for attention-heavy |

---

## 3. Operational phases (do in this order)

### Phase A — Establish a truthful baseline

1. Verify GPU clocks locked: `rocm-smi --showclocks` → sclk=2900 MHz
2. Verify thermal stability: `rocm-smi --showtemp` → note starting temperature
3. Run baseline benchmark with current models
4. Save results JSON with full reproducibility footer (Module 14)

### Phase B — Graph cleanup (cheap, 5-10 minutes)

1. Run `onnxsim` on every model:

```bash
python -m onnxsim model.onnx model_simplified.onnx
```

2. Run shape inference:

```python
model = onnx.load("model.onnx")
model = shape_inference.infer_shapes(model)
onnx.save(model, "model.onnx")
```

3. Verify: node count dropped? Model still runs on CPU EP?

### Phase C — Shape discipline (often huge for MIGraphX, 10 minutes)

1. Run the dynamic shape audit:

```python
for inp in session.get_inputs():
    has_dynamic = any(isinstance(dim, str) for dim in inp.shape)
    if has_dynamic:
        print(f"DYNAMIC: {inp.name}: {inp.shape}")
```

2. Add `dim_overrides` to every ModelSpec with dynamic dims
3. Re-benchmark — compare median latency before/after

### Phase D — EP tuning / caches (one-time cost, 1-3 hours)

1. Create cache directory:

```bash
mkdir -p /home/sudhdevu/R1models/.migraphx_cache
```

2. Enable exhaustive tuning in provider options:

```python
"migraphx_exhaustive_tune": True,
"migraphx_save_compiled_model": True,
"migraphx_load_compiled_model": True,
"migraphx_model_cache_path": "/home/sudhdevu/R1models/.migraphx_cache/"
```

3. Run one pass through all models (this is the slow tuning pass)
4. Verify: second cold start is faster (cached kernels loaded)

### Phase E — Precision (30 min per model for INT8)

1. **FP16:** already done for many models via `ort_migraphx_fp16: True`
2. **INT8 PTQ:** for vision models, using real calibration data:
   - Collect 100-500 representative images
   - Run quantization script (Module 06 section 6)
   - Verify: accuracy sanity check + latency comparison

### Phase F — Transformer graph rewriting (optional, 15 min per model)

1. Save optimized ONNX from ORT and inspect fused op counts
2. Try ORT Transformer Optimizer with correct `model_type`/`num_heads`/`hidden_size`
3. Verify: correctness on golden inputs within numerical tolerance

### Phase G — CPU pinning (2 minutes)

```bash
taskset -c 0-7 python benchmark_cooldown.py ...
```

Verify: reduced variance across runs (tighter std, smaller tails).

### Phase H — Document everything

Record in your benchmark report:
- ORT build flags + commit
- ROCm version
- Kernel cmdline + sysctl deltas
- Model hashes (sha256)
- Exact provider options keys used
- Before/after latency comparison per model

---

## 4. Total potential improvement

| Model type | Expected cumulative gain |
|-----------|------------------------|
| Vision models (YOLO, MobileNet, EfficientNet, DeepSort) | **2-4x** (INT8 + tuning + shapes) |
| Transformer models (Qwen3, CLIP, CrossFormer) | **1.3-2x** (fusion + tuning + shapes) |
| Robotics (OpenVLA, etc.) | **1.3-2x** (shapes + tuning) |

---

## 5. "Do no harm" rules

- **Don't compare** numbers across different thermal states
- **Don't compare** with different `HSA_XNACK` settings without labeling
- **Don't claim** INT8 wins without accuracy validation
- **Don't change** multiple variables simultaneously
- **Don't skip** cooldown between models
- **Don't trust** mean alone — always check median and per-run distribution

---

## 6. Why Phase C (shapes) belongs before Phase E (quantization)

MIGraphX compiles kernels per-shape. If you quantize first but shapes are still dynamic, MIGraphX compiles **generic** INT8 kernels. If you pin shapes first, MIGraphX compiles **specialized** INT8 kernels for your exact dimensions. The combined gain is multiplicative.

---

## Module 18 checklist

- [ ] Can execute phases A→D on a new ONNX model without guidance
- [ ] Can explain why shape pinning should precede quantization for MIGraphX
- [ ] Can reproduce the "already optimized" status table from memory
- [ ] Can list the 5 biggest remaining wins in priority order
- [ ] Can document a before/after comparison with all reproducibility fields

**Congratulations — return to `README.md` and use `glossary.md` as spaced repetition.**
