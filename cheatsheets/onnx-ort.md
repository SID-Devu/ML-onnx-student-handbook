# Cheatsheet: ONNX & ORT quick reference

## ONNX file inspection

```python
import onnx
model = onnx.load("model.onnx", load_external_data=False)
print(f"Opset: {model.opset_import[0].version}")
print(f"Nodes: {len(model.graph.node)}")
for inp in model.graph.input:
    dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  Input: {inp.name} → {dims}")
for out in model.graph.output:
    dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  Output: {out.name} → {dims}")
```

## ORT inference (4 steps)

```python
import onnxruntime as ort
import numpy as np

# 1. Session
session = ort.InferenceSession("model.onnx", providers=["MIGraphXExecutionProvider"])

# 2. Input metadata
inp = session.get_inputs()[0]
print(f"{inp.name}: {inp.shape} {inp.type}")

# 3. Feed
feed = {inp.name: np.random.randn(1, 3, 640, 640).astype(np.float32)}

# 4. Run
outputs = session.run(None, feed)
```

## MIGraphX provider options (full)

```python
opts = {
    "device_id": 0,
    "migraphx_fp16_enable": True,
    "migraphx_exhaustive_tune": True,
    "migraphx_save_compiled_model": True,
    "migraphx_load_compiled_model": True,
    "migraphx_model_cache_path": "/path/to/.migraphx_cache/",
}
providers = [("MIGraphXExecutionProvider", opts)]
```

## SessionOptions

```python
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = "model_optimized.onnx"  # save post-fusion graph
so.log_severity_level = 2  # 0=verbose 1=info 2=warning 3=error
```

## Shape: static vs dynamic

| In `session.get_inputs()[0].shape` | Meaning | MIGraphX impact |
|------------------------------------|---------|-----------------|
| `[1, 3, 640, 640]` (all ints) | Static | Specialized compiled kernel |
| `['batch', 3, 'h', 'w']` (strings) | Dynamic | Generic fallback kernel |

Fix dynamic → static with `dim_overrides` in ModelSpec.

## Common ops

| Op | Role |
|----|------|
| Conv | CNN spatial pattern detection |
| MatMul / Gemm | Linear projections, attention |
| Relu / GELU | Activations |
| Softmax | Probability distribution |
| LayerNorm / BatchNorm | Normalization |
| Attention / MultiHeadAttention | Fused attention (post-optimization) |
| Reshape / Transpose | Layout changes |
| QuantizeLinear / DequantizeLinear | INT8 quant nodes |

## Post-export tools

| Tool | Command |
|------|---------|
| Validate | `onnx.checker.check_model(model)` |
| Shape inference | `onnx.shape_inference.infer_shapes(model)` |
| Simplify | `python -m onnxsim model.onnx model_sim.onnx` |
| Visualize | `netron model.onnx` |

## External data (models >2GB)

```python
onnx.save(model, "model.onnx", save_as_external_data=True,
          all_tensors_to_one_file=True, location="model.onnx.data")
model = onnx.load("model.onnx", load_external_data=True)
```

## Quantization (INT8 PTQ)

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
quantize_static(model_input="model.onnx", model_output="model_int8.onnx",
                calibration_data_reader=reader, quant_format=QuantType.QInt8,
                per_channel=True, weight_type=QuantType.QInt8)
```

## Transformer optimizer

```python
from onnxruntime.transformers import optimizer
opt = optimizer.optimize_model("model.onnx", model_type="gpt2",
                                num_heads=16, hidden_size=2048, use_gpu=True)
opt.save_model_to_file("model_fused.onnx")
```

| Model | model_type | num_heads | hidden_size |
|-------|-----------|-----------|-------------|
| Qwen3-1.7B | `"gpt2"` | 16 | 2048 |
| CLIP ViT-B/32 | `"bert"` | 12 | 768 |
| CLIP ViT-B/16 | `"bert"` | 12 | 768 |

## Check EP status

```python
print(session.get_providers())  # what's available
# If MIGraphXExecutionProvider NOT listed → build issue (Module 21)
```
