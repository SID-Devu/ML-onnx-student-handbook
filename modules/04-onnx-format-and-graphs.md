# Module 04 — ONNX format: graphs, ops, shapes, external data (full depth)

**ONNX** (Open Neural Network Exchange) is a portable **serialized graph** format for ML models. Like PDF is for documents — any framework can export to it, any runtime can load it. Every model you benchmark is an ONNX file.

---

## 1. Graph mental model

An ONNX model is a **directed graph**:

- **Nodes** = operations (Conv, MatMul, Add, Softmax, …)
- **Edges** = tensors flowing between nodes
- **Graph inputs** = the public entry points (your image, tokens, etc.)
- **Graph outputs** = the public exit points (boxes, logits, embeddings, etc.)
- **Initializers** = constant tensors (learned weights, biases)

---

## 2. Operators and opset version

Each ONNX file targets an **opset** version. Opset = version of the operator specification.

| Opset | Key additions |
|-------|---------------|
| 11 | CumSum, Round, Det |
| 13 | ReduceSum shape changes, Squeeze/Unsqueeze changes |
| 16 | `grid_sample` operator (needed by some vision models) |
| 17 | Your models use this. LayerNorm, GroupNorm as native ops |
| 18+ | Newer ops continue to be added |

**Why it matters:** a runtime must support the ops + opset your model requires. If MIGraphX doesn't support an op, ORT may fall back to CPU for that node (graph partitioning — Module 19).

---

## 3. Initializers (weights)

Learned parameters are stored as **initializer tensors** in the graph. These are the big blobs:

```python
import onnx
model = onnx.load("yolo/yolov12m.onnx")
print(f"Initializers: {len(model.graph.initializer)}")  # e.g. 300+
```

---

## 4. External data (large models >2GB)

Protobuf has a ~2GB limit per file. For large models, weights are stored in a sidecar:

- `model.onnx` (graph structure, small)
- `model.onnx.data` (weights, can be many GB)

**Loading with external data:**

```python
import onnx
model = onnx.load("qwen3/qwen3-1.7b.onnx", load_external_data=True)
```

**Saving with external data:**

```python
onnx.save(
    model,
    "qwen3/qwen3-1.7b_simplified.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="qwen3-1.7b_simplified.onnx.data"
)
```

**Your models that need external data:** Qwen3, OpenVLA, LLaMA, DeepSeek (anything >2GB).

---

## 5. Static vs dynamic shapes

Dimensions can be:

- **Static:** fixed integer, e.g. `[1, 3, 224, 224]`
- **Dynamic:** symbolic name, e.g. `["batch", 3, "height", "width"]`

**How you see it in ORT:**

```python
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx")
for inp in sess.get_inputs():
    print(inp.name, inp.shape)
    # might print: images [1, 3, 640, 640]        ← all static
    # or:          images ['batch', 3, 'h', 'w']  ← dynamic
```

A dimension that is a **string** = dynamic. A dimension that is an **integer** = static.

**Performance impact (MIGraphX):** dynamic shapes force MIGraphX to compile generic fallback kernels. Static shapes let it compile a single optimized kernel for that exact size. **This is one of the biggest wins for MIGraphX specifically** (see Module 06 for how to pin shapes).

---

## 6. Shape inference

```python
import onnx
from onnx import shape_inference

model = onnx.load("model.onnx")
model = shape_inference.infer_shapes(model)
onnx.save(model, "model_with_shapes.onnx")
```

This propagates known shapes through the graph so more intermediate tensors have known shapes/types.

**Why it helps:** compilers/backends (MIGraphX) can choose better kernel implementations when shapes are known at compile time.

---

## 7. Model validation

```python
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)  # raises if structurally invalid
print("Model is valid")
```

This checks structural validity — **not** "your model is accurate" or "this will run on MIGraphX." Just "the file is a coherent ONNX graph."

---

## 8. Inspecting models — Netron (visual)

**Install Netron:**

```bash
pip install netron
```

**Open any model visually:**

```bash
netron yolo/yolov12m.onnx
# Opens in browser — you can see every layer, click nodes, see shapes
```

**Also works in browser:** go to https://netron.app/ and drag-drop any `.onnx` file.

**What to look for in Netron:**

- Input names and shapes (click the input node)
- Output names and shapes (click the output node)
- Op types (each node shows its type)
- Whether shapes are static (numbers) or dynamic (strings)
- Weights (click initializer nodes to see shape/dtype)

---

## 9. Inspecting models — Python API (programmatic)

```python
import onnx
from onnx import TensorProto

model = onnx.load("yolo/yolov12m.onnx")

# Opset
print(f"Opset: {model.opset_import[0].version}")

# Node count
print(f"Nodes: {len(model.graph.node)}")

# Inputs (skip initializers that also appear as graph inputs)
init_names = {i.name for i in model.graph.initializer}
print("Inputs:")
for inp in model.graph.input:
    if inp.name in init_names:
        continue  # this is a weight, not a real input
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(d.dim_value)
        elif d.dim_param:
            dims.append(d.dim_param)  # dynamic axis name
        else:
            dims.append("?")
    print(f"  {inp.name}: {dims}")

# Outputs
print("Outputs:")
for out in model.graph.output:
    dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {dims}")

# Op type histogram
from collections import Counter
op_counts = Counter(n.op_type for n in model.graph.node)
print("\nTop ops:")
for op, count in op_counts.most_common(10):
    print(f"  {op}: {count}")
```

---

## 10. `config.json` — model configuration file

HuggingFace models come with `config.json` containing architecture hyperparameters:

```json
{
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "vocab_size": 151936,
  "model_type": "qwen3"
}
```

**Why you need it:** when using the ORT Transformer Optimizer (Module 06), you must provide `num_heads` and `hidden_size` from this file.

---

## Module 04 checklist

- [ ] Can define graph/node/tensor/initializer in your own words
- [ ] Can explain static vs dynamic dimension and how to tell them apart in ORT
- [ ] Know what external data files are for and when they're needed (>2GB models)
- [ ] Can install and use Netron to visually inspect a model
- [ ] Can programmatically list inputs/outputs/ops with the `onnx` Python API
- [ ] Know what `opset` means and why version matters

**Next:** `05-onnx-runtime-sessions-and-eps.md`
