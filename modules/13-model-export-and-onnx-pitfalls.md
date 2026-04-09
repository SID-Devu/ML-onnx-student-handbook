# Module 13 — Model export pipeline and ONNX pitfalls (full depth with workaround table)

---

## 1. Why export exists

Training frameworks (PyTorch, TensorFlow, JAX) train models in Python with dynamic computation. Deployment wants a **portable, optimized artifact** — ONNX provides this.

---

## 2. `torch.onnx.export()` — the most common path

```python
import torch
import torch.onnx

model = MyModel()
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}}
)
```

---

## 3. Tracing vs scripting

| Method | How it works | Limitation |
|--------|-------------|-----------|
| **Tracing** | Runs the model once, records every op that executes | Misses unexecuted branches (if/else paths not taken) |
| **Scripting** (`torch.jit.script`) | Analyzes Python code statically | Can't handle all Python patterns |

**Most exports use tracing.** Problem: if the model has `if` statements that depend on input values, tracing only captures one branch.

---

## 4. Dynamic axes

```python
dynamic_axes = {
    "input": {0: "batch"},         # batch dim can vary
    "output": {0: "batch", 1: "sequence_length"}
}
```

Tells the exporter which dimensions may vary at runtime. Creates dynamic shapes in the ONNX model (Module 04).

---

## 5. Opset version

Higher opset = more operators. Some ops only exist at/after certain opsets:

- `grid_sample`: requires opset 16+
- `GroupNormalization`: native in opset 18+
- Your models use opset 17

---

## 6. HuggingFace Optimum — automated export

```bash
optimum-cli export onnx --model <model_id> <output_dir>
```

Outputs:
- One or more ONNX files (encoder, decoder, etc.)
- `config.json` (architecture hyperparameters)
- Sometimes external data files

---

## 7. `tf2onnx` — TensorFlow/TFLite to ONNX

```bash
python -m tf2onnx.convert --tflite model.tflite --output model.onnx --opset 17
```

Used when the upstream model is TensorFlow-based.

---

## 8. Post-export tools

| Tool | Purpose |
|------|---------|
| `onnx.checker.check_model()` | Validate structural correctness |
| `onnx.shape_inference.infer_shapes()` | Fill in intermediate tensor shapes |
| `onnxsim` (ONNX Simplifier) | Remove redundant ops, fold constants |
| **Netron** | Visual graph inspector |

---

## 9. External data for models >2GB

```python
onnx.save(
    model,
    "model.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="model.onnx.data"
)
```

---

## 10. `config.json` — model configuration

```json
{
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "vocab_size": 151936
}
```

You need these values for the ORT Transformer Optimizer (Module 06).

---

## 11. SafeTensors vs pickle `.bin`

| Format | Safety | Speed | Risk |
|--------|--------|-------|------|
| **SafeTensors** | Safe (no code execution) | Fast (memory-mapped) | None |
| **`.bin` (pickle)** | **Unsafe** — can execute arbitrary code during load | Slower | Malicious checkpoints can run code |

**Prefer SafeTensors** when available. HuggingFace ecosystem is moving to it.

---

## 12. Export workaround table — real problems from your model zoo

| Problem | Workaround | Which models affected |
|---------|------------|----------------------|
| Model too large (>2GB protobuf limit) | Use `save_as_external_data=True` | Qwen3, OpenVLA, LLaMA, DeepSeek |
| MIGraphX template recursion / huge graphs | Split model into sub-graphs (e.g., encoder + decoder as separate ONNX files) | LLaMA-3.2 (vision encoder + decoder) |
| Custom CUDA ops won't trace | Replace with standard PyTorch ops before export | RAFT-Stereo (`CorrSampler` custom op) |
| Dynamic control flow (if/else on input values) | Use `torch.jit.script` or restructure model to remove conditional | Wav2Vec2 (`layer_drop` during training) |
| FP16/FP32 cast mismatches | Insert explicit `Cast` nodes in the ONNX graph post-export | Several models with mixed precision training |
| NaN in weights | Replace NaN values with 0 or small epsilon before saving | Some exported models (numerical issues during training) |
| `tie_weights` error in HuggingFace models | Patch/mock the `tie_weights` function before export | Some HuggingFace encoder-decoder models |
| Meta tensors ("weights not materialized") | Materialize parameters before export: `model.to("cpu")` or load without `device="meta"` | Large models loaded with `device_map="auto"` |

---

## 13. HuggingFace Hub — where you download models

### Authentication

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

Required for **gated models** like LLaMA, Qwen — you must accept a license on the HF web page first.

### Downloading models

```bash
# CLI download
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./qwen3

# Download a specific file
huggingface-cli download Qwen/Qwen3-1.7B config.json --local-dir ./qwen3
```

### `from_pretrained()` — the standard Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
```

Downloads and caches in `~/.cache/huggingface/hub/`.

### `trust_remote_code=True`

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    trust_remote_code=True   # executes Python code from the model repo
)
```

**Security risk:** this allows the model repo to run arbitrary Python on your machine. Only use for repos you trust. Required for models with custom architectures not yet in `transformers`.

### Key concepts

| Concept | What |
|---------|------|
| **Model card** | README on HF page describing model, training data, limitations, license |
| **Gated models** | Require accepting a license before download (LLaMA, some Qwen) |
| **Revisions / commits** | HF repos are git repos — you can pin to a specific version |
| **Git LFS** | Large file storage — HF uses this for model weights (`.safetensors`, `.bin`) |
| **SafeTensors vs .bin** | SafeTensors = safe (no code execution); `.bin` = pickle (can run malicious code on load) |

### Pin to a specific model version

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    revision="abc123def"  # specific commit hash
)
```

---

## Module 13 checklist

- [ ] Explain tracing limitation: "misses unexecuted branches" in one sentence
- [ ] Explain why external data format exists (protobuf >2GB limit)
- [ ] Can name 5 export failure modes from the workaround table
- [ ] Know the difference between SafeTensors and pickle `.bin`
- [ ] Can use `onnx.checker.check_model()` to validate an export
- [ ] Know what `config.json` contains and when you need it
- [ ] Can authenticate with HuggingFace CLI and download a gated model
- [ ] Can explain the security risk of `trust_remote_code=True`
- [ ] Can pin a model download to a specific version

**Next:** `14-benchmarking-methodology-and-stats.md`
