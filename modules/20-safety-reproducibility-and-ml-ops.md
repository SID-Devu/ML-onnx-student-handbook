# Module 20 — Safety, reproducibility, honest benchmarking (full depth)

---

## 1. Pickle and untrusted `.bin` files

Some older weight formats (PyTorch `.pth`, HuggingFace `.bin`) use **Python pickle** for serialization.

**Pickle can execute arbitrary code during `torch.load()`.** A malicious checkpoint can:

- Delete files
- Install malware
- Exfiltrate data

```python
# DANGEROUS — pickle-based loading
import torch
model = torch.load("untrusted_model.pth")  # can run arbitrary code!

# SAFER — restrict to weights only (PyTorch 2.6+)
model = torch.load("model.pth", weights_only=True)  # blocks code execution

# SAFEST — SafeTensors (no pickle at all)
from safetensors.torch import load_file
weights = load_file("model.safetensors")  # only loads tensors, no code execution
```

### SafeTensors vs pickle

| Format | File extension | Safety | Speed | Code execution risk |
|--------|---------------|--------|-------|-------------------|
| **SafeTensors** | `.safetensors` | Safe | Fast (memory-mapped) | **None** |
| **Pickle/bin** | `.bin`, `.pth`, `.pt` | Unsafe | Slower | **Yes — arbitrary code** |

**Rule:** prefer SafeTensors when downloading from HuggingFace or other sources. The HF ecosystem is migrating to SafeTensors.

---

## 2. Supply chain and model provenance

When moving models between machines or downloading from the internet:

### Verify hashes

```bash
# Generate hash of your model
sha256sum yolo/yolov12m.onnx
# Output: a1b2c3d4... yolo/yolov12m.onnx

# Compare with known-good hash
echo "a1b2c3d4...expected_hash...  yolo/yolov12m.onnx" | sha256sum --check
```

### Pin versions in reports

Every benchmark report should include:

| Field | Example value |
|-------|---------------|
| Kernel version | `6.18.0+` |
| ROCm version | `6.3.0` |
| ORT version + build | `1.21.0+custom (CMake: -DUSE_MIGRAPHX=ON -DUSE_HIP_MANAGED_MEM=ON)` |
| Python version | `3.11.9` |
| Model sha256 | `a1b2c3d4e5f6...` |
| HSA_XNACK | `1` |
| amdgpu.gttsize | `28672` |

Without these, a "10% improvement" claim is unverifiable.

---

## 3. Separating science from benchmark theater

### Bad practices (benchmark theater)

- Cherry-picking the fastest run out of 10 and reporting only that
- Changing cooldown only for models that "look bad"
- Comparing results taken at different thermal states
- Comparing Xen dom0 results with bare metal without disclosing
- Reporting mean when there are clear outliers (hiding jitter)
- Running benchmarks with background processes (browsers, updates)
- Claiming INT8 speedup without accuracy validation

### Good practices (honest benchmarking)

- Report **distribution**: median, mean, std, P95, P99
- Report **per-run latencies** (not just aggregates)
- Report **thermal state**: GPU temp before/after, sclk during run
- Disclose **environment**: kernel, ROCm, ORT build, boot params
- Use **same conditions** for before/after comparisons
- Note **cold vs warm** page cache state
- Validate **accuracy** for any quantized model (FP16 or INT8)

---

## 4. Accuracy validation after quantization

When you quantize a model (FP16 or INT8), you **must** check that outputs are still reasonable:

```python
import numpy as np
import onnxruntime as ort

# Load both versions
sess_fp32 = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
sess_int8 = ort.InferenceSession("model_int8.onnx", providers=["CPUExecutionProvider"])

# Same input
inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
feed = {sess_fp32.get_inputs()[0].name: inp}

out_fp32 = sess_fp32.run(None, feed)[0]
out_int8 = sess_int8.run(None, feed)[0]

# Compare
max_diff = np.max(np.abs(out_fp32 - out_int8))
mean_diff = np.mean(np.abs(out_fp32 - out_int8))
print(f"Max absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {mean_diff:.6f}")
print(f"Close enough? {np.allclose(out_fp32, out_int8, atol=0.1)}")
```

**Acceptable thresholds** depend on the task:
- Classification: top-1 class should match in most cases
- Detection: bounding boxes within a few pixels
- LLM: generated text should be coherent (harder to automate)

---

## 5. Data privacy (calibration datasets)

Calibration datasets for INT8 quantization may contain:

- Real images from deployment (faces, medical, industrial)
- Real audio samples (voice recordings)
- Real text (user queries)

**Handle these like production data:**
- Don't commit to public repos
- Don't upload to cloud without permission
- Delete after calibration if not needed

---

## 6. Regression tracking

Keep a **golden JSON** of baseline latencies per model:

```json
{
    "yolov12m": {"median_ms": 12.4, "date": "2025-01-15", "ort": "1.21.0+custom"},
    "mobilenetv2": {"median_ms": 3.2, "date": "2025-01-15", "ort": "1.21.0+custom"},
    "qwen3": {"median_ms": 45.8, "date": "2025-01-15", "ort": "1.21.0+custom"}
}
```

After any change (ORT update, ROCm update, kernel update, new model version):

1. Run the benchmark suite
2. Compare against golden baseline
3. Flag any regressions >5%
4. Investigate before accepting the update

---

## 7. What "reproducibility" means in practice

Someone else (or future you) should be able to:

1. Read your benchmark JSON
2. Set up the same software versions
3. Apply the same kernel/sysctl/service tuning
4. Run the same benchmark script with same flags
5. Get **comparable** results (within statistical noise)

If they can't, you haven't documented enough.

---

## Module 20 checklist

- [ ] Can explain why pickle checkpoints are riskier than SafeTensors
- [ ] Can verify a model file hash with sha256sum
- [ ] Can list 5 fields in a reproducibility footer
- [ ] Can compare FP32 vs INT8 outputs numerically
- [ ] Can name 3 "benchmark theater" bad practices
- [ ] Understand calibration data privacy obligations

**Next:** `21-building-ort-and-version-skew.md`
