# Module 19 — ORT debugging, graph partitioning, telemetry (full depth)

---

## 1. Saving the optimized ONNX graph

ORT can write the **post-optimization** graph to disk so you can inspect what fusions happened:

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "model_optimized.onnx"  # <-- saves here

session = ort.InferenceSession("model.onnx", sess_options, providers=[...])
```

Then inspect:

```python
import onnx
model = onnx.load("model_optimized.onnx")
op_types = [n.op_type for n in model.graph.node]

print(f"Total nodes: {len(model.graph.node)}")
print(f"Attention:           {op_types.count('Attention')}")
print(f"MultiHeadAttention:  {op_types.count('MultiHeadAttention')}")
print(f"FusedConv:           {op_types.count('FusedConv')}")
print(f"Unfused MatMul:      {op_types.count('MatMul')}")
```

**Use case:** prove whether attention fusion happened (or didn't) for a transformer model.

---

## 2. Logging severity

```python
sess_options.log_severity_level = 0   # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal
```

**When to use verbose (0):** debugging EP failures — unsupported ops, shape errors, fallback messages are in the logs.

**When to use warning (2) or higher:** production/benchmark runs — less noise.

---

## 3. Graph partitioning — the hidden performance trap

When MIGraphX can't handle an op, ORT **partitions** the graph:

- Supported subgraphs → MIGraphX (GPU)
- Unsupported nodes → CPU fallback

**The trap:** you think you're "all GPU" because `get_providers()` lists MIGraphX, but **individual hot nodes silently run on CPU**. Data copies between CPU and GPU for each partition boundary add latency.

### How to detect partitioning

1. Set `log_severity_level = 0` (verbose) and look for messages about nodes falling back to CPU
2. Save the optimized graph and check which ops remain unfused/unhandled
3. Compare CPU-only vs MIGraphX-only latency — if MIGraphX is barely faster, partitioning may be eating the gains

### Common ops that cause partitioning on MIGraphX

- Custom ops not in the ONNX standard
- Very new ops from high opset versions
- Ops with unsupported attributes or edge-case shapes

### Fix

- Replace unsupported ops with supported equivalents in the ONNX graph
- Split the model into subgraphs that are fully supported (Module 13 workaround for LLaMA)
- Check MIGraphX version — newer versions support more ops

---

## 4. Custom operators

If a model includes a **custom op** not in the standard ONNX operator set:

| Scenario | What happens |
|----------|-------------|
| Custom op supported by EP | Runs normally |
| Custom op not supported by any EP | Session creation **fails** |
| Custom op supported only by CPU | Partitions to CPU for that node |

**Skill:** read the error message, identify the op type name (e.g., `CorrSampler`), and search ORT op support tables for your version.

---

## 5. Telemetry patterns in benchmark repos

Your `benchmark_cooldown.py` captures system state alongside inference:

### Before/after dmesg

```python
import subprocess

# Before inference
dmesg_before = subprocess.run(
    ["dmesg", "--level=warn,err"], capture_output=True, text=True
).stdout

# ... run inference ...

# After inference
dmesg_after = subprocess.run(
    ["dmesg", "--level=warn,err"], capture_output=True, text=True
).stdout

# Diff — new kernel messages during inference
new_lines = set(dmesg_after.splitlines()) - set(dmesg_before.splitlines())
print(f"New dmesg messages during inference: {len(new_lines)}")
for line in sorted(new_lines):
    print(f"  {line}")
```

### GPU temperature snapshots

```python
import subprocess

def get_gpu_temp():
    result = subprocess.run(
        ["rocm-smi", "--showtemp"], capture_output=True, text=True
    )
    return result.stdout

temp_before = get_gpu_temp()
# ... run model ...
temp_after = get_gpu_temp()
```

### SVM message counting (XNACK experiments)

When testing `HSA_XNACK=0`, count SVM IOCTL messages:

```python
svm_count = sum(1 for line in dmesg_after.splitlines() if "svm" in line.lower())
print(f"SVM IOCTL messages: {svm_count}")
# Your XNACK=0 test showed 750+ of these
```

---

## 6. `KernelTelemetry` style dataclasses

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class KernelTelemetry:
    model_name: str
    dmesg_before: str
    dmesg_after: str
    new_dmesg_lines: int
    svm_message_count: int
    gpu_temp_before_c: float
    gpu_temp_after_c: float
    sclk_during_mhz: int
    per_run_latencies_ms: List[float] = field(default_factory=list)
    throttled: bool = False
```

These are **audit trails** — structured records of system state during each model's benchmark. Essential for explaining anomalous results.

---

## 7. Timing caveats — sync vs async

GPU work may be **asynchronous**:

- CPU call `session.run()` returns → but GPU may still be computing
- If the EP synchronizes at return (common for MIGraphX EP via ORT), your `perf_counter` is correct
- If not, you're measuring "time to enqueue" not "time to complete"

**Verification:** run the same model 100 times and check if latencies are suspiciously low (sub-millisecond for a large model = probably measuring enqueue only).

---

## 8. Memory profiling (conceptual)

For investigating whether a model fits in GTT without swap:

```python
import os
import numpy as np
import onnxruntime as ort

def get_memory_usage_mb():
    """Read current process RSS from /proc."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024  # KB → MB
    return 0

before = get_memory_usage_mb()
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
after_load = get_memory_usage_mb()
inp = session.get_inputs()[0]
feed = {inp.name: np.zeros([1 if isinstance(d, str) else d for d in inp.shape], dtype=np.float32)}
session.run(None, feed)
after_run = get_memory_usage_mb()

print(f"Load: +{after_load - before:.0f} MB")
print(f"Run:  +{after_run - after_load:.0f} MB")
```

---

## 9. `/proc` and `/sys` filesystem navigation

Your `KernelTelemetry` class reads from these. These are **virtual filesystems** — they don't exist on disk. The kernel creates them on the fly.

### `/proc` — process and kernel info

| Path | What's there | Example use |
|------|-------------|-------------|
| `/proc/cpuinfo` | CPU details (model, cores, flags like `sse4_2`, `avx2`) | Check CPU capabilities |
| `/proc/meminfo` | RAM breakdown (MemTotal, MemFree, Buffers, Cached, SwapTotal, SwapFree) | Check if swap is being used |
| `/proc/cmdline` | Kernel boot parameters that were actually used | Verify `amdgpu.gttsize=28672` |
| `/proc/interrupts` | IRQ counts per CPU | Detect interrupt storms affecting benchmarks |
| `/proc/buddyinfo` | Memory fragmentation state | Debug hugepage allocation failures |
| `/proc/pagetypeinfo` | Page allocation statistics | Advanced memory debugging |
| `/proc/slabinfo` | Kernel memory allocator statistics | Debug kernel memory leaks |
| `/proc/<pid>/status` | Per-process memory: VmRSS (actual RAM), VmPeak (max ever) | Track model loading memory |
| `/proc/<pid>/maps` | Memory mappings of a process | See loaded libraries, mmap'd files |

### `/sys` — hardware and driver control

| Path | What's there | Example use |
|------|-------------|-------------|
| `/sys/module/amdgpu/parameters/` | AMD GPU driver runtime parameters | Read `gttsize` |
| `/sys/class/drm/card0/device/` | GPU device info, power, clocks | `pp_od_clk_voltage` |
| `/sys/class/thermal/` | Thermal zones, temperatures | Monitor CPU/GPU temps |
| `/sys/block/nvme*/queue/read_ahead_kb` | NVMe readahead setting | Tuned by `r1-gpu-perf.service` |

### Reading from these in Python

```python
# Check available memory
with open("/proc/meminfo") as f:
    for line in f:
        if "MemAvailable" in line or "SwapFree" in line:
            print(line.strip())

# Check GPU GTT size
with open("/sys/module/amdgpu/parameters/gttsize") as f:
    print(f"GTT size: {f.read().strip()} MB")

# Check kernel boot params
with open("/proc/cmdline") as f:
    params = f.read().strip()
    print(f"Boot: {params}")
```

### Reading from these in bash

```bash
cat /proc/meminfo | grep -i swap
cat /proc/cpuinfo | grep "model name" | head -1
cat /sys/module/amdgpu/parameters/gttsize
cat /proc/<PID>/status | grep VmRSS
```

---

## Module 19 checklist

- [ ] Can use `optimized_model_filepath` to answer "did attention fusion happen?"
- [ ] Can explain graph partitioning and why it breaks naive "GPU-only" assumptions
- [ ] Can capture dmesg before/after inference and count SVM messages
- [ ] Can explain sync vs async GPU timing risk
- [ ] Can use log_severity_level to debug EP failures
- [ ] Can read `/proc/meminfo` to check swap usage
- [ ] Can navigate `/sys/module/amdgpu/parameters/` to verify GPU settings
- [ ] Can read `/proc/<pid>/status` to check per-process memory usage

**Next:** `20-safety-reproducibility-and-ml-ops.md`
