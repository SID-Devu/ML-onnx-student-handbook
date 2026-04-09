# Module 15 — Python patterns in benchmark code (full depth)

This module maps Python patterns to their real usage in `benchmark_cooldown.py` and `inference.py`. Module 01 covered syntax; this module covers **how the pieces fit together in your codebase**.

---

## 1. `argparse` → session configuration flow

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--cooldown", type=int, default=120)
parser.add_argument("--models", nargs="+", default=None)
parser.add_argument("--ep", choices=["migraphx", "cpu"], default="migraphx")
args = parser.parse_args()
```

**Skill:** trace from `args.warmup` through the code to the loop `for i in range(args.warmup): session.run(...)`.

---

## 2. `ModelSpec` dataclass → benchmark matrix

```python
from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, List

@dataclass
class ModelSpec:
    name: str
    onnx_path: str
    build_inputs: Callable[[], Dict[str, "np.ndarray"]]
    dim_overrides: Dict[str, int] = field(default_factory=dict)
    providers: Optional[List[str]] = None
    ort_migraphx_fp16: bool = False

MODELS = [
    ModelSpec(
        name="yolov12m",
        onnx_path="yolo/yolov12m.onnx",
        build_inputs=build_yolo_inputs,
        ort_migraphx_fp16=True,
    ),
    ModelSpec(
        name="mobilenetv2",
        onnx_path="mobilenetv2/mobilenetv2.onnx",
        build_inputs=build_mobilenet_inputs,
        dim_overrides={"batch_size": 1},
    ),
    # ... more models
]
```

The list of `ModelSpec` objects **is** the benchmark matrix. Loop over it:

```python
for spec in MODELS:
    session = create_session(spec)
    feed = spec.build_inputs()
    latencies = time_runs(session, feed, args.runs)
```

---

## 3. `Callable` input builders — deferred execution

```python
def build_yolo_inputs() -> Dict[str, np.ndarray]:
    return {"images": np.random.randn(1, 3, 640, 640).astype(np.float32)}

def build_clip_inputs() -> Dict[str, np.ndarray]:
    return {
        "pixel_values": np.random.randn(1, 3, 224, 224).astype(np.float32),
        "input_ids": np.zeros((1, 77), dtype=np.int64),
        "attention_mask": np.ones((1, 77), dtype=np.int64),
    }
```

**Why Callable:** the function is stored in `ModelSpec.build_inputs` but only **called** when that model is about to run. This means memory for previous models can be freed.

---

## 4. Dictionary comprehension — filtering feeds

```python
# ORT session may not need all inputs we generated
graph_input_names = {inp.name for inp in session.get_inputs()}
filtered_feed = {k: v for k, v in feed.items() if k in graph_input_names}
```

This one-liner builds a new dict containing only the keys that the ORT session expects.

---

## 5. `time.perf_counter()` timing loop

```python
import time

latencies_ms = []
for i in range(args.runs):
    t0 = time.perf_counter()
    session.run(None, feed)
    dt = (time.perf_counter() - t0) * 1000.0
    latencies_ms.append(dt)
```

---

## 6. `subprocess.run()` — telemetry capture

```python
import subprocess

# Capture dmesg
result = subprocess.run(
    ["dmesg", "--level=warn,err"],
    capture_output=True, text=True, timeout=10,
)
dmesg_output = result.stdout

# Capture rocm-smi
result = subprocess.run(
    ["rocm-smi", "--showtemp", "--showclocks"],
    capture_output=True, text=True,
)
gpu_snapshot = result.stdout
```

---

## 7. JSON results I/O

```python
import json

results = {
    "model": spec.name,
    "mean_ms": np.mean(latencies_ms),
    "median_ms": np.median(latencies_ms),
    "std_ms": np.std(latencies_ms),
    "per_run_ms": latencies_ms,
}

with open(f"results/{spec.name}.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 8. `os.makedirs` / `pathlib.Path`

```python
import os
os.makedirs("results/2024", exist_ok=True)

from pathlib import Path
p = Path("results") / "2024" / f"{spec.name}.json"
p.parent.mkdir(parents=True, exist_ok=True)
```

---

## 9. `os.sched_setaffinity` — CPU pinning from Python

```python
import os
os.sched_setaffinity(0, set(range(8)))  # pin to cores 0-7
```

Shell equivalents:

```bash
taskset -c 0-7 python benchmark.py
numactl --cpunodebind=0 --membind=0 python benchmark.py
```

**Why:** reduces cross-core migration jitter during benchmarks. Especially helps models with CPU-bound preprocessing (XTTS, Whisper).

---

## 10. `KernelTelemetry` style dataclasses

```python
@dataclass
class KernelTelemetry:
    dmesg_before: str
    dmesg_after: str
    gpu_temp_before: float
    gpu_temp_after: float
    svm_message_count: int
    per_run_latencies: List[float]
```

These are **audit trails**: structured records of what the system was doing during each model's benchmark.

---

## 11. Linux system file reads from Python

```python
# Read sysctl
with open("/proc/sys/vm/swappiness") as f:
    swappiness = int(f.read().strip())

# Read THP mode
with open("/sys/kernel/mm/transparent_hugepage/enabled") as f:
    thp_mode = f.read().strip()

# Read CPU info
with open("/proc/cpuinfo") as f:
    cpuinfo = f.read()

# Read memory info
with open("/proc/meminfo") as f:
    meminfo = f.read()
```

---

## Module 15 checklist

- [ ] Can trace a CLI flag from argparse → session creation → timing loop
- [ ] Can write a `build_*_inputs()` function for a new model
- [ ] Can read a dict comprehension that filters a feed
- [ ] Can use subprocess to capture dmesg and rocm-smi output
- [ ] Can explain CPU affinity and how to set it (Python + shell)

**Next:** `16-env-libraries-storage-io.md`
