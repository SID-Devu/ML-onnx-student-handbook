# Module 01 — Python for ML / benchmark scripts (full depth)

You need a **specific** subset of Python — the parts that appear in scripts like `benchmark_cooldown.py` and `inference.py`. This module covers every Python concept you will encounter in that codebase.

---

## 1. Variables and types

```python
name = "yolov12m"          # str  — model name
batch = 1                   # int  — batch size
latency = 12.34             # float — milliseconds
fp16 = True                 # bool — is FP16 enabled?
shape = (1, 3, 640, 640)   # tuple — immutable ordered collection
dims = [1, 3, 640, 640]    # list  — mutable ordered collection
```

**`dict`** — key → value map. This is the most important type in your codebase because ORT inference feeds are dicts:

```python
feed = {
    "images": np_array,         # input name → numpy array
    "attention_mask": mask_array,
}
```

**`None`** — "no value" sentinel. You will see it in optional fields:

```python
providers: Optional[List[str]] = None
```

---

## 2. Control flow

```python
# if / elif / else
if has_dynamic:
    print(f"DYNAMIC: {path}")
elif shape[0] == 1:
    print("Static batch=1")
else:
    print("Unknown")

# for loop — iterate over models
for model in MODELS:
    run_benchmark(model)

# for with range — warmup loop
for i in range(warmup):
    session.run(None, feed)

# while — rare, but appears in calibration readers
while self.index < self.num_samples:
    yield self.get_next()
```

---

## 3. Functions (`def`)

Every `build_*_inputs()` function in your benchmark suite is a function:

```python
def build_yolo_inputs():
    """Create random YOLO input tensor."""
    return {
        "images": np.random.randn(1, 3, 640, 640).astype(np.float32)
    }
```

**Default arguments:**

```python
def benchmark(model_path, warmup=3, runs=3, cooldown=120):
    ...
```

**Keyword arguments when calling:**

```python
benchmark("yolo/yolov12m.onnx", warmup=5, runs=10)
```

**Return values:**

```python
def measure_latency(session, feed):
    t0 = time.perf_counter()
    session.run(None, feed)
    return (time.perf_counter() - t0) * 1000  # returns float (ms)
```

---

## 4. Classes (`class`)

Basic class:

```python
class Box:
    def __init__(self, w, h):
        self.w = w          # instance attribute
        self.h = h

    def area(self):         # method
        return self.w * self.h
```

- **`__init__`**: constructor — called when you do `Box(10, 20)`
- **`self`**: the instance being constructed / called on
- **Methods**: functions attached to objects

---

## 5. `@dataclass` — the pattern used in your codebase

**This is critical.** Your `ModelSpec` is a dataclass:

```python
from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, List

@dataclass
class ModelSpec:
    name: str                                          # model name
    onnx_path: str                                     # path to .onnx file
    build_inputs: Callable[[], Dict[str, "np.ndarray"]] # function that creates inputs
    dim_overrides: Dict[str, int] = field(default_factory=dict)  # pin dynamic dims
    providers: Optional[List[str]] = None              # EP override
    ort_migraphx_fp16: bool = False                    # enable FP16 on MIGraphX
```

**What each part means:**

| Syntax | Meaning |
|--------|---------|
| `@dataclass` | Auto-generates `__init__`, `__repr__`, etc. |
| `name: str` | Type hint — `name` should be a string |
| `Callable[[], Dict[str, "np.ndarray"]]` | A function that takes no args and returns a dict of name→array |
| `field(default_factory=dict)` | Default value is a **new empty dict** each time (avoids shared mutable default bug) |
| `Optional[List[str]]` | Either a list of strings or `None` |

**Why `field(default_factory=dict)` instead of `= {}`:**

```python
# BAD — all instances share the SAME dict object
dim_overrides: Dict[str, int] = {}

# GOOD — each instance gets a fresh dict
dim_overrides: Dict[str, int] = field(default_factory=dict)
```

---

## 6. Type hints (used everywhere in your code)

```python
name: str                           # string
batch: int                          # integer
latency: float                      # float
fp16: bool                          # boolean
shape: List[int]                    # list of ints
feed: Dict[str, np.ndarray]        # dict mapping string → numpy array
model: Optional[str] = None        # string or None
Union[int, str]                    # either int or string
```

These are **hints only** — Python won't stop you passing wrong types. ORT will fail at runtime instead.

---

## 7. Imports and modules

```python
import numpy as np                  # import with alias
import onnxruntime as ort           # import with alias
from pathlib import Path            # import specific name
from dataclasses import dataclass, field  # import multiple names
import json                         # standard library
import time                         # standard library
import os                           # standard library
import subprocess                   # for running shell commands
```

**Your codebase pattern:**

```python
from inference import MODELS         # import model specs from another file
```

**Common error:** `ModuleNotFoundError: No module named 'onnxruntime'` → means you need `pip install onnxruntime` or you're in the wrong virtualenv.

---

## 8. `*args` and `**kwargs` — variable arguments

```python
def wrapper(*args, **kwargs):
    # args = tuple of positional arguments
    # kwargs = dict of keyword arguments
    return original_function(*args, **kwargs)
```

You see this in wrapper functions and decorators. It means "pass through whatever arguments were given."

---

## 9. `lambda` — anonymous functions

```python
# Named function
def build_inputs():
    return {"x": np.zeros((1, 3, 224, 224), dtype=np.float32)}

# Same thing as lambda
build_inputs = lambda: {"x": np.zeros((1, 3, 224, 224), dtype=np.float32)}
```

Used for tiny one-line input builders in model spec tables. Prefer named functions when debugging.

---

## 10. Dictionary comprehension

```python
# Filter a feed dict to only include keys the graph expects
graph_names = {"images", "attention_mask"}
filtered = {k: v for k, v in feed.items() if k in graph_names}
```

This is a one-liner `for` loop that builds a new dict. You'll see this pattern in input builders that need to match ORT's expected input names.

---

## 11. `Callable` type hint

```python
build_inputs: Callable[[], Dict[str, np.ndarray]]
```

This means: `build_inputs` is a **function** (callable). The `[]` means it takes **no arguments**. The `Dict[str, np.ndarray]` is the return type.

**Why it's useful:** ModelSpec stores a function reference, not the data itself. The function is called only when needed — so you can free memory between models.

---

## 12. f-strings (formatted string literals)

```python
name = "yolo"
latency = 12.345
print(f"Model {name} took {latency:.2f}ms")  # "Model yolo took 12.35ms"
```

**Format specifiers:**

| Syntax | Output | Meaning |
|--------|--------|---------|
| `{latency}` | `12.345` | Default |
| `{latency:.2f}` | `12.35` | 2 decimal places |
| `{latency:.4f}` | `12.3450` | 4 decimal places |
| `{count:05d}` | `00003` | Zero-padded to 5 digits |
| `{size / 1024:.1f}` | `2.5` | Expression inside f-string |

---

## 13. File I/O

**Text files:**

```python
# Read
with open("results.txt", "r") as f:
    content = f.read()

# Write
with open("results.txt", "w") as f:
    f.write(f"Latency: {lat_ms:.2f}ms\n")
```

**JSON (benchmark results):**

```python
import json

# Write results
results = {"model": "yolo", "mean_ms": 12.34, "runs": [12.1, 12.3, 12.5]}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# Read results
with open("results.json", "r") as f:
    data = json.load(f)
print(data["mean_ms"])  # 12.34
```

---

## 14. Context managers (`with`)

```python
with open("file.txt") as f:
    data = f.read()
# file is automatically closed here, even if an exception occurred
```

The `with` statement guarantees cleanup. You'll see this for files, and sometimes for ORT sessions or GPU contexts.

---

## 15. Error handling (`try` / `except`)

```python
try:
    session = ort.InferenceSession(model_path, providers=["MIGraphXExecutionProvider"])
except Exception as e:
    print(f"Failed to load {model_path}: {e}")
    # fall back to CPU
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
```

**In benchmarks:** never blanket-catch without logging — you'll hide real bugs like missing EPs or wrong shapes.

---

## 16. `argparse` — command-line argument parsing

```python
import argparse

parser = argparse.ArgumentParser(description="Benchmark ONNX models")
parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
parser.add_argument("--runs", type=int, default=3, help="Timed runs")
parser.add_argument("--cooldown", type=int, default=120, help="Cooldown seconds")
parser.add_argument("--models", nargs="+", help="Model names to run")
parser.add_argument("--ep", choices=["migraphx", "cpu"], default="migraphx")
args = parser.parse_args()

print(args.warmup)   # 3
print(args.models)   # ["yolo", "clip"]
```

**Skill to build:** trace from `parse_args()` through the code to where `args.warmup` actually controls the warmup loop.

---

## 17. `subprocess.run()` — running shell commands from Python

```python
import subprocess

# Capture dmesg output
result = subprocess.run(
    ["dmesg", "--level=warn,err"],
    capture_output=True, text=True, timeout=10
)
print(result.stdout)

# Capture rocm-smi
result = subprocess.run(
    ["rocm-smi", "--showtemp", "--showclocks"],
    capture_output=True, text=True
)
print(result.stdout)
```

**Used for:** telemetry (dmesg capture, rocm-smi snapshots, sysfs reads) alongside inference.

**Caveat:** parsing command output is brittle across software versions; but useful for internal tools.

---

## 18. `time.perf_counter()` — high-resolution timer

```python
import time

t0 = time.perf_counter()
session.run(None, feed)          # the work being timed
dt_ms = (time.perf_counter() - t0) * 1000.0

print(f"Latency: {dt_ms:.2f} ms")
```

`perf_counter()` is **monotonic** and **high resolution** — the right choice for benchmarking.

---

## 19. `os` and `pathlib` — file paths

```python
import os
os.makedirs("results/2024", exist_ok=True)    # create nested dirs
os.path.exists("model.onnx")                   # check if file exists
os.path.join("yolo", "yolov12m.onnx")          # build path string

from pathlib import Path
p = Path("/home/sudhdevu/R1models/yolo/yolov12m.onnx")
p.exists()            # True/False
p.stat().st_size      # file size in bytes
p.stem                # "yolov12m"
p.suffix              # ".onnx"
p.parent              # Path("/home/sudhdevu/R1models/yolo")
```

---

## 20. `pip install` and virtual environments

```bash
# Create a virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install packages
pip install numpy onnx onnxruntime onnxsim

# List installed
pip list

# Deactivate
deactivate
```

**Critical concept:** your ORT Python wheel must match the **native** `.so` libraries on the system (ROCm version, MIGraphX version). A `pip install onnxruntime` from PyPI does **not** include MIGraphX EP — that comes from your custom build.

---

## 21. Debugging Python — reading errors and finding bugs

### Reading a traceback (read from bottom up)

```
Traceback (most recent call last):
  File "benchmark.py", line 45, in run_model
    outputs = session.run(None, feed)
  File "onnxruntime/capi/onnxruntime_inference_collection.py", line 220, in run
    return self._sess.run(output_names, input_feed, run_options)
RuntimeError: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: images
```

**Read from the bottom:**
1. `RuntimeError` — the actual error
2. Line above — where ORT raised it
3. Lines above that — your code's call stack. `line 45 in run_model` = your bug location

### Common error types

| Error | What it means | Typical cause |
|-------|--------------|---------------|
| `TypeError` | Wrong argument type | Passed list where numpy array expected |
| `ValueError` | Right type, wrong value | Shape mismatch |
| `KeyError` | Dictionary key doesn't exist | Wrong input name in feed dict |
| `AttributeError` | Object doesn't have that property | Typo in method name |
| `RuntimeError` | Catch-all runtime failure | ORT/MIGraphX errors |
| `ModuleNotFoundError` | `import` can't find the package | Forgot `pip install` or wrong venv |

### `assert` — fail fast with clear message

```python
assert image.shape == (1, 3, 640, 640), f"Bad shape: {image.shape}"
assert image.dtype == np.float32, f"Bad dtype: {image.dtype}"
```

Crashes immediately with your message instead of cryptic errors deep inside ORT.

### `print()` debugging — trace data flow

```python
print(f"shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min():.4f}, max={tensor.max():.4f}")
```

Insert before `session.run()` to verify inputs are correct.

### `type()` and `isinstance()`

```python
print(type(data))               # <class 'numpy.ndarray'> or <class 'list'>?
assert isinstance(data, np.ndarray), f"Expected ndarray, got {type(data)}"
```

### `pdb` / `breakpoint()` — interactive debugger

```python
# Add this line where you want to pause
breakpoint()
# Python drops into interactive prompt: inspect variables, step through code
# Commands: n (next line), c (continue), p variable (print), q (quit)
```

---

## Module 01 checklist

- [ ] Can read a `@dataclass` definition and explain `field(default_factory=dict)`
- [ ] Can write a function that returns a `Dict[str, np.ndarray]` feed dict
- [ ] Can trace an `argparse` flag from CLI through to where it controls behavior
- [ ] Can use `subprocess.run()` to capture `rocm-smi` output
- [ ] Can explain what `Callable[[], Dict[str, np.ndarray]]` means
- [ ] Can read a dictionary comprehension
- [ ] Can load/save JSON results with `json.dump` / `json.load`
- [ ] Can time a code block with `time.perf_counter()`
- [ ] Can read a Python traceback from bottom up and identify the error
- [ ] Can use `assert` to validate tensor shapes before inference
- [ ] Can name 5 common Python error types and what each means

**Next:** `02-numpy-tensors-and-shapes.md`
