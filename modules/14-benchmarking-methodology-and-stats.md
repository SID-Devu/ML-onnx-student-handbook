# Module 14 — Benchmarking methodology, latency metrics, statistics (full depth)

---

## 1. Define what you're measuring FIRST

| Metric | What it measures | When to use |
|--------|-----------------|-------------|
| **ORT run latency** | `session.run()` time only | Pure model performance |
| **End-to-end latency** | Preprocess → ORT run → postprocess | Real-world application speed |
| **TTFT** | Time to first token (LLMs) | User-perceived responsiveness |
| **Throughput (FPS)** | Inferences per second (`1000 / latency_ms` for batch=1) | Capacity planning |
| **Token throughput** | Tokens per second (LLMs) | LLM generation speed |

---

## 2. Your benchmark configuration

| Setting | Your value | Why |
|---------|-----------|-----|
| `--warmup` | **3** | Discard first 3 runs (compilation, caching, page migration) |
| `--runs` | **3** | Minimum for basic statistics (more is better) |
| `--cooldown` | **120** seconds | Let GPU cool between models to prevent thermal coupling |
| `--ep` | `migraphx` | MIGraphXExecutionProvider |

---

## 3. Warmup — not "cheating"

First N runs are slower because of:

- **Kernel compilation** (MIGraphX compiles ONNX ops → GPU kernels)
- **Cache population** (compiled kernels, page table entries)
- **Page migration** (with XNACK=1, pages migrate on first GPU access)
- **Allocator warmup** (memory pools established)

Discarding warmup isolates **steady-state** performance, which is what matters for production.

---

## 4. Cooldown — why 120 seconds

GPUs accumulate heat during inference. Heat → thermal throttling → clocks drop → latency increases.

Without cooldown between models:
- Model A heats GPU to 85°C
- Model B starts immediately at 85°C, throttles to lower clocks
- Model B's results are unfairly slow

**120 seconds** gives the GPU time to return to baseline temperature.

**Verify:** check `rocm-smi --showtemp` before each model. If temperature hasn't dropped below threshold, wait longer.

---

## 5. Timer choice

```python
import time
t0 = time.perf_counter()
session.run(None, feed)
dt_ms = (time.perf_counter() - t0) * 1000.0
```

`time.perf_counter()` is:
- **Monotonic** (never goes backwards)
- **High resolution** (sub-microsecond on modern systems)
- **Elapsed real time** (not CPU time — use `time.process_time()` for CPU-only measurement)

**Async GPU caveat:** if GPU work is asynchronous, wall time may undercount. ORT session.run() typically synchronizes at return for MIGraphX EP, but verify.

---

## 6. Per-run latency — not just averages

You specifically requested per-run latency in your benchmarks. This is critical because:

```
Run 1: 12.1 ms
Run 2: 12.3 ms
Run 3: 45.2 ms  ← outlier! Swap hit? Thermal throttle? Background process?
```

If you only reported mean (23.2 ms), you'd hide that runs 1-2 were consistent and run 3 had an issue.

---

## 7. Statistics

| Statistic | Formula (conceptual) | Use |
|-----------|---------------------|-----|
| **Mean** | Sum / count | Average; sensitive to outliers |
| **Median** | Middle value when sorted | Robust central tendency; use when there are outliers |
| **Std (standard deviation)** | Spread around mean | Shows consistency |
| **Min** | Smallest value | Best-case scenario |
| **Max** | Largest value | Worst case; often caused by jitter |
| **P95** | 95th percentile | 95% of runs complete within this latency (only ~5% are slower) |
| **P99** | 99th percentile | 99% within this; tail latency, SLA-relevant |

**Report at minimum:** median + mean + std. For production: add P95/P99.

---

## 8. Throughput calculation

```
Throughput (FPS) = 1000 / median_latency_ms
```

For batch > 1: `Throughput = batch_size * 1000 / median_latency_ms`

---

## 9. Jitter sources checklist

When results are noisy (high std, or max >> median), check:

- [ ] **Thermal throttling** — `rocm-smi --showclocks` during inference. If sclk drops, clocks are throttling.
- [ ] **Swap activity** — check `vmstat 1` during run. If `si`/`so` columns show activity, pages are swapping.
- [ ] **Background processes** — `top` / `htop`. Browsers, updates, etc. steal CPU/memory.
- [ ] **Filesystem cache cold/hot** — first run after boot reads from disk; subsequent from cache.
- [ ] **Power governor** — verify CPU is in performance mode (or Xen controls it).
- [ ] **Hypervisor scheduling** — if on Xen, VM exits add jitter.
- [ ] **Memory compaction** — if THP is `always`, compaction events cause stalls.
- [ ] **Page migration** — with XNACK=1, first-touch pages cause brief stalls.

---

## 10. Statistical significance (practical)

You don't need formal hypothesis tests for daily work, but you should:

- Use **enough runs** (3 is minimum; 10+ is better for noisy systems)
- **Control temperature** (cooldown between models)
- **Change one variable at a time** (don't change EP + quantization + shapes simultaneously)
- **Compare median to median**, not mean to mean (more robust)

---

## 11. Logging reproducibility artifacts

Every benchmark JSON should record:

```json
{
    "environment": {
        "kernel_version": "6.18.0+",
        "rocm_version": "6.3.0",
        "ort_version": "1.21.0+custom",
        "hsa_xnack": "1",
        "gtt_size_mb": 28672
    },
    "config": {
        "warmup": 3,
        "runs": 3,
        "cooldown_s": 120,
        "providers": ["MIGraphXExecutionProvider"]
    },
    "model": {
        "path": "yolo/yolov12m.onnx",
        "sha256": "abc123...",
        "input_shape": [1, 3, 640, 640]
    },
    "results": {
        "per_run_ms": [12.1, 12.3, 12.2, ...],
        "mean_ms": 12.2,
        "median_ms": 12.2,
        "std_ms": 0.08,
        "p95_ms": 12.35,
        "throughput_fps": 81.97
    },
    "thermal": {
        "gpu_temp_before_c": 42,
        "gpu_temp_after_c": 51,
        "sclk_mhz": 2900,
        "throttled": false
    }
}
```

---

## Module 14 checklist

- [ ] Explain why warmup is necessary (not "cheating")
- [ ] Explain why mean alone is insufficient on a laptop-class system
- [ ] List 5 non-ML reasons for latency spikes
- [ ] Explain TTFT and when it matters (LLMs)
- [ ] Can calculate throughput from median latency
- [ ] Know what fields a reproducibility footer should contain

**Next:** `15-python-patterns-in-benchmark-code.md`
