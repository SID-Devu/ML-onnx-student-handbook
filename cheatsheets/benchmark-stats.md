# Cheatsheet: Benchmarking & statistics quick reference

## Your benchmark configuration

| Setting | Value | Why |
|---------|-------|-----|
| `--warmup` | 3 | Discard compilation/caching runs |
| `--runs` | 3 (min) | Basic statistics; 10+ for production |
| `--cooldown` | 120s | Let GPU return to baseline temp |
| `--ep` | migraphx | MIGraphXExecutionProvider |

## Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| Mean | sum / count | Average (sensitive to outliers) |
| Median | middle value (sorted) | Robust central tendency |
| Std | spread around mean | Consistency measure |
| Min | smallest | Best case |
| Max | largest | Worst case / jitter indicator |
| P95 | 95th percentile | "95% of runs are faster" |
| P99 | 99th percentile | Tail latency, SLA-relevant |
| Throughput | 1000 / median_ms (FPS) | Capacity planning |

**Minimum report:** median + mean + std.
**Production report:** add P95, P99, per-run latencies.

## Timing code

```python
import time
t0 = time.perf_counter()
session.run(None, feed)
dt_ms = (time.perf_counter() - t0) * 1000.0
```

## Statistics code

```python
import numpy as np
arr = np.array(latencies_ms)
print(f"Mean:   {arr.mean():.3f} ms")
print(f"Median: {np.median(arr):.3f} ms")
print(f"Std:    {arr.std():.3f} ms")
print(f"Min:    {arr.min():.3f} ms")
print(f"Max:    {arr.max():.3f} ms")
print(f"P95:    {np.percentile(arr, 95):.3f} ms")
print(f"P99:    {np.percentile(arr, 99):.3f} ms")
print(f"FPS:    {1000.0 / np.median(arr):.1f}")
```

## Latency types

| Metric | What it measures | Models |
|--------|-----------------|--------|
| ORT run latency | `session.run()` only | All |
| End-to-end | Preprocess + ORT + postprocess | Real apps |
| TTFT | Time to first generated token | LLMs (Qwen3, LLaMA) |
| Token throughput | Tokens / second | LLMs |

## Jitter checklist (when results are noisy)

- [ ] Thermal: `rocm-smi --showclocks` — sclk dropped?
- [ ] Swap: `vmstat 1` — si/so non-zero?
- [ ] Background: `top` — browser/updates running?
- [ ] Page cache: cold boot vs warm cache?
- [ ] Power: CPU governor = performance?
- [ ] Xen: VM exit scheduling overhead?
- [ ] THP compaction: `always` → `madvise`?
- [ ] Page migration: first-touch XNACK stalls?

## Reproducibility footer (JSON)

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
        "warmup": 3, "runs": 10, "cooldown_s": 120,
        "providers": ["MIGraphXExecutionProvider"]
    },
    "model": {
        "path": "yolo/yolov12m.onnx",
        "sha256": "abc123...",
        "input_shape": [1, 3, 640, 640]
    },
    "results": {
        "per_run_ms": [12.1, 12.3, 12.2],
        "mean_ms": 12.2, "median_ms": 12.2, "std_ms": 0.08,
        "p95_ms": 12.35, "throughput_fps": 81.97
    },
    "thermal": {
        "gpu_temp_before_c": 42, "gpu_temp_after_c": 51,
        "sclk_mhz": 2900, "throttled": false
    }
}
```

## Bad practices (benchmark theater)

- Cherry-picking fastest run
- Different cooldown per model
- Different thermal states
- Comparing Xen vs bare metal without disclosure
- Reporting only mean
- Claiming INT8 speedup without accuracy check

## Good practices

- Report distribution (median + mean + std + P95)
- Report per-run latencies
- Log thermal state before/after
- Disclose full environment
- Same conditions for all comparisons
- Validate accuracy for quantized models

## Recommended optimization order

| Step | What | Time | Gain |
|------|------|------|------|
| 1 | onnxsim | 5-10 min | 5-15% fewer ops |
| 2 | Static shape pinning | 10 min | 10-30% |
| 3 | Exhaustive tune | 1-3 hrs (once) | 10-30% |
| 4 | INT8 (vision) | 30 min/model | 1.5-2x |
| 5 | Fused attention | 15 min/model | 10-25% |
| 6 | Thread pinning | 2 min | 5-10% less jitter |
