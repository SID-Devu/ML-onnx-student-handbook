# Module 25 — Profiling tools (beyond timing)

Module 14 covers latency measurement. This module covers deeper profiling to find **where** time is spent.

**Permissions note:** Many profiling tools require elevated privileges. If you get "permission denied," try `sudo` or adjust kernel settings (e.g. `sudo sysctl kernel.perf_event_paranoid=-1` for perf, `sudo` for py-spy on hardened systems).

---

## 1. `perf stat` — CPU performance counters

```bash
sudo perf stat python benchmark_cooldown.py --warmup 1 --runs 3
```

Shows: instructions, cycles, cache misses, branch misses, IPC (instructions per cycle).

**When to use:** CPU-bound preprocessing (XTTS mel spectrogram, Whisper audio processing).

---

## 2. `perf record` + `perf report` — CPU profiling

```bash
sudo perf record -g python benchmark_cooldown.py --warmup 1 --runs 3
perf report                     # interactive call-tree viewer
```

Can generate **flame graphs** showing where CPU time is spent.

---

## 3. `rocprofv3` — AMD GPU kernel profiling (ROCm 6.x+)

`rocprofv3` is the modern AMD profiling tool, replacing the older `rocprof`. It supports multiple output formats and tracing modes.

### Kernel trace + statistics (CSV output)

```bash
rocprofv3 --kernel-trace --hip-trace --stats \
  -d profiling/openvla_stats \
  -- python benchmark_cooldown.py --ep migraphx --models openvla_full --warmup 3 --runs 3
```

**Output files** (in the `-d` directory):
- `*_kernel_stats.csv` — per-kernel aggregated times (calls, total, avg, min, max)
- `*_kernel_trace.csv` — every kernel dispatch with timestamps, VGPR/SGPR counts, grid sizes
- `*_hip_api_stats.csv` — HIP API call statistics
- `*_hip_api_trace.csv` — every HIP API call with timestamps
- `*_domain_stats.csv` — high-level breakdown (GPU kernel time vs HIP API time)

### System trace (Perfetto output)

```bash
rocprofv3 --sys-trace -f pftrace \
  -d profiling/perfetto_output \
  -- python benchmark_cooldown.py --ep migraphx --models openvla_full --warmup 3 --runs 3
```

Generates a `.pftrace` file viewable in [Perfetto UI](https://ui.perfetto.dev) — drag and drop the file for an interactive timeline of kernel dispatches, memory copies, and API calls.

### HSA + memory traces (for SVM/XNACK analysis)

```bash
HSA_XNACK=1 rocprofv3 --kernel-trace --hsa-trace --memory-copy-trace \
  -d profiling/openvla_xnack1 \
  -- python benchmark_cooldown.py --ep migraphx --models openvla_full --warmup 3 --runs 3
```

### Key `rocprofv3` flags

| Flag | What it captures |
|------|-----------------|
| `--kernel-trace` | GPU kernel dispatches with timestamps, register counts, grid/block sizes |
| `--hsa-trace` | Low-level HSA runtime API calls |
| `--hip-trace` | HIP API calls (hipMalloc, hipMemcpy, hipLaunchKernel, etc.) |
| `--memory-copy-trace` | Host-to-device / device-to-host memory transfers |
| `--stats` | Aggregated statistics per kernel name |
| `--sys-trace -f pftrace` | Combined system trace as Perfetto-compatible `.pftrace` |
| `-d <dir>` | Output directory |

**When to use:** "Which GPU kernels are slow?", "Is memory copy a bottleneck?", "What does the execution timeline look like?"

### Legacy `rocprof` (ROCm < 6.x)

```bash
rocprof --stats python benchmark_cooldown.py --warmup 1 --runs 1
```

Note: `rocprof` flags vary significantly across ROCm versions. Prefer `rocprofv3` on ROCm 6.x+.

---

## 4. `uftrace` — function-level C/C++ tracing

`uftrace` traces function calls in native libraries (libmigraphx, libamdhip64, librocblas). Useful for understanding the call chain from ORT → MIGraphX → HIP → GPU.

```bash
sudo apt install uftrace

uftrace record --force -D 5 \
  --lib-path /opt/rocm/lib \
  -l libmigraphx_c.so -l libamdhip64.so -l librocblas.so \
  -- python benchmark_cooldown.py --ep migraphx --models openvla_full --warmup 1 --runs 1

uftrace report          # timing summary per function
uftrace graph           # call graph
uftrace dump --chrome   # Perfetto/Chrome-compatible JSON trace
```

The `benchmark_cooldown.py` script has built-in `--uftrace` support that automates this.

**When to use:** "What C++ functions does MIGraphX call?", "How deep is the call stack from ORT to GPU dispatch?"

---

## 5. Perfetto UI — interactive timeline viewer

[Perfetto UI](https://ui.perfetto.dev) is a web-based trace viewer for `.pftrace` files from `rocprofv3 --sys-trace` and Chrome JSON traces from `uftrace dump --chrome`.

1. Open https://ui.perfetto.dev
2. Drag and drop your `.pftrace` or `.json` file
3. Zoom/pan the timeline to inspect kernel dispatches, memory copies, API calls

**When to use:** Visual analysis of GPU execution timelines, identifying gaps/stalls between kernel dispatches.

---

## 6. `py-spy` — Python profiler (no code changes)

```bash
pip install py-spy

# Profile a running process (may need sudo on hardened systems)
sudo py-spy top --pid <PID>

# Record and generate flame graph
sudo py-spy record -o profile.svg -- python benchmark_cooldown.py
# Open profile.svg in browser
```

**No code changes needed.** Attaches to running Python process.

---

## 7. `memory_profiler` — Python memory line-by-line

```bash
pip install memory_profiler

# Add decorator to function you want to profile
# @profile
# def load_model():
#     ...

python -m memory_profiler my_script.py
```

Shows memory usage at each line. Useful for tracking where memory spikes during model loading.

---

## 8. When to use which tool

| Question | Tool |
|----------|------|
| "Is preprocessing CPU-bound?" | `perf stat` |
| "Which Python function is slow?" | `py-spy` |
| "Which GPU kernel is slow?" | `rocprofv3 --kernel-trace --stats` |
| "What does the execution timeline look like?" | `rocprofv3 --sys-trace` + Perfetto UI |
| "What C++ functions are called?" | `uftrace` |
| "Where does memory spike?" | `memory_profiler` |
| "Is my loop hot?" | `perf record` + flame graph |

---

## Module 25 checklist

- [ ] Can run `perf stat` on a benchmark and read IPC
- [ ] Can use `rocprofv3 --kernel-trace --stats` to find slow GPU kernels
- [ ] Can generate a Perfetto trace with `rocprofv3 --sys-trace -f pftrace` and view it
- [ ] Can run `uftrace` on native libraries to see function-level call chains
- [ ] Can attach `py-spy` to a running Python process
- [ ] Know when to use CPU profiling vs GPU profiling vs function tracing

**End of handbook modules.**
