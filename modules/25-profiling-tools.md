# Module 25 — Profiling tools (beyond timing)

Module 14 covers latency measurement. This module covers deeper profiling to find **where** time is spent.

---

## 1. `perf stat` — CPU performance counters

```bash
perf stat python benchmark_cooldown.py --warmup 1 --runs 3
```

Shows: instructions, cycles, cache misses, branch misses, IPC (instructions per cycle).

**When to use:** CPU-bound preprocessing (XTTS mel spectrogram, Whisper audio processing).

---

## 2. `perf record` + `perf report` — CPU profiling

```bash
perf record -g python benchmark_cooldown.py --warmup 1 --runs 3
perf report                     # interactive call-tree viewer
```

Can generate **flame graphs** showing where CPU time is spent.

---

## 3. `rocprof` — AMD GPU kernel profiling

```bash
rocprof --stats python benchmark_cooldown.py --warmup 1 --runs 1
# Outputs: results.stats.csv with per-kernel execution times
```

Shows execution time for **each GPU kernel** MIGraphX compiled. Identifies which ops are slowest.

```bash
rocprof --timestamp on python benchmark.py
# Adds timestamps to see kernel execution timeline
```

**When to use:** "Exhaustive tuning improved 10% — which kernels got faster?"

---

## 4. `py-spy` — Python profiler (no code changes)

```bash
pip install py-spy

# Profile a running process
py-spy top --pid <PID>

# Record and generate flame graph
py-spy record -o profile.svg -- python benchmark_cooldown.py
# Open profile.svg in browser
```

**No code changes needed.** Attaches to running Python process.

---

## 5. `memory_profiler` — Python memory line-by-line

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

## 6. When to use which tool

| Question | Tool |
|----------|------|
| "Is preprocessing CPU-bound?" | `perf stat` |
| "Which Python function is slow?" | `py-spy` |
| "Which GPU kernel is slow?" | `rocprof --stats` |
| "Where does memory spike?" | `memory_profiler` |
| "Is my loop hot?" | `perf record` + flame graph |

---

## Module 25 checklist

- [ ] Can run `perf stat` on a benchmark and read IPC
- [ ] Can use `rocprof --stats` to find slow GPU kernels
- [ ] Can attach `py-spy` to a running Python process
- [ ] Know when to use CPU profiling vs GPU profiling

**End of new modules.**
