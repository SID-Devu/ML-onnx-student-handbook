# Exercises

Do these in order. **Edit paths** to point at ONNX files that exist on your machine.

## Exercise 1 — Python + NumPy warm-up

**Goal:** Create `exercises/solutions/e1_tensor.py` that:

1. Builds `x` with shape `(1, 3, 224, 224)`, dtype `float32`.
2. Prints `x.shape`, `x.dtype`, and `x.strides`.
3. Prints a slice `x[0, :, 0:1, 0:1].shape`.

**Check:** script runs with no exceptions.

---

## Exercise 2 — Inspect ONNX without ORT

**Goal:** Write `exercises/solutions/e2_onnx_inspect.py` that loads an ONNX file and prints:

- opset version
- count of graph nodes
- each graph input: name + shape dimension list (show `dim_param` or `dim_value` per axis)
- whether external data is likely (heuristic: file ends with `.data` sibling exists)

**Check:** output matches what Netron shows for the same model.

---

## Exercise 3 — ORT run (CPU first)

**Goal:** Write `exercises/solutions/e3_ort_cpu_run.py` that:

1. Creates `InferenceSession` with `CPUExecutionProvider` only.
2. Allocates random input **matching** the first input binding shape (replace dynamic dims with 1 for this drill).
3. Runs `session.run` and prints output shapes.

**Check:** no shape errors.

---

## Exercise 4 — Dynamic shape audit

**Goal:** Run `scripts/examples/check_dynamic_shapes.py` after editing `MODEL_ROOTS` / paths.

**Deliverable:** a text file listing every input that still has dynamic axes in your local zoo.

---

## Exercise 5 — Baseline distribution

**Goal:** Write `exercises/solutions/e5_timings.py` that times the same ORT run 50 times (after 5 warmups) and prints mean/median/p95.

**Check:** you can explain one source of jitter you observed.

---

## Exercise 6 — sysctl scavenger hunt

**Goal:** On a Linux machine you are allowed to read:

1. Print current values for: `vm.swappiness`, `vm.max_map_count`, `vm.overcommit_memory`.
2. Print THP mode from `/sys/kernel/mm/transparent_hugepage/enabled`.

**Deliverable:** 5 sentences explaining what each means (use module 09/10).

---

## Exercise 7 — ROCm snapshot

**Goal:** Capture `rocm-smi --showtemp --showclocks --showmeminfo` before and after a 60s synthetic load (any GPU workload you have permission to run).

**Deliverable:** explain whether clocks look stable or throttled.

---

## Exercise 8 — Reproducibility footer

**Goal:** Define a JSON schema (fields + example values) for benchmark results including:

- kernel version (`uname -r`)
- ROCm version (if available)
- ORT version string
- provider list actually used
- model path + sha256
- warmup/runs/cooldown
- per-run latencies array

**Deliverable:** `exercises/solutions/e8_results_footer.json` example file.

---

### Solutions

Full solutions are in `exercises/solutions/`:

| File | Exercise |
|------|----------|
| `e1_tensor.py` | NumPy shape/dtype/strides + slicing |
| `e2_onnx_inspect.py` | ONNX graph inspection (CLI: pass model path) |
| `e3_ort_cpu_run.py` | ORT CPU inference with auto-resolved shapes |
| `e4_dynamic_audit.py` | Walk dirs, report dynamic vs static inputs |
| `e5_timings.py` | 50-run latency distribution with stats |
| `e6_sysctl_hunt.py` | Read + explain vm.* and THP from live system |
| `e7_rocm_snapshot.py` | rocm-smi before/after synthetic GPU load |
| `e8_results_footer.json` | Reproducibility JSON schema with example values |

**Recommendation:** try each exercise yourself first, then compare with the solution.
