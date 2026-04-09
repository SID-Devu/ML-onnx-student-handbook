# How to study from this handbook (and actually retain it)

## Principles

1. **One module at a time** — Finish reading, then do the matching exercise before opening the next module.
2. **Always connect to three questions** — *What is stored?* (weights/graph), *Where does it run?* (CPU/GPU EP), *What shape is the data?* (static vs dynamic).
3. **Write your own one-page summary** after each module in a personal notebook file. Teaching back in your own words is the fastest path to fluency.
4. **Do not skip NumPy (module 02)** even if you dislike math. ML "shapes" are not optional in your job.
5. **After each Python/ONNX chapter**, open `inference.py` or `benchmark_cooldown.py` and try to identify the concepts you just learned.

---

## Full 16-week learning plan with deliverables

| Phase | Week | Modules | Topics | Daily time | Deliverable |
|-------|------|---------|--------|-----------|-------------|
| 1 | 1-2 | 01 | Python basics + all Python specifics (Section J: dataclass, argparse, subprocess, Callable, lambda, dict comprehension, f-strings, context managers, *args/**kwargs) | 1-2 hrs | Can read `inference.py` and understand what each line does |
| 2 | 3 | 02 | NumPy (shapes, dtypes, NCHW, strides, slicing) | 1 hr | Can write a `build_*_inputs()` function from scratch |
| 3 | 4 | 03 | Neural network intuition (CNN, Transformer, Encoder/Decoder, inference vs training) | 1 hr | Can explain CNN vs Transformer to a colleague |
| 4 | 5 | 04 | ONNX format + Netron + onnxsim (graph, ops, opset, static/dynamic, external data) | 1 hr | Can open any model in Netron, read its inputs/outputs/ops |
| 5 | 6 | 05 | ONNX Runtime (sessions, EPs, provider options, warmup) | 1 hr | Can load a model, run inference, print output — without scripts |
| 6 | 7 | 06, 13 | Model export pipeline + all optimization steps (INT8 quantization, onnxsim, shape pinning, exhaustive tuning, attention fusion, thread pinning) | 1 hr | Can quantize a model and compare performance |
| 7 | 8 | 09 | Linux memory management (ALL sysctl params: swappiness=10, overcommit=1, max_map_count=2M, THP, MGLRU, compaction, kswapd, OOM, swap) | 1 hr | Can explain every `vm.*` parameter and its value |
| 8 | 9 | 10, 11 | Kernel boot params (GRUB, amdgpu.gttsize, iommu, cstates, MWAIT, split lock) + systemd (r1-gpu-perf.service) | 1 hr | Can read dmesg and explain what each kernel parameter does |
| 9 | 10 | 07, 08 | AMD GPU architecture + ROCm tools (VRAM, GTT, GART, HSA, XNACK, SVM, hipMalloc, hipMallocManaged, CUs, clocks, rocm-smi, rocminfo) | 1 hr | Can explain why HSA_XNACK=1 matters, what GTT is, why custom ORT build exists |
| 10 | 11 | 12 | Xen hypervisor (dom0, PVH, vCPUs, credit2, cpufreq, xl commands, bare metal vs virtualized) | 1 hr | Can explain dom0 vs bare metal benchmark differences |
| 11 | 12 | 14 | Benchmarking methodology + statistics (warmup, cooldown, TTFT, mean/median/std, p95/p99, jitter sources, thermal throttling) | 1 hr | Can design a fair benchmark and explain why cooldown matters |
| 12 | 13 | 15, 16 | Code patterns (ModelSpec, KernelTelemetry, argparse flow, subprocess telemetry) + environment (PATH, LD_LIBRARY_PATH, NVMe, fstab, page cache) | 1 hr | Can trace a CLI flag from argparse to session creation |
| 13 | 14 | 17, 18 | Model families (all 23 models mapped) + end-to-end optimization playbook (status table, remaining wins, execution order) | 1 hr | Can classify any model by domain and pick optimization strategy |
| 14 | 15 | 19, 20 | ORT debugging (graph partitioning, optimized dumps, telemetry) + safety/reproducibility (pickle risk, supply chain, honest benchmarking) | 1 hr | Can detect graph partitioning and verify model hashes |
| 15 | 16 | 21 | Building ORT from source (CMake, version skew, ldd, wheel installation) | 1 hr | Can explain the version tuple and diagnose library issues |
| 16 | 16+ | All | **End-to-end project:** export a new model, simplify it, pin shapes, tune, benchmark it, document results | 2 hrs | Can add a new model to the benchmark suite **solo** |

---

## Self-check questions (by phase)

After **module 02**: Given shape `(1, 3, 640, 640)`, explain each axis. What dtype does YOLO expect?

After **module 04**: What is the difference between a dimension that is the integer `1` and a dimension that is the string `"batch"` in ORT's reported input shape?

After **module 07**: Why might `hipMalloc` alone be insufficient for a multi-gigabyte model on an APU with 512 MB dedicated VRAM?

After **module 09**: What does `vm.overcommit_memory=1` do and why is it needed for large model loading?

After **module 12**: Why can't you directly compare Xen dom0 benchmarks with bare metal benchmarks?

After **module 14**: Why is the first timed run after a cold boot often misleading for GPU benchmarks?

After **module 18**: Why should static shape pinning happen BEFORE INT8 quantization for MIGraphX?

---

## If you are "very new"

Spend **two full weeks** on module 01 only, typing every example yourself. Speed without foundations produces brittle understanding.

Then spend a full week on module 02 (NumPy). These two alone will make you 10x more effective at your current job.

---

## Honesty about "match an expert level"

Expertise here is **repetition under real constraints**: broken exports, driver limits, thermal throttling, and noisy benchmarks. This handbook gives you the **concept graph**. Your depth comes from applying it on hardware weekly.

## Honesty about "I will use nothing else"

This folder contains the **ideas, vocabulary, procedures, and exercises** you need. It cannot literally replace **running** software on a machine, watching temperatures, or reading **your team's internal** build docs when versions change. Treat those as lab work that turns this handbook into skill.

---

## One piece of advice

Don't try to learn everything at once. Start with **Phase 1 (Python)** and **Phase 4 (ONNX)**. These two alone will make you 10x more effective at your current job. The rest you can pick up as you go.

And the fact that you're asking this question means you're already on the right track. Most people won't admit what they don't know — that's what holds them back, not the lack of knowledge itself.
