# R1 ML / ONNX / AMD Student Handbook

This folder is a **self-contained study path** for working with ONNX models, ONNX Runtime, AMD ROCm / MIGraphX, Linux system tuning, and benchmarking—aligned with a real workflow (Strix Halo–class APU, MIGraphX EP, custom ORT builds, large models with unified memory).

**How to use it**

1. Read `00-how-to-study.md` first (pace, practice, checks).
2. Follow the **numbered modules** in order the first time through (`modules/01-...` through `modules/25-...`; 21 is optional build engineering, 22-25 are practical tooling).
3. Use `cheatsheets/` for quick recall after you have read the corresponding module.
4. Do the tasks in `exercises/`; they mirror what you do at work (load ONNX, inspect shapes, reason about memory).
5. Finish with `glossary.md` as a concept index (search in your editor).

**What “done” means**

You can explain, without notes: what a tensor and an ONNX graph are; how ORT chooses an execution provider; why static shapes help MIGraphX; what GTT, XNACK, and unified memory imply for large models; how warmup and cooldown affect benchmarks; how INT8 calibration differs from FP16; how attention Q/K/V works; how to preprocess an image for YOLO; how to debug a Python traceback; and how to use git/shell/htop in your daily workflow.

**Folder layout**

| Path | Purpose |
|------|---------|
| `modules/` | Deep explanations (this is the main course) |
| `cheatsheets/` | One-page reminders |
| `exercises/` | Hands-on tasks + solutions |
| `scripts/examples/` | Small copy-paste Python examples |
| `glossary.md` | Alphabetical concept index |
| `bibliography-optional.md` | Extra pointers if you later have network (not required to learn from this folder) |

---

## Module list (recommended order)

| # | File | Topics |
|---|------|--------|
| 1 | `modules/01-python-minimum-for-ml-scripts.md` | Variables, control flow, functions, classes, imports, venv, pip, errors, **debugging (tracebacks, pdb, assert, common error types)** |
| 2 | `modules/02-numpy-tensors-and-shapes.md` | Arrays, dtypes, shapes, NCHW, slicing |
| 3 | `modules/03-neural-nets-inference-vs-training.md` | Tensors, layers, CNN vs Transformer, encoder/decoder, **attention Q/K/V, multi-head, cross-attention, autoregressive generation, KV cache, decoding strategies** |
| 4 | `modules/04-onnx-format-and-graphs.md` | Graph, nodes, ops, opset, static/dynamic axes, external data |
| 5 | `modules/05-onnx-runtime-sessions-and-eps.md` | Session, providers, options, run/warmup |
| 6 | `modules/06-optimizations-fusion-quantization.md` | ORT graph opt, onnxsim, shape inference, FP16/INT8, static pinning, **pruning, knowledge distillation, NAS, LoRA, weight sharing** |
| 7 | `modules/07-amd-apu-memory-gtt-unified.md` | VRAM, GTT, GART, unified memory, HSA, XNACK, SVM, migration, hipMalloc |
| 8 | `modules/08-rocm-migraphx-and-ort-migraphx-ep.md` | ROCm stack, MIGraphX, EP options, tuning, caching |
| 9 | `modules/09-linux-memory-sysctl-and-reclaim.md` | vm.*, swap, OOM, kswapd, THP, MGLRU, compaction |
| 10 | `modules/10-kernel-boot-and-io-dma.md` | GRUB, amdgpu params, IOMMU/DMA, C-states, MWAIT, split lock, **full boot chain (UEFI → GRUB → kernel → initramfs → systemd)** |
| 11 | `modules/11-systemd-and-boot-time-tuning.md` | Units, enable/start, journalctl, tying tuning to boot |
| 12 | `modules/12-xen-and-virtualization-basics.md` | Hypervisor, dom0, PVH, vCPUs, scheduler, bare metal vs virt |
| 13 | `modules/13-model-export-and-onnx-pitfalls.md` | torch.export/onnx, dynamic axes, Optimum, tf2onnx, workarounds, **HuggingFace Hub (cli login, download, gated models, from_pretrained, trust_remote_code)** |
| 14 | `modules/14-benchmarking-methodology-and-stats.md` | Warmup, runs, cooldown, TTFT, mean/median/std, percentiles |
| 15 | `modules/15-python-patterns-in-benchmark-code.md` | dataclasses, argparse, perf_counter, JSON, subprocess |
| 16 | `modules/16-env-libraries-storage-io.md` | PATH, LD_LIBRARY_PATH, NVMe, readahead, fstab |
| 17 | `modules/17-model-families-you-will-meet.md` | Detection, CLIP, LLM, TTS/STT, robotics, depth, OCR, **image/audio/text pre-processing code, NMS/argmax/CTC/beam post-processing, evaluation metrics (mAP, IoU, WER, perplexity, AUROC)** |
| 18 | `modules/18-end-to-end-optimization-playbook.md` | Ordered steps: simplify → shapes → tune → quantize → fusion → CPU pin |
| 19 | `modules/19-ort-debugging-and-telemetry.md` | Optimized graph dumps, log severity, graph partitioning traps, dmesg/SVM telemetry, async timing, memory profiling, **/proc and /sys filesystem navigation** |
| 20 | `modules/20-safety-reproducibility-and-ml-ops.md` | Pickle vs SafeTensors, sha256 hashes, supply chain, honest vs theater benchmarking, accuracy validation, regression tracking |
| 21 (opt) | `modules/21-building-ort-and-version-skew.md` | CMake flags, `--use_migraphx`, `hipMallocManaged`, version tuple, `ldd`, wheel install, when to rebuild |
| 22 | `modules/22-shell-bash-for-ml-workflows.md` | Pipes, redirects, variables, quoting, loops, conditionals, `&&`/`;`/`\|\|`, exit codes, `chmod`, `source` vs `./`, `nohup`, `screen`/`tmux` |
| 23 | `modules/23-git-version-control.md` | `status`, `diff`, `add`/`commit`, `log`, branches, `stash`, `pull`/`push`, `blame`, `.gitignore`, Git LFS |
| 24 | `modules/24-process-monitoring.md` | `htop`, `top`, `ps aux`, `kill`, `nice`/`renice`, `watch`, `tail -f`, `iotop`, `lsof`, diagnosis recipes |
| 25 | `modules/25-profiling-tools.md` | `perf stat`, `perf record` + flame graphs, `rocprof --stats`, `py-spy`, `memory_profiler` |

---

## Quick start (first afternoon)

1. Read modules **01–05** (about 2–4 hours depending on background).
2. Run `scripts/examples/inspect_onnx_minimal.py` on a small `.onnx` you have (edit path inside script).
3. Skim `glossary.md` and bookmark terms you do not know; revisit after each module.

You are not expected to memorize every sysctl name on day one; you are expected to understand **what problem each class of tuning solves**.

---

## License / use

Educational notes derived from your project context. Adapt paths and hardware numbers to your machine when you practice.
