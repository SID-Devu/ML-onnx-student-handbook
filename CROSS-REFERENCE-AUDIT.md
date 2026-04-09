# Cross-reference: Source messages → Handbook location

Every line from your original messages mapped to its location in the handbook.

---

## Section 1: Model-Level Optimizations

| Source item | Module | Line/section |
|------------|--------|-------------|
| ORT graph optimizations, `ORT_ENABLE_ALL` | 06 §1 | `sess_options.graph_optimization_level` code |
| Conv+BN+ReLU fusion, MatMul+Add fusion | 06 §1 | "fuses ops: Conv+BN+ReLU → single kernel" |
| onnxsim removes redundant nodes, folds constants | 06 §2 | Full CLI + Python code |
| "Many models already went through this during export" | 06 §2 | Exact quote before install section |
| `onnx.shape_inference.infer_shapes()` | 06 §3 | Full code example |
| FP16 via `ort_migraphx_fp16: True` | 06 §5 | Precision table + status note |
| "~2x faster on RDNA 3.5 CUs" | 06 §5 | In FP16 description |
| INT8 PTQ, QuantizeLinear/DequantizeLinear | 06 §6 | Full CalibrationDataReader code |
| Calibration dataset (100-500 images) | 06 §6 | "What you need" list |
| Mixed precision: attention FP16, conv/dense INT8 | 06 §6 | "Mixed precision (advanced concept)" subsection |
| Fuse LayerNorm, GELU, attention patterns | 06 §10 | "Manual operator fusion targets" |
| Replace Softmax+MatMul with fused MultiHeadAttention | 06 §10 | Listed as fusion target |
| `dim_overrides` in ModelSpec | 06 §4 | Full ModelSpec code example |
| "one of the biggest wins for MIGraphX specifically" | 06 §4 | Bold emphasis in text |

## Section 2: Kernel-Level Optimizations

| Source item | Module | Line/section |
|------------|--------|-------------|
| Exhaustive tuning (assembly, composable, MIOpen) | 06 §7, 08 §2 | Both modules cover it |
| Kernel caching after compilation | 08 §7 | Full section on cache invalidation |
| Memory layout NCHW vs NHWC | 02 §4 | Full section with diagrams |
| "RDNA 3.5 prefers certain layouts" | 08 §9 | Memory layout verification section |
| `migraphx::program::get_parameter_shapes()` | 08 §9 | C++ code snippet + Python alternative |
| "For models under 512 MB, pinned VRAM" | 07 §8 | "Optimization tip for small models" subsection |
| `amdgpu.gttsize=28672` and `hipMallocManaged` | 07 §2, §8 | Both sections |
| Reduce host-device transfers | 07 §12 | "Bad" vs "Good" pattern comparison |
| `r1-gpu-perf.service` locks sclk=2900, fclk=2000, mclk=1000 | 08 §5, 11 §4 | Clock table + service content |
| `rocm-smi --showclocks` verification | 08 §6 | Full command reference |
| `processor.max_cstate=1`, `idle=nomwait` | 10 §2 | Exact GRUB line + explanations |
| taskset/numactl for CPU pinning | 06 §9 | Shell + Python code |
| `transparent_hugepage=madvise` | 09 §12, 10 §2 | Both modules |
| MIGraphX multiple streams | 08 §8 | "Streams and overlapping execution" |
| Pipeline: batch on GPU while next transfers | 08 §8 | Explained in streams section |

## Section 3: What's Already Optimized (status table)

| Source item | Module | Line/section |
|------------|--------|-------------|
| ORT graph optimization (ENABLE_ALL) — Done | 18 §1 | Status table row 1 |
| FP16 for supported models — Done | 18 §1 | Status table row 2 |
| Static shape pinning — Done for several | 18 §1 | Status table row 3 |
| GPU clocks locked to max — Done | 18 §1 | Status table row 4 |
| CPU latency minimized — Done | 18 §1 | Status table row 5 |
| GTT maximized (28 GB) — Done | 18 §1 | Status table row 6 |
| Unified memory (HSA_XNACK=1) — Done | 18 §1 | Status table row 7 |
| Swap for large models (400 GB) — Done | 18 §1 | Status table row 8 |
| MGLRU enabled — Done | 18 §1 | Status table row 9 |
| THP madvise — Done | 18 §1 | Status table row 10 |
| All sysctl vm.* tuned — Done | 18 §1 | Status table row 11 |

## Section 4: Biggest Remaining Wins

| Source item | Module | Line/section |
|------------|--------|-------------|
| INT8 quantization (1.5-2x) | 18 §2 | Priority 1 |
| MIGraphX exhaustive tuning (10-30%) | 18 §2 | Priority 2 |
| ONNX Simplifier pass | 18 §2 | Priority 3 |
| Static shape pinning remaining models | 18 §2 | Priority 4 |
| Kernel fusion for transformers | 18 §2 | Priority 5 |

## Step 1: INT8 Quantization (full code)

| Source item | Module | Line/section |
|------------|--------|-------------|
| `pip install onnxruntime-extensions` | 06 §6 | Install subsection |
| `ImageCalibrationReader(CalibrationDataReader)` class | 06 §6 | Full class code |
| `quantize_static()` call with all params | 06 §6 | Full quantization script |
| `per_channel=True` | 06 §6 | In code + per-channel table |
| `QuantType.QInt8` | 06 §6 | In code |
| Verification code | 06 §6 | "Verify it works" subsection |
| Input shapes table (YOLO/MobileNet/CLIP/DeepSort/EfficientNet) | 06 §6 | "Input shapes for each vision model" table |

## Step 2: ONNX Simplifier (full code)

| Source item | Module | Line/section |
|------------|--------|-------------|
| `pip install onnxsim` | 06 §2 | Install subsection |
| Check node count code | 06 §2 | "Check current node count" code |
| `python -m onnxsim` CLI | 06 §2 | "Simplify (CLI)" code |
| Compare before/after code | 06 §2 | Full comparison code |
| Large model simplification with `load_external_data=True` | 06 §2 | "For large models (>2GB)" code |
| "5-15% fewer ops" | 06 §2 | After code block |

## Step 3: Static Shape Pinning (full code)

| Source item | Module | Line/section |
|------------|--------|-------------|
| `check_dynamic_shapes.py` script | 06 §4 | Full script |
| `isinstance(dim, str)` check | 06 §4 | In loop code |
| `dim_overrides={"batch_size": 1, "sequence_length": 77}` | 06 §4 | ModelSpec code example |

## Step 4: MIGraphX Exhaustive Tuning (full code)

| Source item | Module | Line/section |
|------------|--------|-------------|
| `migraphx_exhaustive_tune: True` | 06 §7 | Provider options dict |
| `migraphx_save_compiled_model: True` | 06 §7 | Provider options dict |
| `migraphx_load_compiled_model: True` | 06 §7 | Provider options dict |
| `migraphx_model_cache_path` | 06 §7 | Provider options dict |
| `mkdir -p .migraphx_cache` | 06 §7 | Bash command |
| Batch tuning command line | 06 §7 | Full benchmark command |
| "minutes to hours per model" | 06 §7 | Important notes |
| "10-30% latency reduction" | 06 §7 | Important notes |

## Step 5: Fused Attention (full code)

| Source item | Module | Line/section |
|------------|--------|-------------|
| Option A: `optimized_model_filepath` | 06 §8 | Full code |
| Inspect fused op counts code | 06 §8 | Attention/MultiHeadAttention count code |
| Option B: `onnxruntime.transformers.optimizer` | 06 §8 | Full optimizer code |
| `model_type="gpt2"` / `"bert"` | 06 §8 | Model type mapping table |
| Qwen3: gpt2, 16 heads, 2048 hidden | 06 §8 | Table row |
| CLIP: bert, 12 heads, 768 hidden | 06 §8 | Table row |

## Step 6: CPU Thread Pinning

| Source item | Module | Line/section |
|------------|--------|-------------|
| `taskset -c 0-7` | 06 §9 | Shell code |
| `numactl --cpunodebind=0 --membind=0` | 06 §9 | Shell code |
| `os.sched_setaffinity(0, set(range(8)))` | 06 §9 | Python code |

## Recommended Execution Order table

| Source item | Module | Line/section |
|------------|--------|-------------|
| 6-row table (onnxsim → shapes → tune → INT8 → fusion → pinning) | 06 §11 | Full table with time + gain |
| "Total: 2-4x vision, 1.3-2x transformers" | 06 §11 | After table |

## Phase 1: Python Foundations

| Source item | Module | Line/section |
|------------|--------|-------------|
| Variables, types, for loops, if/else | 01 §1-2 | Full sections |
| Functions (def), Classes (class) | 01 §3-4 | Full sections |
| `@dataclass` | 01 §5 | Full section |
| Type hints, `Optional`, `List`, `Dict` | 01 §6 | Full section |
| `import` and modules | 01 §7 | Full section |
| `*args`, `**kwargs` | 01 §8 | Full section |
| `lambda` | 01 §9 | Full section |
| Dictionary comprehension | 01 §10 | Full section |
| `Callable` type hint | 01 §11 | Full section |
| f-strings | 01 §12 | Full section |
| File I/O (open, read, write) | 01 §13 | Full section |
| Context managers (`with`) | 01 §14 | Full section |
| `try`/`except` error handling | 01 §15 | Full section |
| `argparse` | 01 §16 | Full section |
| `subprocess.run()` | 01 §17 | Full section |
| `time.perf_counter()` | 01 §18 | Full section |
| `os` and `pathlib` | 01 §19 | Full section |
| `pip install` and virtual environments | 01 §20 | Full section |
| py4e.com reference | bibliography | Full URL |

## Phase 2: NumPy

| Source item | Module | Line/section |
|------------|--------|-------------|
| `np.array`, `np.zeros`, `np.ones`, `np.random.randn` | 02 §1 | Full code examples |
| Shape `(1, 3, 224, 224)` with axis diagram | 02 §3 | ASCII art diagram |
| Shape `(1, 48000)` audio "3 sec × 16000 Hz" | 02 §3 | Audio tensor subsection |
| Shape `(1, 128, 2048)` hidden dimension | 02 §3 | Sequence tensor subsection |
| Dtypes: float32, float16, int64, bool | 02 §2 | Dtype table |
| `.reshape()`, `.transpose()` | 02 §5 | Full section |
| Indexing/slicing `arr[0, :, 32:64]` | 02 §6 | Full section |
| NCHW vs NHWC | 02 §4 | Full section |
| Strides | 02 §7 | Full section |
| `np.ascontiguousarray` | 02 §7 | In strides section |

## Phase 3: Neural Networks

| Source item | Module | Line/section |
|------------|--------|-------------|
| Tensor, Model, Weights, Inference, Training | 03 §1 | Core concepts table |
| Layer, CNN, Transformer | 03 §1, §5 | Table + CNN vs Transformer section |
| Encoder, Decoder | 03 §6 | Full section with examples |
| 3Blue1Brown reference | bibliography | Full YouTube playlist URL |

## Phase 4: ONNX

| Source item | Module | Line/section |
|------------|--------|-------------|
| "Like PDF is for documents" analogy | 04 §1 | In graph mental model |
| Graph = nodes (ops) + edges (tensors) | 04 §1 | Graph mental model section |
| Operators: Conv, Relu, MatMul, Softmax, etc. | 03 §4, 04 §2 | Op tables |
| Opset version (you use 17) | 04 §2 | Opset section |
| Dynamic vs Static shapes | 04 §5 | Full section |
| External data (>2GB split) | 04 §4 | Full section with code |
| Netron install + open command | 04 §8 | "Inspecting models — Netron" |
| Python inspection code (opset, nodes, inputs, outputs) | 04 §9 | "Inspecting models — Python API" |
| `config.json` (hidden_size, num_heads, vocab_size) | 04 §10 | Full section with JSON |

## Phase 5: ONNX Runtime

| Source item | Module | Line/section |
|------------|--------|-------------|
| `InferenceSession` | 05 §1-2 | Key pattern + session section |
| Execution Provider (MIGraphX, CPU) | 05 §3 | EP table |
| Provider options (FP16, device_id, cache) | 05 §7 | Full options dict |
| Session options (graph_optimization_level, threads) | 05 §6 | Full section |
| `session.run()` | 05 §5 | Full section |
| Warmup (first run slow) | 05 §8 | Full section |
| Key 4-step pattern code | 05 §1 | Complete code block |

## Phase 6: GPU & Hardware Concepts

| Source item | Module | Line/section |
|------------|--------|-------------|
| VRAM 512 MB + GTT 28 GB | 07 §1-2 | Sections 1 and 2 |
| Unified Memory, HSA_XNACK=1 | 07 §4-5 | Sections 4 and 5 |
| GTT via `amdgpu.gttsize=28672` | 07 §2 | Exact value |
| `hipMallocManaged` | 07 §8 | Full comparison table |
| FP32 (4 bytes) / FP16 (2 bytes, ~2x) / INT8 (1 byte, ~4x) | 06 §5 | Precision table |
| GPU kernel = compiled function | 08 §2 | MIGraphX section |
| CU = 40 Compute Units | 08 §4 | Full section |
| sclk/mclk/fclk clock speeds | 08 §5 | Clock table |
| Thermal throttling | 08 §10 | Full section |
| ROCm = AMD's CUDA equivalent | 08 §1 | First section |
| MIGraphX = graph compiler for NNs | 08 §2 | Full section |

## Phase 7: Linux Kernel & System Tuning

| Source item | Module | Line/section |
|------------|--------|-------------|
| dmesg | 19 §5 | Telemetry capture code |
| `/proc/cpuinfo` | 16 §9 | /proc table |
| `/proc/meminfo` | 16 §9 | /proc table |
| `/sys/module/amdgpu/parameters/` | 08 §6, 16 §9 | Both modules |
| sysctl (all vm.* params) | 09 §2-15 | Every parameter with exact value |
| GRUB boot parameters | 10 §1-2 | Exact GRUB line + all 7 params |
| `systemctl` / `systemd` | 11 §1-3 | Full command table + unit file |
| `rocm-smi` | 08 §6 | Full flag reference |
| Swap (400 GB) | 09 §15 | Full section |
| Hugepages / THP | 09 §12 | Mode table |
| "How Linux Works" book reference | bibliography | Full entry |

## Phase 8: Model Domains

| Source item | Module | Line/section |
|------------|--------|-------------|
| All 23 models mapped (YOLO → ViNT) | 17 | Complete 23-row catalog |
| Input/output patterns per domain | 17 | Per-domain subsections |
| INT8 suitability per model type | 17 | Notes in each subsection |

## Section A: Xen & Virtualization

| Source item | Module | Line/section |
|------------|--------|-------------|
| Xen = Type 1 hypervisor | 12 §1 | Full explanation |
| PVH mode | 12 §2 | Vocabulary table |
| dom0, domU | 12 §2 | Table |
| `dom0_mem=30G` | 12 §3 | Config table |
| `dom0_max_vcpus=32` | 12 §3 | Config table |
| `sched=credit2` | 12 §5 | Full section |
| `cpufreq=xen:performance` | 12 §6 | Full section |
| `xl info`, `xl list` | 12 §4 | Command table + output |
| Bare metal vs virtualized table | 12 §7 | 7-row comparison table |
| vCPUs vs physical threads | 12 §8 | SMT explanation |

## Section B: AMD GPU Memory

| Source item | Module | Line/section |
|------------|--------|-------------|
| VRAM 512 MB carve-out | 07 §1 | Full section |
| GTT 28 GB via `amdgpu.gttsize=28672` | 07 §2 | Full section |
| GART (address remapping) | 07 §3 | Full section |
| Unified Memory (same physical DRAM) | 07 §4 | Full section |
| HSA framework | 07 §5 | Full section |
| XNACK demand paging | 07 §6 | Full section with comparison table |
| XNACK=0: 750+ SVM messages | 07 §6 | Exact number |
| SVM / SVM IOCTLs | 07 §7 | Full section |
| `hipMalloc` vs `hipMallocManaged` | 07 §8 | Comparison table |
| Page migration | 07 §9 | Full section |
| Page pinning | 07 §9 | Full section |
| TLB + hugepages math (28GB/4KB = 7.3M vs 28GB/2MB = 14K) | 07 §10 | Exact calculation |

## Section C: Linux sysctl

| Source item | Module | Line/section |
|------------|--------|-------------|
| `vm.swappiness=10` | 09 §3 | Full explanation |
| `vm.overcommit_memory=1` | 09 §4 | Full explanation with modes 0/1/2 |
| `vm.vfs_cache_pressure=50` | 09 §5 | Full explanation |
| `vm.watermark_boost_factor=0` | 09 §6 | Full explanation |
| `vm.compaction_proactiveness=0` | 09 §7 | Full explanation |
| `vm.dirty_ratio=10` / `vm.dirty_background_ratio=3` | 09 §8 | Full explanation |
| `vm.zone_reclaim_mode=0` | 09 §9 | Full explanation |
| `vm.max_map_count=2097152` | 09 §10 | Full explanation |
| MGLRU | 09 §11 | Full section |
| THP modes (always/madvise/never) | 09 §12 | Mode table |
| Compaction | 09 §7 | In compaction section |
| OOM Killer | 09 §13 | Full section |
| kswapd | 09 §13 | In kswapd section |
| Page cache | 09 §14 | Full section |
| Swap 400 GB on NVMe | 09 §15 | Full section |
| sysctl read/write/verify commands | 09 §16 | Full bash commands |

## Section D: Kernel Boot Parameters

| Source item | Module | Line/section |
|------------|--------|-------------|
| Exact GRUB line with all 7 params | 10 §1 | `GRUB_CMDLINE_LINUX_DEFAULT=...` |
| `amdgpu.gttsize=28672` | 10 §2 | Full explanation |
| `amdgpu.no_system_mem_limit=1` | 10 §2 | Full explanation |
| `iommu=pt` | 10 §2 | Full explanation + DMA concept |
| `processor.max_cstate=1` with C-state table | 10 §2 | C-state table with wake latencies |
| `idle=nomwait` | 10 §2 | Full MWAIT explanation |
| `split_lock_detect=off` | 10 §2 | Cache line + split lock explanation |
| `transparent_hugepage=madvise` | 10 §2 | Points to Module 09 |
| `update-grub` command | 10 §1 | After GRUB line |
| IOMMU concept | 10 §3 | Full section |
| DMA concept | 10 §3 | Full section |
| Cache lines (64 bytes) | 10 §3 | In concepts section |
| Verification: `cat /proc/cmdline` | 10 §4 | Bash code |

## Section E: Model Export Pipeline

| Source item | Module | Line/section |
|------------|--------|-------------|
| `torch.onnx.export()` | 13 §2 | Full code example |
| Tracing vs Scripting | 13 §3 | Comparison table |
| Dynamic axes | 13 §4 | Code example |
| Opset version (grid_sample at 16+) | 13 §5 | Opset section |
| HuggingFace Optimum CLI | 13 §6 | `optimum-cli` command |
| `tf2onnx` | 13 §7 | Conversion command |
| External data (`save_as_external_data=True`) | 13 §9 | Code example |
| `onnxsim` | 13 §8 | Post-export tools table |
| `onnx.checker.check_model()` | 13 §8 | Table row |
| `onnx.shape_inference` | 13 §8 | Table row |
| Netron | 13 §8 | Table row |
| SafeTensors vs pickle | 13 §11 | Comparison table |
| `config.json` | 13 §10 | JSON example |
| Export workaround table (8 rows) | 13 §12 | Full 8-row table |
| Qwen3 → external data | 13 §12 | Table row |
| LLaMA-3.2 → split subgraphs | 13 §12 | Table row |
| RAFT-Stereo → CorrSampler | 13 §12 | Table row |
| Wav2Vec2 → layer_drop | 13 §12 | Table row |
| Cast mismatches | 13 §12 | Table row |
| NaN in weights | 13 §12 | Table row |
| tie_weights error | 13 §12 | Table row |
| Meta tensors → materialize | 13 §12 | Table row |

## Section F: Benchmarking Methodology

| Source item | Module | Line/section |
|------------|--------|-------------|
| Warmup=3 | 14 §2 | Config table |
| Runs=3 | 14 §2 | Config table |
| Cooldown=120s | 14 §2, §4 | Config table + full section |
| TTFT (Time to First Token) | 14 §1 | Metrics table |
| Per-run latency | 14 §6 | Full section with example |
| Mean / Median / Std | 14 §7 | Statistics table |
| P95 / P99 | 14 §7 | Statistics table |
| Throughput (FPS) | 14 §8 | Formula |
| Thermal throttling | 14 §4, §9 | Cooldown + jitter checklist |
| Statistical significance | 14 §10 | Practical section |
| JSON reproducibility footer | 14 §11 | Full JSON schema |

## Section G: Systemd Services

| Source item | Module | Line/section |
|------------|--------|-------------|
| systemd basics | 11 §1 | Full section |
| `systemctl enable/start/stop/restart/status` | 11 §2 | Command table |
| `journalctl -u service_name` | 11 §2 | Command table |
| Unit file structure (`[Unit]`, `[Service]`, `[Install]`) | 11 §3 | Full INI example |
| `ExecStart`, `After=`, `Requires=`, `WantedBy=` | 11 §3 | Key fields table |
| r1-gpu-perf.service: GPU clocks | 11 §4a | Bash code |
| r1-gpu-perf.service: all sysctl writes | 11 §4b | Full sysctl list |
| r1-gpu-perf.service: THP madvise | 11 §4c | Bash code |
| r1-gpu-perf.service: NVMe readahead=2048 | 11 §4d | Bash code + explanation |
| r1-gpu-perf.service: MGLRU enable | 11 §4e | Bash code |
| Verification commands | 11 §6 | systemctl + sysctl + rocm-smi |
| daemon-reload after editing | 11 §7 | In create/edit section |

## Section H: ROCm / AMD GPU Tools

| Source item | Module | Line/section |
|------------|--------|-------------|
| `rocm-smi` (all flags) | 08 §6 | Full command reference |
| `rocm-smi --showclocks` | 08 §6 | Listed |
| `rocm-smi --showtemp` | 08 §6 | Listed |
| `rocm-smi --showmeminfo` | 08 §6 | Listed |
| `rocm-smi --showall` | 08 §6 | Listed |
| `rocm-smi --setperflevel high` | 08 §6 | Listed |
| `rocminfo` | 08 §6 | Full subsection |
| `hipcc` | 08 §6 | Subsection |
| `hipconfig` | 08 §6 | Subsection |
| `amdgpu` driver sysfs `/sys/module/amdgpu/parameters/*` | 08 §6 | Subsection with `cat` command |

## Section I: Environment & Libraries

| Source item | Module | Line/section |
|------------|--------|-------------|
| Environment variables, `/etc/environment` | 16 §1 | Full section with code |
| `HSA_XNACK=1` in `/etc/environment` | 16 §1 | Exact line |
| PATH | 16 §1 | Environment table |
| LD_LIBRARY_PATH | 16 §2 | Full section with warning |
| Shared libraries (`.so` files) | 16 §2 | Full section with list |
| `ldd` | 16 §2 | Subsection |
| Custom builds vs pip | 16 §3 | Comparison table |
| CMake | 16 §3, 21 §2 | Both modules |

## Section J: Python Specifics

| Source item | Module | Line/section |
|------------|--------|-------------|
| `@dataclass` / `ModelSpec` | 01 §5, 15 §2 | Both modules |
| Type hints (`Optional`, `List`, `Dict`) | 01 §6 | Full section |
| `field(default_factory=dict)` | 01 §5, 15 §2 | Both modules |
| `argparse` | 01 §16, 15 §1 | Both modules |
| `subprocess.run()` | 01 §17, 15 §6 | Both modules |
| `os.path`, `os.makedirs` | 01 §19, 15 §8 | Both modules |
| `json.dump()` / `json.load()` | 15 §7 | Full code |
| `time.perf_counter()` | 01 §18, 15 §5 | Both modules |
| Dictionary comprehension | 01 §10, 15 §4 | Both modules |
| `Callable` | 01 §11, 15 §3 | Both modules |
| `lambda` | 01 §9 | Full section |
| f-strings | 01 §12 | Full section |
| Context managers (`with`) | 01 §14 | Full section |
| `*args`, `**kwargs` | 01 §8 | Full section |

## Section K: Storage & I/O

| Source item | Module | Line/section |
|------------|--------|-------------|
| NVMe (Samsung 990 PRO) | 16 §4 | Speed comparison table |
| Readahead = 2048 KB | 16 §5 | Bash code + explanation |
| IOPS (~500K+) | 16 §4 | Table column |
| Sequential vs Random I/O | 16 §6 | Comparison table |
| Swap on NVMe (400 GB) | 16 §7 | fstab example |
| `/etc/fstab` | 16 §7 | Full section |
| Page cache effects | 16 §8 | Before/after table |

## Learning Roadmaps

| Source item | Module | Line/section |
|------------|--------|-------------|
| 10-week learning plan with deliverables | 00-how-to-study | Subsumed by 16-week plan |
| 14-phase extended roadmap (Sections A-K mapped) | 00-how-to-study | Full 16-week table |
| "One Piece of Advice" | 00-how-to-study | Final section |

## Resources / Bibliography

| Source item | Module | Line/section |
|------------|--------|-------------|
| py4e.com (chapters 1-10, YouTube) | bibliography | Full entry with URL |
| numpy.org quickstart | bibliography | Full entry with URL |
| 3Blue1Brown YouTube playlist | bibliography | Full URL with description |
| onnx.ai/onnx/intro/ | bibliography | Full entry |
| onnxruntime.ai/docs/ | bibliography | Full entry |
| rocm.docs.amd.com | bibliography | Full entry |
| "How Linux Works" by Brian Ward | bibliography | Full entry |
| kernel.org sysctl docs | bibliography | Full URL |
| kernel.org boot parameters | bibliography | Full URL |
| wiki.xenproject.org | bibliography | Full entry |
| netron.app | bibliography | Full entry |
| pytorch.org/docs/stable/onnx.html | bibliography | Full URL |
| HuggingFace Optimum export guide | bibliography | Full URL |
| "HSA Runtime Programmer's Reference Manual" | bibliography | Full entry |
| AMD GPU architecture / RDNA 3.5 whitepapers | bibliography | Full entry |

---

**Total: 250+ items mapped. Zero missing.**

To verify any item: open the referenced module, search for the keyword.
