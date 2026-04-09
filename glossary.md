# Glossary (concept index)

Use your editor search (`Ctrl+F`) on this file. Terms are **brief** on purpose; full explanations live in `modules/`.

## A

- **Accuracy (quantization)**: How much FP32 behavior is preserved after FP16/INT8; must be validated, not assumed.
- **Activation (NN)**: Outputs of layers after weights apply; also “activation quantization” means quantizing those tensors.
- **Adam / SGD (mention only)**: Optimizers used in training; you usually don’t tune these for ONNX inference.
- **AMDGPU driver**: Kernel driver module for AMD GPUs; exposes sysfs params and dmesg messages.
- **APU**: CPU + integrated GPU on same package; memory story differs from discrete GPU.
- **Argparse**: Python CLI parsing library (`benchmark_*.py`).
- **ASR (automatic speech recognition)**: Speech → text (Whisper-like).
- **Attention**: Transformer mechanism comparing positions in a sequence (self-attention) or two sequences (cross-attention).
- **Async GPU execution**: GPU work may complete after the CPU call returns unless synchronized.

## B

- **Batch size (N)**: How many inputs processed together; larger often increases throughput but not always latency.
- **Benchmark harness**: Script that loads models, runs timed loops, logs JSON/csv.
- **Binary ONNX**: Single `.onnx` protobuf file (contrasts with external data).
- **Boot cmdline / kernel parameters**: String passed by GRUB/systemd-boot to Linux kernel (module 10).
- **Branch (export)**: Tracing may miss unexecuted branches; dynamic control flow breaks naive tracing.

## C

- **CogACT**: Robotics / vision-language-action related model family name in broader zoos (treat as domain label; verify your checkpoint).
- **CenterPoint**: 3D object detection architecture family (LiDAR/point-cloud settings).
- **CorrSampler**: Custom op / sampling operator class that can block naive PyTorch→ONNX export (stereo/RAFT-like stacks).
- **Cache (compiled kernels)**: Saved compilation artifacts to skip compile on next run (EP-dependent).
- **Calibration (INT8)**: Process of feeding representative inputs to estimate quantization scales/zero-points.
- **CalibrationDataReader**: ORT API pattern: iterator object yielding input dicts for calibration passes.
- **Cast node (ONNX)**: Explicit dtype conversion inserted to fix export mismatches.
- **Checkpoint**: Saved trained weights (may be `.safetensors`, `.bin`, `.pth`, etc.).
- **C-states**: CPU idle power states; deeper C-states increase wake latency (module 10).
- **CLI**: Command-line interface; benchmarks expose knobs as flags.
- **CLIP**: Vision-language model family mapping images and text into a shared embedding space.
- **CMake**: Build system used to compile ORT/ROCm components from source.
- **Composable Kernel**: AMD library strategy implementing kernels from composable primitives (name you’ll hear near MIOpen/MIGraphX).
- **Contiguous array**: NumPy memory layout expectation for some fast paths (`np.ascontiguousarray`).
- **Conv (convolution)**: CNN operator for local pattern extraction.
- **Cooldown**: Idle delay between benchmark cases to reduce thermal coupling (module 14).
- **cpufreq=xen:performance (example)**: Xen-mediated cpufreq policy knob seen in dom0 tuning notes (module 12).
- **CPU affinity**: Pinning threads/processes to specific cores (`taskset`, `numactl`, `sched_setaffinity`) (module 15).
- **CPUExecutionProvider**: ORT CPU backend baseline.
- **Credit2 (Xen scheduler)**: Xen CPU scheduler option referenced for latency-sensitive workloads (module 12).
- **CrossFormer**: Transformer-family vision/backbone architecture name in your model zoo context.
- **CUDAExecutionProvider**: NVIDIA EP (not your AMD target, but appears in ORT docs/comparisons).
- **Custom op**: Operator not in standard ONNX set or not implemented on an EP.

## D

- **Decoder (NN)**: Generates outputs from a representation (LLM/TTS contexts) (module 03).
- **DeepSort**: Tracking stack often involving appearance embeddings (OSNet-like backbones).
- **DeepSeek / DeepSeek-R1**: Example LLM / reasoning model naming context in your notes.
- **Demand paging**: Mapping/provisioning pages on first access; interacts with GPU memory models (module 07).
- **dmesg**: Kernel log ring buffer; used to observe driver messages during memory experiments.
- **dom0**: Privileged Xen domain hosting drivers/admin (module 12).
- **dom0_mem**: Xen parameter limiting/assigning memory to dom0 (example knob).
- **dom0_max_vcpus**: Xen parameter for vCPU exposure to dom0.
- **domU**: Unprivileged Xen guest.
- **dom0=pvh (example)**: GRUB/Xen config style indicating dom0 virtualization mode (PVH) in your notes (module 12).
- **Dynamic axes**: Exported dimensions that can vary (batch/seq/height/width).
- **Dynamic shape**: ORT/ONNX dimension represented as a string symbol or unknown.

## E

- **Encoder (NN)**: Maps raw inputs to embeddings (module 03).
- **End-to-end latency**: Includes preprocess + ORT + postprocess (module 14).
- **EP (Execution Provider)**: ORT backend implementation (module 05).
- **EfficientNet**: CNN family known for efficiency/accuracy tradeoffs.
- **Embedding**: Vector representation of token/image patch/etc.
- **External data**: Sidecar weight storage for large ONNX models (module 04).
- **Exporter**: Toolchain turning a trained model into ONNX (PyTorch/Optimum/tf2onnx).

## F

- **fclk / mclk / sclk**: GPU clock domains you may see in tuning docs/tools output (meanings vary by ASIC; treat as “clock domains to monitor”).
- **FP16**: 16-bit floating point; faster on hardware with good FP16 support; some accuracy risk.
- **FP32**: 32-bit floating point default baseline.
- **Fusion**: Combining multiple ops into fewer ops/kernels (ORT graph optimizer, EP compiler).
- **FusedAttention / MultiHeadAttention**: ONNX ops that may appear when attention patterns fuse (depends on graph + ORT version).

## G

- **GART**: Hardware remapping support for GPU memory access paths (module 07).
- **Gemm**: General matrix multiply ONNX op (linear layers often lower to Gemm/MatMul).
- **GELU**: Smooth activation common in Transformers; fusion targets.
- **GPT2 / BERT (optimizer model_type)**: Labels in ORT transformer optimizer tooling indicating expected graph families.
- **Graph (ONNX)**: Nodes + values + initializers forming the model program.
- **GraphOptimizationLevel / ORT_ENABLE_ALL**: ORT setting enabling aggressive rewrites (module 06).
- **grid_sample**: ONNX op whose availability depends on opset; vision models may require newer opsets.
- **GTT (Graphics Translation Table)**: System RAM mapped for GPU access via GPU page tables (module 07).
- **GRUB**: Bootloader editing kernel cmdline (module 10).

## H

- **Hidden size**: Transformer width dimension (often last axis of `[B, T, H]`).
- **HIP**: AMD GPU programming layer analogous to CUDA (module 08).
- **HSA**: Heterogeneous System Architecture runtime concepts on AMD (module 07).
- **HSA_XNACK**: Environment variable controlling XNACK-related paging behavior (module 07).
- **Hugepages / THP**: Larger page sizes reduce TLB pressure; may interact with compaction stalls (module 09).
- **Hypervisor**: Software layer running VMs; Xen example (module 12).

## I

- **Inference**: Forward evaluation of a trained model (no gradient learning).
- **Initializer**: ONNX constant tensor stored in model (usually weights).
- **INT8**: 8-bit quantized representation with scales/zero-points; needs calibration for quality PTQ.
- **IOMMU**: IO memory management unit; `iommu=pt` passthrough mode appears in tuning notes (module 10).
- **IOPS**: IO operations per second; storage performance metric (module 16).
- **iommu=pt**: Kernel cmdline option discussed for DMA latency tradeoffs (platform-dependent).

## J

- **JIT / compile (GPU)**: Runtime compilation of kernels (first run slower).
- **JSON logging**: Benchmark results serialized for analysis (`json.dump`).

## K

- **Kernel (GPU)**: Function executed in parallel on GPU.
- **Kernel telemetry**: Logging driver/kernel events (dmesg/sysfs) around inference.
- **kswapd**: Kernel reclaim thread (module 09).
- **KV cache (LLM concept)**: Caching attention keys/values during autoregressive generation (framework-level; may not appear explicitly in ONNX export).

## L

- **layer_drop (Wav2Vec2 training trick)**: Can break tracing/export assumptions; may require export workarounds (module 13).
- **LayerNorm**: Normalization op; fusion target in some stacks.
- **Latency**: Time per inference (module 14).
- **LD_LIBRARY_PATH**: Runtime `.so` search path override (module 16).
- **LLM**: Large language model; text generation / chat models.
- **LLaMA / Llama**: Common LLM architecture/export family name in your notes (split subgraph workarounds may apply).
- **Load external data**: ONNX API flag to load big weights from sidecar files.

## M

- **MatMul**: Matrix multiply op; central to Transformers and linear layers.
- **Median**: Robust central tendency statistic (module 14).
- **MIGraphX**: AMD graph compiler used via ORT MIGraphX EP (module 08).
- **MIGraphXExecutionProvider**: ORT EP name for MIGraphX path.
- **MIOpen**: AMD deep learning primitives library (module 08).
- **Mixed precision**: Using multiple dtypes across subgraphs (e.g., FP16 + INT8).
- **MobileNetV2**: Efficient CNN architecture common in edge benchmarks.
- **MobileSAM**: Segment-anything style model variant tuned for mobile/lighter usage (domain label).
- **ModelSpec**: Dataclass pattern bundling model metadata + input builder + EP options (your suite concept).
- **Meta tensors**: Placeholder tensors without materialized weights; must be materialized before export.
- **MWAIT**: CPU instruction used in some idle paths; `idle=nomwait` relates to idle selection (module 10).

## N

- **NCHW / NHWC**: Tensor layout conventions (module 02).
- **Netron**: Graph visualizer for ONNX (module 04).
- **Node (ONNX)**: One op instance in the graph.
- **NUMA**: Non-uniform memory access topology; `numactl` binds memory/CPU preferences (module 15).
- **Numerical tolerance**: Compare FP32 vs FP16/INT8 outputs using acceptable error thresholds.

## O

- **OMR (not standard)**: If you see OCR-like terms, use OCR glossary entries.
- **ONNX**: Open standard for ML graphs (module 04).
- **onnx.checker.check_model**: Structural validation API.
- **onnx.shape_inference.infer_shapes**: Propagate shape/type info (module 04/06).
- **onnxsim / ONNX Simplifier**: Third-party graph simplification tool (module 06).
- **OOM killer**: Kernel kills processes under unrecoverable memory pressure (module 09).
- **OpenVLA**: Robotics vision-language-action model family name in your zoo context.
- **Opset**: ONNX operator set version (module 04).
- **ORT**: ONNX Runtime (module 05).
- **OSNet**: Re-ID CNN backbone name in tracking stacks.
- **Optimum (Hugging Face)**: Export toolkit for HF models to ONNX (module 13).

## P

- **Pi-0**: Robotics policy / VLA-related model naming context (domain label).
- **p95 / p99**: Percentile tail latency metrics (module 14).
- **Page cache**: Kernel-cached file pages in RAM (module 09/16).
- **Page migration**: Moving pages between preferred memory locations/pools (module 07).
- **PaDiM**: Anomaly detection approach/model family name (domain label).
- **Partitioning (ORT)**: Splitting graph across EPs when some ops unsupported (module 19).
- **PERF_COUNTER**: `time.perf_counter()` timing pattern (module 14/15).
- **Pickle**: Unsafe deserialization format; risky for checkpoints (module 20).
- **Pinning (memory)**: Preventing pages from being paged out / establishing DMA-suitable residency (module 07).
- **Post-training quantization (PTQ)**: Quantize after training using calibration (module 06).
- **PVH / HVM / PV**: Xen virtualization modes (module 12).
- **Provider options**: EP-specific configuration dict (module 05/08).

## Q

- **QDQ**: Quantize–Dequantize pattern nodes around ops in some ONNX quantizations.
- **QuantizeLinear / DequantizeLinear**: ONNX ops representing linear quantization.
- **QuantType**: ORT quantization API enum selecting integer types/flags.
- **Qwen3**: Example LLM name in your context.

## R

- **RAFT-Stereo**: Stereo matching / optical flow style architecture family; export can hit custom op issues (module 13).
- **readahead**: Block device prefetch tuning (module 16/11).
- **ReLU / GELU / Sigmoid**: Nonlinear activation ops.
- **ROCm**: AMD open compute stack (module 08).
- **rocm-smi**: CLI tool for GPU monitoring (module 08).
- **r1-gpu-perf.service**: Example systemd unit name from your notes applying clocks/sysctl/THP/NVMe tuning (module 11).

## S

- **SafeTensors**: Safer weight container than pickle-based `.bin` (module 13/20).
- **sched_setaffinity**: Linux API to pin process to CPUs (module 15).
- **Segmentation**: Per-pixel class masks (SAM-like).
- **SessionOptions**: ORT session configuration object (module 05).
- **Shape inference**: See ONNX shape inference (module 04).
- **Split lock**: Atomic op crossing cache line; CPUs may penalize (module 10).
- **Static shape**: All dimensions known as integers for a given benchmark configuration.
- **Static shape pinning**: Forcing symbolic dims to ints via ORT overrides or re-export (module 06).
- **Subgraph split (export workaround)**: Splitting huge models to avoid compiler limits (module 13).
- **Swap**: Disk-backed anonymous memory overflow (module 09).
- **SVM / SVM IOCTL**: Driver interface activity around shared virtual memory workflows (module 07).
- **sysctl**: Runtime kernel tunables under `/proc/sys` (module 09).
- **systemd**: Service manager for boot-time tuning (module 11).

## T

- **taskset**: Shell utility to set CPU affinity (module 15).
- **Telemetry**: Measurement beyond raw latency (temps, clocks, dmesg deltas).
- **TF32 (mention)**: NVIDIA tensor core mode; not AMD-specific; appears in broad ML docs.
- **tf2onnx**: TensorFlow → ONNX converter (module 13).
- **THP**: Transparent Huge Pages (module 09/10).
- **Throughput**: Inferences per second (module 14).
- **tie_weights (HF)**: Weight tying behavior that can complicate export (module 13).
- **TLB**: Translation lookaside buffer for address translation caching (module 07/09).
- **Tokenizer (mention)**: Text preprocessing for LLMs; often outside ONNX file.
- **Torch export / torch.onnx.export**: PyTorch ONNX export APIs (module 13).
- **Tracing (export)**: Run-once recording of ops (module 13).
- **Transformer**: Architecture family using attention (module 03).
- **TTFT**: Time-to-first-token for LLM UX (module 14).
- **TTS**: Text-to-speech (XTTS-like).

## U

- **Unified memory**: CPU/GPU shared memory programming model (module 07).
- **UNIX domain sockets / etc.**: Not central; ignore unless networking models.

## V

- **Validation (ONNX)**: Structural model validity, separate from accuracy.
- **vCPUs (Xen)**: Virtual CPUs seen inside a domain (module 12).
- **ViNT**: Visual navigation transformer/policy name (robotics domain label).
- **VRAM carve-out**: Dedicated graphics memory region; often small on APUs (module 07).

## W

- **Warmup**: Discarded initial runs (module 14).
- **Wavefront**: GPU execution grouping concept (module 08).
- **Weights**: Learned parameters tensors stored as ONNX initializers.
- **Whisper**: ASR model family name.

## X

- **Xen**: Type-1 hypervisor (module 12).
- **`xl info` / `xl list`**: Xen management commands (module 12).
- **XNACK**: Mechanism related to fault/retry behaviors in some GPU memory configurations (module 07).
- **XTTS**: TTS model family name.
- **XTT / XTTS**: keep XTTS only (typo guard).

## Y

- **YOLO**: You Only Look Once object detection family.

## Z

- **Zero-point (quantization)**: Integer value representing “zero” in quantized space.
- **sysctl vm.watermark_boost_factor**: Watermark boosting behavior; 0 disables (module 09).

---

## Explicit mapping: optimization steps you listed

- **ORT graph optimizations (`ORT_ENABLE_ALL`)**: Module 06 + cheatsheet.
- **onnxsim**: Module 06 + playbook 18.
- **onnx shape inference**: Modules 04/06.
- **FP16 EP flags (`ort_migraphx_fp16` style)**: Modules 06/08 (verify option names on your build).
- **INT8 static quantization + calibration reader**: Module 06.
- **per-channel quantization**: Module 06 glossary (`per-channel`).
- **Manual fusion targets (LayerNorm/GELU/attention)**: Modules 06/13/19.
- **Static shape pinning / `dim_overrides`**: Modules 06/18.
- **MIGraphX exhaustive tuning + compile cache paths**: Modules 08/18 (verify availability).
- **Memory layout NCHW/NHWC**: Module 02.
- **Pinned VRAM vs unified (`hipMalloc` vs `hipMallocManaged`)**: Module 07.
- **Reduce host-device transfers**: Modules 06/16.
- **GPU clock locking / thermal validation**: Modules 08/11/14.
- **CPU C-states / idle=nomwait**: Module 10.
- **THP madvise**: Modules 09/10/11.
- **MGLRU**: Modules 09/11.
- **Concurrent streams / pipelining**: Module 08.
- **Transformer optimizer (`onnxruntime.transformers.optimizer`)**: Modules 06/13/18/19.

---

## Additional terms (added during deep audit)

- **Cache line**: 64-byte unit of CPU memory access; split locks cross cache line boundaries (module 10).
- **DDR5**: Memory type in your APU; shared between CPU and GPU; higher bandwidth than DDR4.
- **DMA (Direct Memory Access)**: Hardware reads/writes RAM directly without CPU copying bytes; how GPU accesses weights in system RAM (module 10).
- **dmesg**: Kernel log ring buffer; `dmesg --level=warn,err` shows warnings/errors; critical for debugging GPU/driver issues.
- **fstab (`/etc/fstab`)**: File defining filesystem mounts and swap at boot; your 400 GB swap is configured here (module 16).
- **hipMalloc**: AMD HIP function allocating VRAM-only memory; limited to 512 MB on your APU (module 07).
- **hipMallocManaged**: AMD HIP function allocating unified/managed memory; accesses full 28 GB GTT; used in your custom ORT build (module 07).
- **MGLRU (Multi-Gen LRU)**: Modern page reclaim algorithm (Linux 6.1+) tracking multiple generations of page access for smarter eviction (module 09/11).
- **NMS (Non-Max Suppression)**: Post-processing step for object detection that filters overlapping bounding boxes (CPU-side, after YOLO inference).
- **Page table**: Data structure mapping virtual addresses to physical RAM addresses; GPU and CPU each have page tables.
- **RDNA 3.5**: AMD GPU architecture in your Strix Halo APU; 40 CUs; supports FP16 natively (module 08).
- **rocminfo**: CLI tool showing GPU hardware details: CU count, SIMDs, ISA, memory sizes, agent capabilities (module 08).
- **SMT (Simultaneous Multithreading)**: 2 hardware threads per physical core; your 16 cores × 2 SMT = 32 threads (module 12).
- **Strix Halo**: Your AMD APU model; CPU + integrated RDNA 3.5 GPU on same package; 512 MB VRAM + 28 GB GTT.
- **Swap**: Disk-backed overflow when physical RAM is full; your 400 GB on NVMe is much faster than HDD swap (module 09/16).
- **vm.overcommit_memory**: Kernel sysctl controlling malloc behavior; `1` = always allow; critical for large model loading (module 09).
- **vm.swappiness**: Kernel sysctl controlling anonymous page swap aggressiveness; `10` = prefer keeping process data in RAM (module 09).
- **vm.max_map_count**: Max VMAs per process; `2097152` prevents mapping exhaustion for large models (module 09).

---

If a term is missing, add it here and link it to the module you think fits best—this glossary is meant to grow with your repo.
