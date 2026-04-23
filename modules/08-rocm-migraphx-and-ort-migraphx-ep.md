# Module 08 — ROCm, MIGraphX, GPU tools, CUs, clocks (full depth)

---

## 1. ROCm — AMD's GPU compute platform

**ROCm** (Radeon Open Compute) is AMD's equivalent of NVIDIA's CUDA. It includes:

- **HIP**: GPU programming layer (write GPU kernels, allocate GPU memory)
- **MIOpen**: optimized deep learning primitives (convolutions, pooling, etc.)
- **MIGraphX**: graph compiler for neural networks
- **rocBLAS, rocFFT, etc.**: math libraries
- **Driver + runtime**: kernel module (`amdgpu`) + user-space libraries

**Critical:** your ORT build, MIGraphX, and HIP must all be built against the **same ROCm version**. Version mismatch = crashes or missing symbols.

---

## 2. MIGraphX — graph compiler for neural networks

MIGraphX takes a computation graph (from ONNX via ORT) and:

1. **Parses** the graph
2. **Optimizes** (fuses ops, selects implementations)
3. **Compiles** each op/fused region into a GPU **kernel**
4. **Executes** the compiled program

**Kernel implementations available to MIGraphX:**
- **Assembly kernels**: hand-written GPU assembly (fastest for specific ops)
- **Composable Kernels**: AMD's library of composable GPU kernel primitives
- **MIOpen**: high-level DL primitives (convolutions, etc.)

**Exhaustive tuning** tries all implementations per op and picks the fastest.

---

## 3. ORT's MIGraphX EP

The `MIGraphXExecutionProvider` routes supported subgraphs/nodes to MIGraphX:

```python
providers = [("MIGraphXExecutionProvider", {
    "device_id": 0,
    "migraphx_fp16_enable": True,
    "migraphx_exhaustive_tune": True,
    "migraphx_save_compiled_model": True,
    "migraphx_load_compiled_model": True,
    "migraphx_model_cache_path": "/home/sudhdevu/R1models/.migraphx_cache/"
})]
```

**What you observe:**
- First run(s) may take minutes (compilation)
- Later runs reuse compiled artifacts (fast)
- Unsupported ops fall back to CPU (graph partitioning)

---

## 4. Compute Units (CUs) — your GPU cores

Your Strix Halo GPU has **40 CUs** (Compute Units).

Each CU contains:
- Multiple **SIMDs** (Single Instruction, Multiple Data units)
- Each SIMD executes a **wavefront** (group of threads executing in lockstep, typically 32 or 64 threads)

**More CUs = more parallel throughput** for compute-heavy ops.

---

## 5. Clock speeds — what to monitor

| Clock | Name | Your locked speed | What it controls |
|-------|------|-------------------|-----------------|
| **sclk** | Shader clock (GPU core) | 2900 MHz | Compute speed |
| **mclk** | Memory clock | 1000 MHz | Memory bandwidth |
| **fclk** | Fabric/Infinity Fabric clock | 2000 MHz | Interconnect speed |

Your `r1-gpu-perf.service` locks these to max to prevent dynamic scaling during benchmarks.

**If clocks drop during inference → thermal throttling.** Results are invalid.

---

## 6. ROCm / AMD GPU tools reference

### `rocm-smi` — GPU monitoring (your most-used tool)

```bash
rocm-smi                          # Quick overview
rocm-smi --showtemp               # GPU temperature
rocm-smi --showclocks             # Current clock speeds (sclk, mclk)
rocm-smi --showmeminfo all        # VRAM and GTT usage
rocm-smi --showall                # Everything at once
rocm-smi --setperflevel high      # Lock GPU to max performance
```

**During benchmarks:** run `rocm-smi --showclocks` in another terminal to verify clocks aren't throttling.

### `rocminfo` — GPU hardware details

```bash
rocminfo
```

Shows: CU count, SIMDs per CU, ISA (instruction set architecture), memory sizes, agent capabilities.

### `hipcc` — HIP compiler

AMD's CUDA-equivalent compiler. Compiles `.hip` / `.cpp` files to GPU code.

```bash
hipcc --version
```

### `hipconfig` — HIP/ROCm configuration

```bash
hipconfig --full
```

Shows: HIP version, platform, compiler, ROCm path.

### `amdgpu` driver sysfs — driver parameters as files

```bash
ls /sys/module/amdgpu/parameters/
cat /sys/module/amdgpu/parameters/gttsize    # current GTT size
```

These are readable (and some writable) files exposing driver configuration.

---

## 7. Kernel caching

After MIGraphX compiles a model's kernels, it can **cache** the compiled program:

- **First run:** slow (compilation)
- **Second run onward:** fast (loads from cache)

Your benchmark already benefits from warmup runs, which trigger compilation so timed runs use cached kernels.

With `migraphx_save_compiled_model=True` + `migraphx_load_compiled_model=True`, the cache persists across process restarts.

**Cache invalidation:** changing ROCm/ORT versions, changing model graph, or changing provider options invalidates the cache.

---

## 8. Streams and overlapping execution (advanced)

GPUs use **streams** to queue work. Independent operations on different streams can overlap:

- While one batch computes on GPU, the next batch's data transfers
- MIGraphX supports multiple streams for this kind of pipelining

**Benchmark caveat:** if you measure host-side wall time without synchronizing, you can miss async behavior. ORT's session.run() typically synchronizes at return for most EP paths, but verify for your build.

---

## 9. Memory layout verification

NCHW vs NHWC: RDNA 3.5 prefers certain data layouts for certain ops. MIGraphX handles layout transforms automatically during compilation, but you can verify what layouts the compiled program uses:

```cpp
// C++ MIGraphX API (for reference — you typically won't call this directly)
auto shapes = program.get_parameter_shapes();
// Returns the shapes and layouts of each parameter in the compiled program
```

From Python via ORT, you can inspect the optimized graph (Module 19) to see if layout transforms were inserted:

```python
import onnx
model = onnx.load("model_optimized.onnx")
transpose_count = sum(1 for n in model.graph.node if n.op_type == "Transpose")
print(f"Transpose nodes (layout transforms): {transpose_count}")
```

If many `Transpose` nodes appear in the optimized graph, the model may benefit from being exported in a different layout.

---

## 10. MIGraphX environment variables — MLIR vs rocBLAS

MIGraphX uses its **MLIR compiler** by default to fuse operations (including GEMMs) into single kernels. You can override this to use hand-tuned **rocBLAS Tensile kernels** instead:

```bash
export MIGRAPHX_DISABLE_MLIR=1            # disable MLIR fusion; GEMMs go to rocBLAS
export MIGRAPHX_SET_GEMM_PROVIDER=rocblas  # force rocBLAS for GEMM ops
```

Or via `benchmark_cooldown.py`:

```bash
python benchmark_cooldown.py --ep migraphx --disable-mlir --gemm-provider rocblas --warmup 3 --runs 3
```

**Performance impact (measured on OpenVLA-7B, gfx1151):**

| Mode | Total GPU kernel time | GEMM avg per call |
|------|-----------------------|-------------------|
| MLIR (default) | 1,798 ms | 1.98 ms |
| rocBLAS (MLIR disabled) | 1,594 ms | 1.72 ms |
| **Delta** | **-11.4% faster** | **-13% faster** |

On gfx1151 (RDNA 3.5), rocBLAS Tensile kernels are faster for GEMMs because they are hand-tuned for the ISA. Non-GEMM kernels (conv, softmax, sigmoid) are unaffected.

**When to try this:** If your model is GEMM-heavy (transformers, LLMs, VLMs), test with `--disable-mlir --gemm-provider rocblas` and compare.

---

## 11. Thermal throttling — the benchmark killer

When GPU temperature exceeds limits:

1. Hardware reduces clock speeds to protect itself
2. Your sclk drops from 2900 MHz to maybe 1800 MHz
3. Latency increases 50%+
4. **Results are invalid** — you're benchmarking thermal management, not model performance

**Monitor:** `rocm-smi --showtemp` before/during/after inference.

**Mitigate:** cooldown periods between models (your `--cooldown 120` flag).

---

## Module 08 checklist

- [ ] Can explain MIGraphX's role: ONNX graph → compiled GPU kernels
- [ ] Can name the three kernel implementation strategies (assembly, composable, MIOpen)
- [ ] Can use `rocm-smi` to check temperature, clocks, and memory
- [ ] Can explain why first run is slow (compilation) and subsequent runs are fast (cache)
- [ ] Can explain what CUs are and why 40 CUs matters
- [ ] Can explain thermal throttling and how to detect it with `rocm-smi --showclocks`
- [ ] Can explain memory layout verification and what many Transpose nodes indicate

**Next:** `09-linux-memory-sysctl-and-reclaim.md`
