# Module 16 — Environment variables, shared libraries, storage & I/O (full depth)

---

## 1. Environment variables — key/value pairs for all processes

Set in `/etc/environment` for system-wide persistence:

```bash
HSA_XNACK=1
```

This ensures **login sessions** see `HSA_XNACK=1` (via PAM). Note: systemd services do **not** automatically inherit `/etc/environment` — use `EnvironmentFile=` in unit files or set vars in the service script.

### Critical environment variables in your setup

| Variable | Value | Purpose |
|----------|-------|---------|
| `HSA_XNACK` | `1` | Enable GPU demand paging (Module 07) |
| `PATH` | system + ROCm + custom paths | Where to find executables (`rocm-smi`, `python`, etc.) |
| `LD_LIBRARY_PATH` | ROCm lib paths | Where to find `.so` shared libraries at runtime |

### Setting environment variables

```bash
# Temporary (current shell only)
export HSA_XNACK=1

# Permanent (all users, all processes)
echo 'HSA_XNACK=1' | sudo tee -a /etc/environment

# In Python
import os
os.environ["HSA_XNACK"] = "1"
print(os.environ.get("HSA_XNACK"))  # "1"
```

---

## 2. Shared libraries (`.so` files)

Linux programs load **shared objects** (`.so`) at runtime via the dynamic linker (`ld.so`).

**ROCm, MIGraphX, and ORT are all `.so` files:**

```
libonnxruntime.so          # ONNX Runtime
libmigraphx.so             # MIGraphX graph compiler
libamdhip64.so             # HIP runtime
librocblas.so              # ROCm BLAS
libMIOpen.so               # MIOpen DL primitives
```

### `ldd` — see what libraries a binary needs

```bash
ldd /path/to/libonnxruntime.so
```

Shows which `.so` files must be found at runtime. If any says "not found," the program will crash on load.

### `LD_LIBRARY_PATH` — where to find `.so` files

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

**Warning:** use sparingly. Wrong `LD_LIBRARY_PATH` can make programs find the **wrong version** of a library, causing subtle crashes.

---

## 3. Custom builds vs pip wheels

| Approach | Pros | Cons |
|----------|------|------|
| `pip install onnxruntime` | Easy, fast | **Does NOT include MIGraphX EP** |
| Custom source build | Full control: choose EPs, allocators, flags | Requires matching ROCm version, CMake knowledge, build time |

**Your setup uses a custom ORT build** because:
- Standard pip wheel doesn't include MIGraphXExecutionProvider
- You need `hipMallocManaged` allocator (not in standard builds)
- Must match your exact ROCm version

### CMake — the build system

CMake configures what gets compiled:

```bash
cmake -DUSE_MIGRAPHX=ON -DUSE_HIP_MANAGED_MEM=ON ...
make -j$(nproc)
```

Exact flags change by ORT version. Always read the build docs for your specific tag.

---

## 4. NVMe storage — your Samsung 990 PRO

**NVMe** (Non-Volatile Memory Express) is a fast SSD interface:

| Metric | HDD | SATA SSD | NVMe SSD (your 990 PRO) |
|--------|-----|----------|------------------------|
| Sequential read | ~150 MB/s | ~550 MB/s | ~7,000 MB/s |
| Random IOPS | ~100 | ~90,000 | ~500,000+ |
| Latency | ~10 ms | ~0.1 ms | ~0.02 ms |

**Why it matters:** loading a 3.4 GB model from NVMe takes ~0.5 seconds. From HDD it would take ~23 seconds.

---

## 5. Readahead — NVMe optimization

Your `r1-gpu-perf.service` sets:

```bash
echo 2048 > /sys/block/nvme0n1/queue/read_ahead_kb
```

This tells the OS to pre-fetch **2048 KB (2 MB)** of upcoming data when reading sequentially. Helps when loading large ONNX model files.

---

## 6. Sequential vs random I/O

| Pattern | Example | NVMe speed |
|---------|---------|-----------|
| **Sequential** | Loading a 3 GB model file end-to-end | Very fast (~7 GB/s) |
| **Random** | Many small reads from scattered dataset files | Fast but limited by IOPS |

Model weight reads are mostly large sequential reads — good for SSDs.

---

## 7. `/etc/fstab` — mount configuration at boot

Your `fstab` defines:
- **Filesystem mounts** (root, boot, data partitions)
- **Swap files** — your 400 GB swap across 4 files on NVMe

```
/swapfile1  none  swap  sw  0 0
/swapfile2  none  swap  sw  0 0
/swapfile3  none  swap  sw  0 0
/swapfile4  none  swap  sw  0 0
```

**400 GB swap on NVMe** means:
- ML processes almost never get OOM-killed
- Swap access is fast (~7 GB/s vs ~150 MB/s on HDD)
- But swap during timed inference = latency spike (Module 14)

---

## 8. Page cache effects on benchmarks

Linux caches file reads in RAM (**page cache**):

| Run | What happens | Speed |
|-----|-------------|-------|
| First `onnx.load("model.onnx")` after boot | Reads from NVMe | ~7 GB/s |
| Second `onnx.load("model.onnx")` | Reads from page cache in RAM | ~40+ GB/s |

**Benchmark note:** if you're comparing "before optimization" vs "after optimization" runs, both should be warm-cache (or both cold-cache). Mixing invalidates the comparison.

---

## 9. `/proc` and `/sys` — Linux virtual filesystems

| Path | What it exposes |
|------|----------------|
| `/proc/cpuinfo` | CPU details (model, cores, frequencies, flags) |
| `/proc/meminfo` | RAM usage (total, free, available, swap) |
| `/proc/cmdline` | Kernel boot parameters |
| `/sys/module/amdgpu/parameters/` | amdgpu driver parameters (gttsize, etc.) |
| `/sys/kernel/mm/transparent_hugepage/enabled` | THP mode |
| `/sys/block/nvme0n1/queue/read_ahead_kb` | NVMe readahead setting |

---

## Module 16 checklist

- [ ] Can explain `LD_LIBRARY_PATH` and its risk (wrong library version)
- [ ] Can explain why swap on NVMe is different from swap on HDD
- [ ] Can explain page cache effect on repeated model loads
- [ ] Know why a custom ORT build is required (MIGraphX EP + hipMallocManaged)
- [ ] Can use `ldd` to check library dependencies
- [ ] Know where `/etc/environment` sets system-wide env vars

**Next:** `17-model-families-you-will-meet.md`
