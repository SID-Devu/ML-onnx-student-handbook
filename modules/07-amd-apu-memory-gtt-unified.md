# Module 07 — AMD APU memory: VRAM, GTT, unified memory, HSA, XNACK, SVM (full depth)

This module explains **why your APU with "512 MB VRAM" can run multi-GB models** — and the costs you pay.

---

## 1. VRAM (dedicated framebuffer carve-out)

Your Strix Halo APU has **512 MB** of dedicated VRAM (Video RAM) carved out from system memory at boot.

This is fast, local GPU memory — but **512 MB is far too small** for most ML models (YOLO alone is ~80 MB, Qwen3 is ~3.4 GB).

**With only hipMalloc (VRAM only):** you'd be limited to models under 512 MB. That's why your setup uses unified memory.

---

## 2. GTT (Graphics Translation Table) — the big pool

**GTT** = system RAM made accessible to the GPU through GPU page tables.

Your kernel boot parameter:

```
amdgpu.gttsize=28672
```

This sets **28 GB** of system RAM (out of your 30 GB total) as GPU-accessible via the GTT.

**Not the same as "free RAM":** GTT is an address-space / mapping budget. The GPU can access pages within this budget via address remapping.

---

## 3. GART (Graphics Address Remapping Table)

**GART** is the hardware support that translates GPU virtual addresses to physical addresses in system RAM.

You rarely configure GART directly — it's the mechanism underneath GTT that makes the mapping work. When the GPU accesses system RAM through GTT, GART handles the address translation.

---

## 4. Unified Memory — CPU and GPU share the same physical RAM

On your APU:

- CPU and GPU are on the **same chip** (same package)
- They share the **same physical DRAM** (30 GB DDR5)
- With proper configuration, they can share the **same virtual address space**

This is fundamentally different from discrete GPUs where CPU RAM and GPU VRAM are separate physical pools connected by a PCIe bus.

**Advantage:** no PCIe bottleneck, can oversubscribe "GPU memory" using system RAM
**Cost:** bandwidth is shared, and page management overhead exists

---

## 5. HSA (Heterogeneous System Architecture)

**HSA** is AMD's framework for CPU-GPU cooperation on APUs.

Key idea: CPU and GPU are "peers" that can access each other's memory, dispatch work to each other, and share virtual addresses.

Your environment variable:

```
HSA_XNACK=1
```

Set in `/etc/environment` so ALL processes see it.

---

## 6. XNACK — demand paging for the GPU

**XNACK** is a hardware retry mechanism for GPU memory accesses:

### `HSA_XNACK=1` (your production setting)

- When the GPU accesses a page **not yet mapped**, it triggers a **page fault**
- The system maps/migrates the page on demand and **retries** the access
- This is **demand paging** — like virtual memory for the GPU
- **Why it works for large models:** pages are brought in as needed, not all upfront
- **Cost:** first-touch latency (each new page access may stall briefly)

### `HSA_XNACK=0` (the test you ran)

- GPU accesses pages that **must be pre-mapped/pre-pinned**
- No retry on fault — pages must be explicitly managed
- Runtime uses **SVM IOCTLs** to pin/migrate pages between CPU and GPU
- **What you saw in dmesg:** 750+ SVM messages as the driver explicitly pinned/migrated pages
- **Cost:** upfront setup time, explicit management overhead

### Which is better?

| | XNACK=1 | XNACK=0 |
|--|---------|---------|
| First-access behavior | Page fault + retry (demand paging) | Must pre-pin (explicit SVM IOCTLs) |
| dmesg noise | Quiet | 750+ SVM messages per model load |
| Steady-state performance | Good (pages resident after first touch) | Potentially better (no fault overhead) |
| Ease of use | Simple | Complex driver interaction |
| **Your choice** | **YES — production setting** | Test only |

---

## 7. SVM (Shared Virtual Memory) and SVM IOCTLs

**SVM** = CPU and GPU see the same virtual addresses. A pointer allocated on one side is valid on the other.

**SVM IOCTLs** = system calls the GPU driver uses to:
- **Pin** pages in physical memory (prevent swapping, make DMA-safe)
- **Migrate** pages between CPU and GPU preferred locations

With `XNACK=0`, you saw 750+ of these messages in `dmesg` because the runtime had to explicitly pre-pin every memory region the GPU would access.

---

## 8. `hipMalloc` vs `hipMallocManaged` — why your custom ORT build exists

| Allocator | What it does | Memory limit | Your use |
|-----------|-------------|-------------|----------|
| `hipMalloc` | Allocates in VRAM only | **512 MB** (your VRAM carve-out) | Standard ORT builds use this |
| `hipMallocManaged` | Allocates in unified memory (GPU-visible system RAM) | **~28 GB** (your GTT) | **Your custom ORT build uses this** |

**This is why you have a custom ORT build:** the standard build uses `hipMalloc`, which would fail for any model larger than 512 MB. Your custom build patches the allocator to use `hipMallocManaged`, allowing models to use the full 28 GB GTT.

### Optimization tip for small models

For models that fit entirely within 512 MB VRAM (e.g., MobileNetV2 ~14 MB, DeepSort ~20 MB), using `hipMalloc` (pinned VRAM) instead of `hipMallocManaged` would **avoid page migration overhead**. Pages stay in fast VRAM without demand-paging faults. However, your custom ORT build uses `hipMallocManaged` for all models — the unified path is simpler and the performance difference for small models is usually minor compared to the benefit of supporting large models.

---

## 9. Page migration and pinning

### Page migration

When the OS/driver determines a page would be faster in a different location (e.g., closer to GPU), it copies the page:

- **CPU → GPU-preferred:** happens when GPU repeatedly accesses a CPU-resident page
- **GPU → CPU-preferred:** can happen when CPU needs to read GPU output

**Benchmark impact:** cold start (first inference after load) triggers migration → slower. Steady state (pages already in preferred locations) → fast.

### Page pinning

Locking pages in physical memory so they:
- Can't be swapped to disk
- Are safe for DMA (GPU hardware can read them directly)
- Stay in a known location

---

## 10. TLB and hugepages

**TLB (Translation Lookaside Buffer)** caches virtual-to-physical address translations.

With standard 4 KB pages and 28 GB GTT:
- Need 28 GB / 4 KB = ~7.3 million TLB entries
- TLBs are small (thousands of entries) → many TLB misses → expensive

With **2 MB huge pages** (THP):
- Need 28 GB / 2 MB = ~14,000 entries
- Far fewer TLB misses → better GPU memory access performance

This is why your setup uses `transparent_hugepage=madvise` (Module 10).

---

## 11. Memory bandwidth considerations

Your DDR5 memory is shared between CPU and GPU. When both are active:

- GPU inference competes with CPU preprocessing for memory bandwidth
- This is inherent to APU architecture (no separate VRAM with dedicated bandwidth)
- Mitigated by: keeping GPU work on-device (avoid host↔device round-trips), using FP16 (halves bandwidth needs)

---

## 12. Reduce host-device transfers

ORT's MIGraphX EP keeps intermediate tensors on GPU. But custom pre/post-processing on CPU can cause round-trips:

- **Bad:** CPU preprocess → copy to GPU → inference → copy to CPU → CPU postprocess → copy to GPU → next inference
- **Good:** CPU preprocess → copy to GPU → inference → next inference (keep data on GPU between stages)

---

## 13. Verifying your memory setup (commands)

### Check GTT size

```bash
cat /sys/module/amdgpu/parameters/gttsize
# Should output: 28672  (in MB)
```

### Check VRAM and GTT usage

```bash
rocm-smi --showmeminfo all
```

### Check XNACK setting

```bash
echo $HSA_XNACK
# Should output: 1

# Or check from /etc/environment
cat /etc/environment | grep HSA_XNACK
```

### Check kernel boot params

```bash
cat /proc/cmdline
# Look for: amdgpu.gttsize=28672 amdgpu.no_system_mem_limit=1
```

### Monitor memory during model load (Python)

```python
import os

def get_rss_mb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0

print(f"Before load: {get_rss_mb():.0f} MB")
# ... load model ...
print(f"After load:  {get_rss_mb():.0f} MB")
```

### Count SVM messages (XNACK experiments)

```bash
dmesg | grep -ci "svm"
# With XNACK=0 you saw 750+ messages per model load
# With XNACK=1 you see far fewer (demand paging handles it silently)
```

---

## 14. XNACK=0 vs XNACK=1 — performance tradeoffs (measured)

On OpenVLA-7B (14 GB model on Strix Halo with 32 GB unified memory):

| Metric | XNACK=1 (demand paging) | XNACK=0 (explicit) | Winner |
|--------|-------------------------|---------------------|--------|
| GPU kernel compute | 1,696 ms | 1,603 ms | XNACK=0 (-5.5%) |
| Memory copies (H2D) | 220 ms | 421 ms | XNACK=1 (-48%) |
| Memory allocation | 1,339 ms | 1,084 ms | XNACK=0 (-19%) |
| Memory free | 184 ms | 99 ms | XNACK=0 (-46%) |

**Rule of thumb:**
- **XNACK=0 is faster** when the model fits in available memory (no oversubscription) — avoids TLB page-fault overhead on every kernel dispatch
- **XNACK=1 is essential** when the model oversubscribes GPU memory — the GPU can demand-page from system RAM instead of crashing with out-of-memory

For your 28 GB GTT: models under ~25 GB → prefer XNACK=0. Models that exceed available memory → must use XNACK=1.

---

## 15. `hipMemAdviseSetCoarseGrain` — coherence optimization

When using `hipMallocManaged`, the default is fine-grained coherence (CPU and GPU see each other's writes immediately). For read-only weights, **coarse-grained** coherence avoids the per-access coherence overhead:

```c
hipMemAdviseSetCoarseGrain(ptr, size, device);
```

Your custom ORT build uses this for weight buffers that don't change after loading. This reduces memory access latency for large models on unified memory.

---

## Module 07 checklist

- [ ] Explain VRAM carve-out (512 MB) vs GTT (28 GB) in your own words
- [ ] Explain why large models (Qwen3 = 3.4 GB) can run despite 512 MB VRAM
- [ ] Explain XNACK=1 vs XNACK=0 and what you observed (750+ SVM messages)
- [ ] Explain why `hipMallocManaged` is needed and why your ORT build is custom
- [ ] Explain why THP helps with TLB pressure for large GPU memory mappings
- [ ] Explain demand paging in one sentence

**Next:** `08-rocm-migraphx-and-ort-migraphx-ep.md`
