# Module 09 — Linux virtual memory: sysctl tuning with your exact values (full depth)

Every `vm.*` knob from your `r1-gpu-perf.service` and sysctl config is explained here with your actual values.

---

## 1. Virtual memory basics

Each process has a **virtual address space** mapped to physical RAM (and sometimes swap) via **page tables**. The kernel balances fast access, fairness, and OOM avoidance.

**Page size:** 4 KB (standard) or 2 MB (huge pages / THP).

---

## 2. Your exact sysctl settings

| Parameter | Your value | Default | What it does |
|-----------|-----------|---------|-------------|
| `vm.swappiness` | **10** | 60 | How aggressively kernel swaps anonymous pages |
| `vm.overcommit_memory` | **1** | 0 | Memory allocation policy |
| `vm.vfs_cache_pressure` | **50** | 100 | How aggressively kernel reclaims inode/dentry cache |
| `vm.watermark_boost_factor` | **0** | 15000 | Boost kswapd watermarks after reclaim events |
| `vm.compaction_proactiveness` | **0** | 20 | How aggressively kernel compacts memory for hugepages |
| `vm.dirty_ratio` | **10** | 20 | Max % of RAM as dirty pages before writer blocks |
| `vm.dirty_background_ratio` | **3** | 10 | % of RAM as dirty pages before background flush starts |
| `vm.zone_reclaim_mode` | **0** | 0 | NUMA local zone reclaim behavior |
| `vm.max_map_count` | **2097152** | 65530 | Max memory-mapped regions per process |

---

## 3. `vm.swappiness = 10` — keep process memory in RAM

Controls how readily the kernel moves **anonymous** (non-file-backed) pages to swap vs reclaiming **page cache** (file-backed pages).

- **0:** avoid swap as much as possible (but doesn't disable swap — kernel will still swap under extreme pressure)
- **10 (yours):** strongly prefer keeping process data in RAM; reclaim page cache first
- **60 (default):** balanced
- **100:** swap aggressively

**Why 10:** ML model tensors are anonymous memory. Swapping them to disk mid-inference causes massive latency spikes (milliseconds of SSD I/O instead of nanoseconds of RAM).

---

## 4. `vm.overcommit_memory = 1` — never fail malloc

Linux can allow allocations that sum to more than physical RAM because many programs over-allocate.

- **0 (default):** heuristic — kernel guesses whether the allocation is reasonable, may reject
- **1 (yours):** always allow allocation (never fail malloc) — the OOM killer is still the last resort if you actually *use* more than available
- **2:** strict — limit total commit to RAM + swap × ratio

**Why 1:** ML frameworks often allocate large arenas they may never fully use. With mode 0, `malloc` can fail during model loading and crash the process. Mode 1 defers the problem to actual usage (OOM killer handles true exhaustion).

---

## 5. `vm.vfs_cache_pressure = 50` — keep filesystem cache longer

Controls how aggressively the kernel reclaims **inode/dentry cache** (filesystem metadata kept in RAM).

- **0:** never reclaim VFS cache
- **50 (yours):** half the default pressure — keep filesystem cache longer
- **100 (default):** balanced

**Why 50:** when loading large ONNX files from disk, cached directory entries and inodes help repeated loads.

---

## 6. `vm.watermark_boost_factor = 0` — disable watermark boosting

When memory is reclaimed, the kernel can temporarily **boost kswapd watermarks** (making kswapd start reclaiming earlier). Setting to 0 disables this.

**Why 0:** avoids unnecessary early reclaim that can cause latency perturbations during benchmarks.

---

## 7. `vm.compaction_proactiveness = 0` — disable proactive compaction

**Compaction** = kernel rearranges physical pages to create contiguous free blocks (needed for huge pages).

Proactive compaction runs in the background to prepare huge pages in advance. This causes **latency stalls** — the kernel pauses to move pages around.

**Why 0:** latency stability is more important than having huge pages ready instantly. With `madvise` THP mode, huge pages are created only when explicitly requested.

---

## 8. `vm.dirty_ratio = 10` / `vm.dirty_background_ratio = 3`

**Dirty pages** = RAM pages modified but not yet flushed to disk.

- `dirty_background_ratio = 3`: when 3% of RAM is dirty, start **background** flush (non-blocking)
- `dirty_ratio = 10`: when 10% of RAM is dirty, **block** the writing process until flush catches up

**Why lower than defaults:** prevents large dirty-page backlogs from causing sudden I/O stalls when writing benchmark results or logs.

---

## 9. `vm.zone_reclaim_mode = 0` — no NUMA zone reclaim

On NUMA machines, the kernel can try to reclaim memory from the local zone before going to remote zones.

**Why 0:** your APU is effectively single-node; zone reclaim adds overhead without benefit.

---

## 10. `vm.max_map_count = 2097152` — allow many memory mappings

Maximum number of **VMAs** (Virtual Memory Areas / memory-mapped regions) a process can have.

Default is 65530. Large ONNX loads + mmap'd weights + frameworks can exceed this.

**Why 2097152 (2 million):** prevents `cannot allocate memory` errors during session creation for multi-GB models with many mapped regions.

---

## 11. MGLRU (Multi-Gen LRU) — smarter page reclaim

**MGLRU** is a modern page reclaim algorithm (merged in Linux 6.1+) that tracks page access patterns across multiple "generations."

Standard LRU: recently accessed = keep, old = evict (simple but wrong when a scan passes through once).

MGLRU: tracks **multiple generations** of access → smarter eviction that doesn't throw away hot pages just because a sequential scan pushed them to the "old" end.

**Enabled in your `r1-gpu-perf.service`** via sysctl writes.

---

## 12. THP (Transparent Huge Pages)

THP automatically promotes 4 KB pages to 2 MB huge pages when possible.

**Modes:**

| Mode | Behavior | Risk |
|------|----------|------|
| `always` | THP everywhere | Compaction stalls (latency spikes) |
| `madvise` **(yours)** | THP only where apps request via `madvise()` | Best balance for latency-sensitive work |
| `never` | No THP | More TLB misses |

Your boot param: `transparent_hugepage=madvise`
Your service also writes to `/sys/kernel/mm/transparent_hugepage/enabled`.

---

## 13. kswapd and OOM killer

- **kswapd:** kernel daemon that reclaims pages in the background when memory gets low. Controlled by swappiness and watermark settings.
- **OOM killer:** when memory is truly exhausted and nothing can be reclaimed, kernel kills a process (chosen by heuristics — usually the biggest memory consumer).

**Benchmark reality:** OOM during a suite = not comparable results. Your `overcommit_memory=1` + 400 GB swap makes this very rare.

---

## 14. Page cache

The kernel caches file reads/writes in RAM:

- First `onnx.load("model.onnx")`: reads from NVMe SSD
- Second `onnx.load("model.onnx")`: reads from page cache in RAM (much faster)

**Benchmark note:** repeated benchmark runs may be faster because of warm page cache. Note cold vs warm cache when comparing.

---

## 15. Swap — your 400 GB configuration

Your `/etc/fstab` has **400 GB** of swap across 4 files on NVMe:

- Much faster than HDD swap (NVMe has ~500K+ IOPS)
- Prevents OOM killer from firing during large model loads
- **But:** swap hit during timed inference = huge latency spike → invalid measurement

---

## 16. Verifying and setting sysctl values (commands)

### Read current values

```bash
sysctl vm.swappiness
sysctl vm.overcommit_memory
sysctl vm.max_map_count
sysctl vm.vfs_cache_pressure
sysctl vm.watermark_boost_factor
sysctl vm.compaction_proactiveness
sysctl vm.dirty_ratio
sysctl vm.dirty_background_ratio
sysctl vm.zone_reclaim_mode
```

### Set values temporarily (reverts on reboot)

```bash
sudo sysctl -w vm.swappiness=10
sudo sysctl -w vm.overcommit_memory=1
sudo sysctl -w vm.max_map_count=2097152
```

### Set values permanently (persists across reboots)

Add to `/etc/sysctl.d/99-r1-tuning.conf`:

```
vm.swappiness=10
vm.overcommit_memory=1
vm.vfs_cache_pressure=50
vm.watermark_boost_factor=0
vm.compaction_proactiveness=0
vm.dirty_ratio=10
vm.dirty_background_ratio=3
vm.zone_reclaim_mode=0
vm.max_map_count=2097152
```

Then apply: `sudo sysctl --system`

Or let your `r1-gpu-perf.service` write them at boot (Module 11).

### Read raw values from /proc/sys

```bash
cat /proc/sys/vm/swappiness          # 10
cat /proc/sys/vm/overcommit_memory   # 1
cat /proc/sys/vm/max_map_count       # 2097152
```

### Check THP mode

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# Output: always [madvise] never   (brackets = active mode)
```

### Check MGLRU status

```bash
cat /sys/kernel/mm/lru_gen/enabled
# Shows 0x0007 (or 7) when all features enabled
```

### Monitor swap activity during benchmark

```bash
vmstat 1
# Watch the 'si' (swap in) and 'so' (swap out) columns
# Any non-zero during timed runs = latency spike
```

---

## Module 09 checklist

- [ ] Can explain every `vm.*` parameter in the table above with your exact value
- [ ] Can explain swap vs page cache in one paragraph
- [ ] Can explain why `max_map_count` matters for big models
- [ ] Can explain THP `madvise` vs `always` tradeoff in latency terms
- [ ] Can explain what MGLRU improves over standard LRU
- [ ] Can explain why `overcommit_memory=1` helps ML framework loading

**Next:** `10-kernel-boot-and-io-dma.md`
