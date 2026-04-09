# Cheatsheet: Linux & AMD APU tuning quick reference

## Your kernel boot parameters (GRUB)

```
GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.gttsize=28672 amdgpu.no_system_mem_limit=1 iommu=pt processor.max_cstate=1 idle=nomwait split_lock_detect=off transparent_hugepage=madvise"
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `amdgpu.gttsize` | 28672 | 28 GB GPU-accessible system RAM |
| `amdgpu.no_system_mem_limit` | 1 | Remove GPU memory allocation cap |
| `iommu` | pt | DMA passthrough (no translation overhead) |
| `processor.max_cstate` | 1 | Prevent deep CPU sleep (C2+ wake latency) |
| `idle` | nomwait | Simple halt instead of MWAIT |
| `split_lock_detect` | off | No penalty for cache-line-crossing atomics |
| `transparent_hugepage` | madvise | THP only when apps request it |

After editing: `sudo update-grub && sudo reboot`

## Your sysctl values (all from r1-gpu-perf.service)

| Parameter | Value | Why |
|-----------|-------|-----|
| `vm.swappiness` | 10 | Keep model tensors in RAM |
| `vm.overcommit_memory` | 1 | Never fail malloc |
| `vm.vfs_cache_pressure` | 50 | Keep filesystem cache longer |
| `vm.watermark_boost_factor` | 0 | No kswapd boosting |
| `vm.compaction_proactiveness` | 0 | No background compaction stalls |
| `vm.dirty_ratio` | 10 | Block writer at 10% dirty |
| `vm.dirty_background_ratio` | 3 | Background flush at 3% dirty |
| `vm.zone_reclaim_mode` | 0 | No NUMA zone reclaim |
| `vm.max_map_count` | 2097152 | 2M VMAs for large models |

Read: `sysctl vm.swappiness`
Write: `sudo sysctl -w vm.swappiness=10`

## rocm-smi commands

```bash
rocm-smi                          # Quick GPU overview
rocm-smi --showtemp               # Temperature
rocm-smi --showclocks             # Current sclk/mclk
rocm-smi --showmeminfo all        # VRAM + GTT usage
rocm-smi --showall                # Everything
rocm-smi --setperflevel high      # Lock to max clocks
```

## Other GPU tools

```bash
rocminfo                          # CU count, ISA, memory
hipcc --version                   # HIP compiler version
hipconfig --full                  # ROCm/HIP config
cat /sys/module/amdgpu/parameters/gttsize  # GTT size (MB)
```

## Clock speeds (locked by r1-gpu-perf.service)

| Clock | Name | Your value |
|-------|------|-----------|
| sclk | Shader/GPU core | 2900 MHz |
| mclk | Memory | 1000 MHz |
| fclk | Fabric/Infinity Fabric | 2000 MHz |

## Memory architecture

| Pool | Size | Access |
|------|------|--------|
| VRAM (carve-out) | 512 MB | Fast, GPU-local |
| GTT | 28 GB | System RAM, GPU-accessible via page tables |
| Total system RAM | 30 GB DDR5 | Shared CPU + GPU |
| Swap | 400 GB on NVMe | Overflow, fast but not during benchmarks |

## Environment variables

```bash
# /etc/environment (system-wide)
HSA_XNACK=1

# Verify
echo $HSA_XNACK
cat /proc/cmdline
```

## systemd service commands

```bash
systemctl status r1-gpu-perf.service
systemctl enable r1-gpu-perf.service    # auto-start on boot
systemctl start r1-gpu-perf.service     # run now
journalctl -u r1-gpu-perf.service       # view logs
sudo systemctl daemon-reload            # after editing unit file
```

## Verification after boot

```bash
cat /proc/cmdline                                        # boot params
sysctl vm.swappiness vm.overcommit_memory vm.max_map_count   # sysctl
cat /sys/kernel/mm/transparent_hugepage/enabled           # THP mode
rocm-smi --showclocks                                     # GPU clocks locked
echo $HSA_XNACK                                           # XNACK enabled
```

## C-state wake latencies

| C-state | Wake time | Impact |
|---------|-----------|--------|
| C0 (active) | 0 | None |
| C1 (halt) | ~1 us | Minimal |
| C2 (stop-clock) | ~10 us | Noticeable |
| C3+ (deep) | ~100+ us | Benchmark jitter |

Your `max_cstate=1` prevents C2+.

## Xen (if applicable)

```bash
xl info       # hypervisor details
xl list       # running domains
```

| Xen param | Your value |
|-----------|-----------|
| dom0_mem | 30G |
| dom0_max_vcpus | 32 |
| dom0 | pvh |
| sched | credit2 |
| cpufreq | xen:performance |
