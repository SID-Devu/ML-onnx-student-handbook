# Module 10 — Kernel boot parameters, IOMMU/DMA, CPU power states (full depth)

---

## 1. GRUB and the kernel command line

**GRUB** is the bootloader that loads the Linux kernel. Boot parameters are set in `/etc/default/grub`:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.gttsize=28672 amdgpu.no_system_mem_limit=1 iommu=pt processor.max_cstate=1 idle=nomwait split_lock_detect=off transparent_hugepage=madvise"
```

After editing, you **must run**:

```bash
sudo update-grub
```

Then **reboot** for changes to take effect.

---

## 2. Your exact boot parameters explained

### `amdgpu.gttsize=28672`

Sets GPU-accessible system RAM (GTT) to **28 GB** (28672 MB).

Without this, the GPU can only access a small default GTT (varies by driver version, often 1-8 GB). Your models need much more.

### `amdgpu.no_system_mem_limit=1`

Removes an artificial cap on how much system memory the GPU driver will allocate.

Without this, some driver versions refuse to allocate beyond a built-in limit, even if GTT and RAM are available.

### `iommu=pt` (passthrough)

**IOMMU** (Input/Output Memory Management Unit) translates DMA addresses for device isolation.

**Passthrough mode (`pt`):** devices (GPU) perform DMA directly without address translation overhead.

**Why it helps:** reduces GPU memory access latency by avoiding IOMMU translation for every DMA operation.

**Concept — DMA (Direct Memory Access):** hardware (GPU, NVMe, etc.) reads/writes RAM directly without the CPU copying bytes in a loop. DMA is how the GPU accesses model weights in system RAM.

### `processor.max_cstate=1`

**C-states** are CPU idle power-saving states:

| C-state | Name | Wake latency | What happens |
|---------|------|-------------|-------------|
| C0 | Active | 0 | CPU is running |
| C1 | Halt | ~1 μs | Light sleep, quick wake |
| C2 | Stop-Clock | ~10 μs | Deeper sleep |
| C3+ | Deep sleep | ~100 μs+ | Very deep, slow to wake |

Setting `max_cstate=1` prevents CPU from going deeper than C1.

**Why it matters:** if a CPU core is in C3 when an inference callback arrives, it takes ~100 μs to wake up. That's 0.1 ms of pure latency jitter — unacceptable for precision benchmarks.

### `idle=nomwait`

**MWAIT** is a CPU instruction that enters a power-saving idle state with various sub-states.

`idle=nomwait` forces the CPU to use a simpler halt instruction instead of MWAIT.

**Why:** more deterministic idle behavior — fewer surprises from power management interactions.

### `split_lock_detect=off`

A **split lock** happens when an atomic memory operation (like compare-and-swap) crosses a **cache line boundary** (64 bytes). Modern CPUs detect this and can:

- Penalize the operation (200+ cycle stall)
- Trap to the kernel

Some ML libraries accidentally trigger split locks. Disabling detection avoids the penalty.

### `transparent_hugepage=madvise`

Sets THP to `madvise` mode at boot (before userspace starts). See Module 09 for details.

---

## 3. Concepts behind the parameters

### IOMMU

Hardware unit that translates DMA addresses. Purpose:

- **Isolation:** prevents a buggy device from corrupting arbitrary RAM
- **Address translation:** maps device virtual addresses to physical addresses

In passthrough mode, the IOMMU doesn't translate — direct access = faster.

### DMA (Direct Memory Access)

GPU reads weights from system RAM via DMA — the CPU is not involved in the data transfer. This is why IOMMU and memory pinning settings matter for GPU performance.

### Cache lines

CPU memory is managed in 64-byte **cache lines**. When you read one byte, the entire 64-byte line is loaded. Split locks cross line boundaries, causing expensive hardware synchronization.

---

## 4. Full boot process — end to end

Understanding where your configurations apply:

```
Power on
  → UEFI/BIOS (hardware init, POST — power-on self-test)
    → GRUB bootloader (reads /etc/default/grub, presents menu)
      → Linux kernel loads (applies boot parameters from cmdline)
        → initramfs (early userspace, loads essential drivers including amdgpu)
          → systemd (PID 1, manages all services)
            → r1-gpu-perf.service (your GPU/IO tuning — Module 11)
            → multi-user.target (system ready)
              → your SSH session / terminal
```

| Stage | Config location | Your settings applied |
|-------|----------------|----------------------|
| UEFI/BIOS | Firmware settings | Secure boot, boot order |
| GRUB | `/etc/default/grub` | `amdgpu.gttsize`, `iommu=pt`, C-states, THP |
| Kernel | Applied from cmdline | GTT carved out, IOMMU passthrough enabled |
| initramfs | `/etc/initramfs-tools/` | `amdgpu` driver loaded early |
| systemd | `/etc/systemd/system/` | `r1-gpu-perf.service` sets clocks, sysctl, readahead |
| Your shell | `/etc/environment`, `.bashrc` | `HSA_XNACK=1`, `PATH` |

### Where things go wrong

| Symptom | Check | Likely cause |
|---------|-------|-------------|
| GTT still small after reboot | `cat /proc/cmdline` | Forgot `update-grub` |
| GPU clocks not locked | `rocm-smi --showclocks` | `r1-gpu-perf.service` failed — check `journalctl` |
| `HSA_XNACK` not set | `echo $HSA_XNACK` | Not in `/etc/environment` or not sourced |
| Boot hangs | Connect display / serial console | Bad GRUB parameter, driver crash in initramfs |

---

## 5. Verifying boot parameters

```bash
# Check current kernel command line
cat /proc/cmdline

# Check specific values
cat /sys/module/amdgpu/parameters/gttsize    # GTT size in MB

# Verify all your parameters took effect
grep -o 'amdgpu.gttsize=[0-9]*' /proc/cmdline
grep -o 'iommu=[a-z]*' /proc/cmdline
grep -o 'processor.max_cstate=[0-9]' /proc/cmdline
```

---

## Module 10 checklist

- [ ] Can recite all 7 boot parameters and their purpose
- [ ] Can explain IOMMU's purpose and why passthrough reduces latency
- [ ] Can explain C-states and why C3+ wake latency hurts benchmarks
- [ ] Can explain DMA in one sentence
- [ ] Can explain split locks and why detection is disabled
- [ ] Know to run `update-grub` after editing `/etc/default/grub`
- [ ] Can draw the boot chain from UEFI → GRUB → kernel → initramfs → systemd → your service
- [ ] Can diagnose "GTT still small" by checking `/proc/cmdline`

**Next:** `11-systemd-and-boot-time-tuning.md`
