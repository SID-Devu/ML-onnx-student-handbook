# Module 12 — Xen hypervisor: dom0, PVH, scheduling, bare metal comparison (full depth)

You ran your first benchmarks on Xen, then rebooted to bare metal. This module explains the virtualization layer you worked with.

---

## 1. What a hypervisor is

A **hypervisor** is software that runs **virtual machines** (domains/VMs) on shared hardware.

**Type 1 (bare metal) hypervisor:** runs directly on hardware — Xen, VMware ESXi
**Type 2 (hosted) hypervisor:** runs inside an existing OS — VirtualBox, QEMU/KVM (simplified)

**Xen** is a Type 1 hypervisor. It sits **between the hardware and all operating systems**, including your "host" OS.

---

## 2. Xen-specific vocabulary

| Concept | What it is |
|---------|-----------|
| **dom0** | The **privileged** "host" domain. Controls hardware, runs device drivers, administers other VMs. **Your machine runs as dom0.** |
| **domU** | Unprivileged guest VMs (you weren't running any — just dom0) |
| **PV** | Para-Virtualization — guest OS is modified to cooperate with hypervisor |
| **HVM** | Hardware Virtual Machine — full hardware virtualization (unmodified guest OS) |
| **PVH** | Para-Virtualized Hardware — hybrid mode combining PV efficiency with HVM features. **Your dom0 used PVH** (`dom0=pvh` in GRUB) |

---

## 3. Your Xen configuration

These were in your GRUB config (Xen hypervisor cmdline):

| Parameter | Value | What it does |
|-----------|-------|-------------|
| `dom0_mem` | **30G** | Allocate 30 GB RAM to dom0 (all your RAM) |
| `dom0_max_vcpus` | **32** | Expose all 32 threads (16 cores × 2 SMT) to dom0 |
| `dom0` | **pvh** | dom0 runs in PVH mode |
| `sched` | **credit2** | Use Credit2 CPU scheduler |
| `cpufreq` | **xen:performance** | Xen controls CPU frequency, locks to max |

---

## 4. `xl` commands — Xen management

| Command | What it shows |
|---------|--------------|
| `xl info` | Hypervisor details: memory, CPUs, Xen version, scheduler |
| `xl list` | Running domains (VMs): names, IDs, memory, vCPUs, state |
| `xl dmesg` | Xen hypervisor log messages |

```bash
xl info
# Shows: total_memory, free_memory, nr_cpus, xen_version, xen_scheduler

xl list
# Shows:
# Name     ID   Mem   VCPUs   State   Time(s)
# Domain-0  0   30720  32      r-----   1234.5
```

---

## 5. Credit2 scheduler

Xen's CPU scheduler assigns physical CPU time to virtual CPUs:

- **Credit1 (older):** simpler, fair-share scheduler
- **Credit2 (yours):** improved latency handling, better for latency-sensitive workloads like GPU inference benchmarks

`sched=credit2` selects Credit2.

---

## 6. `cpufreq=xen:performance`

On bare metal, Linux controls CPU frequency via the `cpufreq` governor.

Under Xen, the **hypervisor** controls CPU frequency instead:
- `cpufreq=xen:performance` = Xen locks all CPU cores to maximum frequency
- This is the Xen equivalent of setting the Linux governor to "performance"

---

## 7. Bare metal vs virtualized — benchmark implications

You switched from Xen dom0 to bare metal mid-session. Key differences:

| Aspect | Xen dom0 | Bare metal |
|--------|----------|------------|
| **Timer behavior** | Xen intercepts timer interrupts (VM exits) | Direct hardware timers |
| **CPU scheduling** | Credit2 scheduler adds scheduling overhead | Linux CFS directly on hardware |
| **Device access** | Through Xen's device model / passthrough | Direct hardware access |
| **Memory** | Xen carves out memory for hypervisor first | All RAM available to OS |
| **GPU passthrough** | May involve vfio/IOMMU mediation by Xen | Direct amdgpu driver access |
| **Benchmark noise** | Higher (VM exits, scheduling jitter) | Lower |
| **Comparability** | Results NOT directly comparable to bare metal | Clean baseline |

**Critical rule:** never compare Xen dom0 benchmark numbers with bare metal numbers without disclosing the environment difference.

---

## 8. vCPUs vs physical threads

| Term | Meaning |
|------|---------|
| Physical cores | Actual CPU cores on the chip (16 on your Strix Halo) |
| SMT threads | Simultaneous multithreading (2 threads per core = 32 threads) |
| vCPUs | Virtual CPUs assigned to a domain. `dom0_max_vcpus=32` = all 32 threads exposed to dom0 |

If vCPUs are **overcommitted** (more vCPUs across all domains than physical threads), performance degrades due to time-sharing.

---

## 9. When you'd use Xen vs bare metal

| Scenario | Use |
|----------|-----|
| Running multiple isolated workloads | Xen (multiple domUs) |
| Maximum benchmark performance / minimal noise | Bare metal |
| GPU passthrough to a guest VM | Xen with vfio-pci |
| Simple development / testing | Bare metal (simpler) |

---

## Module 12 checklist

- [ ] Define dom0 vs domU in one sentence each
- [ ] Explain PVH mode at a high level
- [ ] Explain why GPU benchmarks on dom0 might differ from bare metal
- [ ] Name `xl info` and `xl list` and what they show
- [ ] Explain Credit2 scheduler purpose
- [ ] Explain `cpufreq=xen:performance` vs Linux governor

**Next:** `13-model-export-and-onnx-pitfalls.md`
