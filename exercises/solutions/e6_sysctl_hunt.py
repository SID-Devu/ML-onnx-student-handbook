#!/usr/bin/env python3
"""
Exercise 6 — sysctl scavenger hunt

Reads current kernel tunable values and THP mode, then prints an explanation
of each one. Runs only on Linux.

Run:  python e6_sysctl_hunt.py
Requires: Linux with /proc/sys and /sys/kernel available (no pip deps)
"""
from __future__ import annotations

import sys
from pathlib import Path


def read_sysfs(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "<not available>"
    return p.read_text().strip()


def read_sysctl(name: str) -> str:
    """Read a sysctl by mapping dots to slashes under /proc/sys."""
    path = "/proc/sys/" + name.replace(".", "/")
    return read_sysfs(path)


def main() -> None:
    if sys.platform != "linux":
        print("This exercise requires Linux.")
        sys.exit(1)

    print("=== Exercise 6: sysctl scavenger hunt ===\n")

    # --- 1. vm.swappiness ---
    val = read_sysctl("vm.swappiness")
    print(f"vm.swappiness = {val}")
    print(
        "  Explanation: Controls how readily the kernel moves anonymous (non-file-backed)\n"
        "  memory pages to swap. Lower values (e.g. 10) tell the kernel to prefer keeping\n"
        "  process memory in RAM and reclaim page cache instead. A value of 0 does not\n"
        "  disable swap entirely—it just makes the kernel very reluctant to swap until\n"
        "  memory pressure is high. For ML benchmarks, low swappiness avoids surprise\n"
        "  latency spikes caused by model tensors being paged out mid-inference.\n"
    )

    # --- 2. vm.max_map_count ---
    val = read_sysctl("vm.max_map_count")
    print(f"vm.max_map_count = {val}")
    print(
        "  Explanation: Maximum number of memory-mapped regions (VMAs) a single process\n"
        "  can have. The default (~65530) can be too low when loading large ONNX models,\n"
        "  mmap-ing weights, or running frameworks that create many small mappings.\n"
        "  Raising to ~2097152 prevents 'cannot allocate memory' failures during session\n"
        "  creation for multi-GB models.\n"
    )

    # --- 3. vm.overcommit_memory ---
    val = read_sysctl("vm.overcommit_memory")
    print(f"vm.overcommit_memory = {val}")
    mode_names = {"0": "heuristic", "1": "always-succeed", "2": "strict"}
    mode = mode_names.get(val, "unknown")
    print(
        f"  Explanation: Controls the kernel's memory overcommit policy (current: {mode}).\n"
        "  Mode 0 (heuristic) rejects clearly excessive allocations. Mode 1 never refuses\n"
        "  malloc—useful when ML frameworks over-allocate arena buffers they may never touch.\n"
        "  Mode 2 strictly limits commit to physical RAM + swap × ratio. For large model\n"
        "  workloads, mode 1 prevents early malloc failures, though the OOM killer remains\n"
        "  the last resort if physical memory is truly exhausted.\n"
    )

    # --- 4. THP mode ---
    thp_path = "/sys/kernel/mm/transparent_hugepage/enabled"
    raw = read_sysfs(thp_path)
    print(f"THP enabled = {raw}")
    print(
        "  Explanation: Transparent Huge Pages (THP) let the kernel automatically use\n"
        "  2 MB pages instead of 4 KB, reducing TLB misses for large allocations.\n"
        "  'always' applies THP broadly—can cause latency stalls from compaction.\n"
        "  'madvise' applies THP only where applications explicitly opt in via madvise().\n"
        "  'never' disables THP entirely. For latency-sensitive GPU inference, 'madvise'\n"
        "  is the common recommendation: you get huge-page benefits where requested without\n"
        "  surprise compaction pauses.\n"
    )

    # --- Bonus: vm.dirty_ratio ---
    val = read_sysctl("vm.dirty_ratio")
    print(f"vm.dirty_ratio = {val}")
    print(
        "  Explanation: The percentage of system memory that can be filled with 'dirty'\n"
        "  (modified, not-yet-flushed-to-disk) pages before processes writing to disk are\n"
        "  forced to wait for writeback. A lower value forces more frequent flushing,\n"
        "  preventing large dirty-page backlogs from causing I/O stalls.\n"
    )

    print("[OK] Exercise 6 complete.")


if __name__ == "__main__":
    main()
