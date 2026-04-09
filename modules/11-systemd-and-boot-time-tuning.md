# Module 11 — systemd services: your r1-gpu-perf.service (full depth)

---

## 1. systemd basics

**systemd** is the init system and service manager on most modern Linux. It controls what starts at boot and how services are managed.

---

## 2. Essential commands

| Command | What it does |
|---------|-------------|
| `systemctl status r1-gpu-perf.service` | Is it running? Recent log snippet |
| `systemctl enable r1-gpu-perf.service` | Start automatically on every boot |
| `systemctl disable r1-gpu-perf.service` | Don't start on boot |
| `systemctl start r1-gpu-perf.service` | Start now (manually) |
| `systemctl stop r1-gpu-perf.service` | Stop now |
| `systemctl restart r1-gpu-perf.service` | Stop + start |
| `journalctl -u r1-gpu-perf.service` | Full logs for this service |
| `journalctl -u r1-gpu-perf.service -f` | Follow logs in real time |

---

## 3. Unit file structure

Service unit files live in `/etc/systemd/system/` (user-created) or `/lib/systemd/system/` (package-managed).

Structure:

```ini
[Unit]
Description=R1 GPU Performance Tuning
After=multi-user.target        # run after basic system is up
# After=rocm.target            # if you have a ROCm-specific target

[Service]
Type=oneshot                    # runs once at boot, then exits
RemainAfterExit=yes             # systemd considers it "active" even after script finishes
ExecStart=/path/to/tuning-script.sh

[Install]
WantedBy=multi-user.target      # enable = link into multi-user boot
```

**Key fields:**

| Field | Meaning |
|-------|---------|
| `After=` | Run after these targets/services are ready |
| `Requires=` | Hard dependency — if this fails, we fail too |
| `Type=oneshot` | Script runs once and exits (not a daemon) |
| `RemainAfterExit=yes` | systemd shows "active" even after the script finishes |
| `ExecStart=` | The command to run |
| `WantedBy=multi-user.target` | Standard boot target (like runlevel 3) |

---

## 4. What your `r1-gpu-perf.service` does

Your service applies all of these on **every boot** to ensure reproducible benchmark conditions:

### a) GPU clocks — lock to maximum

```bash
rocm-smi --setperflevel high
# Or specific clock setting commands for sclk=2900, mclk=1000, fclk=2000
```

Prevents dynamic clock scaling during benchmarks.

### b) sysctl writes — all the vm.* parameters from Module 09

```bash
sysctl -w vm.swappiness=10
sysctl -w vm.overcommit_memory=1
sysctl -w vm.vfs_cache_pressure=50
sysctl -w vm.watermark_boost_factor=0
sysctl -w vm.compaction_proactiveness=0
sysctl -w vm.dirty_ratio=10
sysctl -w vm.dirty_background_ratio=3
sysctl -w vm.zone_reclaim_mode=0
sysctl -w vm.max_map_count=2097152
```

### c) THP (Transparent Huge Pages) — ensure madvise mode

```bash
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
echo madvise > /sys/kernel/mm/transparent_hugepage/defrag
```

### d) NVMe readahead — optimize large sequential reads

```bash
echo 2048 > /sys/block/nvme0n1/queue/read_ahead_kb
```

Sets readahead to **2048 KB** (2 MB) — the OS pre-fetches 2 MB of upcoming disk blocks when reading sequentially. Helps when loading large ONNX model files.

### e) MGLRU — enable modern page reclaim

```bash
# Enable MGLRU (exact sysctl path varies by kernel version)
echo Y > /sys/kernel/mm/lru_gen/enabled
# or: sysctl -w vm.lru_gen.enabled=7
```

---

## 5. Why boot-time tuning matters

**Without the service:** after every reboot, you'd manually re-run all these commands. If you forget one, your benchmark conditions differ from last time → results aren't comparable.

**With the service:** every boot reaches the exact same tuned state. Reproducibility.

---

## 6. Verifying the service worked

```bash
# Check status
systemctl status r1-gpu-perf.service

# Verify GPU clocks are locked
rocm-smi --showclocks

# Verify sysctl values
sysctl vm.swappiness vm.overcommit_memory vm.max_map_count

# Verify THP mode
cat /sys/kernel/mm/transparent_hugepage/enabled
# Should show: always [madvise] never  (brackets = active mode)
```

---

## 7. Creating or editing a service

```bash
# Create/edit the unit file
sudo nano /etc/systemd/system/r1-gpu-perf.service

# After editing, reload systemd's knowledge of unit files
sudo systemctl daemon-reload

# Enable for boot
sudo systemctl enable r1-gpu-perf.service

# Run it now to test
sudo systemctl start r1-gpu-perf.service

# Check for errors
systemctl status r1-gpu-perf.service
journalctl -u r1-gpu-perf.service
```

---

## 8. Separating experiments from production

| Situation | Approach |
|-----------|----------|
| Testing a new sysctl value | Manual `sysctl -w` in shell (reverts on reboot) |
| Confirmed it helps | Add to `r1-gpu-perf.service` (persists across reboots) |
| Debugging a boot issue | `systemctl disable` the service, reboot, test without it |

---

## 9. Logging and reproducibility

For benchmark reports, record:

- Kernel version (`uname -r`)
- ROCm version (`rocm-smi --version` or package query)
- ORT build info (commit hash + CMake flags)
- systemd unit file content (or hash)
- `cat /proc/cmdline` (kernel boot params)

---

## Module 11 checklist

- [ ] Can enable/start/status a service with systemctl
- [ ] Can read a unit file and identify ExecStart, After=, WantedBy=
- [ ] Can list the 5 categories of tuning your service applies (GPU clocks, sysctl, THP, NVMe readahead, MGLRU)
- [ ] Can explain why boot-time tuning improves benchmark reproducibility
- [ ] Know to run `daemon-reload` after editing a unit file

**Next:** `12-xen-and-virtualization-basics.md`
