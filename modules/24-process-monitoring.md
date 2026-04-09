# Module 24 — Process monitoring and system observation

You monitor benchmarks constantly. These tools tell you what's happening on the system.

---

## 1. `htop` — interactive process viewer

```bash
htop
```

Shows: CPU per-core usage, RAM, swap, all processes sorted by resource use.

**Key columns:** PID, USER, CPU%, MEM%, TIME, COMMAND

**Shortcuts inside htop:**
- `F6` sort by column
- `F9` kill a process
- `F5` tree view (shows parent/child)
- `q` quit

---

## 2. `top` — simpler built-in alternative

```bash
top
```

Available on every Linux system. Less visual than htop but always present.

---

## 3. `ps` — list processes

```bash
ps aux                          # all processes, detailed
ps aux | grep python            # find Python processes
ps aux | grep benchmark         # find benchmark processes
ps -ef --forest                 # show parent-child tree
```

| Column | Meaning |
|--------|---------|
| PID | Process ID |
| %CPU | CPU usage |
| %MEM | Memory usage |
| VSZ | Virtual memory size |
| RSS | Resident set size (actual RAM) |
| STAT | State (S=sleeping, R=running, Z=zombie) |
| COMMAND | Command that started the process |

---

## 4. `kill` — stop processes

```bash
kill <PID>                      # graceful shutdown (SIGTERM)
kill -9 <PID>                   # force kill (SIGKILL) — use when graceful fails
killall python                  # kill all Python processes

# Find and kill a hung benchmark
ps aux | grep benchmark_cooldown
kill <PID from above>
```

---

## 5. `nice` / `renice` — process priority

```bash
# Start benchmark with high priority (lower nice = higher priority)
nice -n -10 python benchmark_cooldown.py

# Change priority of running process
renice -n -10 -p <PID>
```

Nice values: -20 (highest priority) to 19 (lowest). Default is 0.

---

## 6. `watch` — repeat a command

```bash
watch -n 2 rocm-smi --showtemp --showclocks
# Runs rocm-smi every 2 seconds, refreshes display

watch -n 5 'nvidia-smi || rocm-smi'
watch -n 1 'cat /proc/meminfo | grep -i swap'
```

Essential for live GPU monitoring during benchmarks.

---

## 7. `tail -f` — live-stream log files

```bash
tail -f results/benchmark.log          # follow new lines as they appear
tail -f -n 50 results/benchmark.log    # show last 50 lines, then follow

# Multiple files
tail -f results/*.log
```

---

## 8. `iotop` — disk I/O per process

```bash
sudo iotop
```

Shows which processes are reading/writing to disk. Critical when diagnosing swap thrashing during large model inference.

---

## 9. `lsof` — list open files

```bash
lsof -p <PID>                   # files open by a process
lsof /path/to/model.onnx        # which process has this file open
lsof -i :8080                   # what's using port 8080
```

Useful for debugging "file busy" errors when a model file won't delete.

---

## 10. Combining tools for benchmark monitoring

```bash
# Terminal 1: run benchmark
python benchmark_cooldown.py --warmup 3 --runs 10 --cooldown 120

# Terminal 2: watch GPU (tmux pane or second SSH)
watch -n 2 rocm-smi --showtemp --showclocks --showmeminfo all

# Terminal 3: watch system resources
htop

# Terminal 4: watch swap activity
watch -n 1 'cat /proc/meminfo | grep -i swap'
```

---

## 11. Quick diagnosis recipes

| Symptom | Check with | Look for |
|---------|-----------|----------|
| Benchmark hangs | `ps aux \| grep python` | Process stuck (high CPU or 0 CPU) |
| GPU not being used | `rocm-smi` | 0% utilization, low clocks |
| System slow | `htop` | High CPU/RAM by unexpected process |
| Swap thrashing | `iotop` or `vmstat 1` | High si/so, disk I/O by kswapd |
| Model file locked | `lsof /path/to/model.onnx` | Which PID holds the file |
| OOM kills | `dmesg \| grep -i "oom\|killed"` | OOM killer messages |

---

## Module 24 checklist

- [ ] Can use `htop` to identify which process is consuming resources
- [ ] Can find and kill a hung benchmark process
- [ ] Can use `watch` to monitor GPU during inference
- [ ] Can use `tail -f` to stream benchmark logs
- [ ] Can diagnose swap thrashing with `iotop` or `vmstat`

**Next:** `25-profiling-tools.md`
