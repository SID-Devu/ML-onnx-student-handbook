# Module 22 — Shell / Bash scripting for ML workflows

You use the terminal every day. This module covers the patterns that appear in your benchmark and tuning work.

---

## 1. Pipes (`|`) — chaining commands

```bash
dmesg | grep amdgpu                    # filter dmesg for GPU messages
rocm-smi --showtemp | grep "Temperature"
cat /proc/meminfo | grep -i swap
dmesg | grep -c "svm"                  # count SVM messages
```

The output of the left command becomes the input of the right command.

---

## 2. Redirects (`>`, `>>`, `2>&1`)

```bash
rocm-smi --showall > snapshot.txt       # overwrite file with output
rocm-smi --showall >> log.txt           # append to file

python benchmark.py > out.txt 2>&1      # stdout AND stderr to same file
python benchmark.py 2>/dev/null         # discard errors
```

| Symbol | Meaning |
|--------|---------|
| `>` | Write stdout to file (overwrite) |
| `>>` | Append stdout to file |
| `2>` | Redirect stderr |
| `2>&1` | Redirect stderr to same place as stdout |
| `/dev/null` | Discard output |

---

## 3. Variables

```bash
export HSA_XNACK=1
export MODEL_PATH=/home/sudhdevu/R1models/yolo/yolov12m.onnx

echo $MODEL_PATH
echo "Model at: ${MODEL_PATH}"

# Local (current shell only)
WARMUP=3
RUNS=10

# In /etc/environment (persist across reboots, all users)
HSA_XNACK=1
```

---

## 4. Quoting — the source of most bash bugs

```bash
FILE="path with spaces/model.onnx"

# WRONG — bash splits on spaces
cp $FILE /dest/             # tries to copy "path", "with", "spaces/model.onnx"

# RIGHT — double quotes preserve spaces
cp "$FILE" /dest/

# Single quotes: literal, no variable expansion
echo '$FILE'                # prints: $FILE
echo "$FILE"                # prints: path with spaces/model.onnx
```

**Rule:** always double-quote `"$VARIABLE"` unless you specifically want word splitting.

---

## 5. Loops

```bash
# Simplify all ONNX files in current directory
for f in *.onnx; do
    echo "Simplifying $f ..."
    python -m onnxsim "$f" "${f%.onnx}_sim.onnx"
done

# Loop over model directories
for d in yolo crossformer mobilenetv2 clip_vit_b32; do
    echo "=== $d ==="
    ls "$d"/*.onnx 2>/dev/null
done

# While loop reading lines
while IFS= read -r line; do
    echo "Processing: $line"
done < model_list.txt
```

`${f%.onnx}` strips the `.onnx` suffix from `$f`.

---

## 6. Conditionals

```bash
if rocm-smi --showtemp | grep -q "80"; then
    echo "WARNING: GPU temperature at 80°C"
fi

# Check exit code
if python benchmark.py; then
    echo "Benchmark succeeded"
else
    echo "Benchmark FAILED"
fi

# File tests
if [ -f "model.onnx" ]; then
    echo "Model exists"
fi
```

---

## 7. Command chaining (`&&`, `;`, `||`)

```bash
mkdir -p logs && python benchmark.py    # run second ONLY if first succeeds
mkdir -p logs; python benchmark.py      # run second regardless
python benchmark.py || echo "FAILED"    # run second ONLY if first fails

# Common pattern: create dir, run, report
mkdir -p results && \
    python benchmark_cooldown.py --warmup 3 --runs 10 && \
    echo "Done"
```

| Operator | Meaning |
|----------|---------|
| `&&` | Run next only if previous succeeded (exit code 0) |
| `;` | Run next regardless |
| `\|\|` | Run next only if previous failed (exit code != 0) |

---

## 8. Exit codes (`$?`)

```bash
python benchmark.py
echo $?   # 0 = success, non-zero = error

# Use in scripts
python benchmark.py
if [ $? -ne 0 ]; then
    echo "Benchmark failed with exit code $?"
    exit 1
fi
```

---

## 9. Making scripts executable

```bash
chmod +x my_script.sh
./my_script.sh                  # runs in subshell

# Script header (shebang)
#!/bin/bash
set -euo pipefail               # exit on error, undefined vars, pipe failures
```

`set -euo pipefail` is the safety net for production scripts.

---

## 10. `source` vs `./`

```bash
source /etc/environment         # loads variables INTO current shell
. /etc/environment              # same as source

./benchmark.sh                  # runs in a NEW subshell (variables don't leak back)
bash benchmark.sh               # same as ./
```

**When it matters:** `export HSA_XNACK=1` in a script run with `./` won't affect your current shell. Use `source` to load environment variables.

---

## 11. Long-running benchmarks: `nohup`, `screen`, `tmux`

### `nohup` — survive SSH disconnect

```bash
nohup python benchmark_cooldown.py --warmup 3 --runs 10 > bench.log 2>&1 &
# Process runs even if you close the terminal
# Check output: tail -f bench.log
# Find process: ps aux | grep benchmark
```

### `disown` — detach an already-running process

```bash
python benchmark_cooldown.py &       # start in background
disown                               # detach from shell — survives SSH disconnect
```

Use `disown` when you forgot to use `nohup` and the job is already running.

### `screen` — persistent terminal session

```bash
screen -S bench                  # create named session
python benchmark_cooldown.py     # run inside screen
# Ctrl+A, D                     # detach (keeps running)
screen -r bench                  # reattach later
screen -ls                       # list sessions
```

### `tmux` — modern alternative to screen

```bash
tmux new -s bench                # create named session
python benchmark_cooldown.py     # run inside tmux
# Ctrl+B, D                     # detach
tmux attach -t bench             # reattach
tmux ls                          # list sessions
```

Use `screen` or `tmux` for **any benchmark that takes more than a few minutes**.

---

## 12. Useful one-liners for your workflow

```bash
# Count ONNX files across all model dirs
find /home/sudhdevu/R1models -name "*.onnx" | wc -l

# Check total size of all ONNX models
find /home/sudhdevu/R1models -name "*.onnx" -exec du -sh {} + | sort -rh

# Monitor GPU during benchmark (updates every 2 seconds)
watch -n 2 rocm-smi --showtemp --showclocks

# Live-stream benchmark log
tail -f results/benchmark.log

# Find which process is using the GPU
ps aux | grep python
```

---

## Module 22 checklist

- [ ] Can pipe `dmesg | grep amdgpu` and explain what each part does
- [ ] Can redirect stdout and stderr to a file
- [ ] Can write a for loop over `.onnx` files
- [ ] Know when to use `&&` vs `;`
- [ ] Can keep a benchmark running after SSH disconnect using screen/tmux
- [ ] Can explain `source` vs `./` for environment variables

**Next:** `23-git-version-control.md`
