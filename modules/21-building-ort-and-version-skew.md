# Module 21 — Building ORT from source, CMake, version skew (full depth)

---

## 1. Why custom ORT builds exist

Pre-built pip wheels are convenient but **do not include**:

- **MIGraphXExecutionProvider** (not in PyPI wheels)
- **`hipMallocManaged` allocator** (your APU needs unified memory, not VRAM-only)
- Specific compiler flags your team requires

**Your custom build exists because:** standard `pip install onnxruntime` would give you an ORT that can only use CPU — no GPU acceleration on your AMD hardware.

---

## 2. CMake — configuring what gets compiled

ORT uses **CMake** as its build system. You configure features via CMake options:

```bash
# Example ORT build command (simplified)
./build.sh \
    --config Release \
    --use_migraphx \
    --cmake_extra_defines \
        USE_HIP_MANAGED_MEM=ON \
        CMAKE_HIP_ARCHITECTURES=gfx1151  # your Strix Halo GPU architecture
```

Key CMake flags for your setup:

| Flag | Purpose |
|------|---------|
| `--use_migraphx` | Enable MIGraphXExecutionProvider |
| `USE_HIP_MANAGED_MEM=ON` | Use `hipMallocManaged` instead of `hipMalloc` |
| `CMAKE_HIP_ARCHITECTURES=gfx1151` | Target your specific GPU ISA |
| `--config Release` | Optimized build (not debug) |
| `--parallel` | Parallel compilation (faster build) |

**Build time:** typically 30-90 minutes depending on machine and options.

---

## 3. Version skew — the #1 cause of mysterious crashes

**Version skew** = ORT Python package, native `.so` libraries, and ROCm are built against different versions.

### Symptoms of version skew

| Symptom | Likely cause |
|---------|-------------|
| `ImportError: undefined symbol: ...` | `.so` compiled against different ROCm/HIP version |
| EP loads but crashes on first `session.run()` | ABI mismatch between ORT and MIGraphX |
| `ImportError: No module named 'onnxruntime'` | Python package not installed or wrong venv |
| `MIGraphXExecutionProvider not available` | ORT wasn't built with `--use_migraphx` |
| Segfault during session creation | Library version conflict (e.g., two ROCm versions on `LD_LIBRARY_PATH`) |

### The version tuple that must agree

All of these must be built against the **same** ROCm version:

```
ROCm version (e.g., 6.3.0)
  ├── libamdhip64.so
  ├── libmigraphx.so
  ├── libMIOpen.so
  └── libonnxruntime.so (your custom build)
      └── onnxruntime Python wheel (built from same source)
```

---

## 4. `ldd` — debugging library dependencies

```bash
# Check what libraries ORT needs
ldd /path/to/onnxruntime/libonnxruntime.so

# Example output:
#   libamdhip64.so.6 => /opt/rocm/lib/libamdhip64.so.6
#   libmigraphx.so => /opt/rocm/lib/libmigraphx.so
#   libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6

# If any line says "not found", that library is missing
# and the program will crash on load
```

### Check Python bindings

```bash
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

If `MIGraphXExecutionProvider` is not in the list, your wheel wasn't built with it.

---

## 5. Known-good version tuple

Document and share with your team:

```
# known_good_versions.txt
kernel:    6.18.0+
rocm:      6.3.0
ort_commit: abc123def456  (git tag/hash)
ort_cmake:  --use_migraphx USE_HIP_MANAGED_MEM=ON CMAKE_HIP_ARCHITECTURES=gfx1151
python:    3.11.9
migraphx:  built with ROCm 6.3.0
```

When **anything** in this tuple changes, re-benchmark to detect regressions.

---

## 6. Building a wheel from your custom ORT

After building ORT from source, you typically get a `.whl` file:

```bash
# After build completes
ls build/Release/dist/*.whl
# onnxruntime_rocm-1.21.0-cp311-cp311-linux_x86_64.whl

# Install it
pip install build/Release/dist/onnxruntime_rocm-*.whl
```

**Never mix** a pip-installed `onnxruntime` from PyPI with your custom `.so`. Uninstall any previous onnxruntime first:

```bash
pip uninstall onnxruntime onnxruntime-gpu onnxruntime-rocm  # remove all
pip install your_custom_wheel.whl                            # install yours
```

---

## 7. When to rebuild

| Trigger | Action |
|---------|--------|
| ROCm version update | Rebuild ORT against new ROCm |
| New ORT release with bug fixes | Pull new ORT source, rebuild |
| New GPU architecture (different gfx target) | Rebuild with correct `CMAKE_HIP_ARCHITECTURES` |
| Need a new EP feature (e.g., new provider option) | Rebuild from newer ORT source |
| Random crashes after system update | Check `ldd` for broken links, may need rebuild |

---

## Module 21 checklist

- [ ] Can explain why `pip install onnxruntime` from PyPI won't work for your GPU
- [ ] Can explain what CMake does in the ORT build process
- [ ] Can use `ldd` to check if all `.so` dependencies resolve
- [ ] Can list the "version tuple" that must be consistent
- [ ] Know to capture CMake flags in your team's build docs
- [ ] Can explain why mixing pip wheels with custom `.so` files is dangerous

**End of numbered modules.**
