# Optional external references (not required to complete this course)

Use these only when you have network and want official docs, video, or books. Everything **essential** to understand the topics is also explained inside `modules/` in this handbook.

---

## Python

- **"Python for Everybody"** by Charles Severance — https://www.py4e.com/ (do chapters 1-10 only; skip web/database chapters; also free on YouTube, ~10 hours)

## NumPy

- NumPy quickstart — https://numpy.org/doc/stable/user/quickstart.html (1-2 hours)

## Neural networks (intuition, no math)

- **3Blue1Brown "Neural Networks" playlist** — https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi (4 videos, ~1 hour total; best visual explanation that exists)

## ONNX

- ONNX official docs — https://onnx.ai/onnx/intro/
- Netron (graph viewer) — https://netron.app/ or local `pip install netron` (drag-drop any `.onnx` file)

## ONNX Runtime

- ORT documentation — https://onnxruntime.ai/docs/
- ORT quantization docs (INT8 PTQ) — https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

## Model export

- PyTorch ONNX export guide — https://pytorch.org/docs/stable/onnx.html
- HuggingFace Optimum ONNX export — https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model
- `tf2onnx` (TensorFlow → ONNX) — https://github.com/onnx/tensorflow-onnx

## AMD / ROCm / GPU

- ROCm documentation — https://rocm.docs.amd.com/
- ROCm memory management — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html
- **"HSA Runtime Programmer's Reference Manual"** — search on AMD developer site (https://developer.amd.com); covers HSA, XNACK, SVM internals
- AMD GPU architecture whitepapers — ask your team for RDNA 3.5 docs, or check https://www.amd.com/en/technologies/rdna

## Linux kernel & system tuning

- **"How Linux Works"** by Brian Ward (book) — covers everything: dmesg, /proc, /sys, sysctl, systemd, boot process, memory management. Excellent for your role.
- Kernel sysctl vm.* reference — https://www.kernel.org/doc/html/latest/admin-guide/sysctl/vm.html
- Kernel boot parameters — https://www.kernel.org/doc/html/latest/admin-guide/kernel-parameters.html

## Xen virtualization

- Xen Project documentation — https://wiki.xenproject.org/wiki/Main_Page

---

Everything above is **supplementary**. The `modules/` folder in this handbook covers all concepts at the depth you need for your work.
