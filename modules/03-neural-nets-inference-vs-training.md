# Module 03 — Neural networks: inference vs training (full depth)

You do not need calculus or linear algebra to be effective. You need a **mechanical mental model** of what a model is, what inference does, and the vocabulary your team uses.

---

## 1. Core concepts table

| Concept | Simple explanation |
|---------|-------------------|
| **Tensor** | A multi-dimensional array of numbers. An image is a 3D tensor (H × W × C). A batch of images is 4D (N × C × H × W). |
| **Model** | A function: input tensor in → output tensor out. That's it. |
| **Weights / Parameters** | Numbers the model learned during training. Stored in `.onnx`, `.safetensors`, `.pth` files. That's why model files are large (millions to billions of numbers). |
| **Inference** | Running a trained model on new data. **This is what your benchmarks do. No learning happens.** |
| **Training** | Teaching the model by showing it millions of examples and adjusting weights. **You are NOT doing this.** |
| **Layer** | One step in the model's computation. Conv layer, Attention layer, Linear layer, etc. |
| **CNN (Convolutional Neural Network)** | Good at images. Detects local patterns (edges, textures, shapes) hierarchically. Used in: YOLO, MobileNetV2, EfficientNet, DeepSort |
| **Transformer** | Good at sequences (text, video frames). Compares all positions via attention. Used in: Qwen3, CLIP, CrossFormer, LLaMA, DeepSeek-R1 |
| **Encoder** | Compresses input into a compact representation (embedding). CLIP encodes images and text into shared embedding space. |
| **Decoder** | Generates output from a representation (text token by token, audio sample by sample). Qwen3 decoder generates one token at a time. |

---

## 2. Model as a function

```
outputs = f(inputs; θ)
```

- **inputs**: your data tensor(s) — image, text tokens, audio
- **θ (theta)**: learned parameters (weights/biases) — stored in the model file
- **outputs**: predictions — bounding boxes, class scores, text tokens, audio waveforms

**Inference** = evaluate `f` on new inputs. The weights θ are frozen (not changing).

**Training** = adjust θ using data + optimizer (Adam, SGD, etc.). You don't do this.

### What inference looks like in Python

```python
import onnxruntime as ort
import numpy as np

# Load model (weights θ are inside the .onnx file)
session = ort.InferenceSession("yolo/yolov12m.onnx")

# Create input tensor (1 image, 3 channels, 640x640 pixels)
image = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Run inference: outputs = f(image; θ)
outputs = session.run(None, {"images": image})

# outputs[0] contains the predictions (bounding boxes, class scores)
print(f"Output shape: {outputs[0].shape}")  # e.g. (1, 84, 8400)
```

That's it. Load, feed input, get output. The model does not learn or change.

---

## 3. Weights / parameters — why model files are big

A model's `.onnx` file contains:

1. **The graph** (nodes = operations, edges = data flow) — relatively small
2. **The weights** (initializer tensors) — the bulk of the file size

Example sizes:

| Model | Parameter count | File size (approx) |
|-------|----------------|---------------------|
| MobileNetV2 | ~3.4M params | ~14 MB |
| YOLO v12m | ~20M params | ~80 MB |
| CLIP ViT-B/32 | ~151M params | ~600 MB |
| Qwen3-1.7B | ~1.7B params | ~3.4 GB FP16 / ~6.8 GB FP32 (needs external data) |
| DeepSeek-R1 | larger | even bigger |

Each float32 parameter = 4 bytes. So 1 billion params ≈ 4 GB.

---

## 4. Layers / operators (building blocks)

These are the ONNX operator names you'll see in Netron and in ORT logs:

| Op | What it does | Where you see it |
|----|-------------|-----------------|
| **Conv** | Convolution — detects local spatial patterns in images | YOLO, MobileNetV2, EfficientNet |
| **MatMul / Gemm** | Matrix multiplication — linear projections | Every model (attention, dense layers) |
| **Relu** | Activation — zeroes negative values, passes positive | After Conv/Linear in CNNs |
| **GELU** | Smooth activation — common in Transformers | Qwen3, CLIP, CrossFormer |
| **Sigmoid** | Squashes values to 0-1 range | YOLO detection heads |
| **Softmax** | Turns logits into probabilities (sum to 1) | Attention weights, classification |
| **BatchNormalization** | Normalizes activations per batch — stabilizes training | MobileNetV2, EfficientNet |
| **LayerNorm** | Normalizes per-layer — standard in Transformers | Qwen3, CLIP |
| **Attention / MultiHeadAttention** | Compares all positions in a sequence | Transformers (may be fused or unfused) |
| **Reshape / Transpose** | Data layout changes | Everywhere |
| **Add / Mul** | Element-wise arithmetic | Residual connections, scaling |
| **Concat** | Join tensors along an axis | Multi-scale feature maps (YOLO) |

---

## 5. CNN vs Transformer — when each is used

### CNN (Convolutional Neural Network)

- **Strength:** local pattern detection (edges → textures → objects), spatial hierarchy
- **Your models:** YOLO, MobileNetV2, EfficientNet, DeepSort/OSNet, PaDiM
- **Key ops:** Conv, BatchNorm, Relu, MaxPool
- **Input:** images `(N, C, H, W)`
- **Output:** feature maps, bounding boxes, class scores

### Transformer

- **Strength:** comparing all positions in a sequence via attention ("what relates to what")
- **Your models:** Qwen3, CLIP, CrossFormer, LLaMA, DeepSeek-R1
- **Key ops:** MatMul, LayerNorm, Softmax, GELU, Attention
- **Input:** token sequences `(N, SeqLen)` or image patches `(N, NumPatches, HiddenDim)`
- **Output:** logits, embeddings, generated tokens

### Vision Transformer (ViT) — hybrid

- Splits image into patches, treats each patch as a "token"
- Your CLIP models use ViT
- Still uses attention, but input is image-derived

---

## 6. Encoder vs Decoder

### Encoder

- Maps raw input → compact representation (embedding)
- **Example:** CLIP image encoder takes `(1, 3, 224, 224)` → `(1, 512)` embedding vector
- **Example:** Whisper encoder takes mel spectrogram → audio features

### Decoder

- Generates output step by step from a representation
- **Example:** Qwen3 decoder generates one token at a time, each step conditioned on all previous tokens
- **Example:** XTTS decoder generates audio waveform from text + speaker embedding

### Encoder-Decoder (both)

- **Example:** Whisper: encoder (audio → features) + decoder (features → text tokens)
- **Example:** Some robotics models: vision encoder + action decoder

Not every model neatly splits — some are encoder-only (CLIP, BERT), some are decoder-only (Qwen3, LLaMA).

---

## 7. What "logits" means

You'll see this word everywhere. **Logits** = raw, unnormalized scores from the model's last layer.

- For classification: logits shape `(1, 1000)` means scores for 1000 classes. Apply softmax to get probabilities.
- For LLMs: logits shape `(1, SeqLen, VocabSize)` means score for each possible next token.

---

## 8. Attention mechanism — Q, K, V (how Transformers actually work)

You work with Transformers daily (Qwen3, CLIP, CrossFormer, Whisper). Attention fusion is one of your top optimization targets.

### The core idea

Each position in a sequence asks: "Who should I pay attention to?"

| Component | Role | Analogy |
|-----------|------|---------|
| **Query (Q)** | "What am I looking for?" | Each position asks a question |
| **Key (K)** | "What do I contain?" | Each position advertises its content |
| **Value (V)** | "What information do I give?" | The actual content to retrieve |

### The math (simplified)

```
scores    = Q × K^T           # dot product: who should I attend to?
weights   = softmax(scores / sqrt(head_dim))  # normalize to probabilities
output    = weights × V       # weighted sum of values
```

**Scaling by `sqrt(head_dim)`** prevents softmax from saturating (all probability on one position).

### Multi-head attention

Run **multiple Q/K/V sets in parallel** — each "head" learns to attend to different patterns (syntax, semantics, position, etc.). Then concatenate all heads.

```
head_1 = Attention(Q_1, K_1, V_1)
head_2 = Attention(Q_2, K_2, V_2)
...
output = Concat(head_1, head_2, ...) × W_out
```

For Qwen3-1.7B: 16 heads, hidden_size 2048, so head_dim = 2048/16 = 128.

### Cross-attention

Q comes from one modality, K+V from another:

- **CLIP:** Q from text, K+V from image (or vice versa) → aligns text and image
- **Whisper decoder:** Q from text tokens, K+V from audio features
- **OpenVLA:** Q from language instructions, K+V from image observations

### Why this matters for optimization

Attention fusion (Module 06 §8, Module 18 Phase F) replaces the sequence of MatMul→Scale→Softmax→MatMul with a **single fused kernel** — 10-25% faster.

---

## 9. KV Cache (LLM concept)

When LLMs generate text token by token, they compute attention over all previous tokens. **KV cache** stores the Key and Value tensors from previous steps so they don't need to be recomputed.

- Without KV cache: generating token N recomputes K and V for all N-1 previous tokens
- With KV cache: only compute K, V for the new token; reuse cached values for all previous tokens
- Cache size grows linearly with sequence length — affects memory usage
- Relevant to TTFT (time to first token) metrics

---

## 10. Autoregressive generation — how LLMs produce text

You benchmark Qwen3 and LLaMA. Understanding token-by-token generation explains TTFT and token throughput.

### Step-by-step

| Step | What happens |
|------|-------------|
| 1. **Encode prompt** | Feed all input tokens at once. This is the "prefill" step. **TTFT measures this.** |
| 2. **Get logits** | Model outputs probability for every possible next token (`vocab_size` scores) |
| 3. **Sample one token** | Pick the next token (greedy = argmax, temperature sampling = weighted random) |
| 4. **Append to input** | Add the chosen token to the input sequence |
| 5. **Repeat from step 2** | Run model again with extended input. KV cache avoids recomputing previous tokens. |
| 6. **Stop** | When model generates EOS (end-of-sequence) token or max length reached |

### Decoding strategies

| Term | Meaning |
|------|---------|
| **Temperature** | Controls randomness. 0 = always pick highest probability. 1 = sample proportionally. >1 = more random |
| **Top-k** | Only consider the k most likely tokens |
| **Top-p (nucleus)** | Only consider tokens whose cumulative probability exceeds p |
| **Greedy decoding** | Always pick the most probable token (temperature=0) |
| **Beam search** | Track N parallel hypotheses, pick best overall sequence |

### In code (simplified)

```python
import numpy as np

input_ids = tokenizer.encode("Hello, how are")  # [15496, 11, 703, 527]

for _ in range(50):  # generate up to 50 tokens
    logits = session.run(None, {"input_ids": np.array([input_ids], dtype=np.int64)})[0]
    next_token_logits = logits[0, -1, :]         # scores for next position
    next_token = int(np.argmax(next_token_logits))  # greedy: pick highest
    if next_token == tokenizer.eos_token_id:
        break
    input_ids.append(next_token)

print(tokenizer.decode(input_ids))
```

---

## Module 03 checklist

- [ ] Can explain inference vs training in one sentence
- [ ] Can name 5 ONNX op types and what they roughly do
- [ ] Can explain why weights make ONNX files large (params × 4 bytes for float32)
- [ ] Can classify your models: which are CNN-based, which are Transformer-based
- [ ] Can explain encoder vs decoder with a concrete example from your model zoo
- [ ] Know what "logits" means
- [ ] Can explain Q, K, V and why multi-head attention uses multiple heads
- [ ] Can explain cross-attention and give one example from your models
- [ ] Can describe the 6 steps of autoregressive generation
- [ ] Can explain temperature, top-k, top-p in one sentence each
- [ ] Can explain what KV cache avoids recomputing

**Next:** `04-onnx-format-and-graphs.md`
