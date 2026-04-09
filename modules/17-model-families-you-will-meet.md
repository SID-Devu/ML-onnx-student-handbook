# Module 17 — Model families: all your models mapped (full depth)

This maps **every model in your benchmark zoo** to its domain, architecture, inputs/outputs, and practical notes.

---

## Complete model catalog

| # | Domain | Model name | Architecture | What it does |
|---|--------|-----------|-------------|-------------|
| 1 | Object Detection | **YOLO v12m** | CNN (CSPDarknet + FPN) | Finds and labels objects in images with bounding boxes |
| 2 | Image Classification | **MobileNetV2** | CNN (inverted residuals) | Classifies images: "this is a cat" |
| 3 | Image Classification | **EfficientNet** | CNN (compound scaling) | Classifies images; known for efficiency/accuracy tradeoff |
| 4 | Vision-Language | **CLIP ViT-B/32** | Transformer (ViT encoder) | Maps images + text into shared embedding space |
| 5 | Vision-Language | **CLIP ViT-B/16** | Transformer (ViT encoder) | Same as CLIP B/32 but with 16×16 patches (higher res) |
| 6 | Object Tracking | **DeepSort (OSNet)** | CNN (Re-ID backbone) | Generates appearance embeddings to track objects across video frames |
| 7 | LLM | **Qwen3-1.7B** | Transformer (decoder-only) | Text generation, chat, reasoning |
| 8 | LLM | **DeepSeek-R1** | Transformer (decoder-only) | Reasoning-focused LLM |
| 9 | LLM | **LLaMA-3.2** | Transformer (decoder-only) | General-purpose LLM (may have vision encoder component) |
| 10 | Speech (TTS) | **XTTS** | Encoder-decoder + vocoder | Converts text → speech audio with speaker voice cloning |
| 11 | Speech (ASR) | **Whisper** | Transformer encoder-decoder | Converts speech audio → text |
| 12 | Robotics VLA | **OpenVLA** | Vision-Language-Action | Takes images + language instructions → robot actions |
| 13 | Robotics | **CrossFormer** | Transformer (cross-attention) | Cross-embodiment robot policy |
| 14 | Robotics | **CogACT** | Transformer variant | Cognitive action planning |
| 15 | Robotics | **Pi-0** | Transformer policy | Robot manipulation policy |
| 16 | Stereo Depth | **RAFT-Stereo** | CNN + correlation + GRU | Estimates depth/disparity from two camera images |
| 17 | Anomaly Detection | **PaDiM** | CNN (pretrained backbone + Mahalanobis) | Detects defects/anomalies in manufactured parts |
| 18 | Segmentation | **MobileSAM** | Transformer + CNN | Cuts out objects from images (segment-anything) |
| 19 | OCR | **EasyOCR** | CNN + LSTM/Transformer | Reads text in images |
| 20 | 3D Detection | **CenterPoint** | CNN (pillar/voxel encoder) | Detects objects in 3D point clouds (LiDAR data) |
| 21 | Navigation | **ViNT** | Transformer/CNN policy | Robot visual navigation |
| 22 | Wav2Vec2 | **Wav2Vec2** | Transformer (self-supervised) | Audio representation learning / ASR features |
| 23 | (others) | Various | Various | Your zoo may grow — apply the same analysis pattern |

---

## Input/output patterns by domain

### Object detection (YOLO)

- **Input:** `images` shape `(1, 3, 640, 640)` float32
- **Output:** `detections` shape `(1, 84, 8400)` — 8400 candidate boxes, 84 = 4 coords + 80 class scores
- **Postprocessing:** NMS (non-max suppression) on CPU to filter overlapping boxes
- **INT8 candidate:** Yes — CNNs tolerate quantization well

### Classification (MobileNetV2, EfficientNet)

- **Input:** `input` shape `(1, 3, 224, 224)` float32
- **Output:** `logits` shape `(1, 1000)` — scores for 1000 ImageNet classes
- **Postprocessing:** argmax to get predicted class
- **INT8 candidate:** Yes

### Vision-Language (CLIP)

- **Image input:** `pixel_values` shape `(1, 3, 224, 224)` float32
- **Text input:** `input_ids` shape `(1, 77)` int64 + `attention_mask` shape `(1, 77)` int64
- **Output:** image embedding + text embedding (compare with cosine similarity)
- **INT8 candidate:** Possible for image encoder; text encoder may be sensitive

### Tracking (DeepSort / OSNet)

- **Input:** `input` shape `(1, 3, 256, 128)` float32 — person crop
- **Output:** embedding vector (e.g., 512-dim)
- **Used for:** comparing person appearances across video frames
- **INT8 candidate:** Yes

### LLM (Qwen3, DeepSeek-R1, LLaMA)

- **Input:** `input_ids` shape `(1, seq_len)` int64 + `attention_mask` int64
- **Output:** `logits` shape `(1, seq_len, vocab_size)` — next-token scores
- **Key metric:** TTFT (time to first token)
- **Size:** Qwen3-1.7B ≈ 3.4 GB (needs external data)
- **Quantization:** INT8 risky for attention; mixed precision may help

### Speech (XTTS, Whisper)

- **Whisper input:** mel spectrogram (audio features)
- **Whisper output:** text tokens
- **XTTS input:** text tokens + speaker conditioning
- **XTTS output:** audio waveform
- **CPU-bound preprocessing:** heavy (FFT, mel computation)
- **Thread pinning helps** (Module 06)

### Robotics (OpenVLA, CrossFormer, CogACT, Pi-0)

- **Input:** images + language instructions (multimodal)
- **Output:** action vectors (robot joint commands, gripper actions)
- **Challenges:** large graphs, dynamic shapes, sometimes custom ops
- **Size:** OpenVLA is multi-GB (needs external data)

### Stereo depth (RAFT-Stereo)

- **Input:** pair of images (left + right camera)
- **Output:** disparity map (depth estimation)
- **Export challenge:** `CorrSampler` is a custom CUDA op → must be replaced for ONNX export

### Anomaly detection (PaDiM)

- **Input:** image features from a pretrained CNN backbone
- **Output:** anomaly score map
- **Small model** — good INT8 candidate

### Segmentation (MobileSAM)

- **Input:** image + optional prompts (point/box)
- **Output:** segmentation masks
- **Variable prompt handling** in the graph

### OCR (EasyOCR)

- **Input:** text region image
- **Output:** recognized text string (via pipeline: detection → recognition)
- **Pipeline model** — may be multiple ONNX files

### 3D Detection (CenterPoint)

- **Input:** point cloud tensors (from LiDAR)
- **Output:** 3D bounding boxes
- **Different preprocessing** from image-based models (pillar/voxel encoding)

### Navigation (ViNT)

- **Input:** visual observations (camera images)
- **Output:** navigation actions/waypoints
- **Robot visual navigation policy**

---

## Pre-processing by domain — the code that runs BEFORE the model

### Image pre-processing (YOLO, MobileNetV2, CLIP, EfficientNet, DeepSort)

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg")                       # (H, W, 3) BGR uint8

# 1. Resize to model's expected size
img = cv2.resize(img, (640, 640))                    # YOLO expects 640x640

# 2. BGR → RGB (OpenCV loads BGR, models expect RGB)
img = img[:, :, ::-1]

# 3. Normalize pixel values 0-255 → 0.0-1.0
img = img.astype(np.float32) / 255.0

# 4. ImageNet normalization (MobileNetV2, EfficientNet, CLIP)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img = (img - mean) / std

# 5. HWC → NCHW layout
img = img.transpose(2, 0, 1)[np.newaxis, ...]       # (1, 3, H, W)
```

| Step | What | Code |
|------|------|------|
| Resize | Scale image to model input size | `cv2.resize(img, (640, 640))` |
| Channel order | BGR (OpenCV) to RGB | `img = img[:, :, ::-1]` |
| Normalize | Scale pixels 0-255 → 0.0-1.0 | `img = img / 255.0` |
| ImageNet norm | Subtract mean, divide by std | `(img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]` |
| Layout | HWC to NCHW | `img.transpose(2, 0, 1)[np.newaxis, ...]` |

### Audio pre-processing (Whisper, Wav2Vec2, XTTS)

```python
import numpy as np

# 1. Resample to model's expected rate (16 kHz for Whisper/Wav2Vec2)
# Use librosa or torchaudio for resampling
# audio = librosa.resample(audio, orig_sr=44100, target_sr=16000)

# 2. Mel spectrogram (Whisper) — converts waveform to frequency-domain
# whisper uses 80 mel bins, 30s window → shape (1, 80, 3000)

# 3. Padding/trimming — ensure fixed-length input
target_len = 16000 * 30  # 30 seconds at 16kHz
if len(audio) < target_len:
    audio = np.pad(audio, (0, target_len - len(audio)))
else:
    audio = audio[:target_len]
```

| Step | What |
|------|------|
| Resample | Convert any sample rate to model's expected rate (16 kHz) |
| Mel spectrogram | Convert waveform to frequency-domain representation (Whisper) |
| Padding/trimming | Ensure fixed-length input |

### Text pre-processing (Qwen3, CLIP text, LLaMA)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

text = "What is the capital of France?"
encoded = tokenizer(text, padding="max_length", max_length=128, return_tensors="np")

input_ids      = encoded["input_ids"]        # (1, 128) int64 — token IDs
attention_mask = encoded["attention_mask"]    # (1, 128) int64 — 1=real, 0=padding
```

| Step | What |
|------|------|
| Tokenization | Convert text string to integer token IDs |
| Padding | Pad shorter sequences to fixed length |
| Attention mask | Binary mask: 1 for real tokens, 0 for padding |
| BPE / SentencePiece | Specific tokenizer algorithms that break words into subwords |

---

## Post-processing by domain — the code that runs AFTER the model

| Domain | Step | What it does | Code |
|--------|------|-------------|------|
| Detection (YOLO) | **NMS** (Non-Maximum Suppression) | Filters overlapping bounding boxes, keeps best ones | `cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.45)` |
| Detection | **Score thresholding** | Discard detections below confidence threshold | `boxes[scores > 0.5]` |
| Classification | **argmax** | Picks the class with highest score | `predicted_class = np.argmax(logits)` |
| LLM (Qwen3) | **Greedy / Beam search** | Picks next token(s) from probability distribution | `next_token = np.argmax(logits[0, -1, :])` |
| OCR (EasyOCR) | **CTC decoding** | Converts frame-level predictions to text string | Library handles this internally |
| Speech (Whisper) | **Token decoding** | Converts token IDs back to text | `tokenizer.decode(token_ids)` |

---

## Evaluation metrics by domain — how you know if a model is accurate

You need these when validating INT8 quantization — "did quantization break accuracy?"

| Domain | Metric | What it measures |
|--------|--------|-----------------|
| Detection (YOLO) | **mAP** (mean Average Precision) | How well boxes match ground truth at various IoU thresholds |
| Detection | **IoU** (Intersection over Union) | Overlap between predicted and actual box (0-1) |
| Classification | **Top-1 / Top-5 accuracy** | Is the correct class the #1 prediction? In top 5? |
| LLM | **Perplexity** | How "surprised" the model is by correct text (lower = better) |
| Speech (ASR) | **WER** (Word Error Rate) | Fraction of words incorrectly transcribed |
| Text generation | **BLEU / ROUGE** | How similar generated text is to reference text |
| Segmentation | **mIoU** | Average IoU across all classes |
| Anomaly detection | **AUROC** | Area under ROC curve (separates normal from anomalous) |
| Depth estimation | **Abs Rel / RMSE** | Error between predicted and actual depth |

### Quick accuracy check after INT8 quantization

```python
import numpy as np

# Run same input through FP32 and INT8 models
out_fp32 = session_fp32.run(None, feed)[0]
out_int8 = session_int8.run(None, feed)[0]

# Compare
max_diff = np.max(np.abs(out_fp32 - out_int8))
mean_diff = np.mean(np.abs(out_fp32 - out_int8))
print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

# For classification: do they agree on top-1?
agree = np.argmax(out_fp32) == np.argmax(out_int8)
print(f"Top-1 agreement: {agree}")
```

---

## Cross-cutting analysis for any new model

For **every** model you add to the benchmark:

1. **Input names + dtypes + shapes** — `session.get_inputs()`
2. **Output names + dtypes + shapes** — `session.get_outputs()`
3. **Dynamic axes?** — any string dims → add `dim_overrides`
4. **External data?** — check for `.onnx.data` sidecar
5. **Size** — can it fit in your GTT without swap?
6. **INT8 candidate?** — CNNs usually yes, attention layers usually no
7. **CPU preprocessing?** — heavy preprocessing → consider thread pinning

---

## Module 17 checklist

- [ ] Can classify each of your models by domain
- [ ] Can state the typical input shape for YOLO, CLIP, and Qwen3
- [ ] Can preprocess an image for YOLO (resize, normalize, HWC→NCHW)
- [ ] Can explain NMS and why detection needs post-processing
- [ ] Can tokenize text for an LLM and explain attention masks
- [ ] Can name the accuracy metric for detection (mAP), classification (Top-1), LLM (perplexity), ASR (WER)
- [ ] Can compare FP32 vs INT8 outputs to check if quantization broke accuracy
- [ ] For any new ONNX model, can determine inputs/outputs/shapes/dynamic dims

**Next:** `18-end-to-end-optimization-playbook.md`
