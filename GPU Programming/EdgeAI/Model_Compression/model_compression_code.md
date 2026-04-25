# 💻 EdgeAI · Model Compression — CODE

### *Runnable Python — quantize, prune, distill, factorize*

> **Nav:** [← Model Compression README](README.md) | [📖 THEORY](model_compression_theory.md) | **CODE** | [🎯 PRACTICE →](model_compression_practice.md)

---

## 🏗️ Setup

Colab with **T4 GPU**. Install what's needed:

```python
!pip install -q torch torchvision tensorflow onnx onnxruntime bitsandbytes
```

---

## Ex 1 — FP16 weight-only (2× shrink, zero calibration)

### 👶 What this does
The safest compression: convert FP32 weights to FP16 at rest. Runtime
casts back to FP32 or runs in FP16 natively if the chip supports it.

```python
import torch, torchvision, os

m = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
torch.save(m.state_dict(), "mb2_fp32.pt")

# Convert weights to FP16 in-place
for p in m.parameters():
    p.data = p.data.half()
torch.save(m.state_dict(), "mb2_fp16.pt")

fp32 = os.path.getsize("mb2_fp32.pt") / 1024**2
fp16 = os.path.getsize("mb2_fp16.pt") / 1024**2
print(f"FP32: {fp32:5.2f} MB   FP16: {fp16:5.2f} MB   shrink: {fp32/fp16:.2f}×")
```

Expect **~2×** shrink with essentially zero accuracy drop on vision
models.

---

## Ex 2 — INT8 PTQ dynamic-range (TFLite, 1-line)

### 👶 What this does
TFLite's easiest quant mode. No calibration data needed; activations
are quantized on the fly.

```python
import tensorflow as tf
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]
model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation="softmax")])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["acc"])
model.fit(x_train, y_train, epochs=1, batch_size=256, verbose=0)

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]     # this one line does it
open("model_dynrange.tflite", "wb").write(conv.convert())
```

---

## Ex 3 — INT8 PTQ full-integer with calibration (TFLite)

### 👶 What this does
The mode every NPU (Coral, Hexagon, Ethos-U55) actually wants —
everything INT8, even inputs and outputs.

```python
def representative():
    for i in range(100):
        yield [x_train[i:i+1]]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
open("model_full_int8.tflite", "wb").write(conv.convert())
```

---

## Ex 4 — INT8 PTQ with PyTorch (`torch.ao.quantization`)

### 👶 What this does
PyTorch's native quant API. Per-channel weights, asymmetric
activations, fused modules.

```python
import torch.ao.quantization as Q
import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = Q.QuantStub()
        self.conv    = nn.Conv2d(1, 16, 3, padding=1)
        self.relu    = nn.ReLU()
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(16, 10)
        self.dequant = Q.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return self.dequant(x)

m = TinyCNN().eval()
m.qconfig = Q.get_default_qconfig("x86")     # per-channel weights
m_fused   = Q.fuse_modules(m, [["conv", "relu"]])
m_prep    = Q.prepare(m_fused)

# Calibrate
import torch
for i in range(30):
    m_prep(torch.randn(1, 1, 28, 28))

m_int8 = Q.convert(m_prep)
print(m_int8)   # layers are now QuantizedConv2d, QuantizedLinear, …
```

---

## Ex 5 — Quantization-Aware Training (QAT)

### 👶 What this does
Insert **fake-quant** nodes before training so the model learns to be
robust to rounding.

```python
m_qat = TinyCNN().train()
m_qat.qconfig = Q.get_default_qat_qconfig("x86")
Q.fuse_modules(m_qat, [["conv", "relu"]], inplace=True)
Q.prepare_qat(m_qat, inplace=True)

opt = torch.optim.Adam(m_qat.parameters(), lr=1e-3)
for epoch in range(2):
    for step in range(100):
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        loss = nn.functional.cross_entropy(m_qat(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

m_qat.eval()
m_int8_qat = Q.convert(m_qat)
```

Expect **~0.3–0.8 %** better accuracy than PTQ on the same
architecture — the extra training teaches the weights to "be okay"
with INT8 rounding.

---

## Ex 6 — Pruning with `torch.nn.utils.prune` (magnitude, unstructured)

### 👶 What this does
Zero out the lowest-magnitude weights. Returns a sparse model.

```python
import torch.nn.utils.prune as P
layer = nn.Linear(256, 256)
P.l1_unstructured(layer, name="weight", amount=0.5)   # 50 % sparsity
print("Sparsity:", (layer.weight == 0).float().mean().item())

# Iterative pruning schedule (grow sparsity over 5 rounds)
schedule = [0.2, 0.4, 0.55, 0.65, 0.75]
layer2 = nn.Linear(256, 256)
for s in schedule:
    P.l1_unstructured(layer2, name="weight", amount=s)
    # retrain a few epochs here to recover accuracy (not shown)

# When you're done, bake the mask in so the layer is truly dense-sparse
P.remove(layer2, "weight")
```

### Structured pruning (whole channels — gives real speed-up)

```python
conv = nn.Conv2d(32, 64, 3)
# remove 30 % of output channels with lowest L2 norm
P.ln_structured(conv, name="weight", amount=0.3, n=2, dim=0)
```

---

## Ex 7 — 2:4 structured sparsity (Ampere+ GPUs, Jetson Orin)

### 👶 What this does
Two zeros in every contiguous group of four. TensorRT / cuSPARSELt
gives a deterministic ~2× speedup.

```python
import torch

def enforce_2_to_4(w):
    """Zero the 2 smallest-magnitude weights in every group of 4
       along the last dim."""
    orig_shape = w.shape
    w = w.reshape(-1, 4)
    # indices of the 2 smallest |w| in each row
    _, idx = torch.topk(w.abs(), 2, dim=1, largest=False)
    mask = torch.ones_like(w, dtype=torch.bool)
    mask.scatter_(1, idx, False)
    return (w * mask).reshape(orig_shape)

fc = nn.Linear(1024, 1024)
fc.weight.data = enforce_2_to_4(fc.weight.data)
# Verify
g4 = fc.weight.data.reshape(-1, 4)
zeros_per_group = (g4 == 0).sum(dim=1)
print("2:4 check — should all be 2:", zeros_per_group.unique())
```

Export to ONNX, build with `trt.BuilderFlag.SPARSE_WEIGHTS` — engine
runs ~1.8× faster on H100 / Orin.

---

## Ex 8 — Logit distillation (10-line training loop)

### 👶 What this does
Small student learns from a big teacher's *soft* predictions.

```python
import torch.nn.functional as F
T, alpha = 4.0, 0.1     # temperature + hard/soft mix

def distill_step(student, teacher, x, y, opt):
    student_logits = student(x)
    with torch.no_grad():
        teacher_logits = teacher(x)

    hard = F.cross_entropy(student_logits, y)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits  / T, dim=-1),
        reduction="batchmean") * (T * T)

    loss = alpha * hard + (1 - alpha) * soft
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

# Usage:
# teacher = torchvision.models.resnet50(weights="DEFAULT").eval()
# student = torchvision.models.mobilenet_v2(weights=None).train()
# opt     = torch.optim.Adam(student.parameters(), lr=3e-4)
# for x, y in loader: distill_step(student, teacher, x, y, opt)
```

The **`T²`** factor keeps the soft-loss gradient magnitude comparable
to the hard-loss term. Without it, soft loss vanishes.

---

## Ex 9 — Feature distillation (hint layer)

### 👶 What this does
Match the *intermediate activations* of teacher and student at a
chosen layer — stronger signal than logits alone.

```python
teacher_feat, student_feat = [], []

def teacher_hook(m, i, o): teacher_feat.append(o)
def student_hook(m, i, o): student_feat.append(o)

# teacher.layer3.register_forward_hook(teacher_hook)
# student.features[8].register_forward_hook(student_hook)

def feature_loss():
    # Align shapes via a 1×1 conv if channel counts differ
    return F.mse_loss(student_feat[-1], teacher_feat[-1].detach())
```

Typical combination: `loss = CE + λ1·logit_KL + λ2·feature_mse`, with
`λ1≈0.9, λ2≈0.1`.

---

## Ex 10 — SVD low-rank factorization of a Linear layer

### 👶 What this does
Replace one big matmul with two smaller ones.

```python
import torch

def svd_compress(linear, rank):
    W = linear.weight.data             # (out, in)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    U = U[:, :rank] * S[:rank]         # (out, r)
    V = Vt[:rank, :]                   # (r, in)

    factored = torch.nn.Sequential(
        torch.nn.Linear(linear.in_features, rank, bias=False),
        torch.nn.Linear(rank, linear.out_features, bias=linear.bias is not None))
    factored[0].weight.data = V
    factored[1].weight.data = U
    if linear.bias is not None:
        factored[1].bias.data = linear.bias.data
    return factored

fc = torch.nn.Linear(1024, 1024)
fc_sm = svd_compress(fc, rank=128)
p_before = sum(p.numel() for p in fc.parameters())
p_after  = sum(p.numel() for p in fc_sm.parameters())
print(f"Params before: {p_before:,}   after: {p_after:,}   shrink: {p_before/p_after:.2f}×")
```

---

## Ex 11 — 4-bit / NF4 for LLMs with BitsAndBytes

### 👶 What this does
Load an LLM in 4-bit NF4 quantization. One of the few ways to fit 7 B
models on a single 8 GB GPU.

```python
# pip install bitsandbytes accelerate transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-1B", quantization_config=bnb, device_map="auto")
# print("Footprint:", model.get_memory_footprint() / 1024**2, "MB")
```

A 7 B model drops from ~14 GB (FP16) to ~4 GB (NF4). QLoRA fine-tuning
runs on a single 16 GB card.

---

## Ex 12 — The full Pareto loop (size vs accuracy vs latency)

### 👶 What this does
A stub you can fill in with real numbers after running Exercises 1–6
on the same dataset. Produces a chart you can drop in a design review.

```python
import matplotlib.pyplot as plt

configs = [
    ("FP32",            100, 76.0, 30),
    ("FP16",             50, 75.9, 20),
    ("INT8 PTQ",         25, 75.2, 10),
    ("INT8 + 50% prune", 13, 74.8, 8),
    ("INT4 NF4",         12, 73.5, 9),
    ("INT8 + distilled",  6, 74.4, 5),
]
names, size, acc, lat = zip(*configs)
plt.figure(figsize=(7, 5))
plt.scatter(size, acc, s=[l * 20 for l in lat])
for n, s, a in zip(names, size, acc):
    plt.annotate(n, (s, a), xytext=(4, 4), textcoords="offset points", fontsize=8)
plt.xlabel("Size (MB)"); plt.ylabel("Top-1 Accuracy (%)")
plt.title("Compression Pareto — bubble size = latency (ms)")
plt.grid(alpha=0.3); plt.show()
```

---

## 📝 Summary

| Exercise | Technique | Typical result |
|---|---|---|
| 1 | FP16 weights | 2× shrink, 0 acc drop |
| 2 | Dyn-range INT8 (TFLite) | 4× shrink, 0.5 % drop |
| 3 | Full-integer INT8 (TFLite) | NPU-ready |
| 4 | INT8 PTQ (PyTorch) | 4× shrink, 0.5 % drop |
| 5 | QAT (PyTorch) | recovers ~0.5 % vs PTQ |
| 6 | Unstructured pruning | 50 % sparsity |
| 7 | 2:4 structured sparsity | ~2× GPU speed-up |
| 8 | Logit distillation | 2–10× smaller student |
| 9 | Feature distillation | +0.3 % over logit-only |
| 10 | SVD low-rank | dim-r factor shrink |
| 11 | NF4 LLM | ~3.5× footprint cut |
| 12 | Pareto plot | ship-ready chart |

Now put them **together** in the [practice notebook](model_compression_practice.md).

---

> *GPU Programming · EdgeAI · Model Compression · CODE · github.com/rpaut03l/TS-02*
