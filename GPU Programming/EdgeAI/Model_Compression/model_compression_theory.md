# 📖 EdgeAI · Model Compression — THEORY

### *Quantization · Pruning · Distillation · Low-rank — the four levers*

> **Nav:** [← Model Compression README](README.md) | **THEORY** | [💻 CODE](model_compression_code.md) | [🎯 PRACTICE](model_compression_practice.md)

---

## 🧠 MNEMONIC: **"Q-P-D-L"**

> **Q**uantize · **P**rune · **D**istill · **L**ow-rank

Four levers. You pull one, two, or all of them. More levers = more
shrink, usually at the cost of more accuracy loss and more engineering.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Why we compress | [§1](#1-why-we-compress) |
| 2 | **Quantization** — turning big numbers into small ones | [§2](#2-quantization--turning-big-numbers-into-small-ones) |
| 3 | PTQ vs QAT (and "fake quant") | [§3](#3-ptq-vs-qat-and-fake-quant) |
| 4 | **Pruning** — cutting away what doesn't matter | [§4](#4-pruning--cutting-away-what-doesnt-matter) |
| 5 | **Knowledge Distillation** — small student, smart teacher | [§5](#5-knowledge-distillation--small-student-smart-teacher) |
| 6 | **Low-Rank Factorization** — thin-out fat layers | [§6](#6-low-rank-factorization--thin-out-fat-layers) |
| 7 | Stacking all four | [§7](#7-stacking-all-four) |
| 8 | Accuracy ↔ size ↔ latency ↔ power Pareto | [§8](#8-accuracy--size--latency--power-pareto) |
| 9 | Decision tree: what to try first | [§9](#9-decision-tree-what-to-try-first) |
| 10 | Cheat sheet | [§10](#10-cheat-sheet--red-flags) |

---

## 1. Why we compress

### 👶 Easy Story
Your cloud model is a **grand piano**. Beautiful, but it doesn't fit in
your backpack. Compression is **rebuilding the same melody on a
toy keyboard** — smaller, lighter, battery-powered, and (almost) as
beautiful to listen to.

### The formal reasons
Every pillar from [Fundamentals/](../Fundamentals/README.md) § 4 pulls
in the direction of compression:

- **Low latency** — smaller models = fewer ops per forward pass.
- **Low power** — less memory traffic = fewer joules per inference.
- **Privacy + offline** — smaller models *fit* on the device at all.
- **Bandwidth** — smaller models download faster for OTA updates.

### A single number to remember
> **Memory movement, not math, is the #1 cost of inference on the edge.**
> Shrinking weights 4× usually speeds up inference 2–4×, because you
> spend less time reading DRAM.

[↑ Back to Top](#-edgeai--model-compression--theory)

---

## 2. Quantization — turning big numbers into small ones

### 👶 Easy Story
You don't need to write every price with 10 decimal places. "₹99.99"
and "₹100" buy the same coffee. Quantization rounds every weight to a
short, cheap representation.

### The number formats, ranked

```
 ┌──────────┬────────┬────────┬───────────────┬────────────────────┐
 │ Format   │ Bits   │ Shrink │ Accuracy hit  │ Where it shines    │
 ├──────────┼────────┼────────┼───────────────┼────────────────────┤
 │ FP32     │ 32     │ 1×     │ baseline      │ training / debug   │
 │ TF32     │ 19     │ ~1.7×  │ ~0 %          │ A100/H100 training │
 │ BF16     │ 16     │ 2×     │ ~0 %          │ training, robust   │
 │ FP16     │ 16     │ 2×     │ ~0 %          │ Jetson inference   │
 │ FP8 (E4M3)│ 8     │ 4×     │ 0–0.5 %       │ H100/B200 inference│
 │ INT8     │ 8      │ 4×     │ 0.2–1 %       │ the edge default   │
 │ INT4     │ 4      │ 8×     │ 1–3 %         │ LLMs, some CNNs    │
 │ NF4      │ 4      │ 8×     │ 0.5–2 % (LLMs)│ QLoRA, 4-bit LLMs  │
 │ INT2/bin │ 1-2    │ 16–32× │ 5–15 %        │ research / BNNs    │
 └──────────┴────────┴────────┴───────────────┴────────────────────┘
```

### The math, in one equation
Quantization maps a float `r` to an integer `q` using a **scale** `s`
and a **zero-point** `z`:

```
 q = round(r / s) + z
 r̂ = (q − z) · s           ← de-quantized back
```

- **Symmetric** (common for weights): `z = 0`. Range is `[-s·2^(b-1),
  s·(2^(b-1)-1)]`.
- **Asymmetric** (common for activations like ReLU): `z ≠ 0`. Range is
  `[0, s·(2^b − 1)]` (good for non-negative activations).

### Per-tensor vs per-channel
- **Per-tensor** — one scale for the whole weight tensor. Simple but
  loses accuracy when channel ranges differ a lot.
- **Per-channel** (along the output-channel axis of a conv/linear) — one
  scale per channel. Much better accuracy, tiny overhead. **Use this
  for weights by default.**

### What actually gets quantized

```
 ┌───────────────────────────────┐
 │ A convolution / linear layer   │
 │                                │
 │  input x   →  [weights W]  →  output y + activation  │
 │   (INT8 or FP16)   (INT8 per-channel)    (INT8 asymmetric) │
 └───────────────────────────────┘
```

- **Weights**: static; quantize once at conversion time.
- **Activations**: dynamic; need **calibration** data to estimate the
  range.
- **Biases**: usually stored in INT32 (sum of INT8 × INT8 fits there).

### "Dynamic-range" vs "full-integer" quant (TFLite terms)
- **Dynamic-range**: weights INT8, activations quantized *at runtime*
  on the fly. Zero calibration needed. Decent speed-up on CPU.
- **Full-integer**: everything INT8, needs representative data. Fastest
  on INT-only NPUs (Coral, Hexagon, Ethos-U55).
- **Float-fallback**: some ops remain FP32 where INT8 kernels don't
  exist. A compatibility escape hatch.

### So what?
> Default: **weights INT8 per-channel, activations INT8 asymmetric,
> biases INT32.** 90 % of real edge models ship exactly this way.

[↑ Back to Top](#-edgeai--model-compression--theory)

---

## 3. PTQ vs QAT (and "fake quant")

### 👶 Easy Story
Two ways to shrink a cake from FP32 to INT8:

- **PTQ (Post-Training Quantization)** — bake the cake as usual, then
  *cut it smaller* with a knife. Fast. Sometimes the cake crumbles.
- **QAT (Quantization-Aware Training)** — bake the cake *pretending*
  it's already small. The batter knows how it will be cut. Slower, but
  the result is always pretty.

### The formal story

```
 PTQ (minutes)                     QAT (hours–days)
 ─────────────                     ─────────────
 train FP32  ──▶  calibrate  ──▶   train with fake-quant nodes
        100 batches             (inserted at every weight and
                                 activation → forward uses INT8
                                 semantics, backward uses
                                 straight-through estimator)
        ↓                         ↓
 INT8 model                       INT8 model with ~0.5 % better
                                  accuracy than PTQ on the
                                  same architecture
```

### Fake-quant in one sentence
> A **fake-quant** op is a layer that, in the forward pass, quantizes
> then de-quantizes its input (so downstream layers see the *rounded*
> value), but in the backward pass passes gradients through unchanged
> (Straight-Through Estimator). The network learns to be robust to
> rounding during training.

### When to use which

| Situation | Choice |
|---|---|
| "I just need it working fast" | **PTQ dynamic-range** |
| Full-integer NPU target (Coral, Hexagon) | **PTQ full-integer with calibration** |
| Accuracy drop > 1 % after PTQ | **QAT** |
| Very small model (< 1 M params) | **QAT** (rounding errors bite harder) |
| Transformer / LLM | **PTQ with GPTQ / AWQ / SmoothQuant** tricks |

### STE pseudo-code
```python
# Forward
y_q = round(x / s) * s          # introduces rounding error

# Backward
grad_x = grad_y                 # pass through unchanged (STE)
```

[↑ Back to Top](#-edgeai--model-compression--theory)

---

## 4. Pruning — cutting away what doesn't matter

### 👶 Easy Story
A tree with 1,000,000 leaves. Many leaves do almost nothing — you could
cut them off and the tree would still be green. Pruning is **finding
the useless weights and zeroing them out**.

### The 3 granularities

```
 ┌──────────────────┬────────────────────────────┬──────────────────────┐
 │ Granularity      │ What you zero              │ Speed-up on hardware  │
 ├──────────────────┼────────────────────────────┼──────────────────────┤
 │ Unstructured     │ Any weight, anywhere       │ Little (sparse math)  │
 │ Structured       │ Whole neurons / filters /   │ Real — smaller dense  │
 │                  │ attention heads             │ matrices              │
 │ N:M (semi-       │ N out of every M contiguous │ Real — 2:4 on Ampere+ │
 │ structured)      │ weights                     │                       │
 └──────────────────┴────────────────────────────┴──────────────────────┘
```

### The criteria for "useless"
- **Magnitude** — remove weights with smallest |w|. Classic, simple,
  surprisingly good.
- **Gradient / saliency** — remove weights whose removal raises loss
  least (Taylor expansion).
- **Movement pruning** — track how weights move during fine-tuning;
  prune the ones moving toward zero.
- **Lottery ticket** — find a sparse sub-network that trains to full
  accuracy from scratch using the original init.

### Iterative magnitude pruning (the workhorse)

```
 repeat until sparsity target reached:
   1. train a few epochs
   2. rank all weights by |w|
   3. zero out the bottom k %
   4. fine-tune to recover
```

`k` grows gradually (e.g. 0 → 50 → 70 → 85 % across 3–5 rounds).
Sudden cuts hurt accuracy; gradual cuts heal.

### 2:4 structured sparsity (NVIDIA Ampere+)
Exactly 2 zeros in every contiguous group of 4 weights. The hardware
reads only the 2 non-zeros — giving a **deterministic ~2× speed-up** on
A100, H100, and Jetson Orin. Supported in TensorRT, cuSPARSELt, cuDNN.

### Pruning vs quantization — they stack
A **50 % magnitude-pruned + INT8** MobileNetV2 can be **~6× smaller**
than FP32 with < 1 % accuracy drop. That's the everyday combo.

[↑ Back to Top](#-edgeai--model-compression--theory)

---

## 5. Knowledge Distillation — small student, smart teacher

### 👶 Easy Story
Imagine a kid learning math. Option A: copy exam answers from a
textbook. Option B: learn from a smart teacher *who shows her thinking*
("the answer is 7, and by the way here's why 6 and 8 look close but
are wrong"). Option B wins — the student ends up a much better
problem solver.

Knowledge distillation is Option B for neural networks. The **student**
learns not just from labels, but from the **soft probabilities** of a
bigger **teacher**.

### The loss function

```
 L = α · CE(student_logits, hard_labels)
   + (1 − α) · T² · KL( softmax(student_logits / T),
                        softmax(teacher_logits / T) )
```

- `T` is a **temperature** — higher T softens the softmax and reveals
  more information about classes the teacher "almost" picked.
  Typical `T = 4`.
- `α` balances hard labels vs soft labels. Typical `α = 0.1`.
- `T²` keeps gradient magnitudes comparable across terms.

### The 3 "knowledge types"

| Type | What's distilled | Example |
|---|---|---|
| **Logit / response** | Final softmax probabilities | Hinton 2015 (the classic) |
| **Feature** | Intermediate activations | FitNets, AT, CRD |
| **Relational** | Pairwise relations *between* samples' features | RKD, PKT |

### Why it works so well
The teacher's soft distribution encodes **dark knowledge** — tiny
probabilities on wrong classes that tell the student about class
similarity (a student model learns that "cat" and "tiger" look
similar even if the dataset never tags them that way).

### When to use distillation
- You want a **much smaller architecture** (not just smaller weights).
- You have an **already-trained big model** (off-the-shelf teachers
  are everywhere).
- Accuracy-per-parameter is your metric.

### When not to
- Your teacher is barely better than a random small model. Distillation
  can't invent knowledge that isn't in the teacher.

[↑ Back to Top](#-edgeai--model-compression--theory)

---

## 6. Low-Rank Factorization — thin-out fat layers

### 👶 Easy Story
A 1000×1000 matrix has 1 M numbers. But most real weight matrices
have **a lot of redundancy**. You can write it as `A × B` with
`A` being 1000×64 and `B` being 64×1000 — total **128 k numbers,
~8× smaller**, approximately the same function.

### The math
For a weight matrix `W ∈ R^(m×n)`:

```
 W  ≈  U · V          with  U ∈ R^(m×r),  V ∈ R^(r×n)
 params:  m·n         →     r·(m + n)
 shrink:  (m·n) / (r·(m+n))
```

`r` is the **rank**. Pick it so `r·(m+n) < m·n`, i.e. `r < m·n/(m+n)`.

### Two common recipes
- **Truncated SVD** — compute `W = U Σ Vᵀ`, keep top-`r` singular
  values. One-shot, no training. Quick accuracy check needed.
- **Low-rank training** — replace `W` with `U·V` from the start and
  train them together. Related to **LoRA** for fine-tuning.

### CNN analogues
- **Depthwise-separable conv** (MobileNet, Xception) is a learned
  low-rank factorization — split a 3×3×C→C conv into 3×3 depthwise
  (C params) plus 1×1 pointwise (C² params). **Huge** savings.
- **Bottleneck blocks** (ResNet, SqueezeNet) use 1×1 → 3×3 → 1×1 to
  shrink the channel count inside the 3×3.

### So what?
> If you designed the architecture from scratch, low-rank is baked in.
> If you inherited a big matrix, SVD is a quick way to compress it
> before training even starts.

[↑ Back to Top](#-edgeai--model-compression--theory)

---

## 7. Stacking all four

### The "compound interest" of compression
Each lever multiplies.

```
 Start:      FP32 ResNet-50     25.6 M params    ~100 MB   76 % acc
 + INT8:                                         ~25 MB    75.5 %
 + 50 % prune (magnitude):                       ~13 MB    75 %
 + 2× distill to smaller student:                ~6  MB    74.5 %
 + low-rank on remaining FC:                     ~4  MB    74 %
```

**25× smaller** at ~2 % accuracy drop. Real ship-able edge model.

### The common orderings

1. **Design small first** (MobileNet, EfficientNet-Lite, MobileViT).
2. **Distill** from a big teacher.
3. **Prune** gradually.
4. **Fine-tune** after every prune round.
5. **Quantize** last (PTQ or QAT).

Many teams stop at step 5 — quantization alone gets them 80 % of the
way there.

[↑ Back to Top](#-edgeai--model-compression--theory)

---

## 8. Accuracy ↔ size ↔ latency ↔ power Pareto

### 👶 Easy Story
You can't have everything. Compression is always a trade — you're
moving a point along a curve. The **Pareto frontier** is the set of
configurations that are not dominated by any other on both axes.

### Typical shape

```
 accuracy
    ▲
 76% │  ● FP32 baseline
    │    \
    │     ● INT8 (PTQ)
 75% │      \
    │       ● INT8 + 30% prune
    │         \
 74% │          ● INT8 + 50% prune + distilled smaller
    │            \
 73% │             ● INT4 NF4
    │              \
 72% │               ● INT4 + 70% prune
    └──────────────────────────────────────────────►  size (MB)
    100MB   50MB   25MB   12MB   6MB    3MB
```

You pick a point based on your **deployment budget**: chip flash,
RAM, battery, or FPS requirement.

### Four axes you should plot
1. **Top-1 accuracy** (or your task metric)
2. **Model size on disk** (MB)
3. **Inference latency** (ms)
4. **Energy per inference** (mJ)

Two-axis charts hide bad trade-offs. Always plot at least 3.

[↑ Back to Top](#-edgeai--model-compression--theory)

---

## 9. Decision tree: what to try first

```
 What accuracy drop can you tolerate?

 < 0.5 %   →  FP16 weight-only OR INT8 PTQ dynamic-range
 < 1.5 %   →  INT8 PTQ full-integer (per-channel weights)
 < 3 %     →  INT8 + unstructured 30–50 % prune
 < 5 %     →  INT8 + 50 % structured prune + distillation
 ≥ 5 %     →  Redesign the architecture from scratch (MobileNet, etc.)

 Target chip?
 • NPU / INT-only  →  full-integer INT8 (no FP fallback)
 • Jetson / Orin   →  FP16 first, INT8 only if needed
 • Phone (ANE/Hex) →  INT8 full-integer (the native mode)
 • MCU (Ethos-U55) →  INT8 + aggressive prune; < 100 KB target
 • GPU with Ampere+→  INT8 + 2:4 structured sparsity
```

[↑ Back to Top](#-edgeai--model-compression--theory)

---

## 10. Cheat sheet & red flags

### Cheat sheet
```
 LEVERS          Q · P · D · L  (quant, prune, distill, low-rank)
 QUANT FORMATS   FP16 / BF16 / FP8 / INT8 / INT4 / NF4 / binary
 DEFAULT COMBO   INT8 per-channel weights + asymmetric activations
 PTQ vs QAT      PTQ first; QAT only if PTQ drops > 1 %
 PRUNING         magnitude + iterative + fine-tune; try 2:4 on Ampere+
 DISTILL         T=4, α=0.1, T² factor; "dark knowledge" is free data
 LOW-RANK        SVD once, or train U·V from scratch
 STACK           design-small → distill → prune → fine-tune → quant
 METRIC          always plot 3-axis: accuracy vs size vs latency
```

### Red flags 🚩
- 🚩 Quantizing a model whose BN layers weren't folded → activations
  have huge ranges, INT8 collapses.
- 🚩 Per-tensor quant on a depthwise conv → terrible accuracy; always
  use per-channel.
- 🚩 Prune 90 % in one shot → can't recover.
- 🚩 Distill with T = 1 → soft labels are basically hard labels again.
- 🚩 Compressing a model that doesn't even meet accuracy targets at
  FP32. Fix accuracy first; compress never helps *accuracy*.

### Green flags ✅
- ✅ Baseline FP32 accuracy is recorded and reproducible.
- ✅ Calibration set looks like production (same distribution).
- ✅ You have a deployment-target-specific op coverage list.
- ✅ You measure **on the target chip**, not on the desktop.
- ✅ Pareto chart committed alongside the model.

---

## 🔭 Next up

Now that you can **shrink** a model, the next folder
[`Deployment_Frameworks/`](../Deployment_Frameworks/README.md) is how
you actually **run** it — TensorFlow Lite, ONNX Runtime, OpenVINO.

---

> *GPU Programming · EdgeAI · Model Compression · THEORY · github.com/rpaut03l/TS-02*
