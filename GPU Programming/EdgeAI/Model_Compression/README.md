# 🗜️ EdgeAI · Model Compression

### *Shrink the model · Speed it up · Keep the accuracy*

> **Nav:** [← EdgeAI](../README.md) | **Model Compression** | [Deployment Frameworks →](../Deployment_Frameworks/README.md) | [TinyML →](../TinyML/README.md) | [Federated Learning →](../Federated_Learning/README.md) | [Edge MLOps →](../Edge_MLOps/README.md) | [Security & Privacy →](../Security_Privacy/README.md)

---

## 👶 30-second story

Imagine you've written a huge **500-page novel** (the cloud model). Now
you need to fit the same story into a **50-page comic book** (the edge
model) — same storyline, 10× fewer pages, still enjoyable.

The four "editor tricks" you use:

1. **Quantization** → use shorter words (32-bit → 8-bit → 4-bit).
2. **Pruning** → cut boring paragraphs the reader skips anyway.
3. **Distillation** → a smart teacher writes a shorter book that reads
   *like* the original.
4. **Low-rank / factorization** → rewrite fat chapters as two thin ones.

Combine two or three of these and you routinely get **4–20× smaller**
models with **< 1 %** accuracy drop. That's this whole folder.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [model_compression_theory.md](model_compression_theory.md) | The full theory — quantization (INT8, INT4, FP16, BF16, FP8, NF4), PTQ vs QAT, symmetric vs asymmetric, per-tensor vs per-channel, **pruning** (unstructured, structured, magnitude, lottery ticket, N:M sparsity), **knowledge distillation** (logit, feature, relational), low-rank factorization & tensor decomposition, the **compression decision tree** |
| [model_compression_code.md](model_compression_code.md) | Runnable Python — dynamic-range INT8 quant, full-integer INT8 with calibration, FP16 weight-only, `torch.ao.quantization` API tour, **pruning with `torch.nn.utils.prune`**, a 10-line **logit distillation** loop, SVD low-rank factorization of a `Linear` layer, BitsAndBytes 4-bit for LLMs |
| [model_compression_practice.md](model_compression_practice.md) | **Colab notebook** — take MobileNetV2 → apply all 4 techniques → plot a **Pareto chart** of (accuracy, size, latency). End with a "stacked" version that uses quant + prune + distill together |

---

## 🎯 After reading this you should be able to…

- Name the **4 main compression families** and the 1-line intuition
  of each
- Tell **INT8 symmetric** from **asymmetric** and pick the right one
- Explain when to use **PTQ** vs **QAT** (and what "fake quant" means)
- Run a **pruning schedule** (sparsity target over epochs) with
  PyTorch built-ins
- Train a student model with **logit distillation** in ≤ 15 lines
- Draw the **size-vs-accuracy Pareto curve** from your own experiment
- Know when **4-bit (NF4)** is the right choice vs **INT8**

---

## ⚡ One-line cheat per technique

| Technique | Typical shrink | Typical speed-up | Typical acc drop |
|---|---|---|---|
| FP16 weight-only | 2× | 1.5–2× | ~0 % |
| INT8 PTQ (per-channel) | 4× | 2–4× | 0.2–1 % |
| INT4 / NF4 (LLMs) | 8× | 1.5–3× | 1–3 % |
| Unstructured pruning 50 % | ~1× size, 1× speed* | 1–1.2× on NPU/GPU | < 0.5 % |
| Structured pruning 30 % | 1.4× | 1.3–2× | 0.5–2 % |
| N:M (2:4) sparsity | 1.5× | 1.5–2× (Ampere+) | < 0.5 % |
| Knowledge distillation | 2–10× | 2–10× | 0.5–2 % |
| Low-rank SVD (r/min_dim) | depends on rank | depends on rank | depends on rank |

*Unstructured pruning shrinks *sparse-storage* size but rarely speeds up
dense hardware — structured or 2:4 sparsity does both.

---

> *GPU Programming · EdgeAI · Model Compression · github.com/rpaut03l/TS-02*
