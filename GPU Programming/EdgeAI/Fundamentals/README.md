# 🌱 EdgeAI · Fundamentals

### *What Edge AI is, why it matters, where it fits*

> **Nav:** [← EdgeAI](../README.md) | **Fundamentals** | [GPU Types →](../GPU_Types/README.md) | [Hardware →](../Hardware/README.md) | [CUDA for Edge →](../CUDA_for_Edge/README.md)

---

## 👶 30-second story

You drop your glass of milk. **SPLASH.**

- **Cloud way:** take a photo, send it to the cloud, wait for the cloud
  to say "yes, that's a spill," then beep the robot vacuum. Total time:
  **2 seconds**. In 2 seconds the puppy has already licked the floor.

- **Edge way:** the camera's own chip recognises "spill!" in **20 ms**
  and the vacuum is moving before you've even stood up. No internet
  needed, no photo uploaded, no bill.

That's the whole point of Edge AI: **make the decision right there,
right now, on the device itself**. Every chapter after this is just
details of *how*.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [edge_ai_fundamentals_theory.md](edge_ai_fundamentals_theory.md) | The full concept tour — Edge vs Cloud, the 5 pillars, the edge continuum (device → gateway → edge server → cloud), real-world use cases, the Sense → Preprocess → Infer → Act pipeline, and the trade-offs |
| [edge_ai_fundamentals_code.md](edge_ai_fundamentals_code.md) | Runnable Python — check your environment, train a tiny MNIST CNN as a baseline, convert to TFLite, INT8 quantize, measure size + latency + accuracy drop |
| [edge_ai_fundamentals_practice.md](edge_ai_fundamentals_practice.md) | **Colab-ready notebook** — full cat-vs-dog mini-CNN to edge model with a "latency budget" exercise for a 30 FPS camera |

---

## 🎯 After reading this you should be able to…

- Explain **Edge AI vs Cloud AI** using one sentence that a 10-year-old
  gets on first read
- Name the **5 pillars** of Edge AI and give one real-world example
  of each
- Draw the **4-tier edge continuum** (device → gateway → edge server
  → cloud) from memory
- List **5 real-world Edge AI use cases** and the pillar each one
  relies on most
- Describe the **Sense → Preprocess → Infer → Act** pipeline with a
  real example (e.g., a smart doorbell)
- State why a 10 ms inference budget changes everything about how
  you design your model

---

> *GPU Programming · EdgeAI · Fundamentals · github.com/rpaut03l/TS-02*
