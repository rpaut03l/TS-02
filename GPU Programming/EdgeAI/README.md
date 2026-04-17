# 🌐 Edge AI — Running AI Where the Data Is Born

### *A self-study deep-dive — part of the GPU Programming track*

> 🔗 **Repo:** [github.com/rpaut03l/TS-02](https://github.com/rpaut03l/TS-02) · GPU Programming › **EdgeAI**
>
> **What this is:** This is **not** a lecture series. It is my personal
> topic-by-topic deep dive into Edge AI, built because Edge AI is the
> natural next step after learning GPU programming.
>
> **Style:** Easy format to unerstand things included, then the technical depth, then
> real runnable code. Same style as the rest of this repo.

---

## 👶 What even is "Edge AI"?

Think about your **homework**.

- **Way 1 — send it to the teacher's house.** You write it down, put it in
  an envelope, walk to the post-box, wait 3 days for the postman to take
  it to the teacher, and wait another 3 days for the answer to come back.
  The teacher is very smart — she can grade **anything**. But it takes
  **6 days**, and if the post office is closed, you get **nothing**.
  This is **Cloud AI**.

- **Way 2 — check it yourself at your desk.** You have a small answer-key
  at home. You check your own answers the moment you finish. It's not as
  smart as the teacher, but the answer comes in **half a second**, it
  works even if the internet is down, and nobody else ever sees your
  notebook. This is **Edge AI**.

> **Edge AI = do the AI thinking RIGHT WHERE the data is born**
> (the phone, the camera, the car, the watch, the factory robot) —
> **instead of sending it all to a far-away cloud**.

```
   ☁️  CLOUD AI                        📱 EDGE AI
   ─────────────                       ─────────────
   Big brain, far away.                Small brain, right here.
   Slow round-trip (50–500 ms).        Super fast  (1–20 ms).
   Needs internet.                     Works offline.
   Your data leaves your device.       Your data stays with you.
   Cheap per chip, huge at scale.      Cheap per inference, no cloud bill.
```

---

## 🤔 Why does this live inside "GPU Programming"?

Because **Edge AI is mostly the same GPU parallelism tricks you already
know, squeezed into a tiny low-power package**.

- A data-center GPU (H100, A100) = **700 W**, hundreds of TFLOPS, huge
  memory.
- A Jetson Orin Nano = **7–15 W**, ~40 TOPS, 8 GB memory — **same CUDA,
  same kernels, same `__global__` functions** — just smaller, cooler,
  and running on a battery.

If you already learned how to write CUDA for a desktop GPU, **90 % of
that skill transfers straight to the edge**. The new things you'll
learn here are: **how to shrink the model, how to pick the right chip,
how to stay inside a 10-watt power budget, and how to wire it all up
with NVIDIA JetPack / TensorRT**.

---

## 📁 Contents of this folder

| # | Topic | Folder | What you'll learn |
|---|---|---|---|
| 1 | **Fundamentals** | [Fundamentals/](Fundamentals/) | What Edge AI is, why it matters, Edge vs Cloud, the 5 pillars, real-world use cases, the end-to-end Edge AI pipeline |
| 2 | **GPU Types for Edge AI** | [GPU_Types/](GPU_Types/) | Every variety of GPU used at the edge — NVIDIA Jetson family (Nano, TX2, Xavier, Orin), discrete edge GPUs (RTX 4000 Ada SFF, T4), integrated GPUs (iGPUs), mobile SoC GPUs (Adreno, Mali, Apple GPU) — with specs, prices, use cases |
| 3 | **Edge Hardware (beyond GPUs)** | [Hardware/](Hardware/) | The *non-GPU* chips: NPUs (Apple Neural Engine, Qualcomm Hexagon, Google Coral, Hailo-8), MCUs for TinyML (Cortex-M, ESP32), FPGAs at the edge, the **Edge-vs-Cloud GPU comparison**, power & thermal budgets, benchmarks |
| 4 | **EdgeAI with CUDA** | [CUDA_for_Edge/](CUDA_for_Edge/) | How CUDA changes when you move from desktop to Jetson — JetPack SDK, unified memory on Tegra, TensorRT, DeepStream, INT8 calibration, `nvpmodel` / `jetson_clocks`, cross-compile workflow |

Each topic folder has the same **trio** of files:

| File | Purpose |
|---|---|
| `*_theory.md` | **5-year-old story first**, then deeper technical depth. Mnemonic → TOC → numbered sections with boxed ASCII diagrams → cheat sheet. |
| `*_code.md` | **Runnable code** — Python (TFLite, ONNX Runtime, TensorRT) and a few CUDA snippets. Every concept from theory gets a snippet. |
| `*_practice.md` | **Google Colab or Jetson ready** — paste cells, run, see the edge inference actually work. |

---

## 🧭 How to use this folder

1. **Start with `Fundamentals/`.** Zero code. If the easy stories make
   sense, the rest will too.
2. **Move to `GPU_Types/`.** Now you know *why* Edge AI matters — next,
   learn *which chips* are in the market and what each one is good at.
3. **Read `Hardware/`.** Fill in the non-GPU picture — NPUs, MCUs,
   FPGAs — and the Edge-vs-Cloud comparison.
4. **Finish with `CUDA_for_Edge/`.** Bring your CUDA skills to the
   Jetson. Now you can actually build an edge product.

At every step: **read theory → type out the code → run practice on a
free Colab GPU or on a real Jetson**.

> 💡 **No physical Jetson needed for most exercises.** Google Colab's free
> T4 GPU is enough to simulate most edge inference workloads. Where a
> real Jetson is required, the code is still shown so you understand
> the workflow.

---

## 📚 Topic roadmap

Planned topics in this sub-track (I'll add more over time):

- ✅ **Fundamentals** — What is Edge AI, Edge vs Cloud, the 5 pillars, use cases
- ✅ **GPU Types for Edge AI** — Jetson, discrete edge, iGPU, mobile SoC GPU varieties
- ✅ **Edge Hardware (beyond GPUs)** — NPUs, MCUs, FPGAs, Edge-vs-Cloud comparison
- ✅ **EdgeAI with CUDA** — JetPack, TensorRT, unified memory on Tegra
- 🔭 **Model compression** — quantization (INT8, INT4), pruning, knowledge distillation
- 🔭 **Deployment frameworks** — TensorFlow Lite, ONNX Runtime, OpenVINO deep-dives
- 🔭 **TinyML** — AI on microcontrollers (TF Lite Micro, CMSIS-NN)
- 🔭 **Federated Learning & on-device training**
- 🔭 **Edge MLOps** — OTA updates, monitoring, drift at the edge
- 🔭 **Security & privacy** on edge devices

---

## 🔗 Useful links for self-study

- [NVIDIA Jetson Developer Site](https://developer.nvidia.com/embedded-computing) — official docs, JetPack SDK, tutorials
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — the "turn your PyTorch model into a lightning-fast edge engine" tool
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) — multi-camera video AI pipelines
- [TensorFlow Lite](https://www.tensorflow.org/lite) — Google's edge runtime
- [ONNX Runtime](https://onnxruntime.ai/) — vendor-neutral edge runtime
- [Intel OpenVINO](https://docs.openvino.ai/) — Intel's edge runtime (CPU + iGPU + VPU)
- [Google Coral (Edge TPU)](https://coral.ai/) — Google's tiny edge accelerator
- [Hailo-8 / Hailo-15](https://hailo.ai/) — dedicated edge NPU, up to 26 TOPS at 2.5 W
- [MLPerf Tiny](https://mlcommons.org/benchmarks/inference-tiny/) — benchmarks for microcontroller-class Edge AI
- [MLPerf Edge / Inference](https://mlcommons.org/benchmarks/inference-edge/) — benchmarks for Jetson-class edge devices
- [Edge Impulse](https://www.edgeimpulse.com/) — end-to-end Edge AI platform, especially for TinyML
- [Google Colab](https://colab.research.google.com) — free CUDA-capable GPU, good enough to simulate most edge workloads

---

## 🎓 Where this fits in my MTech study plan

In the GPU Programming course, I've learned how data-center GPUs do massive parallel work. 
Edge AI is the "other half" of the same story — the *same* GPU tricks, pushed into
tiny, power-starved, always-on devices that you can actually ship in a
product. 
Together, they cover the full **cloud → edge** spectrum that modern AI systems run on.

---

> *GPU Programming · EdgeAI · github.com/rpaut03l/TS-02*
