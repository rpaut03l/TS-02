# ⚡ EdgeAI · CUDA for the Edge

### *How CUDA changes when you move from desktop to Jetson — JetPack, TensorRT, unified memory, DeepStream*

> **Nav:** [← EdgeAI](../README.md) | [← Fundamentals](../Fundamentals/README.md) | [← GPU Types](../GPU_Types/README.md) | [← Hardware](../Hardware/README.md) | **CUDA for Edge**

---

## 👶 30-second story

You learned CUDA on a desktop RTX card. You wrote `__global__` kernels,
you moved data with `cudaMemcpy`, you built beautiful tiled matrix
multiplies. Now your boss says: **"Ship this on a Jetson Orin."**

Good news: **95 % of what you learned still works.**

Small but important news: **5 % is different and will ruin your day if
you don't know about it.**

The five differences:

1. The **CPU and GPU share physical memory** — `cudaMemcpy` is often
   a copy from RAM to… the same RAM. Use **unified memory** instead.
2. You don't just install CUDA — you install **JetPack**, which
   bundles CUDA + cuDNN + TensorRT + DeepStream + BSP.
3. The fastest way to ship a model is **TensorRT**, not raw CUDA.
4. The GPU throttles based on **`nvpmodel`** power modes. Always check
   what mode you're in before benchmarking.
5. The ARM CPU is weaker than your desktop CPU — **move more work onto
   the GPU** than you would on desktop.

This folder is the tour.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [edge_cuda_theory.md](edge_cuda_theory.md) | The full theory — JetPack SDK stack, **unified/zero-copy memory on Tegra**, TensorRT workflow (ONNX → engine → runtime), DeepStream for video pipelines, **INT8 calibration**, `nvpmodel` / `jetson_clocks`, cross-compilation workflow, Jetson ↔ desktop CUDA differences, when to drop to raw CUDA vs use TensorRT |
| [edge_cuda_code.md](edge_cuda_code.md) | Runnable code — Python with `pycuda` and `tensorrt`, a **hello-world kernel on T4** (Colab proxy for Jetson), **ONNX → TensorRT engine** build, **INT8 calibration** routine, **CUDA Graphs** for low-latency inference, energy-mode snippets, `tegrastats` automation |
| [edge_cuda_practice.md](edge_cuda_practice.md) | **Colab notebook** — full end-to-end: take your cat-vs-dog model from Fundamentals → convert to ONNX → build FP16 and INT8 TensorRT engines → benchmark three precisions (FP32 Torch, FP16 TRT, INT8 TRT) → see the ~5× speedup that TensorRT delivers |

---

## 🎯 After reading this you should be able to…

- Name the **5 big components** of JetPack SDK
- Explain why `cudaMemcpy` is **usually wrong** on a Jetson and what
  to use instead (unified / zero-copy memory)
- Describe the **TensorRT workflow** in 4 steps: ONNX → parser →
  builder → runtime
- Understand the difference between **FP32, FP16, and INT8** TensorRT
  precisions and when to use each
- Write an **INT8 calibrator** in Python
- Use **`nvpmodel`** and **`jetson_clocks`** correctly before
  benchmarking
- Know when to write a **custom CUDA kernel** versus let TensorRT
  handle everything

---

## 🧠 Two cheat lines for the rest of your career

> **On Jetson, "memory transfer" is usually a lie.** Learn unified memory.
>
> **On Jetson, "PyTorch inference" is leaving 3–5× speed on the table.**
> Learn TensorRT.

---

> *GPU Programming · EdgeAI · CUDA for Edge · github.com/rpaut03l/TS-02*
