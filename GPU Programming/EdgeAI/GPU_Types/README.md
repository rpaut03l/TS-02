# 🎮 EdgeAI · GPU Types

### *Every variety of GPU that shows up on the edge — Jetson, discrete, iGPU, mobile SoC*

> **Nav:** [← EdgeAI](../README.md) | [← Fundamentals](../Fundamentals/README.md) | **GPU Types** | [Hardware →](../Hardware/README.md) | [CUDA for Edge →](../CUDA_for_Edge/README.md)

---

## 👶 30-second story

A GPU is a GPU is a GPU… **until you look at how much power it eats and
how big the box is**.

- A **gaming RTX 4090** under your desk = 450 W, size of a brick.
- A **Tesla T4** in an edge server = 70 W, size of a chocolate bar.
- A **Jetson Orin Nano** in a robot = 7–15 W, size of a credit card.
- An **Intel iGPU** in a factory PC = 15 W, already inside the CPU.
- An **Adreno GPU** in your phone = 1–5 W, size of a rice grain.

Same *architecture ideas* (thousands of tiny cores doing the same thing
at once). Different **packaging**, **power**, **price**, and **use
case**. This folder is the map.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [edge_gpu_varieties_theory.md](edge_gpu_varieties_theory.md) | The full atlas — NVIDIA Jetson family deep dive (Nano, TX2, Xavier NX, Orin Nano, Orin NX, AGX Orin), discrete edge GPUs (RTX 4000 Ada SFF, T4, RTX A2000), integrated GPUs (Intel UHD/Iris Xe, AMD Radeon iGPU), mobile SoC GPUs (Qualcomm Adreno, ARM Mali, Apple GPU, Samsung Xclipse) — with full spec tables and decision trees |
| [edge_gpu_varieties_code.md](edge_gpu_varieties_code.md) | Runnable Python — detect which GPU you're on, read the spec sheet programmatically, benchmark the same model across simulated edge-GPU families, read `tegrastats` output, compare CPU vs GPU-delegate on Colab |
| [edge_gpu_varieties_practice.md](edge_gpu_varieties_practice.md) | **Colab-ready notebook** — build your own "which edge GPU should I buy?" decision notebook. Measure FLOPs, TOPS, latency, and power estimates for a sample workload |

---

## 🎯 After reading this you should be able to…

- Name the **four families** of Edge GPUs and give one example of each
- Read a Jetson spec table and translate **TOPS**, **CUDA cores**, and
  **Tensor cores** into "will my model fit?"
- Pick between **Jetson Orin Nano** and **Jetson AGX Orin** for a
  concrete use case, using a 3-criteria rule
- Explain what an **iGPU** is, why your laptop has one, and when it's
  the right edge choice
- Understand what makes a **mobile SoC GPU** (Adreno / Mali) different
  from a desktop or Jetson GPU
- Read a `torch.cuda` / `tf.config` report and map it to the chip you're
  actually running on

---

## ⚡ One-line summary of each family

| Family | Form | Power | Typical TOPS | Sweet spot |
|---|---|---|---|---|
| **Jetson** (embedded) | SoM / carrier board | 5 – 60 W | 20 – 275 | Robots, drones, smart cameras, kiosks |
| **Discrete edge GPU** | PCIe card | 50 – 150 W | 20 – 65 | Edge servers, rugged industrial PCs |
| **Integrated GPU (iGPU)** | Inside the CPU | 5 – 45 W | 1 – 11 | Consumer laptops, light industrial PCs |
| **Mobile SoC GPU** | Inside phone SoC | 1 – 5 W | 5 – 50 | Phones, tablets, wearables |

---

> *GPU Programming · EdgeAI · GPU Types · github.com/rpaut03l/TS-02*
