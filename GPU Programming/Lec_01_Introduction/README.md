# 🎓 Lecture 1 — Introduction to GPU Programming

### *Different brains for different jobs · Why GPUs exist · Amdahl's Law*

> **Nav:** [← GPU Programming](../README.md) | **Lec 1** | [Lec 2 →](../Lec_02_CPU_CUDA_Basics/README.md)

---

## 👶 30-second story

Your brain has **two modes**:

1. **Boss mode** — you're planning your week. You juggle 5 different things, make decisions, change your mind, and handle surprises. One person doing one complex thing. **That's a CPU.**

2. **Worker-bee mode** — you're colouring 100 sheep in a colouring book. Every sheep is exactly the same task: pick up crayon, fill the shape, put it down. You don't think. If 100 of your friends showed up and each took one sheep, you'd finish **100× faster** because the job is boring and identical. **That's a GPU.**

AI work — training neural networks, running image filters, rendering 3D games — is 95% "colour the sheep." That's why GPUs beat CPUs for AI by 10×, 100×, sometimes 1000×.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [gpu_lec01_intro_theory.md](gpu_lec01_intro_theory.md) | The full tour — hardware families (CPU/GPU/TPU/NPU/FPGA), GPU vs CPU architecture, compute + memory hierarchy, Amdahl's Law with worked math, why GPUs dominate AI |
| [gpu_lec01_intro_code.md](gpu_lec01_intro_code.md) | Runnable Python code: inspect your GPU with `nvidia-smi`, NumPy vs CuPy vector add, measure actual speedup, first taste of `numba.cuda` |
| [gpu_lec01_intro_practice.md](gpu_lec01_intro_practice.md) | **Colab-ready notebook** — end-to-end benchmarks of vector add and matrix multiply across CPU, NumPy, CuPy, and pure CUDA. You'll literally watch the speedup climb |

---

## 🎯 After this lecture you should be able to…

- Explain in **one sentence** why a GPU beats a CPU at AI workloads
- Compare **CPU, GPU, TPU, NPU, FPGA** along at least 3 axes
- Draw the **CPU and GPU compute + memory hierarchies** from memory
- State **Amdahl's Law** and use it to compute the max possible speedup for a program that's 90% parallel
- Run a **CuPy vector add on Google Colab** and see the actual speedup

---

> *GPU Programming · Lec 1 · github.com/rpaut03l/TS-02*
