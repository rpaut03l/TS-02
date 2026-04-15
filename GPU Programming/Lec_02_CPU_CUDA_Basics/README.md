# 🎓 Lecture 2 — CPU Basics · CUDA Programming Model · GPU Architecture

### *How the CPU actually runs a program · CUDA threads, blocks, grids, warps · GPU memory hierarchy*

> **Nav:** [← Lec 1](../Lec_01_Introduction/README.md) | [← GPU Programming](../README.md) | **Lec 2**

---

## 👶 30-second story

In Lec 1 we said "GPUs have thousands of cores that all do the same thing at once." Now we're going to zoom in and ask **how exactly**.

It turns out:
- **Threads** — each is one tiny worker that does a little piece of the job.
- **Thread blocks** — groups of ~256-1024 threads that are assigned to the same Streaming Multiprocessor (SM) and can *talk to each other* via fast shared memory.
- **Warps** — inside each block, threads are bundled into groups of **32** that all execute the *same instruction* at the same time (SIMT).
- **Grid** — all the blocks, together, form the whole job.

That's the CUDA programming model. Once you understand it, everything else in GPU programming clicks into place.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [gpu_lec02_cuda_theory.md](gpu_lec02_cuda_theory.md) | Full theory — CPU + ISA, measuring CPU performance, CUDA programming model (threads/blocks/grids), warps, SIMT vs SIMD, Fermi SM architecture, full GPU memory hierarchy, workload flow, programming models |
| [gpu_lec02_cuda_code.md](gpu_lec02_cuda_code.md) | Runnable CUDA C (compiled with `nvcc` inside Colab via `%%writefile`) + numba equivalents — hello world kernel, vector add, thread indexing, grid/block config, shared memory tiling |
| [gpu_lec02_cuda_practice.md](gpu_lec02_cuda_practice.md) | Colab-ready notebook — compile + run real CUDA on a free T4 GPU, measure warp efficiency, play with block sizes, write a tiled matrix multiply |

---

## 🎯 After this lecture you should be able to…

- Explain **ISA (Instruction Set Architecture)** in one sentence
- Name the **three types of instructions** a CPU supports
- Write the **CPI** (cycles per instruction) formula for CPU performance
- Define **thread**, **thread block**, **grid**, and **warp** and explain how they relate
- Explain **SIMT** and how it differs from **SIMD**
- Sketch the **GPU memory hierarchy** (registers → shared → L1 → L2 → global → texture)
- Trace the **7-step lifecycle** of a CUDA workload from CPU launch to CPU return
- Compile a **tiny CUDA C program** with `nvcc` on Colab

---

> *GPU Programming · Lec 2 · github.com/rpaut03l/TS-02*
