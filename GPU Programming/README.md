#  GPU Programming

### *How to make a computer do a million things at the same time — notes, code, and practice*

> 🔗 **Repo:** [github.com/rpaut03l/TS-02](https://github.com/rpaut03l/TS-02) · GPU Programming track
>
> **Style:** Every topic explained (easy story + picture), then deeper technical details, then real runnable code.

---

##  What even is a GPU?

### 👶 Easy Story
Imagine a **kitchen** that has to cook dinner for 1000 people.

- **Option A — one super-chef.** Very smart, very fast, can do anything — chop, sauté, plate, decorate. But she can only cook **one dish at a time**. That's a **CPU** (Central Processing Unit). Great at hard, branchy, "thinking" tasks. Bad at doing 1000 copies of the same easy task.

- **Option B — an army of 1000 line-cooks.** Each one isn't as smart as the super-chef. They can only follow simple instructions like "chop this onion." But there are **a thousand of them**, and they all chop at the same time. In the time the super-chef chops 1 onion, they chop **1000**. That's a **GPU** (Graphics Processing Unit).

### The big idea
> **GPUs are not "faster CPUs." They are "wider" CPUs.**

Each GPU core is much simpler than a CPU core — but a GPU has *thousands* of them and runs them in **parallel**. That's why GPUs crush tasks like graphics (every pixel can be computed independently) and deep learning (every matrix entry can be computed independently).

```
 CPU:   few VERY smart cores     → sequential, branchy work
 GPU:   thousands of simple cores → massively parallel, uniform work
```

---

## 📁 Contents of this folder

| # | Lecture | Folder |
|---|---|---|
| 1 | **Introduction to GPUs** — The 6 processors powering modern AI (CPU / GPU / TPU / NPU / LPU / DPU) + FPGA, memory hierarchy, Amdahl's Law, why GPUs for AI | [Lec_01_Introduction/](Lec_01_Introduction/) |
| 2 | **CPU Basics · CUDA Programming Model · GPU Architecture** — ISA, threads/blocks/grids, warps, SIMT vs SIMD, memory hierarchy, full workload flow | [Lec_02_CPU_CUDA_Basics/](Lec_02_CPU_CUDA_Basics/) |
| ⭐ | **EdgeAI sub-track** *(self-study deep-dive, not a lecture)* — Fundamentals, every Edge GPU family (Jetson / discrete / iGPU / mobile SoC), non-GPU hardware (NPUs / MCUs / FPGAs) + Edge-vs-Cloud comparison, and CUDA for the edge (JetPack, TensorRT, unified memory) | [EdgeAI/](EdgeAI/) |

Each lecture folder has the same **trio** of files:

| File | Purpose |
|---|---|
| `*_theory.md` | **5-year-old stories first**, then deeper technical depth. Mnemonic → TOC → numbered sections with boxed diagrams → cheat sheet → red-flag quick-reference |
| `*_code.md` | **Runnable code** — CUDA C (compiled with `nvcc`) and Python equivalents (`numba.cuda`, `cupy`). Every concept from theory gets a snippet |
| `*_practice.md` | **Google Colab ready** (free T4 GPU!) or Kaggle — paste cells, run, see the GPU actually work |

---

## 🧭 How to use this folder

1. **Read `theory.md` first.** No code required. If the 5-year-old story makes sense, the technical part will too.
2. **Go through `code.md`.** Short snippets, heavily commented. Type them out; don't just read.
3. **Open `practice.md` in Google Colab.** On Colab, **Runtime → Change runtime type → T4 GPU** to get a free GPU. Paste cells one at a time.

> 💡 **No physical GPU needed.** Google Colab's free tier gives you a real NVIDIA T4 GPU with CUDA pre-installed. Kaggle also offers free GPU notebooks. Everything in this folder is designed to run on those free tiers.

---

## 📚 Topic roadmap

Planned topics in this track (more folders will be added over time):

- ✅ **Lec 1** — Introduction to GPU Programming (hardware families, parallelism, Amdahl's Law)
- ✅ **Lec 2** — CPU basics, CUDA programming model, GPU architecture
- ✅ **EdgeAI sub-track** — full deep-dive on running GPUs at the edge (see [EdgeAI/](EdgeAI/))
- 🔭 Memory optimization (coalescing, shared memory tiling)
- 🔭 Reductions, prefix sums, histograms
- 🔭 Matrix multiplication deep dive
- 🔭 Streams, concurrency, multi-GPU
- 🔭 Profiling and optimization (nsys, ncu)

---

## 🔗 Useful links for self-study

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — authoritative reference
- [NVIDIA CUDA by Example (free PDF chapters)](https://developer.nvidia.com/cuda-example) — gentle intro
- [Numba CUDA documentation](https://numba.readthedocs.io/en/stable/cuda/index.html) — Python-first CUDA
- [CuPy](https://docs.cupy.dev/) — NumPy API on the GPU
- [Google Colab](https://colab.research.google.com) — free CUDA-capable GPU

---

> *github.com/rpaut03l/TS-02*
