# 📖 GPU Programming Lec 1 — Introduction: THEORY

### *Hardware families · CPU vs GPU · Memory hierarchy · Amdahl's Law*

> **Nav:** [← Lec 1 README](README.md) | **THEORY** | [💻 CODE](gpu_lec01_intro_code.md) | [🎯 PRACTICE](gpu_lec01_intro_practice.md)

---

## 🧠 MNEMONIC: **"CGTNF-HAP"**

> **C**PU · **G**PU · **T**PU · **N**PU · **F**PGA · **H**ierarchy · **A**mdahl · **P**arallel

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Why Hardware Matters for AI | [§1](#1-why-hardware-matters-for-ai) |
| 2 | The Processor Families You'll Meet | [§2](#2-the-processor-families-youll-meet) |
| 3 | Side-by-Side Comparison + When to Use Which | [§3](#3-side-by-side-comparison--when-to-use-which) |
| 4 | What is a GPU? | [§4](#4-what-is-a-gpu) |
| 5 | GPU vs CPU — The Core Architecture Difference | [§5](#5-gpu-vs-cpu--the-core-architecture-difference) |
| 6 | Compute & Memory Hierarchy | [§6](#6-compute--memory-hierarchy) |
| 7 | Amdahl's Law | [§7](#7-amdahls-law) |
| 8 | Impact of Hardware on AI Execution | [§8](#8-impact-of-hardware-on-ai-execution) |
| 9 | Program Execution on a GPU (preview) | [§9](#9-program-execution-on-a-gpu-preview) |
| 10 | Cheat Sheet | [§10](#10-cheat-sheet--red-flags) |

---

## 1. Why Hardware Matters for AI

### 👶 Easy Story
Say you want to eat an ice-cream sundae. You can:
- Eat it with a **fork** (terrible — it falls through).
- Eat it with a **spoon** (great).
- Eat it with a **giant ladle** (overkill, messy).
- Eat it with a **1000-spoon robot** (absurd for a single sundae).

**Different tools are good at different jobs.** Picking the wrong tool wastes time, power, and money. AI workloads are **huge arithmetic tasks on giant arrays of numbers**, and the right tool for the job turns out to be a chip with **thousands of small cores** working in parallel — not a chip with a few very smart cores.

### The real-world question from Lecture 1
> **Different hardware platforms give very different performance on the same AI model.**

You can run the *same* neural network on a CPU, a GPU, a TPU, or an FPGA and see **10× to 1000× differences** in speed, energy use, and cost. Choosing the right platform is half the battle of making AI real.

### Running example — Image recognition
Image recognition is the "Hello world" of modern AI:
```
  input:   a 224×224 colour image (150,528 numbers)
  output:  "cat", "dog", "car"…
  inside:  convolutions + matrix multiplications
           → billions of multiply-add operations per image
```

Billions of operations. Per image. Sequential hardware would take minutes per image. Parallel hardware (GPU, TPU) can do hundreds of images per second. This is why hardware matters.

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 2. The Processor Families You'll Meet

Modern AI runs on a **zoo** of specialized chips, each one good at a different slice of the workload. The popular short list — the "**6 processors powering modern AI**" — is **CPU, GPU, TPU, NPU, LPU, DPU**. On top of those, **FPGA** is worth knowing as the "reconfigurable" option. Let's walk through all seven with a 5-year-old story + the formal definition.

### 1️⃣ CPU — Central Processing Unit

```
 "The boss / the backbone."
```

- **👶 Story:** The super-smart chef who can do *anything*. Reads any recipe. Handles surprises. But only one dish at a time.
- **Formal:** A general-purpose processor with **few** (4-128) **very sophisticated** cores. Each core has deep pipelines, branch predictors, out-of-order execution, and lots of cache. Great at branchy, irregular, sequential code.
- **Role in an AI system:** the **orchestrator**. It receives the user request, schedules the job, routes it to the right accelerator (GPU/TPU/NPU/LPU/DPU), manages I/O and the OS, and assembles the final response.
- **Pros:** most flexible, best single-thread performance, runs the OS and the whole software stack, handles anything thrown at it.
- **Cons:** few cores, cannot train large models, low throughput for AI workloads.
- **Examples:** Intel Core i9, AMD Ryzen, Apple M3/M4, ARM Cortex-A78.

### 2️⃣ GPU — Graphics Processing Unit

```
 "The army of line cooks."
```

- **👶 Story:** A thousand simple cooks who all chop onions at once. Can't improvise, can't make decisions. But if the job is *the same thing copied a thousand times*, nothing beats them.
- **Formal:** A processor with **thousands** of **simple** cores grouped into Streaming Multiprocessors (SMs). Built for **SIMD/SIMT** — single instruction running on many data items at the same time. A modern consumer flagship like the **NVIDIA RTX 4090** has ~**16,896 CUDA cores**; datacenter parts like the H100 have ~**14,592** plus specialized tensor cores.
- **Role in AI:** the **workhorse**. Matrix multiply → backprop → weight update, all parallelized across thousands of cores.
- **Pros:** massive parallelism, excellent for both training *and* inference, huge ecosystem (CUDA, cuDNN, PyTorch, TensorFlow).
- **Cons:** ~**300–700 W** per unit, very expensive (**$30k+** for an H100), overkill for small tasks.
- **Examples:** NVIDIA RTX 4090, A100, H100, B200; AMD Instinct MI300; Apple M3 GPU.

### 3️⃣ TPU — Tensor Processing Unit

```
 "The specialized matrix-multiply factory."
```

- **👶 Story:** A kitchen that ONLY makes one thing — giant pancakes, as fast as physically possible. Can't cook anything else, but the best pancake machine on Earth.
- **Formal:** An Application-Specific Integrated Circuit (ASIC) designed by Google for **neural network tensor operations** — primarily matrix multiply and convolutions. The key structural idea is a **systolic array** of multiply-accumulate (MAC) units where data flows *through* the array while partial sums accumulate, without being written back to memory between steps. A single TPU pod can chain **thousands of TPU chips** (8,192 on some generations) into one gigantic matrix machine.
- **Role in AI:** Google-scale **training and inference** of neural networks, especially very large models.
- **Pros:** systolic-array dataflow gives the highest FLOPs-per-watt on tensor ops, tight integration with TensorFlow / JAX and Google Cloud.
- **Cons:** Google Cloud only, less flexible than a GPU, limited framework support outside the Google stack.
- **Examples:** Google TPU v4, TPU v5e, TPU v5p, TPU Ironwood.

### 4️⃣ NPU — Neural Processing Unit

```
 "The tiny AI chip inside your phone."
```

- **👶 Story:** A small, energy-efficient chef built into your phone that specializes in recognizing your face, translating languages, and running ML features *without* the phone getting hot or the battery dying.
- **Formal:** Low-power accelerators optimized for **on-device AI inference**. They use **quantized** math (INT8 / INT4), small memories, and tight power budgets (typically **< 5 W**). Every modern smartphone SoC includes one.
- **Role in AI:** runs models **entirely on-device** — no cloud round-trip, millisecond-level latency, privacy-preserving.
- **Pros:** inference in milliseconds, ultra-low power, no network needed, privacy by default.
- **Cons:** **inference only** (no training), limited model size (fits in on-chip memory), less accurate than full-precision GPU inference.
- **Examples:** Apple Neural Engine (every iPhone since X), Qualcomm Hexagon NPU, Samsung Exynos NPU, MediaTek APU, Google Edge TPU.

### 5️⃣ LPU — Language Processing Unit

```
 "Groq's deterministic inference beast."
```

- **👶 Story:** Imagine the fastest talker in the world — a chef who has *every* ingredient already arranged on a single giant counter in front of her, so she never has to walk to the fridge. That's an LPU: one enormous on-chip cache that fits the model weights, so it *never* waits for memory.
- **Formal:** A new class of inference accelerator (pioneered by **Groq**) designed specifically for **low-latency sequence/LLM inference**. Instead of slow HBM with caches, an LPU puts **hundreds of MB of on-chip SRAM** (e.g., ~230 MB per chip) directly next to the compute, giving **zero cache misses** and **fully deterministic execution**. Latency per token is predictable to the cycle.
- **Role in AI:** **real-time LLM serving** — token-by-token generation where every millisecond matters (chat, streaming, agents).
- **Pros:** **fastest token-generation** latency available (public demos show **500+ tokens/sec** on 70B-class models), zero cache misses, deterministic (no tail latency spikes).
- **Cons:** **inference only**, limited memory per chip (so very large models need to be sharded across many chips), very new technology, small ecosystem.
- **Examples:** Groq LPU. (Other vendors like Cerebras Wafer-Scale Engine, SambaNova, Tenstorrent occupy nearby niches in the "not-a-GPU inference chip" space.)

### 6️⃣ DPU — Data Processing Unit

```
 "The data-center's traffic cop."
```

- **👶 Story:** The clerk at the busy airport who checks passports, directs luggage, manages security lines, and handles the paperwork — so the *pilots* (GPUs) can focus on flying, not passport-stamping.
- **Formal:** A **SmartNIC** on steroids — a programmable accelerator for **network, storage, and security** work that used to be done by the CPU. Offloads things like packet processing, TLS termination, firewall rules, RDMA, and storage protocols to free up CPU cycles for the actual workload.
- **Role in AI:** **not** for ML compute itself — it sits at the edge of each server and keeps data flowing to and from the GPUs at 200–800 Gb/s, handles security, and offloads virtualization. Critical for running GPU clusters at scale.
- **Pros:** frees CPU for core workloads, hardware-level security, 400+ Gb/s networking, RDMA & NVMe offload.
- **Cons:** **not an AI/ML compute chip**, complex to program (usually requires vendor SDK and low-level networking expertise), niche outside large datacenters.
- **Examples:** NVIDIA BlueField-3, AMD Pensando, Intel IPU, Marvell OCTEON.

### 7️⃣ FPGA — Field-Programmable Gate Array *(bonus — the reconfigurable option)*

```
 "The rewire-it-yourself chip."
```

- **👶 Story:** A box of LEGO that you build into a cookie-cutter machine this week, then tear apart and build into a toaster next week. You decide what the chip *is*.
- **Formal:** A chip containing a grid of **configurable logic blocks** that you can re-wire (in software, at boot) into almost any digital circuit. Used when you need custom hardware but can't afford to manufacture a real ASIC.
- **Role in AI:** custom inference engines with very specific latency / power constraints; prototyping next-generation accelerators; smart-NICs; medical and scientific devices.
- **Pros:** ultra-low latency, reconfigurable, great for niche workloads.
- **Cons:** you are basically designing a circuit — the hardest of all these to program.
- **Examples:** AMD–Xilinx (UltraScale+, Versal), Intel (Altera) Stratix, Lattice.

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 3. Side-by-Side Comparison + When to Use Which

### Dots-out-of-5 along the five axes that matter most
(Higher dots = better on that axis.)

```
┌───────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Axis          │  CPU    │  GPU    │  TPU    │  NPU    │  LPU    │  DPU    │ FPGA    │
├───────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ Flexibility   │ ●●●●●   │  ●●●    │   ●●    │   ●●    │   ●     │   ●●    │ ●●●●    │
│ Throughput    │   ●     │ ●●●●●   │ ●●●●●   │  ●●     │ ●●●●    │  ●●     │ ●●●     │
│ Latency       │  ●●●    │  ●●●    │   ●●    │ ●●●●    │ ●●●●●   │ ●●●●    │●●●●●    │
│ Power eff.    │   ●●    │   ●     │  ●●●    │●●●●●    │  ●●●    │ ●●●●    │ ●●●     │
│ Ease of use   │ ●●●●●   │ ●●●●    │  ●●     │  ●●     │   ●●    │   ●     │  ●      │
└───────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

### When to use which — at-a-glance table

```
┌──────────┬────────────────────────────────────────────────────┐
│ Processor│ Primary job                                        │
├──────────┼────────────────────────────────────────────────────┤
│ CPU      │ Orchestration · preprocessing · OS · "glue"         │
│ GPU      │ Training & deep learning · general AI workhorse    │
│ TPU      │ Google-scale tensor workloads (training + infer.)  │
│ NPU      │ Edge / mobile / on-device inference                │
│ LPU      │ Real-time LLM serving (lowest token latency)       │
│ DPU      │ Data-center network / storage / security offload   │
│ FPGA     │ Custom accelerators · ultra-low-latency niches     │
└──────────┴────────────────────────────────────────────────────┘
```

### Choose based on these five factors
When picking hardware for an AI workload, weigh:

1. **Latency** — how quickly must a single request return?
2. **Parallelism** — can the work be split into many identical pieces?
3. **Power** — is this in a phone, a laptop, or a datacenter?
4. **Cost** — chip price, operating electricity, engineering time.
5. **Scale** — one user, one million users, or one billion?

### The "specialization vs flexibility" trade-off
Moving left (CPU) to right (DPU/FPGA) in the comparison above, you trade **generality** for **speed/efficiency** on a narrower workload. A CPU can run anything but none of it is world-class fast. A TPU or LPU is incredibly fast at its specialty but can't even load an OS.

### How the processors work together in one AI system
Here's how all 6 of the "modern AI" processors line up in a real deployment:

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   User Request                                                     │
│        │                                                           │
│        ▼                                                           │
│     ┌─────┐      schedule + preprocess                             │
│     │ CPU │─────────────────────────────────────────────┐          │
│     └─────┘                                              │          │
│        │                                                 │          │
│        │                                                 │          │
│        ▼                                                 │          │
│    ┌────────────────────────────────────────────────┐   │          │
│    │  Choose the right accelerator for the job…      │   │          │
│    └────────────────────────────────────────────────┘   │          │
│        │                                                 │          │
│        ├── heavy training / dense matmul  →  GPU / TPU   │          │
│        ├── real-time LLM tokens           →  LPU         │          │
│        ├── on-device inference            →  NPU         │          │
│        └── network + security offload     →  DPU         │          │
│                                                          │          │
│    After compute, result goes back → CPU → user         │          │
│                                                          │          │
└────────────────────────────────────────────────────────────────────┘
```

**The key point:** these chips are **collaborators**, not competitors. A real AI platform uses several at once — CPUs for orchestration, GPUs for training, LPUs or NPUs for serving, DPUs for networking. "Which is best" is always answered with "best at *what*, under *which* constraints."

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 4. What is a GPU?

### Formal definition
> A **Graphics Processing Unit (GPU)** is a specialized processor originally designed to accelerate 2D/3D graphics rendering, now widely used for general-purpose parallel computing — especially neural network training and inference.

### The history in one paragraph
GPUs were invented in the late 1990s to draw polygons fast for video games. Each pixel on the screen could be computed *independently* from every other pixel, so hardware designers built chips with dozens of simple shader cores that all processed pixels in parallel. By 2006, NVIDIA shipped **CUDA** — a way to program those shader cores for general-purpose math, not just graphics. AI researchers quickly discovered that training a neural network looks **exactly** like a graphics workload (the same operation applied to millions of numbers), and GPUs went from gaming accessory to AI backbone.

### What's inside (at a high level)
```
┌──────────────────────────────────────────────────┐
│                    GPU                           │
│                                                  │
│  [SM 0]  [SM 1]  [SM 2]  …  [SM 107]              │
│   │        │        │           │                │
│   └────────┴────────┴───────────┘                │
│                │                                 │
│         L2 cache (~40-50 MB)                     │
│                │                                 │
│         HBM / GDDR memory (16-80 GB)             │
│                                                  │
│  Each SM = many parallel cores + its own L1,    │
│            shared memory, and register file      │
└──────────────────────────────────────────────────┘
```

- **SM (Streaming Multiprocessor)** = the unit of parallelism. An A100 has 108 SMs; an H100 has 132. Each SM has many CUDA cores (up to ~128 each).
- **CUDA core** = one simple arithmetic unit. Total cores = SMs × cores-per-SM. An A100 has ~6,912 cores; an H100 has ~14,592.
- **Memory hierarchy** — fast tiny memories close to the cores, big slow memory further away (more on this in §6).

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 5. GPU vs CPU — The Core Architecture Difference

### 👶 Easy Story
Look at a **CPU** under a microscope and you'll see a few big, complicated cores with lots of cache, branch predictors, and control logic around them. Each core is **smart**.

Look at a **GPU** and you'll see **rows upon rows** of tiny identical cores. Almost no cache, minimal control logic — most of the silicon is just **arithmetic units**. Each core is **dumb but there are a thousand of them**.

### The technical reason
Chip designers have a **silicon budget**. They can spend it on:
1. **Making each core smarter** (branch prediction, out-of-order execution, deep caches) → CPU strategy.
2. **Packing more cores in the same space** (keep each core simple, give them simple memory systems) → GPU strategy.

Which strategy wins depends on the workload:
- **Branchy, sequential, different work per item** → CPU wins. The smart cores keep speculating ahead, avoiding stalls.
- **Uniform, parallel, same work per item** → GPU wins. The 1000 dumb cores do the same thing at once.

### Side-by-side
```
┌──────────────────┬──────────────────────┬──────────────────────┐
│                  │ CPU                  │ GPU                  │
├──────────────────┼──────────────────────┼──────────────────────┤
│ # of cores       │  8–128               │  2,000–20,000        │
│ Core complexity  │  very high           │  very low            │
│ Cache per core   │  large (~MB)         │  tiny (~KB)          │
│ Branch handling  │  great               │  poor                │
│ Memory bandwidth │  ~100 GB/s           │  ~1,000–3,000 GB/s   │
│ Memory latency   │  low (good caches)   │  high (hide via      │
│                  │                      │  parallelism)        │
│ Best at          │  sequential, branchy │  SIMD parallel loops │
│ Worst at         │  massive parallel    │  branchy, serial     │
└──────────────────┴──────────────────────┴──────────────────────┘
```

### Hiding memory latency
A GPU has high memory latency (hundreds of cycles!) but hides it **by swapping**. When one warp (group of 32 threads) stalls waiting for memory, the SM instantly runs *another* warp. With enough warps in flight, the compute units stay busy. CPUs hide latency with caches; GPUs hide it with parallelism.

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 6. Compute & Memory Hierarchy

Data doesn't magically appear at the cores. It travels through a **pyramid** — fast and tiny at the top, slow and huge at the bottom.

### CPU memory pyramid
```
   [registers]                ← single cycle,  ~KB total
   [L1 cache]                 ← ~4 cycles,     ~32 KB per core
   [L2 cache]                 ← ~12 cycles,    ~256 KB per core
   [L3 cache]                 ← ~40 cycles,    ~10-32 MB shared
   [DRAM / main memory]       ← ~200 cycles,   ~32 GB
   [SSD / disk]               ← ~millions of cycles, ~TB
```

### GPU memory pyramid
```
   [registers]                ← 1 cycle,     ~64 KB per SM
   [shared memory / L1]       ← ~30 cycles,  ~100-164 KB per SM
   [L2 cache]                 ← ~200 cycles, ~40-50 MB (shared)
   [global memory (HBM)]      ← ~400 cycles, ~16-80 GB
   [PCIe transfer to host]    ← ~milliseconds — the slowest thing
```

### Key observations
1. **Closer to the cores = smaller + faster.** This is physics — signals take time to travel.
2. **Getting data off the GPU (over PCIe) is glacial** compared to everything else on the GPU. If you keep bouncing data back and forth between CPU and GPU, the transfer cost eats all the speedup. **Rule of thumb: move data to the GPU once, do a lot of work, move result back once.**
3. **Shared memory** on a GPU is a programmer-managed cache. Unlike CPU L1 (which is invisible to the programmer), shared memory is explicitly allocated and shared by threads in the same block — used for stencil kernels, matrix tiling, and reductions.
4. **HBM (High Bandwidth Memory)** on top-tier GPUs has **10-30× more bandwidth** than CPU DRAM — that's how GPUs feed thousands of cores.

### The "memory wall"
> Fetching a number from DRAM costs roughly as much as **200 arithmetic operations**. If you fetch a number and do only 1 operation on it, you're **200× slower** than if you fetch once and do 200 operations on it.

This is called the **memory wall**. Writing fast GPU code is 80% about data movement: load once, compute a lot, store once. Not about the math itself.

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 7. Amdahl's Law

The single most important equation in parallel computing.

### 👶 Easy Story
Suppose you have a job that takes 60 minutes, split into:
- **30 minutes of stuff that can be done in parallel** (many workers help)
- **30 minutes of stuff that can't be parallelized** (only you can do it)

If you get a million helpers, what's the fastest the job can finish? **30 minutes** — because the 30 minutes of non-parallel work still has to happen. Even infinite workers can't make it faster.

**Amdahl's Law** captures this: *the non-parallel part limits your maximum speedup.*

### The formula
> **Speedup(N) = 1 / ((1 − P) + P/N)**

Where:
- **P** = the fraction of the program that is **parallelizable** (0 ≤ P ≤ 1)
- **N** = the number of parallel workers (CPUs, GPU cores, etc.)
- **Speedup** = how many times faster the parallel version is vs the serial version

### Intuition
As **N → ∞**:
> **max speedup = 1 / (1 − P)**

### Worked example
Your program is 90% parallel (P = 0.9), 10% serial. How much speedup can you get?

| # workers N | Speedup |
|---|---|
| 1 | 1.0× (baseline) |
| 2 | 1 / (0.1 + 0.45) = **1.82×** |
| 4 | 1 / (0.1 + 0.225) = **3.08×** |
| 10 | 1 / (0.1 + 0.09) = **5.26×** |
| 100 | 1 / (0.1 + 0.009) = **9.17×** |
| 1000 | 1 / (0.1 + 0.0009) = **9.91×** |
| ∞ | 1 / 0.1 = **10×** ceiling |

### The brutal punch
Even with **infinite** GPU cores, a 90% parallel program maxes out at **10× speedup**. The 10% serial part **wins**.

### 👶 What this means in practice
- **Profile first.** Find where the time *actually* goes. Don't optimize a 5% step and expect a 20× speedup.
- **Parallelism alone isn't enough.** You also have to attack the serial parts — algorithm changes, data reshaping, batch size tuning.
- **Small serial fraction matters hugely at scale.** Going from P = 0.95 to P = 0.99 changes the ceiling from 20× to 100×.

### Extensions
- **Gustafson's Law** — "make the problem bigger instead of the machine" — gives a less pessimistic view. If you use more workers to solve a *bigger* problem (rather than the same problem), speedup scales almost linearly. Relevant for AI: more GPUs → bigger models → still useful.
- **Communication overhead** is not in Amdahl's Law but matters a lot. If workers spend all their time talking to each other, adding more workers *hurts*. That's **strong scaling failure**.

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 8. Impact of Hardware on AI Execution

### The case study
Take an image classification model and profile its inner building blocks on different hardware:

```
 Layer     What it does
 ─────     ──────────────
 C         Conv2D — the heart of a CNN
 M         MaxPooling2D
 F         Flatten
 D         Dense (fully connected)
 DR        Dropout
 BN        BatchNormalization
 DC        DepthwiseConv2D
 AP        AveragePooling2D
 GA        GlobalAveragePooling2D
```

Each layer has wildly different compute and memory patterns. Convolutions are arithmetic-heavy; flattening is memory-move only; dense layers are matrix multiplies.

### Why the hardware platform changes the order of which layer is "slow"
On a CPU, convolutions dominate — they do the most math.
On a GPU, **dense layers and large convolutions run blazingly fast** (GPU's sweet spot) but tiny layers (like 1×1 depthwise convs, activations) pay per-kernel launch overhead. The **bottleneck shifts**.
On an FPGA or edge board, **memory transfers between layers can dominate** because the chip has much less on-chip memory.

### Key takeaway
> **There is no universally fastest layer.** The "slow" layer depends on the hardware. Optimizing a model means **co-designing** the layer sizes, batch sizes, and precision to match the hardware you'll actually run on.

### Measurement metric: FPS (Frames Per Second)
How many input images can the system classify per second. Reported on evaluation boards:
- A consumer GPU: thousands of FPS on small models.
- An edge FPGA: tens to hundreds of FPS at much lower power.

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 9. Program Execution on a GPU (preview)

> Full detail lands in **Lec 2** — this is just the skeleton.

A GPU program's lifecycle:

```
 1. CPU starts the program (the "host").
 2. CPU allocates memory on the GPU (the "device").
 3. CPU copies input data over PCIe → GPU memory.
 4. CPU launches a CUDA "kernel" — a function that runs on thousands
    of GPU threads simultaneously.
 5. GPU executes the kernel. Threads are grouped into blocks,
    blocks into a grid. Each thread processes a tiny slice of data.
 6. When the kernel finishes, CPU copies the result back over PCIe.
 7. CPU continues with whatever came next.
```

Key ideas to remember:
- **Host** = CPU side. **Device** = GPU side.
- A **kernel** is a function you write that runs on the GPU.
- **Threads** are grouped into **blocks**; blocks are grouped into a **grid**.
- **Data must travel over PCIe** in both directions — and that's usually the slowest step.

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 10. Cheat Sheet & Red Flags

```
╔══════════════════════════════════════════════════════════════╗
║  GPU LEC 1 ONE-LINERS                                        ║
╠══════════════════════════════════════════════════════════════╣
║  CPU = few smart cores, orchestration & glue                ║
║  GPU = thousands of simple cores, training + inference      ║
║  TPU = ASIC systolic array, Google-scale tensor workloads   ║
║  NPU = low-power on-device AI (phones, edge)                ║
║  LPU = zero-cache-miss SRAM, real-time LLM serving (Groq)   ║
║  DPU = SmartNIC offload: network + storage + security       ║
║  FPGA = reconfigurable logic (custom accelerators)          ║
║                                                             ║
║  GPUs hide memory latency with PARALLELISM, not caches      ║
║  Memory wall: 1 DRAM fetch ≈ 200 arithmetic ops             ║
║  Shared memory = programmer-managed cache on a GPU SM       ║
║  HBM → 10-30× more bandwidth than CPU DRAM                  ║
║                                                             ║
║  Amdahl: Speedup = 1 / ((1−P) + P/N)                        ║
║  90% parallel → max 10× speedup no matter how many cores    ║
║                                                             ║
║  Host = CPU,  Device = GPU                                  ║
║  Kernel = a function that runs on the GPU                   ║
║  Data flows: host → PCIe → device → compute → back → host   ║
║                                                             ║
║  Choose processor on: latency · parallelism · power ·       ║
║                       cost · scale                          ║
╚══════════════════════════════════════════════════════════════╝
```

### ⚡ Red-flag questions
1. **"Why are GPUs good for AI and CPUs aren't?"** — Deep learning is matrix multiply + convolution, which are thousands of independent arithmetic operations per input. GPUs have thousands of parallel cores; CPUs have a few dozen. GPUs can feed the cores with high-bandwidth HBM. The workload matches the hardware exactly.

2. **"Name the 6 processor families powering modern AI."** — CPU, GPU, TPU, NPU, LPU, DPU. (Bonus: FPGA is the "reconfigurable" seventh; ASIC is the generic term — TPU, NPU, and LPU are all ASICs.)

3. **"State Amdahl's Law."** — `Speedup(N) = 1 / ((1 − P) + P/N)`, where P is the parallel fraction and N is the number of workers. Max speedup as N → ∞ is `1 / (1 − P)`.

4. **"Why does a 95% parallel program plateau?"** — The 5% serial part is unscaled. Even infinite workers can't speed it up, so the total time bottoms out at 5% × original + 0 × parallel = 5% of original, i.e. 20× max speedup.

5. **"What is the memory wall?"** — The fact that fetching a number from DRAM costs ~200× more than doing an arithmetic operation on it. Fast code re-uses each loaded value many times before storing it back.

6. **"GPU vs CPU cache strategy?"** — CPU has huge caches to hide latency on every access. GPU has tiny caches but hides latency by keeping many warps in flight — if one stalls, another runs.

7. **"What is the host-device split in GPU programming?"** — The host is the CPU that launches work; the device is the GPU that executes the kernel. Data has to be explicitly copied between them over PCIe.

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

> **Next:** [💻 CODE →](gpu_lec01_intro_code.md) · [🎯 PRACTICE →](gpu_lec01_intro_practice.md)
>
> *GPU Programming · Lec 1 · github.com/rpaut03l/TS-02*
