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
| 2 | The Five Hardware Families | [§2](#2-the-five-hardware-families) |
| 3 | Comparing the Five Families | [§3](#3-comparing-the-five-families) |
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

## 2. The Five Hardware Families

You'll hear these five abbreviations all the time. Here's the **5-year-old story** for each, then the formal definition.

### 1️⃣ CPU — Central Processing Unit

```
 "The boss."
```

- **👶 Story:** The super-smart chef who can do *anything*. Reads any recipe. Handles surprises. But only one dish at a time.
- **Formal:** A general-purpose processor with **few** (4-128) **very sophisticated** cores. Each core has deep pipelines, branch predictors, out-of-order execution, and lots of cache. Great at branchy, irregular, sequential code.
- **Best for:** operating systems, databases, browsers, most day-to-day code.
- **Bad at:** massively parallel arithmetic on arrays.
- **Examples:** Intel Core i9, AMD Ryzen, Apple M3, ARM Cortex-A78.

### 2️⃣ GPU — Graphics Processing Unit

```
 "The army of line cooks."
```

- **👶 Story:** A thousand simple cooks who all chop onions at once. Can't improvise, can't make decisions. But if the job is *the same thing copied a thousand times*, nothing beats them.
- **Formal:** A processor with **thousands** (2,000-20,000) of **simple** cores grouped into Streaming Multiprocessors (SMs). Built for **SIMD/SIMT** — single instruction running on many data items at the same time.
- **Best for:** graphics, AI training & inference, scientific simulations, crypto mining.
- **Bad at:** branchy code, irregular memory access, tasks that can't be parallelized.
- **Examples:** NVIDIA RTX 4090, A100, H100; AMD Radeon; Apple M3 GPU.

### 3️⃣ TPU — Tensor Processing Unit

```
 "The specialized matrix-multiply factory."
```

- **👶 Story:** A kitchen that ONLY makes one thing — giant pancakes, as fast as physically possible. Can't cook anything else, but the best pancake machine on Earth.
- **Formal:** An Application-Specific Integrated Circuit (ASIC) designed by Google for **neural network tensor operations** — primarily matrix multiply and convolutions. Uses a **systolic array** of multiply-accumulate units.
- **Best for:** large-scale training and inference of neural networks (Google's own models and Google Cloud customers).
- **Bad at:** anything that isn't a tensor op.
- **Examples:** Google TPU v4, TPU v5e, TPU Ironwood.

### 4️⃣ NPU — Neural Processing Unit

```
 "The tiny AI chip inside your phone."
```

- **👶 Story:** A small, energy-efficient chef built into your phone that specializes in recognizing your face, translating languages, and running ML features *without* the phone getting hot or the battery dying.
- **Formal:** Low-power accelerators optimized for on-device AI inference. Use **quantized (int8)** math, small memories, and tight power budgets (~1 W).
- **Best for:** phone cameras, speech, on-device AI without cloud round-trips.
- **Bad at:** training, high-precision (fp32) math, general compute.
- **Examples:** Apple Neural Engine (in every iPhone since X), Qualcomm Hexagon NPU, Samsung Exynos NPU.

### 5️⃣ FPGA — Field-Programmable Gate Array

```
 "The rewire-it-yourself chip."
```

- **👶 Story:** A box of LEGO that you build into a cookie-cutter machine this week, then tear apart and build into a toaster next week. You decide what the chip *is*.
- **Formal:** A chip containing a grid of **configurable logic blocks** that you can re-wire into almost any digital circuit. Used when you need custom hardware but can't afford to manufacture a real ASIC.
- **Best for:** networking gear, medical devices, prototyping custom accelerators, very specific number-crunching at ultra-low latency.
- **Bad at:** ease-of-use — you basically design the circuit yourself.
- **Examples:** AMD-Xilinx (UltraScale+, Versal), Intel (Altera) Stratix, Lattice.

[↑ Back to Top](#-gpu-programming-lec-1--introduction-theory)

---

## 3. Comparing the Five Families

Five dimensions that matter for AI. Higher dots = better on that axis.

```
┌──────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│          │    CPU    │    GPU    │    TPU    │    NPU    │   FPGA    │
├──────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│Flexibility ●●●●●      │   ●●      │    ●      │    ●      │   ●●●●    │
│Throughput│    ●      │  ●●●●●    │  ●●●●●    │   ●●      │   ●●●     │
│Latency   │  ●●●      │   ●●●     │   ●●      │  ●●●●     │  ●●●●●    │
│Power eff.│   ●●      │    ●      │   ●●●     │  ●●●●●    │   ●●●     │
│Ease of   │ ●●●●●     │  ●●●●     │   ●●      │   ●●      │    ●      │
│  use     │           │           │           │           │           │
└──────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```

### Reading the table
- **CPU** is the most **flexible** and easy to program but has low throughput.
- **GPU** is the sweet-spot for AI training — high throughput, good ecosystem, not *too* hard to program.
- **TPU** goes further on throughput but locks you into Google's stack.
- **NPU** wins on power efficiency — that's why it's in your phone.
- **FPGA** wins on latency but is a pain to program.

### The "specialization vs flexibility" trade-off
Moving left to right, you trade **generality** for **speed/efficiency** on a narrower workload. A CPU can run anything but none of it is world-class fast. A TPU or NPU is incredibly fast at tensor ops but can't even load an OS.

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
║  CPU = few smart cores, low throughput, great flexibility    ║
║  GPU = thousands of simple cores, massive parallelism       ║
║  TPU = ASIC for tensor ops (Google)                         ║
║  NPU = low-power on-device AI (phones)                      ║
║  FPGA = reconfigurable logic (custom accelerators)          ║
║                                                             ║
║  GPUs hide memory latency with PARALLELISM, not caches      ║
║  Memory wall: 1 DRAM fetch ≈ 200 arithmetic ops             ║
║  Shared memory = programmer-managed cache on a GPU SM        ║
║  HBM → 10-30× more bandwidth than CPU DRAM                  ║
║                                                             ║
║  Amdahl: Speedup = 1 / ((1−P) + P/N)                        ║
║  90% parallel → max 10× speedup no matter how many cores    ║
║                                                             ║
║  Host = CPU,  Device = GPU                                  ║
║  Kernel = a function that runs on the GPU                   ║
║  Data flows: host → PCIe → device → compute → back → host   ║
╚══════════════════════════════════════════════════════════════╝
```

### ⚡ Red-flag questions
1. **"Why are GPUs good for AI and CPUs aren't?"** — Deep learning is matrix multiply + convolution, which are thousands of independent arithmetic operations per input. GPUs have thousands of parallel cores; CPUs have a few dozen. GPUs can feed the cores with high-bandwidth HBM. The workload matches the hardware exactly.

2. **"Name 5 hardware families used for AI."** — CPU, GPU, TPU, NPU, FPGA. (Bonus: ASIC is the generic term; TPU and NPU are both ASICs.)

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
