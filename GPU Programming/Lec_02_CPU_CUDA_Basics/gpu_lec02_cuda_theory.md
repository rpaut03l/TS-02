# 📖 GPU Programming Lec 2 — CPU · CUDA · GPU Architecture: THEORY

### *ISA · threads/blocks/grids · warps · SIMT · memory hierarchy · workload flow*

> **Nav:** [← Lec 2 README](README.md) | **THEORY** | [💻 CODE](gpu_lec02_cuda_code.md) | [🎯 PRACTICE](gpu_lec02_cuda_practice.md)

---

## 🧠 MNEMONIC: **"ICP-TBGWS-MW"**

> **I**SA · **C**PI · **P**rogram run · **T**hread · **B**lock · **G**rid · **W**arp · **S**IMT · **M**emory hierarchy · **W**orkload flow

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Hardware Accelerators for Edge AI | [§1](#1-hardware-accelerators-for-edge-ai) |
| 2 | CPU & Instruction Set Architecture (ISA) | [§2](#2-cpu--instruction-set-architecture-isa) |
| 3 | Types of Instructions | [§3](#3-types-of-instructions) |
| 4 | Measuring CPU Performance (CPI, MIPS, clock) | [§4](#4-measuring-cpu-performance) |
| 5 | How a Program Actually Runs on a CPU | [§5](#5-how-a-program-actually-runs-on-a-cpu) |
| 6 | The CUDA Programming Model — Threads, Blocks, Grids | [§6](#6-the-cuda-programming-model) |
| 7 | Warps | [§7](#7-warps) |
| 8 | The Streaming Multiprocessor (SM) | [§8](#8-the-streaming-multiprocessor-sm) |
| 9 | SIMT vs SIMD | [§9](#9-simt-vs-simd) |
| 10 | GPU Memory Hierarchy (the complete picture) | [§10](#10-gpu-memory-hierarchy-the-complete-picture) |
| 11 | Workload Execution — 7 steps | [§11](#11-workload-execution--7-steps) |
| 12 | Programming Models & CUDA | [§12](#12-programming-models--cuda) |
| 13 | Cheat Sheet | [§13](#13-cheat-sheet--red-flags) |

---

## 1. Hardware Accelerators for Edge AI

### 👶 Easy Story
Your phone is a **tiny computer** that needs to do big-computer things — recognize faces, filter photos, transcribe speech — without eating the battery. It can't afford to ship every image to the cloud and back. So engineers add **tiny specialized chips** *right next to* the CPU to handle specific AI tasks efficiently. Those chips are **accelerators**.

### Formal definition
> An **accelerator** is a specialized processor that sits alongside the general-purpose CPU and performs a specific class of computations much faster and/or more energy-efficiently than the CPU could.

### Examples of accelerators
| Name | What it accelerates | Where you'll find it |
|---|---|---|
| **GPU** | dense parallel math | gaming PCs, datacenters, laptops |
| **TPU** | tensor ops (matmul, convolution) | Google datacenters, Coral edge boards |
| **NPU** | low-power AI inference | every modern phone |
| **DSP** | signal processing (audio, radio) | phones, hearing aids |
| **VPU** | video encode/decode | every SoC (System on Chip) |
| **FPGA** | user-defined custom logic | networking, medical devices, prototypes |
| **ASIC** | fixed custom silicon | Bitcoin miners, crypto engines, TPUs |

### Why "edge"?
**Edge AI** = running AI **on the device** instead of sending data to the cloud. Why? Privacy (your photos stay on your phone), latency (no network round-trip), cost (no cloud API bill), reliability (works offline). Accelerators make edge AI possible because a pure CPU would be too slow and hot for real-time inference.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 2. CPU & Instruction Set Architecture (ISA)

### 👶 Easy Story
The **ISA** is the "language" a CPU understands. Different CPUs speak different languages.
- An **x86** CPU (your laptop) understands one set of instructions.
- An **ARM** CPU (your phone) understands a different set.
- **RISC-V** is a newer, open language.

A program compiled for one language **won't run** on a CPU that speaks a different one. That's why your iPhone apps don't directly run on your laptop — different ISA.

### Formal definition
> The **Instruction Set Architecture (ISA)** is the contract between hardware and software. It specifies:
> - What instructions exist (add, load, jump, compare, …)
> - How registers are organized (how many, how wide)
> - How memory is addressed
> - What the program's "programmer-visible state" is

Anything **above** the ISA (compilers, operating systems, applications) is software. Anything **below** (microarchitecture, transistors) is hardware. The ISA is the boundary where software meets silicon.

### Two big families
```
CISC (Complex Instruction Set Computer)     RISC (Reduced Instruction Set Computer)
────────────────────────────────            ────────────────────────────────────
  Example: x86-64                            Example: ARM, RISC-V, MIPS, SPARC
  Many instructions (~1000s)                 Few instructions (~100s)
  Instructions can do a lot                  Each instruction does one small thing
  Variable instruction length                Fixed instruction length
  Hard to decode, easy to program            Easy to decode, harder to program
  Older, dominant on laptops/desktops        Newer, dominant on phones/servers/edge
```

Modern CPUs blur the line — x86 decodes its complex instructions into simple RISC-like *micro-ops* internally — but the visible ISA is still what matters for the programmer.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 3. Types of Instructions

Every ISA has instructions that fall into three buckets:

### 1️⃣ Data Transfer
Move data around. **No math.** Examples:
```
 LOAD    R1, [1000]    ; read memory address 1000 into register R1
 STORE   R2, [2000]    ; write register R2 to memory address 2000
 MOV     R1, R2        ; copy R2 into R1
```

### 2️⃣ Arithmetic/Logic
Actual computation on data. Examples:
```
 ADD     R3, R1, R2    ; R3 = R1 + R2
 SUB     R3, R1, R2    ; R3 = R1 - R2
 MUL     R3, R1, R2    ; R3 = R1 * R2
 AND     R3, R1, R2    ; bitwise AND
 OR      R3, R1, R2    ; bitwise OR
 SHIFT   R3, R1, 4     ; left-shift R1 by 4
```

### 3️⃣ Control
Change which instruction runs next. Without these, programs would just march straight down. Examples:
```
 JUMP    L1            ; unconditional: go to label L1
 BEQ     R1, R2, L1    ; conditional: branch if R1 == R2
 CALL    func          ; call function, remember return address
 RET                   ; return to the caller
```

### Why this matters for GPU programming
GPUs are **bad at control** instructions, especially branches where different threads in the same warp take different paths (called **warp divergence** — §7). GPUs are **excellent at** arithmetic and data transfer in bulk. So when you write GPU code, you want to minimize branching and maximize straight-line arithmetic.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 4. Measuring CPU Performance

How do you say "CPU A is faster than CPU B"? Four big metrics, each measuring something different.

### Clock frequency
> **Frequency (Hz)** = how many clock ticks per second.

- A 3 GHz CPU does 3 billion ticks per second.
- Once upon a time, higher clock = faster. Then physics hit the wall (~4 GHz) and single-core clock stopped climbing. Now everyone goes wider (more cores) instead of faster (higher GHz).

### CPI — Cycles Per Instruction
> **CPI = total cycles / total instructions executed**

- CPI of 1 → one instruction per clock tick (ideal).
- Modern superscalar CPUs can hit CPI < 1 (pipelined, multiple instructions per cycle).
- CPI of 5 → most instructions are slow (loads from DRAM, branch mispredicts).

### MIPS — Millions of Instructions Per Second
> **MIPS = (Clock frequency / CPI) × 10⁻⁶**

- A 3 GHz CPU with CPI of 1 does 3,000 MIPS.
- Same CPU with CPI of 3 only does 1,000 MIPS — 3× slower even though the clock didn't change.

### Execution time
> **Execution time = (instruction count × CPI) / clock frequency**

This is the **only** metric that really matters for the user. The other three are components you can tune.

### The performance equation
> **Execution time = IC × CPI × T**
>
> where IC = instruction count, CPI = cycles per instruction, T = clock period (1 / frequency)

Three levers to make a program faster:
1. **Reduce IC** — smarter algorithm, better compiler.
2. **Reduce CPI** — pipelining, better branch prediction, caching.
3. **Reduce T** (= raise frequency) — physics, power, cooling.

### GPUs have a very different metric
Instead of CPI on a single core, GPUs are measured in **TFLOPS** (trillions of floating-point ops per second) and **memory bandwidth (GB/s)**. Different game, different scoreboard.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 5. How a Program Actually Runs on a CPU

### 👶 Easy Story
You write `print("hello")`. A long chain of tools turns that into machine code the CPU can run.

```
 Your source code   (.py, .c, .cpp, .rs)
       ↓     [compiler / interpreter]
 Assembly code      (MOV, ADD, LOAD, …)
       ↓     [assembler]
 Machine code       (0s and 1s)
       ↓     [linker]
 Executable         (.exe, .elf, ELF binary)
       ↓     [OS loader]
 In RAM, ready to run
       ↓     [CPU fetch + decode + execute]
 Instructions run → your program does something
```

### The fetch-decode-execute cycle
At the very bottom, every CPU runs the same tiny loop:

```
┌────────────────────────────────────────────────────┐
│ 1. FETCH     get the next instruction from memory  │
│              (at the address in the Program        │
│              Counter, PC)                          │
│ 2. DECODE    figure out what the instruction means │
│ 3. EXECUTE   do it (ALU, memory, branch, …)        │
│ 4. WRITEBACK store the result                      │
│ 5. ADVANCE PC                                      │
│ (repeat forever)                                   │
└────────────────────────────────────────────────────┘
```

Modern CPUs **pipeline** this cycle — instead of running one instruction at a time, they keep many in flight at different stages. A 5-stage pipeline can process ~1 instruction per cycle on average even though each instruction takes 5 cycles to finish.

### Pipelining analogy
A laundromat with **5 machines**: sort → wash → dry → fold → put-away. If you do one load at a time, each load takes 5 phases. But if you overlap — start washing load 2 while load 1 is drying — you finish a load per phase instead of per 5 phases. That's 5× throughput. CPUs do this with **instructions**.

### Hazards — why pipelining isn't free
Sometimes the next instruction depends on the previous one's result. Pipelines must **stall** for that. Branch mispredictions are even worse — the CPU has to throw away half-executed instructions and restart. This is why **CPI = 1** is the ideal, not the typical.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 6. The CUDA Programming Model

This is **the** core abstraction you'll use for every CUDA program.

### 👶 Easy Story — the 1000-sheep colouring book
You have 10,000 sheep to colour. The way CUDA assigns the work:
- **1 thread = 1 sheep.** Each thread (worker) colours exactly one sheep.
- **Threads are bundled into "blocks"** — e.g. 256 threads per block. Each block is like a team of 256 kids sitting at one table, sharing crayons.
- **Blocks are arranged in a "grid"** — one grid covers the whole job. 10,000 sheep / 256 per block = about 40 blocks.
- **The same function (the "kernel") runs on every thread, but each thread sees a different index** — so thread 7 knows "colour sheep 7."

### The three-level hierarchy
```
┌────────────────────────── GRID ────────────────────────────┐
│                                                            │
│  ┌─BLOCK 0─┐   ┌─BLOCK 1─┐   ┌─BLOCK 2─┐   ┌─BLOCK 3─┐     │
│  │ t t t t │   │ t t t t │   │ t t t t │   │ t t t t │     │
│  │ t t t t │   │ t t t t │   │ t t t t │   │ t t t t │     │
│  │ t t t t │   │ t t t t │   │ t t t t │   │ t t t t │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│    (e.g. 256                                               │
│    threads                                                 │
│    each)                                                   │
└────────────────────────────────────────────────────────────┘
```

### What is a kernel?
> A **CUDA kernel** is a function, written in CUDA C/C++ (or Python via numba/cupy), that runs **on the GPU** in parallel on many threads.

```c
// Every thread executes this function body.
__global__ void vector_add(float *a, float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // my unique index
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
```

- `__global__` — "I'm a kernel, launch me from the CPU, run me on the GPU."
- `blockIdx.x` — which block am I in? (0, 1, 2, …)
- `blockDim.x` — how many threads per block? (e.g. 256)
- `threadIdx.x` — my index *within* my block? (0, 1, …, 255)
- `i = blockIdx.x * blockDim.x + threadIdx.x` — my global unique index across the whole grid.

### Launch syntax
```c
// Launch with 40 blocks × 256 threads = 10,240 threads total
vector_add<<<40, 256>>>(d_a, d_b, d_c, N);
```

That `<<<grid_size, block_size>>>` triple-bracket thing is the CUDA C launch syntax. In Python (numba), it's `kernel[grid_size, block_size](...)`.

### Why three levels?
| Level | Purpose |
|---|---|
| **Thread** | the unit of work — one index of the output |
| **Block** | a group of threads that can **share memory** and **synchronize**. Think "team sharing a whiteboard." |
| **Grid** | all the blocks launched by one kernel call. No direct communication between blocks — they're independent. |

Blocks are independent so the GPU scheduler can launch them on any available SM in any order. That's how CUDA scales from a small GPU (few SMs) to a big GPU (hundreds of SMs) with zero code changes.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 7. Warps

### The 32-thread bundle
Inside a thread block, threads are further grouped into **warps of 32 threads**. A warp is the true unit of execution on NVIDIA hardware — **all 32 threads in a warp execute the same instruction at the same clock cycle**. This is called **SIMT** (Single Instruction, Multiple Threads).

```
 Block of 256 threads = 8 warps × 32 threads each

 Warp 0: t0  t1  t2  t3  ... t31   ─┐
 Warp 1: t32 t33 t34 ... t63        │  each warp
 Warp 2: t64 ... t95                │  executes the
 Warp 3: t96 ... t127                │  SAME instruction
 ...                                 │  on all 32 threads
 Warp 7: t224 ... t255              ─┘  in lockstep
```

### 👶 Easy Story — the 32-seat bus
Imagine a **32-seat tour bus**. The tour guide announces "everyone, take a photo of the tree!" and all 32 passengers raise their cameras and click at the same moment. That's a warp: 32 passengers, one instruction, everyone obeys at once.

### Warp divergence
What happens if half the passengers are told "take a photo of the tree" and the other half "take a photo of the bench"? A normal bus could split — but a warp cannot. Instead, the warp has to:
1. **Mask off** the bench-photographers. Only the 16 tree-people are active.
2. Execute the "tree" instruction. The 16 bench-people do nothing.
3. **Flip the mask.** Only the 16 bench-people are now active.
4. Execute the "bench" instruction. The 16 tree-people do nothing.

**Total time = time for instruction 1 + time for instruction 2.** The warp effectively ran **twice**. This is **warp divergence** — when threads in the same warp take different branches, the warp runs both paths sequentially, halving (or worse) your throughput.

### Rule of thumb
- Branches that the **whole warp agrees on** are cheap.
- Branches that **split the warp** are expensive.
- If you see an `if` inside a GPU kernel, ask: "will all 32 threads in each warp take the same path?" If not, you're losing performance.

### Warp size is always 32 on NVIDIA
It's been 32 since the earliest CUDA GPUs, and it's likely to stay 32. On AMD GPUs ("wavefronts") it's traditionally 64, though newer architectures moved to 32. Your **block size** should always be a **multiple of 32** so you don't waste a partial warp.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 8. The Streaming Multiprocessor (SM)

### What's an SM?
The GPU is divided into **Streaming Multiprocessors**. Each SM is an independent unit that can execute thread blocks. A GPU with 108 SMs (like an A100) can have 108 blocks running *at the same time*.

### What's inside an SM (Fermi-era example from the slides)
```
┌─────────────────────── SM ────────────────────────┐
│                                                   │
│  [Warp Schedulers]                                │
│      ▼                                            │
│  [Register File]   (thousands of 32-bit regs)     │
│      ▼                                            │
│  [32 CUDA Cores]   (each does 1 FP add per cycle) │
│  [16 Load/Store units]                            │
│  [4 Special Function Units (SFU)]                 │
│          sin, cos, sqrt, exp, log                 │
│                                                   │
│  [Shared memory / L1 cache]   ~48-164 KB          │
│  [Texture memory]                                 │
└───────────────────────────────────────────────────┘
```

From the slides:
- **16 load/store units** — one SM can calculate source and destination addresses for **16 threads per clock** (half a warp).
- **4 Special Function Units (SFUs)** — execute transcendentals (`sin`, `cos`, `reciprocal`, `sqrt`). Slower than normal arithmetic but cheap compared to calculating these in software.

### Modern SMs (Ampere, Hopper) are much bigger
- **A100 SM** — 64 FP32 cores + 64 INT32 cores + 32 FP64 cores + 4 tensor cores + 192 KB L1/shared memory.
- **H100 SM** — 128 FP32 cores + 128 INT32 cores + 64 FP64 cores + 4 4th-gen tensor cores + 228 KB L1/shared memory.

Tensor cores deserve their own deep dive in a later lecture — they are specialized units that do a 4×4 matrix multiply-accumulate in a single instruction.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 9. SIMT vs SIMD

Two related ideas, one important distinction.

### SIMD — Single Instruction, Multiple Data
**One instruction operates on a vector of data.** Example: an AVX-512 instruction on x86 can add 16 floats at once. The programmer writes *one* instruction; the hardware treats the 16 floats as a single packed vector.

```
 SIMD add:  v3 = v1 + v2   where v1, v2, v3 each hold 16 floats
```

**Programming model:** the compiler/programmer has to think in terms of *vectors*. If your loop doesn't vectorize cleanly, you lose the SIMD benefit.

### SIMT — Single Instruction, Multiple Threads
**One instruction is executed by many (conceptually independent) threads.** Each thread has its own registers, its own program counter, its own sense of identity. But the hardware groups 32 threads into a warp and executes them lock-step.

```
 SIMT kernel:  each of 1,000,000 threads runs "c[i] = a[i] + b[i]"
               with its own i
```

**Programming model:** the programmer writes code as if each thread is an independent worker. Easier to read, easier to scale.

### The key difference
| SIMD | SIMT |
|---|---|
| Operates on vectors | Operates on threads |
| Programmer thinks in vectors | Programmer thinks in threads |
| Hard to write branching | Branches allowed (but can diverge) |
| Fixed vector width | Warp width hidden from programmer |
| CPU-style (AVX, NEON) | GPU-style (CUDA, ROCm) |

**Both are "Single Instruction, Multiple Things" but SIMT gives each "thing" the illusion of being an independent thread**, even though under the hood it still runs in warps of 32.

### Why SIMT is easier
If you've written SIMD code, you know the pain — special intrinsics, hand-unrolling, hating on loops with branches. SIMT code reads like normal C — you just index by `threadIdx + blockIdx*blockDim` and let the hardware sort it out. This is why CUDA took off.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 10. GPU Memory Hierarchy (the complete picture)

### 👶 Easy Story
A GPU has **many kinds of memory**, each a different size, speed, and purpose.
- **Registers** — the very fast notebook in front of each worker. Few pages.
- **Shared memory** — the team whiteboard. Only workers on the same team see it.
- **L1 cache** — automatic shortcut for the team.
- **L2 cache** — the office-wide bookshelf.
- **Global memory** — the warehouse. Big, slow, everyone shares it.
- **Constant memory** — read-only reference books cached for fast reads.
- **Texture memory** — special cache tuned for 2D/3D spatial access (graphics).

### The hierarchy in full

```
┌────────────────────────────────────────────────────────────────┐
│  per-thread      REGISTERS       ~32 32-bit regs per thread    │
│                  ── 1 cycle ──   ~KB per SM                     │
├────────────────────────────────────────────────────────────────┤
│  per-block       SHARED MEMORY   48-164 KB per SM              │
│                  ── ~30 cycles ──  programmer-managed cache     │
│                  L1 CACHE        combined with shared memory   │
├────────────────────────────────────────────────────────────────┤
│  per-device      L2 CACHE        ~40-50 MB (one copy per GPU)  │
│                  ── ~200 cycles ──                              │
├────────────────────────────────────────────────────────────────┤
│  per-device      GLOBAL (HBM)    16-80 GB                       │
│                  ── ~400 cycles ──  ~1-3 TB/s bandwidth         │
│                  CONSTANT        64 KB, cached, read-only       │
│                  TEXTURE         cached, 2D/3D optimized        │
├────────────────────────────────────────────────────────────────┤
│  over PCIe       HOST MEMORY     (system DRAM, your OS's RAM)  │
│                  ── milliseconds ── 6-64 GB/s                   │
└────────────────────────────────────────────────────────────────┘
```

### Each type in detail

#### Registers
Per-thread. The fastest. Your compiler decides how many each thread gets. Running out of registers = "register spill" = the compiler has to put some variables in local memory (slow). Watch for this in profiling.

#### Shared memory
Per-block. The "programmer-managed L1." When threads in a block need to share data or re-use a loaded value, you put it in shared memory. Typical use: **tiling** for matrix multiply (load a tile of A and B into shared memory, multiply-accumulate, move to next tile).

#### Global memory (HBM / GDDR)
The main GPU memory. When you do `cudaMalloc` or `cp.asarray`, this is where it goes. Visible to every thread. High bandwidth but high latency. Always the bottleneck for memory-bound kernels.

#### Constant memory
64 KB of read-only data. Cached aggressively because reads are broadcast to all threads in a warp — perfect for kernel-wide constants like filter weights.

#### Texture memory (from the slides)
> **Texture memory** on a GPU is a dedicated memory system for storing texture data (typically 2D or 3D images and patterns, used to render detailed 3D scenes). It's **read-only**, features **hardware caching** for efficient **spatial access**, and supports specialized addressing modes like **clamping** and **interpolation**.

Originally designed for graphics, texture memory is now also used for scientific workloads that access 2D/3D data with spatial locality — the texture cache is optimized for "read `[i, j]` and nearby pixels."

### Access patterns — why **coalesced** memory access matters
The GPU reads global memory in **chunks of 32 or 128 bytes at a time**. If all 32 threads in a warp ask for consecutive addresses, the hardware can combine all 32 reads into **1 memory transaction**. This is called **coalesced access**. If they ask for scattered addresses, the hardware makes **32 separate transactions** — **32× slower**.

**Rule:** write kernels so thread `i` reads `a[i]`, not `a[perm[i]]`. The stride-1 pattern is coalesced; strided or random patterns are not.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 11. Workload Execution — 7 Steps

Here's the full life of a GPU workload, start to finish.

```
 1. CPU (host) starts the program.

 2. CPU allocates memory on the GPU (cudaMalloc / cp.zeros / torch.empty).

 3. CPU copies input data from host memory to device memory over PCIe.
    This is the SLOW step. Measured in ms for large tensors.

 4. CPU launches a CUDA kernel.
    The kernel is scheduled onto the GPU's SMs. Blocks are distributed;
    warps execute; threads compute.

 5. Kernel runs to completion. The CPU can do other work in parallel,
    or call cudaDeviceSynchronize to wait.

 6. CPU copies the result back from device to host memory over PCIe.
    Another slow step.

 7. CPU continues with whatever comes next (post-processing,
    storing to disk, calling another kernel, etc.).
```

### The implicit rule
**Steps 3 and 6 are the expensive transfers.** Step 5 is the cheap computation. Fast GPU code minimizes 3+6 by:
- **Moving data once**, doing many kernels on it, and only bringing the final result back.
- **Pinning host memory** (faster DMA transfers — `cudaHostAlloc`).
- **Overlapping** transfers with computation using CUDA streams (Lec later).

### Asynchronous launches
When the CPU "launches" a kernel, it just **enqueues** the work and returns immediately. The GPU runs asynchronously. This is why:
- You **don't see timing from the CPU** unless you sync.
- You **can overlap CPU work with GPU work** for free.
- `cudaDeviceSynchronize()` is required before measuring GPU time.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 12. Programming Models & CUDA

### What is CUDA?
> **CUDA** = **C**ompute **U**nified **D**evice **A**rchitecture.
>
> A parallel computing platform and programming model created by NVIDIA. It lets developers write C/C++/Fortran/Python code that runs directly on NVIDIA GPUs.

### The CUDA stack
```
 Your code (C++, Python, Fortran)
        ↓
 CUDA Runtime API  (cudaMalloc, cudaMemcpy, cudaLaunchKernel)
        ↓
 CUDA Driver API   (lower level, lets you talk to the GPU driver)
        ↓
 NVIDIA driver     (talks to the actual hardware)
        ↓
 GPU silicon
```

In practice, 99% of the time you use the **Runtime API** from C++ (via `nvcc` compiler) or from Python (via `numba`, `cupy`, `pytorch`, or `tensorflow`, which all use the runtime API under the hood).

### Alternative GPU programming models
CUDA is the dominant one but not the only one:

| Platform | Works on | Notes |
|---|---|---|
| **CUDA** | NVIDIA only | The de-facto standard |
| **HIP** | AMD + NVIDIA | AMD's CUDA-alike; 95% API-compatible |
| **OpenCL** | Everyone | Older, open standard; clunkier API |
| **SYCL** | Everyone | C++-based, modern, Intel's oneAPI |
| **Metal** | Apple | Mac/iOS GPUs |
| **Vulkan Compute** | Everyone | Graphics API that also does compute |

For AI/ML practically everyone writes CUDA (directly or indirectly) because NVIDIA dominates the market and the CUDA ecosystem (cuDNN, cuBLAS, TensorRT, Triton, etc.) is enormous.

### The Python abstraction layers
You'll see these tools — each abstracts CUDA at a different level:

| Tool | Level | What you write |
|---|---|---|
| **CUDA C++** | lowest | full kernels, manage everything |
| **Numba `@cuda.jit`** | mid | CUDA kernels in Python |
| **CuPy** | high | NumPy replacement |
| **PyTorch / TensorFlow** | highest | you don't see the kernels; they're autogenerated |

Start high, drop a level when you need more control.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

## 13. Cheat Sheet & Red Flags

```
╔══════════════════════════════════════════════════════════════╗
║  GPU LEC 2 ONE-LINERS                                        ║
╠══════════════════════════════════════════════════════════════╣
║  ISA = the language the CPU speaks (x86, ARM, RISC-V)       ║
║  3 instruction types: data transfer / arithmetic / control  ║
║  CPI: cycles per instruction                                ║
║  Exec time = IC × CPI × T                                   ║
║                                                              ║
║  CUDA hierarchy: thread → block → grid                      ║
║  Thread: 1 worker = 1 output element                        ║
║  Block:  shares memory + synchronization                     ║
║  Grid:   all blocks of one kernel launch                    ║
║                                                              ║
║  Warp = 32 threads executing the same instruction          ║
║  Warp divergence = branch taken by some threads             ║
║                    → paths serialized → slower              ║
║  Block size MUST be a multiple of 32                        ║
║                                                              ║
║  SM = Streaming Multiprocessor (unit of parallelism)        ║
║  SIMT = per-thread programming model over SIMD hardware     ║
║                                                              ║
║  Memory hierarchy (fast → slow):                             ║
║    registers → shared → L1 → L2 → global → (PCIe) → host    ║
║  Coalesced access: thread i reads a[i] → 1 transaction      ║
║  Scattered access: thread i reads a[perm[i]] → 32 txns     ║
║                                                              ║
║  Workload lifecycle: host → alloc → H2D → launch →          ║
║    compute → D2H → back on host                             ║
║                                                              ║
║  CUDA = Compute Unified Device Architecture                 ║
╚══════════════════════════════════════════════════════════════╝
```

### ⚡ Red-flag questions

1. **"What's an ISA?"** — The Instruction Set Architecture — the contract between hardware and software; it specifies what instructions exist, what registers are visible, and how memory is addressed. Examples: x86, ARM, RISC-V.

2. **"Name the 3 types of instructions."** — Data transfer (LOAD, STORE, MOV), arithmetic/logic (ADD, SUB, MUL, AND, OR), control (JUMP, BRANCH, CALL, RET).

3. **"Write the CPU performance equation."** — Execution time = Instruction Count × CPI × Clock Period. Three knobs: reduce IC (algorithm), reduce CPI (microarchitecture), raise frequency (physics).

4. **"What's a CUDA thread, block, grid?"** — A thread is one unit of work running the kernel body. A block is a group of threads that share memory and can synchronize. A grid is all the blocks launched by one kernel call. Grid → blocks → threads.

5. **"What's a warp?"** — A group of 32 threads within a block that execute the same instruction in lockstep. Warp size = 32 on NVIDIA. Block sizes should be multiples of 32 to avoid waste.

6. **"What is warp divergence and why is it bad?"** — When threads in the same warp take different branches, the hardware masks off some threads and executes the branches *sequentially*. This serializes work that should have been parallel — a 2-way divergence halves throughput.

7. **"SIMT vs SIMD?"** — SIMD: one instruction over a vector of data — the programmer writes vectorized code. SIMT: one instruction over many *threads* — the programmer writes per-thread code and the hardware groups threads into warps. SIMT is an easier programming model built on SIMD-like hardware.

8. **"What is texture memory?"** — A read-only, hardware-cached memory system on the GPU specialized for 2D/3D spatial access patterns, with built-in clamping and interpolation, originally designed for graphics but useful for scientific 2D/3D data.

9. **"Trace the 7-step workload lifecycle."** — (1) CPU starts, (2) allocate device memory, (3) copy H2D over PCIe, (4) launch kernel, (5) GPU executes, (6) copy D2H over PCIe, (7) CPU continues.

10. **"What does CUDA stand for?"** — Compute Unified Device Architecture.

11. **"Why do block sizes of 32, 64, 128, 256 work and not 50?"** — The hardware runs warps of 32 threads. If you pick 50, you get one full warp of 32 and one warp of 18 with 14 "dead" threads that still occupy a warp slot. Multiples of 32 avoid that waste.

12. **"What is coalesced memory access?"** — When all 32 threads in a warp read consecutive addresses, the GPU can service them with a single wide memory transaction. Non-coalesced access requires multiple transactions and is dramatically slower.

[↑ Back to Top](#-gpu-programming-lec-2--cpu--cuda--gpu-architecture-theory)

---

> **Next:** [💻 CODE →](gpu_lec02_cuda_code.md) · [🎯 PRACTICE →](gpu_lec02_cuda_practice.md)
>
> *GPU Programming · Lec 2 · github.com/rpaut03l/TS-02*
