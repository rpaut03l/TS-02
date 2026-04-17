# 📖 EdgeAI · CUDA for Edge — THEORY

### *Desktop CUDA meets Jetson — JetPack, unified memory, TensorRT, DeepStream*

> **Nav:** [← CUDA for Edge README](README.md) | **THEORY** | [💻 CODE](edge_cuda_code.md) | [🎯 PRACTICE](edge_cuda_practice.md)

---

## 🧠 MNEMONIC: **"J-U-T-D-N"**

> **J**etPack · **U**nified memory · **T**ensorRT · **D**eepStream · **N**vpmodel

Five things that *look* like small extras on top of CUDA but actually
make or break every real Jetson project.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Desktop CUDA vs Jetson CUDA — what's different | [§1](#1-desktop-cuda-vs-jetson-cuda--whats-different) |
| 2 | **JetPack SDK** — the edge CUDA stack | [§2](#2-jetpack-sdk--the-edge-cuda-stack) |
| 3 | **Unified / zero-copy memory** on Tegra | [§3](#3-unified--zero-copy-memory-on-tegra) |
| 4 | **TensorRT** — how your model becomes an "engine" | [§4](#4-tensorrt--how-your-model-becomes-an-engine) |
| 5 | **INT8 calibration** | [§5](#5-int8-calibration) |
| 6 | **DeepStream** — video AI pipelines | [§6](#6-deepstream--video-ai-pipelines) |
| 7 | Power modes — **`nvpmodel` & `jetson_clocks`** | [§7](#7-power-modes--nvpmodel--jetson_clocks) |
| 8 | When to write a custom CUDA kernel | [§8](#8-when-to-write-a-custom-cuda-kernel) |
| 9 | Cross-compilation & deployment | [§9](#9-cross-compilation--deployment) |
| 10 | Cheat sheet | [§10](#10-cheat-sheet--red-flags) |

---

## 1. Desktop CUDA vs Jetson CUDA — what's different

### 👶 Easy Story
Imagine two houses:

- **Desktop house.** Chef (CPU) has her kitchen upstairs; line-cooks
  (GPU) are downstairs. Food travels up and down the stairs
  (PCIe bus). Lots of trips.
- **Jetson house.** Chef and line-cooks share **one** big kitchen
  (unified physical memory on the same die). No stairs. If you keep
  sending people up and down stairs that aren't there, you just run
  around in circles — wasting time.

| Aspect | Desktop CUDA | Jetson CUDA |
|---|---|---|
| CPU & GPU memory | **Separate** (DRAM + VRAM, PCIe in between) | **Shared** (one LPDDR5 pool) |
| `cudaMemcpy` cost | Real (PCIe bandwidth bound) | Often wasted (copy within same RAM) |
| Preferred alloc | `cudaMalloc` | `cudaMallocManaged` (unified) |
| CPU cores | 8–32 × x86 (fast) | 6–12 × ARM (slower per-core) |
| GPU compute cap. | 7.5 / 8.0 / 8.6 / 8.9 / 9.0 | **8.7** (Orin) / 7.2 (Xavier) |
| Storage | NVMe, fast SSD | eMMC / microSD (often slow) |
| Power | Plug the wall (450 W) | Battery or 19 V barrel (10–60 W) |
| Dev workflow | SSH locally, native build | Cross-compile from x86, copy binary |

### The single biggest surprise
On a Jetson, the **same DRAM** is seen by both CPU and GPU. This is
called **"shared physical memory"**. The consequence:

> **`cudaMemcpy` from host to device on a Jetson is a copy *within*
> the same physical memory.** It wastes cycles. Replace it with
> unified / zero-copy memory.

[↑ Back to Top](#-edgeai--cuda-for-edge--theory)

---

## 2. JetPack SDK — the edge CUDA stack

### 👶 Easy Story
On a desktop, you install the **NVIDIA driver**, then **CUDA toolkit**,
maybe **cuDNN**, maybe **TensorRT** — each as its own package.

On Jetson, you install **one thing** called **JetPack**, and it brings
the whole stable stack at once. Think of it as a **"Jetson starter
pack"** blessed by NVIDIA.

### What's inside JetPack (as of JetPack 6.x)

```
 ┌────────────────────────────────────────────────────────────────┐
 │                       JetPack 6 SDK                            │
 ├────────────────────────────────────────────────────────────────┤
 │ 1. L4T  (Linux for Tegra)      → Ubuntu-based OS image         │
 │ 2. CUDA toolkit                → the language + runtime        │
 │ 3. cuDNN                       → deep-learning kernel library  │
 │ 4. TensorRT                    → inference optimizer + runtime │
 │ 5. DeepStream                  → video AI pipeline SDK         │
 │ 6. VisionWorks / VPI           → image-processing primitives   │
 │ 7. Multimedia API              → camera, encode, decode        │
 │ 8. NSight tools                → profilers (nsys / ncu / graph)│
 │ 9. Developer tools             → gcc, gdb, Python, OpenCV      │
 └────────────────────────────────────────────────────────────────┘
```

### Install flow (abbreviated)
1. Download **SDK Manager** on a host Linux x86 machine.
2. Connect Jetson via USB in recovery mode (`REC` pin shorted).
3. SDK Manager flashes the L4T image, then installs the rest over
   Ethernet.
4. 60–90 minutes later, you have a Jetson that speaks CUDA.

### So what?
> **Never install CUDA packages on Jetson by hand.** Use JetPack. It
> pins tested versions that NVIDIA's compilers, drivers, and kernels
> all agree on.

[↑ Back to Top](#-edgeai--cuda-for-edge--theory)

---

## 3. Unified / zero-copy memory on Tegra

### 👶 Easy Story
The CPU and GPU are roommates. They share one fridge. If the CPU
writes "leftover pizza" on a note, the GPU sees the pizza **without
opening the fridge door**. Telling the chef to "carry the pizza from
the kitchen to the kitchen" is silly.

### The 3 memory models on Jetson

```
 ┌─────────────────────────────────┬────────────────────────────────┐
 │ Allocator                       │ What it actually does          │
 ├─────────────────────────────────┼────────────────────────────────┤
 │ cudaMalloc (pure device)        │ Allocates in the LPDDR pool,   │
 │                                 │ only the GPU sees it.          │
 ├─────────────────────────────────┼────────────────────────────────┤
 │ cudaMallocHost / cudaHostAlloc  │ Allocates pinned CPU memory.   │
 │ (pinned/zero-copy)              │ GPU accesses it over the same  │
 │                                 │ memory bus (no copy).          │
 ├─────────────────────────────────┼────────────────────────────────┤
 │ cudaMallocManaged (UVM)         │ One pointer works on both CPU  │
 │                                 │ and GPU. On Tegra it's backed  │
 │                                 │ by the shared LPDDR pool — no  │
 │                                 │ page-migration cost.           │
 └─────────────────────────────────┴────────────────────────────────┘
```

### Practical rule
On Jetson:
- **New code** → prefer `cudaMallocManaged` (unified). One pointer, no
  copies.
- **Camera / producer-consumer** → `cudaHostAlloc` with the
  `cudaHostAllocMapped` flag (zero-copy; the camera fills host memory
  and the GPU reads it in place).
- **`cudaMalloc` + `cudaMemcpy`** → still works, but you're paying
  for copies that are no-ops on desktop and actually *slower* on
  Tegra than using unified.

### Why "slower" on Tegra?
Because `cudaMemcpy` triggers a full synchronization and a copy through
the system DMA — when the source and destination are the same physical
DRAM, that's a wasted trip.

### A mental picture
```
  ┌──────────────────────────────────────────┐
  │          TEGRA SoC  (Orin)               │
  │  ┌──────────┐           ┌──────────┐     │
  │  │  ARM CPU │ ←─ L3 ──→ │   GPU    │     │
  │  └──────────┘           └──────────┘     │
  │        │                     │           │
  │        └───── SoC fabric ────┘           │
  │                 │                        │
  │         ┌───────┴──────────┐             │
  │         │  LPDDR5 (shared) │ ◄─ one pool │
  │         └──────────────────┘             │
  └──────────────────────────────────────────┘
```

[↑ Back to Top](#-edgeai--cuda-for-edge--theory)

---

## 4. TensorRT — how your model becomes an "engine"

### 👶 Easy Story
You trained a model in PyTorch. It runs at 30 FPS on your Jetson —
great. TensorRT asks: "give me that model, and let me **re-cook it**
for this specific GPU." You wait 30 seconds. Out comes a **.engine**
file. Load it — the same model now runs at **120 FPS**. Same accuracy,
4× faster, 3× less power.

### What TensorRT actually does
1. **Graph optimisations** — fuses convolution + batchnorm + ReLU into
   one kernel. Kills dozens of separate CUDA launches.
2. **Precision calibration** — drops FP32 to FP16 or INT8 where safe.
3. **Kernel auto-tuning** — benchmarks multiple CUDA kernel
   implementations on *your specific* GPU and picks the fastest.
4. **Memory planning** — allocates one contiguous workspace rather
   than letting PyTorch churn.

### The standard 4-step flow

```
  ┌──────────────┐    ┌────────────┐    ┌──────────────┐    ┌──────────┐
  │ 1. TRAIN     │ -> │ 2. EXPORT  │ -> │ 3. BUILD     │ -> │ 4. RUN   │
  │ PyTorch / TF │    │ ONNX file  │    │ trtexec or   │    │ Python or│
  │              │    │            │    │ Python API   │    │ C++ API  │
  │ .pth / .h5   │    │ .onnx      │    │ .engine plan │    │ inference│
  └──────────────┘    └────────────┘    └──────────────┘    └──────────┘
```

### Precision choices

| Precision | Why | Typical speedup vs FP32 | Accuracy drop |
|---|---|---|---|
| **FP32** | safest; debug baseline | 1× | 0 % |
| **FP16** | fast; no calibration; Orin has strong FP16 | 2–3× | usually < 0.2 % |
| **INT8** | fastest + lowest power; needs calibration | 3–5× | 0.5–2 % (if done right) |

### So what?
> Any production Jetson inference loop is a TensorRT engine. **PyTorch
> on Jetson is for development only.**

[↑ Back to Top](#-edgeai--cuda-for-edge--theory)

---

## 5. INT8 calibration

### 👶 Easy Story
To pack 32-bit floats into 8-bit integers, TensorRT needs to know
**the range of values** each tensor actually takes during real
inference. You hand it **a few hundred sample inputs**; it runs them,
records min/max per layer, and derives scale/zero-point. That's
calibration.

### The calibration interfaces

- **EntropyCalibratorV2** — the classic. Minimises KL divergence.
  Safe default.
- **MinMaxCalibrator** — use when activations are bounded.
- **PostTrainingQuantization (PTQ)** — use TensorRT's calibrator
  (what we're doing).
- **QAT (Quantization-Aware Training)** — train with fake-quant nodes
  in PyTorch, then export ONNX with QDQ nodes. Best accuracy, most
  work.

### A tiny checklist
1. Gather **100–500** *representative* samples (same distribution as
   production).
2. Feed them through a `Python class(trt.IInt8EntropyCalibrator2)`
   implementing `get_batch`.
3. Save the calibration **cache** so you don't recompute on each
   build.
4. Compare INT8 accuracy vs FP16 baseline. Expect **< 1 %** drop if
   calibration data is good.

See [code.md §3](edge_cuda_code.md#ex-3--full-int8-calibration-with-tensorrt)
for a runnable example.

[↑ Back to Top](#-edgeai--cuda-for-edge--theory)

---

## 6. DeepStream — video AI pipelines

### 👶 Easy Story
You've built a cool model. Now you need: **camera → decode → resize
→ batch → infer → track → overlay → encode → stream**, all **real-
time**, for **8 cameras at once**. Writing that in Python would be
slow and ugly. DeepStream is NVIDIA's **ready-made Lego kit** built
on GStreamer + TensorRT.

### DeepStream in one diagram

```
 ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌────────────┐
 │ nvarguss│→  │  nvv4l2-  │→  │ nvstream│→  │ nvinfer    │→ classify
 │ (CSI)   │   │  decoder  │   │  mux    │   │ (TensorRT) │
 └─────────┘   └───────────┘   └─────────┘   └────────────┘
                                    ▲
                                    │            ↓
                               ┌─────────┐   ┌────────────┐
                               │ nvtracker│←  │ nvinfer-sec│
                               └─────────┘   └────────────┘
                                    ↓
                               ┌─────────┐   ┌────────────┐
                               │nvdsosd  │→  │ nvv4l2-enc │→ RTSP
                               └─────────┘   └────────────┘
```

All communication happens on the GPU with **NVMM buffers** — zero-copy
all the way. That's why an AGX Orin can do **12+ simultaneous 1080p30
analytics streams** without falling over.

### When to use DeepStream
- Multi-camera video analytics.
- You need TensorRT + tracking + encoding in one pipeline.
- You care about **cost per stream**.

### When not to
- You're building a non-video product (audio, industrial sensor,
  robotics control loop). Use raw CUDA / TensorRT.

[↑ Back to Top](#-edgeai--cuda-for-edge--theory)

---

## 7. Power modes — `nvpmodel` & `jetson_clocks`

### 👶 Easy Story
A Jetson wakes up in a **"polite mode"** — slow, cool, quiet. To
measure real performance you need to wake it up fully.

### The two tools
- **`nvpmodel`** — picks a **power mode** (a ceiling on watts and
  which cores are enabled). Orin NX has 3 (10 W / 15 W / 25 W),
  AGX Orin has more.
- **`jetson_clocks`** — locks CPU, GPU, EMC clocks to their **maximum
  allowed for the current nvpmodel**. No thermal scaling, no DVFS.

### The canonical "benchmark mode" dance on Jetson

```bash
sudo nvpmodel -m 0         # MAXN / highest TDP for this module
sudo jetson_clocks         # lock to max allowed clocks
sudo tegrastats            # watch live power & temps
# … run your benchmark …
sudo jetson_clocks --restore
```

### When to use which
- **`jetson_clocks`** → benchmarks & short hot loops.
- **`nvpmodel`** → pick a power mode for the production deployment.

### Red flag 🚩
Benchmark numbers from blogs with no mention of `nvpmodel` are
**meaningless**. Always post both numbers.

[↑ Back to Top](#-edgeai--cuda-for-edge--theory)

---

## 8. When to write a custom CUDA kernel

### 👶 Easy Story
TensorRT already wrote the kernel for convolution, pooling, attention,
GEMM, softmax. You should only write your **own** kernel when:

1. The op doesn't exist in TensorRT and isn't easy to express in ONNX
   (custom preprocessing, non-standard loss, odd convolution shapes).
2. The op is your **hot path** and you can beat TensorRT by exploiting
   structure (rare but possible).
3. **Preprocessing on the GPU** — resize, colour-convert, normalise.
   If your camera delivers NVMM buffers and you need them in NCHW
   float, a ~50-line CUDA kernel avoids a CPU round-trip.

### The interface: TensorRT plugins
- Subclass **`IPluginV2IOExt`** (C++) or **`IPluginV2DynamicExt`**
  (Python via `tensorrt` APIs).
- Provide: `enqueue()` (your kernel launch), `supportsFormatCombination`,
  `getOutputDataType`, serialization hooks.
- Register the plugin with the ONNX parser so the engine builder can
  stitch it in.

This is exactly the same CUDA you wrote on desktop — the wrapper just
lets TensorRT *call* it between its own kernels.

[↑ Back to Top](#-edgeai--cuda-for-edge--theory)

---

## 9. Cross-compilation & deployment

### 👶 Easy Story
Your laptop is **x86**. Your Jetson is **ARM64**. Compiling on the
Jetson itself is slow (eMMC flash is tiny and not very fast). So you
**cross-compile**: big x86 machine builds an ARM64 binary, you copy
it across, it runs.

### Two common flows

| Flow | Tooling | When |
|---|---|---|
| **Native on device** | `apt install build-essential`, `cmake`, `nvcc` | Early prototyping; small projects |
| **Cross-compile** | **JetPack SDK Manager** → install x86 cross toolchain + sysroot; **CMake** with toolchain file | Real teams, CI, reproducible builds |
| **Docker** | `nvidia/cuda:12.2-devel-ubuntu22.04` for x86; **`l4t-jetpack` base images** for aarch64 | Modern default — push a signed image to the device |

### Best practice
- Pin JetPack version in Dockerfile.
- Keep a single `Dockerfile.jetson` that uses `l4t-pytorch` or
  `l4t-ml` as the base.
- Run CI that builds an **aarch64** image using `docker buildx`.

[↑ Back to Top](#-edgeai--cuda-for-edge--theory)

---

## 10. Cheat sheet & red flags

### Cheat sheet
```
 JETSON CUDA     Shared DRAM → use cudaMallocManaged.
                 Don't cudaMemcpy without thinking.
 JETPACK         One install does it: L4T + CUDA + cuDNN + TensorRT
                 + DeepStream + NSight.
 TRT FLOW        Train → export ONNX → build engine → run (FP16/INT8).
 TRT GAIN        2–5× speed-up vs PyTorch on the same chip, 1 % acc. cost.
 INT8            Calibrate with 100–500 real samples; cache the result.
 DEEPSTREAM      Use it for multi-camera video analytics. Skip otherwise.
 POWER MODES     `nvpmodel -m <N>` + `jetson_clocks` before every bench.
 CUSTOM KERNEL   Only for preprocessing or ops TensorRT can't fuse.
 DEPLOYMENT      Cross-compile or Docker `l4t-*` base images.
```

### Red flags
- 🚩 Benchmarks published without `nvpmodel` / `jetson_clocks` state.
- 🚩 PyTorch in production on a Jetson. Convert to TensorRT.
- 🚩 INT8 with only 5 calibration samples — accuracy will be terrible.
- 🚩 Big `cudaMemcpy` loops in Jetson code — probably wasted copies.
- 🚩 Building on eMMC with `make -j12` — you'll melt the SSD *and*
  wait an hour.

### Green flags
- ✅ `cudaMallocManaged` or `cudaHostAllocMapped` everywhere.
- ✅ Every model ships as a **TensorRT engine** plus an **ONNX
  fallback**.
- ✅ INT8 calibration cache checked in (reproducible builds).
- ✅ `tegrastats` output in CI logs.
- ✅ `Dockerfile.jetson` based on `l4t-pytorch` or `l4t-ml`.

---

## 🔭 Where to go next

You've now seen:
- **What Edge AI is** ([Fundamentals/](../Fundamentals/README.md))
- **Every edge GPU family** ([GPU_Types/](../GPU_Types/README.md))
- **Every non-GPU edge chip + Edge-vs-Cloud** ([Hardware/](../Hardware/README.md))
- **How CUDA changes at the edge** (this folder)

Planned next topics (the 🔭 list in the top-level README):
- **Model compression** (quantization, pruning, distillation)
- **Deployment frameworks** (TFLite, ONNX Runtime, OpenVINO deep-dives)
- **TinyML** (TF Lite Micro on Cortex-M + Ethos-U55)
- **Federated Learning** on the edge
- **Edge MLOps** and OTA updates
- **Security & privacy** on edge devices

---

> *GPU Programming · EdgeAI · CUDA for Edge · THEORY · github.com/rpaut03l/TS-02*
