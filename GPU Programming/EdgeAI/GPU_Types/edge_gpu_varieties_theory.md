# 📖 EdgeAI · GPU Types — THEORY

### *The full atlas of GPUs that live on the edge*

> **Nav:** [← GPU Types README](README.md) | **THEORY** | [💻 CODE](edge_gpu_varieties_code.md) | [🎯 PRACTICE](edge_gpu_varieties_practice.md)

---

## 🧠 MNEMONIC: **"J-D-I-M"**

> **J**etson · **D**iscrete-edge · **I**ntegrated · **M**obile-SoC

The four families of Edge GPUs. Remember this one letter-mix and you
can place any new chip you read about into its slot.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | The 3 super-powers of edge GPUs | [§1](#1-the-3-super-powers-of-edge-gpus) |
| 2 | The 4 families — at a glance | [§2](#2-the-4-families--at-a-glance) |
| 3 | Family 1 — **Jetson** (embedded SoMs) | [§3](#3-family-1--nvidia-jetson-embedded-soms) |
| 4 | Family 2 — **Discrete edge GPUs** | [§4](#4-family-2--discrete-edge-gpus) |
| 5 | Family 3 — **Integrated GPUs (iGPUs)** | [§5](#5-family-3--integrated-gpus-igpus) |
| 6 | Family 4 — **Mobile SoC GPUs** | [§6](#6-family-4--mobile-soc-gpus) |
| 7 | How to read a GPU spec sheet | [§7](#7-how-to-read-a-gpu-spec-sheet) |
| 8 | "Which edge GPU should I use?" decision tree | [§8](#8-which-edge-gpu-should-i-use-decision-tree) |
| 9 | Cheat sheet & red flags | [§9](#9-cheat-sheet--red-flags) |

---

## 1. The 3 super-powers of edge GPUs

### 👶 Easy Story
Why is it always a **GPU** people reach for at the edge, not a CPU?
Because GPUs have three super-powers that match what Edge AI actually
needs.

### Super-power 1 — **Massive parallelism**
- Unlike CPUs (4–16 big cores), a GPU has **hundreds to thousands of
  tiny cores**.
- Neural networks are mostly **matrix multiplies** — thousands of tiny
  multiply-adds that can all happen at the same time.
- A Jetson Orin NX has **1,024 CUDA cores** + **32 Tensor cores** in a
  chip the size of your thumb.

### Super-power 2 — **Low latency & high speed**
- A GPU kernel can launch and finish in **< 1 ms** for small models.
- That's fast enough for real-time robotics, autonomous driving,
  industrial defect detection.
- Measured by **throughput** (images/sec) and **latency** (ms/inference).

### Super-power 3 — **Local processing**
- The GPU lives **on the device**, so data never leaves.
- Better **privacy**, better **uptime** (works offline), and you don't
  need a gigabit uplink.

> These three powers are the ones highlighted in the image that kicked
> off this whole folder: *massive parallelism*, *low latency*, *local
> processing*. The rest of this chapter is about the **four families of
> chips** that deliver those powers in different shapes.

[↑ Back to Top](#-edgeai--gpu-types--theory)

---

## 2. The 4 families — at a glance

```
                          EDGE GPU FAMILIES
                          ─────────────────
   ┌────────────────────┐  ┌────────────────────┐
   │ 1. JETSON (SoM)    │  │ 2. DISCRETE EDGE   │
   │ NVIDIA embedded    │  │ PCIe card in a     │
   │ module + carrier   │  │ rugged edge server │
   │ 5 – 60 W           │  │ 50 – 150 W         │
   │ Orin, Xavier, Nano │  │ T4, RTX 4000 Ada,  │
   │                    │  │ RTX A2000          │
   └────────────────────┘  └────────────────────┘
   ┌────────────────────┐  ┌────────────────────┐
   │ 3. INTEGRATED GPU  │  │ 4. MOBILE SoC GPU  │
   │ Inside the CPU die │  │ Inside phone chip  │
   │ 5 – 45 W           │  │ 1 – 5 W            │
   │ Intel UHD, Iris Xe │  │ Adreno, Mali,      │
   │ AMD Radeon iGPU    │  │ Apple GPU, Xclipse │
   └────────────────────┘  └────────────────────┘
```

Each family is optimised for a **different point** on the
**cost / power / compute / size** curve. They rarely overlap.

[↑ Back to Top](#-edgeai--gpu-types--theory)

---

## 3. Family 1 — NVIDIA Jetson (embedded SoMs)

### 👶 Easy Story
Take a gaming GPU. Shrink it. Put it on a **credit-card-sized board
that also has the CPU, RAM, video encoders, and I/O**. Drop the power
to 5–60 W. That's a **Jetson**.

### Formal
A Jetson is a **System-on-Module (SoM)** by NVIDIA. It combines in
one package:
- An **ARM CPU** (usually 6–12 ARM cores).
- A **GPU** with **CUDA cores** and **Tensor cores**.
- **LPDDR memory** shared between CPU and GPU (this is important —
  see the CUDA folder).
- Hardware video encoders / decoders (so you can pipe in cameras).
- I/O (PCIe, CSI cameras, GPIO, USB, Ethernet, CAN, …).

You plug the SoM into a **carrier board** (which has the actual
connectors) to make a dev kit or a product.

### The lineup (most common, past-to-present)

```
 ┌───────────────────┬────────┬──────────┬─────────┬──────────┬────────────┐
 │ Module            │ Power  │ CUDA c.  │ Tensor  │ AI perf  │ Typical    │
 │                   │ (W)    │          │ cores   │ (TOPS)   │ use case   │
 ├───────────────────┼────────┼──────────┼─────────┼──────────┼────────────┤
 │ Jetson Nano       │  5–10  │  128     │   —     │   0.47   │ Hobby, ed. │
 │ Jetson TX2        │  7–15  │  256     │   —     │   1.3    │ Drones     │
 │ Jetson Xavier NX  │ 10–20  │  384     │  48     │  21      │ Smart cam  │
 │ Jetson AGX Xavier │ 10–30  │  512     │  64     │  32      │ Robotics   │
 │ Jetson Orin Nano  │  7–15  │  1024    │  32     │  40      │ New hobby  │
 │ Jetson Orin NX    │ 10–25  │  1024    │  32     │  70–100  │ Prod cam   │
 │ Jetson AGX Orin   │ 15–60  │  2048    │  64     │ 200–275  │ Robots/AV  │
 └───────────────────┴────────┴──────────┴─────────┴──────────┴────────────┘
```

(TOPS figures are the **sparse INT8** figures NVIDIA publishes. Real
workloads often see 30–70 % of peak.)

### What each row tells you
- **Power:** most Jetsons have multiple **power modes** (`nvpmodel`)
  — you trade TOPS for watts.
- **CUDA cores** = the general-purpose parallel units. All CUDA code
  runs here.
- **Tensor cores** = specialised units for **matrix multiply** (INT8 /
  FP16 / BF16). This is where your neural-net speed really comes from.
- **TOPS** = Trillion Operations Per Second. The single most cited
  number. Look at the **data type** it's measured in (usually INT8).

### Why Jetson dominates "industrial edge"
- **Full CUDA ecosystem** — any PyTorch / TensorFlow model, any CUDA
  kernel you wrote for a desktop GPU, runs here.
- **JetPack SDK** bundles CUDA, cuDNN, TensorRT, DeepStream,
  VisionWorks, OpenCV — you get a ready-made edge ML stack.
- **Rich I/O** — up to **6 simultaneous CSI cameras** on AGX Orin.
- **Long product availability** — NVIDIA commits to 5–7 years of
  supply, which is what industrial customers need.

### Pros / Cons
- **Pros:** most flexible edge GPU on the market, full CUDA, huge
  community, great tools (`tegrastats`, `jtop`, TensorRT).
- **Cons:** $$$ (Orin NX module alone can be $600+), power budget
  still bigger than phone NPUs, custom carrier board adds time to
  ship a real product.

[↑ Back to Top](#-edgeai--gpu-types--theory)

---

## 4. Family 2 — Discrete edge GPUs

### 👶 Easy Story
Same shape as a gaming card — a PCIe board you slide into a computer.
But these are built for **server-style edge boxes**: fanless or
single-slot, modest power (50–150 W), and **certified for 24/7 duty**
at industrial temperatures.

### The lineup

```
 ┌────────────────────────┬────────┬─────────┬──────────┬──────────────┐
 │ GPU                    │ Power  │ Memory  │ INT8 TOPS│ Role         │
 ├────────────────────────┼────────┼─────────┼──────────┼──────────────┤
 │ NVIDIA Tesla T4        │  70 W  │ 16 GB   │   130    │ Inference    │
 │ NVIDIA RTX A2000       │  70 W  │ 12 GB   │    40    │ Workstation  │
 │ NVIDIA RTX 4000 Ada SFF│  70 W  │ 20 GB   │   153    │ Edge server  │
 │ NVIDIA L4              │  72 W  │ 24 GB   │   242    │ Video AI     │
 │ NVIDIA RTX 6000 Ada    │ 300 W  │ 48 GB   │ 1,457    │ Edge DC      │
 └────────────────────────┴────────┴─────────┴──────────┴──────────────┘
```

### When to use discrete edge
- Your edge box is a **rugged industrial PC** or a **small edge server**.
- You need to run **many cameras** (8–64) through one AI pipeline.
- You need to serve **several models** concurrently (detection +
  tracking + pose).
- You need **more memory** than a Jetson has (some models need 16+ GB).

### Sweet-spot: Tesla T4
- The T4 is the **work-horse of edge inference**.
- **70 W, single-slot, passive cooling**, 16 GB GDDR6.
- Runs anywhere a data-center puts it — including telco 5G MEC
  cabinets, retail stores, and small factories.

### Sweet-spot: RTX 4000 Ada SFF
- **"Small form factor"** = fits in a short, 2-slot industrial PC.
- 70 W, 20 GB — latest Ada Lovelace (2024), strong INT8/FP8 TOPS.
- Replaces the T4 for new deployments.

### Pros / Cons
- **Pros:** far more compute than a Jetson, standard PCIe (no custom
  carrier board), works with regular Linux servers.
- **Cons:** 70–150 W is too hot for small enclosures; needs real
  cooling; bigger BOM (you need a server to plug it into).

[↑ Back to Top](#-edgeai--gpu-types--theory)

---

## 5. Family 3 — Integrated GPUs (iGPUs)

### 👶 Easy Story
Open up any recent Intel or AMD CPU. A piece of the same silicon is a
**tiny GPU**. That's the **iGPU**. It's been there for years; you've
been using it without noticing. For **light Edge AI** it's often
"good enough" — and **free**, because you already paid for the CPU.

### The lineup

```
 ┌───────────────────────────────────┬───────┬──────────┬─────────────────┐
 │ iGPU                              │ Power │ INT8 TOPS│ Typical host    │
 ├───────────────────────────────────┼───────┼──────────┼─────────────────┤
 │ Intel UHD Graphics (older)        │  5 W  │   ~1     │ NUC, thin PC    │
 │ Intel Iris Xe (Gen12)             │ 15 W  │   ~4     │ Laptops         │
 │ Intel Arc iGPU (Meteor Lake+)     │ 18 W  │   ~11    │ "Core Ultra"    │
 │ AMD Radeon Vega iGPU (Ryzen APUs) │ 10 W  │   ~2     │ Mini-PC         │
 │ AMD Radeon 780M (RDNA3)           │ 15 W  │   ~8     │ Ryzen 7040+     │
 └───────────────────────────────────┴───────┴──────────┴─────────────────┘
```

### Use cases
- **Industrial PCs** that already have an Intel or AMD CPU — use the
  iGPU as a "free" inference accelerator.
- **Kiosks, digital signage, point-of-sale** — small models like
  person-count, product recognition, speech-to-text.
- **Home NAS / small business servers** — transcoding + light AI.

### Programming iGPUs
- **Intel iGPUs** → **OpenVINO** (Intel's runtime) with the GPU plugin.
- **AMD iGPUs** → **ROCm** (limited on iGPUs), or **Vulkan/SYCL**
  cross-vendor.
- **Cross-vendor** → **ONNX Runtime** with the appropriate execution
  provider.
- **NOT CUDA** — these are not NVIDIA chips.

### Pros / Cons
- **Pros:** free (comes with the CPU), easy to deploy, low power, no
  extra cooling.
- **Cons:** modest TOPS, lives on the same thermal budget as the CPU
  (so heavy CPU work steals GPU headroom), smaller software ecosystem
  than CUDA.

[↑ Back to Top](#-edgeai--gpu-types--theory)

---

## 6. Family 4 — Mobile SoC GPUs

### 👶 Easy Story
Your phone has a GPU. Yes, really. It's tiny, sips milliwatts, and
runs at 1–5 W. It's there for games, but it's also a **first-class
edge AI accelerator** when you use the right APIs.

### The lineup

```
 ┌──────────────────────────┬──────────┬────────────┬────────────────────┐
 │ Mobile GPU               │ Typical  │ INT8 TOPS  │ Example SoC / Phone│
 │                          │ power    │ (estimate) │                    │
 ├──────────────────────────┼──────────┼────────────┼────────────────────┤
 │ Qualcomm Adreno 750      │ 2 – 5 W  │   ~45      │ Snapdragon 8 Gen 3 │
 │ Qualcomm Adreno 730      │ 2 – 5 W  │   ~26      │ Snapdragon 8 Gen 2 │
 │ ARM Mali-G720 Immortalis │ 1 – 4 W  │   ~32      │ MediaTek Dim. 9300 │
 │ ARM Mali-G78             │ 1 – 4 W  │   ~18      │ Dimensity 9000     │
 │ Apple GPU (A17 Pro)      │ 2 – 5 W  │   ~35      │ iPhone 15 Pro      │
 │ Samsung Xclipse 940      │ 2 – 5 W  │   ~26      │ Exynos 2400        │
 └──────────────────────────┴──────────┴────────────┴────────────────────┘
```

> **Note:** on a phone, AI rarely runs on the GPU alone. The SoC also
> has a **dedicated NPU** (Apple Neural Engine, Hexagon, Samsung NPU),
> which is often faster *and* more power-efficient than the GPU. The
> GPU is used when: the NPU lacks a needed op, the model was compiled
> only for GPU, or when doing graphics-heavy AR/VR work.

### Programming mobile GPUs
- **Android** → TensorFlow Lite **GPU Delegate** (uses OpenCL / Vulkan),
  or **NNAPI** which dispatches to NPU/GPU/DSP, or **SNPE** (Qualcomm).
- **iOS** → **Core ML** (Apple's runtime dispatches across ANE / GPU /
  CPU automatically), or **Metal Performance Shaders**.
- **Cross-platform** → **ONNX Runtime Mobile**.

### Pros / Cons
- **Pros:** incredibly low power, already in the user's pocket, very
  fast for everyday models, no extra BOM cost.
- **Cons:** tiny memory budget, each vendor has its own toolchain,
  thermal throttling kicks in fast during games.

[↑ Back to Top](#-edgeai--gpu-types--theory)

---

## 7. How to read a GPU spec sheet

Every edge-GPU datasheet hands you roughly the same 8 numbers. Here's
what each one really means.

```
 ┌───────────────────┬─────────────────────────────────────────────────┐
 │ Number            │ What it really tells you                        │
 ├───────────────────┼─────────────────────────────────────────────────┤
 │ CUDA cores        │ "How many simple workers?" More = more parallel │
 │ Tensor cores      │ "How many matrix-multiply helpers?" Key for AI  │
 │ FP32 TFLOPS       │ Peak speed for general-purpose float math       │
 │ FP16 / BF16 TFLOPs│ Same, at half precision (usually 2× FP32)       │
 │ INT8 TOPS         │ Peak integer speed — the "AI inference" number  │
 │ Sparse INT8 TOPS  │ INT8 with 2:4 structured sparsity (2× INT8)     │
 │ Memory capacity   │ How big a model you can load                    │
 │ Memory bandwidth  │ How fast the GPU can fetch weights from memory  │
 │ TDP (watts)       │ Power budget — heat you must remove             │
 └───────────────────┴─────────────────────────────────────────────────┘
```

### Rules of thumb
- **Memory bandwidth** is usually the bottleneck for inference, not
  compute. Look at **GB/s**, not only TOPS.
- **Sparse** TOPS is marketing — you only get it if your model *is*
  actually 2:4-sparse. Treat it as an upper bound.
- Always divide **TOPS by TDP** to get **TOPS/Watt** — the true
  edge-friendliness score.

### Worked example — Jetson Orin NX 16GB (15 W mode)

- CUDA cores: 1,024
- Tensor cores: 32 (3rd-gen Ampere)
- INT8 TOPS: 100 (sparse), 50 (dense)
- Memory: 16 GB LPDDR5, 102.4 GB/s
- TDP: 15 W
- **TOPS/Watt** (dense INT8) = 50 / 15 ≈ **3.3 TOPS/W** ✅

[↑ Back to Top](#-edgeai--gpu-types--theory)

---

## 8. "Which edge GPU should I use?" decision tree

```
                       START
                         │
           ┌─────────────┴─────────────────┐
           │    Is the device a phone?     │
           └─┬─────────────────────────────┘
             │ YES                       NO
             ▼                            ▼
      ┌──────────────┐                Do I have a
      │ Use MOBILE   │                PCIe slot and
      │ SoC GPU      │                a 70–150 W power
      │ (Adreno,     │                budget?
      │  Mali, ANE)  │                   │
      └──────────────┘                   │
                                ┌────────┴─────────┐
                                │ YES              │ NO
                                ▼                  ▼
                         Use DISCRETE       Is there a CPU
                         EDGE GPU           already in the box?
                         (T4 / RTX 4000 Ada)        │
                                                    │
                                              ┌─────┴─────┐
                                              │ YES       │ NO
                                              ▼           ▼
                                       iGPU is free.   Need a
                                       Use it if       full SoM?
                                       TOPS enough.    → JETSON
                                       Else add a
                                       Jetson.
```

### Three-line shortcuts
- **Phone product?** → Mobile SoC GPU (+ NPU).
- **Battery-powered camera / robot / drone?** → Jetson.
- **Rugged PC or edge server?** → Discrete edge GPU (or iGPU for light
  work).

[↑ Back to Top](#-edgeai--gpu-types--theory)

---

## 9. Cheat sheet & red flags

### Cheat sheet
```
 FAMILIES       Jetson · Discrete-Edge · iGPU · Mobile-SoC
 KEY NUMBER     TOPS / Watt  (higher = more edge-friendly)
 JETSON TIPS    "Orin Nano = hobby, Orin NX = product,
                 AGX Orin = robots/AV"
 DISCRETE TIPS  "T4 = old reliable, RTX 4000 Ada SFF = new default,
                 L4 = video AI"
 iGPU TIPS      "Free with the CPU. OpenVINO on Intel, ROCm on AMD,
                 ONNX Runtime cross-vendor."
 MOBILE TIPS    "Use the NPU first; GPU is plan B, CPU is plan C.
                 Android = TFLite GPU delegate or NNAPI;
                 iOS = Core ML."
 ALWAYS ASK     TOPS ✕ data-type? Memory BW? Power mode?
                Thermal throttle? Ecosystem/SDK?
```

### Red flags
- 🚩 A vendor quotes only **sparse TOPS** with no dense number — they
  are hiding something.
- 🚩 A vendor quotes **FP32 TFLOPS** for an AI chip — almost no modern
  edge model runs in FP32. Ask for INT8 / FP16 numbers.
- 🚩 A chip promises 40 TOPS at 5 W but only in a "burst" mode. Always
  check **sustained** performance under real thermal conditions.
- 🚩 Choosing a Jetson **without** a carrier board plan. Prototype on
  the NVIDIA dev kit, but plan custom carrier early.

### Green flags
- ✅ Dense + sparse TOPS both quoted.
- ✅ Sustained-vs-burst benchmarks in the datasheet.
- ✅ MLPerf Edge or MLPerf Tiny submission available.
- ✅ 5+ years of supply committed (critical for industrial).

---

## 🔭 Next up

You now know every *GPU* variety that matters on the edge. Next folder
[`Hardware/`](../Hardware/README.md) covers the **non-GPU** hardware —
NPUs, MCUs, FPGAs — **and** the big **Edge GPU vs Cloud GPU**
comparison table (the lower half of the image that started this whole
deep-dive).

---

> *GPU Programming · EdgeAI · GPU Types · THEORY · github.com/rpaut03l/TS-02*
