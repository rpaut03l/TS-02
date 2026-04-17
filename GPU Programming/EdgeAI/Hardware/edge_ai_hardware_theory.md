# 📖 EdgeAI · Hardware — THEORY

### *Non-GPU edge chips + the big Edge-vs-Cloud GPU comparison*

> **Nav:** [← Hardware README](README.md) | **THEORY** | [💻 CODE](edge_ai_hardware_code.md) | [🎯 PRACTICE](edge_ai_hardware_practice.md)

---

## 🧠 MNEMONIC: **"N-M-F-D"** (+ the big table)

> **N**PU · **M**CU · **F**PGA · **D**SP

And the one table every Edge AI engineer memorises:
**Edge GPU vs Cloud / Data-center GPU** (§6 below).

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Why we need non-GPU chips | [§1](#1-why-we-need-non-gpu-chips) |
| 2 | Edge **NPUs** — the purpose-built AI chips | [§2](#2-edge-npus--the-purpose-built-ai-chips) |
| 3 | **MCUs** for TinyML | [§3](#3-mcus-for-tinyml) |
| 4 | **FPGAs** at the edge | [§4](#4-fpgas-at-the-edge) |
| 5 | **DSPs** and signal processors | [§5](#5-dsps-and-signal-processors) |
| 6 | **Edge vs Cloud GPU** — full comparison | [§6](#6-edge-vs-cloud-gpu--full-comparison) |
| 7 | Power & thermal budgets | [§7](#7-power--thermal-budgets) |
| 8 | Benchmarks — MLPerf Tiny & Edge | [§8](#8-benchmarks--mlperf-tiny--edge) |
| 9 | Cheat sheet | [§9](#9-cheat-sheet--red-flags) |

---

## 1. Why we need non-GPU chips

### 👶 Easy Story
GPUs are amazing, but they're **Swiss-army knives**. If you need to
open a thousand cans of tomato, a **can-opener** will do it faster,
cheaper, and with less effort than the Swiss army knife — even though
the Swiss army knife *could* do it.

- **GPUs** are the Swiss-army knife: can do anything, good at a lot.
- **NPUs** are the can-opener: only does matrix multiplies, but 5–10×
  more efficient per watt for that one job.
- **MCUs** are the cheap knife in your drawer: tiny, everywhere, costs
  pennies, runs forever on a coin battery.
- **FPGAs** are a **configurable** tool: you *build* the exact tool
  you need out of logic gates.

### The real-world driver
Every watt, dollar, and square millimetre counts on a consumer device.
Specialized silicon buys you **5–10× efficiency** for the specific
workload you care about, at the cost of flexibility.

[↑ Back to Top](#-edgeai--hardware--theory)

---

## 2. Edge NPUs — the purpose-built AI chips

### 👶 Easy Story
An NPU is a **GPU that forgot how to do graphics, and got really good at
matrix multiply instead**. It's smaller, cheaper, cooler, and eats fewer
watts — but it can only run neural networks.

### Formal
A **Neural Processing Unit** is an **ASIC** (application-specific
integrated circuit) optimised for the **multiply-accumulate (MAC)**
operations that dominate neural networks, usually in **INT8 or lower**
precision. Many use a **systolic array** architecture (data flowing
through a grid of MAC units without writing back to RAM between steps).

### The lineup

```
 ┌───────────────────────────┬──────────┬──────────┬───────────────────────┐
 │ NPU                       │ TOPS     │ Power    │ Where it lives        │
 │                           │          │          │                       │
 ├───────────────────────────┼──────────┼──────────┼───────────────────────┤
 │ Apple Neural Engine (A17) │    35    │  ~2 W    │ Every iPhone/iPad/Mac │
 │ Qualcomm Hexagon (Gen 3)  │    45    │  2–5 W   │ Flagship Androids     │
 │ Google Edge TPU (Coral)   │     4    │  0.5–2 W │ USB dongle / dev kits │
 │ Hailo-8                   │    26    │  2.5 W   │ Smart camera SoM      │
 │ Hailo-15                  │    20    │  5 W     │ Vision SoC            │
 │ Intel NPU (Meteor Lake)   │    11    │  ~6 W    │ Core Ultra laptops    │
 │ AMD XDNA (Ryzen AI 300)   │    50    │  ~5 W    │ Ryzen 7040+ laptops   │
 │ Tesla FSD HW4             │   121    │  ~40 W   │ Tesla cars            │
 │ Qualcomm Cloud AI 100     │   400    │  75 W    │ 5G MEC, edge servers  │
 │ Mythic M1076 (analog)     │    25    │  4 W     │ Analog AI, ultra-low-P│
 └───────────────────────────┴──────────┴──────────┴───────────────────────┘
```

### Programming NPUs
Every vendor has its own runtime — this is the biggest **fragmentation
pain** in Edge AI.

| NPU | SDK / Runtime |
|---|---|
| Apple ANE | **Core ML** (transparent; dispatcher picks ANE/GPU/CPU) |
| Qualcomm Hexagon | **SNPE** or **Qualcomm AI Engine** + NNAPI on Android |
| Google Edge TPU | **PyCoral** (TFLite compiled through `edgetpu_compiler`) |
| Hailo-8 / 15 | **Hailo SDK** (DFC compiler + HailoRT) |
| Intel NPU | **OpenVINO** with the `NPU` device plugin |
| AMD XDNA | **ROCm-AI** / **Ryzen AI Software** |
| Cross-vendor | **ONNX Runtime** with the right **Execution Provider** |

### Pros / Cons
- **Pros:** 3–10× better TOPS/Watt than an equivalent GPU. Smaller.
  Cheaper at scale. Often "free" because it's inside a SoC you already
  pay for.
- **Cons:** less flexible — some operators are not supported, forcing
  parts of the graph to fall back to CPU/GPU. Toolchain fragmentation.
  Mostly **inference only** (no training).

[↑ Back to Top](#-edgeai--hardware--theory)

---

## 3. MCUs for TinyML

### 👶 Easy Story
A microcontroller is a **tiny whole computer on a chip** — usually a
1-cent-to-10-dollar chip that runs on a button-cell battery for years.
**TinyML** is the art of running (small) neural networks on these
chips.

### Why it matters
- A smart lightbulb sells for $8. Its chip budget is **$1**. There's no
  room for a Jetson.
- Your smoke detector must wake from sleep, classify smoke-vs-steam in
  100 ms, and go back to sleep. Total energy budget: **micro-joules**.
- Farm sensors solar-powered with a supercap must do ML and survive
  winter nights. Total budget: **10 mW**.

### The lineup

```
 ┌─────────────────────────────────┬──────────┬──────────┬───────────────┐
 │ MCU                             │ CPU      │ Memory   │ AI helper     │
 ├─────────────────────────────────┼──────────┼──────────┼───────────────┤
 │ ARM Cortex-M4F (e.g. STM32F4)   │ 80 MHz   │ 256 KB   │ (DSP ISA only)│
 │ ARM Cortex-M7 (STM32H7)         │ 480 MHz  │ 1 MB RAM │ (DSP ISA)     │
 │ ARM Cortex-M33 + Helium (M55)   │ 400 MHz  │ 1 MB RAM │ Helium SIMD   │
 │ ARM Cortex-M55 + Ethos-U55      │ 400 MHz  │ 1 MB RAM │ Ethos-U55 NPU │
 │                                 │          │          │ (~0.5 TOPS)   │
 │ ESP32-S3 (Xtensa)               │ 240 MHz  │ 512 KB   │ Vector DSP    │
 │ Raspberry Pi Pico (RP2040)      │ 133 MHz  │ 264 KB   │ none (PIO)    │
 │ Alif Ensemble (M55 + U55)       │ 400 MHz  │ 13 MB    │ Ethos-U55 NPU │
 └─────────────────────────────────┴──────────┴──────────┴───────────────┘
```

### Programming TinyML
- **TensorFlow Lite Micro** — the dominant framework. C++ runtime,
  no OS, no malloc, deterministic memory.
- **CMSIS-NN** — ARM's hand-tuned kernels used under TFLite Micro.
- **Edge Impulse** — end-to-end studio that collects data, trains, and
  compiles to a firmware library for most of the boards above.
- **microTVM** — Apache TVM cross-compiler for MCUs.

### TinyML models (typical sizes)

| Task | Model | Size | TOPS needed |
|---|---|---|---|
| Keyword spotting ("Hey XYZ") | DS-CNN | ~30 KB | ~3 MOPS |
| Anomaly detection (vibration) | Autoencoder | ~5 KB | ~1 MOPS |
| Person detection (320×240 grayscale) | MobileNetV1-quant | ~250 KB | ~60 MOPS |
| Gesture recognition (IMU) | Small 1D-CNN | ~20 KB | ~2 MOPS |

Yes, those are **KB and MOPS**, not MB and GOPS. TinyML is a different
planet from Jetson — but the **ideas are the same** (quantize, shrink,
compile, deploy).

[↑ Back to Top](#-edgeai--hardware--theory)

---

## 4. FPGAs at the edge

### 👶 Easy Story
An **FPGA** is a box of **Lego logic gates**. You don't use pre-built
chips — you **wire up your own chip** from tiny pieces. Slower to
develop, but once you've wired it, you get **ultra-predictable,
ultra-low-latency** behaviour.

### Formal
A **Field-Programmable Gate Array** is a chip of configurable logic
blocks connected by configurable interconnect. You program the
**hardware itself** (in Verilog/VHDL or via HLS from C++) rather than
writing software for fixed hardware.

### Edge FPGA families

| Chip | Typical use | Power |
|---|---|---|
| **Xilinx Zynq UltraScale+** (ARM + FPGA SoC) | Automotive, aerospace, industrial | 10 – 30 W |
| **Intel Agilex** (small) | 5G MEC, low-latency trading, radar | 30 – 80 W |
| **Xilinx Spartan / Artix** | Low-power industrial | 1 – 5 W |
| **Lattice CrossLink-NX** | TinyML + always-on vision | < 1 W |

### When FPGAs win
- **Ultra-low latency** (< 1 ms) for deterministic control loops.
- **Custom data types** (INT4, INT2, binary nets, analogue-like).
- **Sensor fusion** at line-rate (multiple cameras / lidars coming in
  simultaneously).
- **Long product lifecycles** (avionics, defence, medical) where fixed
  silicon gets obsolete.

### When FPGAs lose
- Development is **10× slower** than writing PyTorch.
- Far fewer engineers who can do it.
- Toolchains (Vitis AI, OpenVINO-FPGA) are smaller than CUDA's
  ecosystem.

### The compromise
Most teams reach for **FPGA SoCs** (Zynq) which put an ARM CPU plus
FPGA fabric on one chip — you do most of the work in software and
only drop to hardware for the hottest path.

[↑ Back to Top](#-edgeai--hardware--theory)

---

## 5. DSPs and signal processors

### 👶 Easy Story
A **DSP** is a CPU that grew up listening to music. It specialises in
the kind of math that audio, radio, and signal-processing need: fast
multiplies, ring buffers, fixed-point.

### In Edge AI
- **Qualcomm Hexagon** started as a DSP and grew AI extensions — it's
  now the NPU in every Snapdragon.
- **Cadence Tensilica Vision DSPs** show up inside ISPs (image signal
  processors) to run AI *inside the camera pipeline*.
- **CEVA** and **Synopsys ARC** DSPs are common in hearing aids,
  earbuds, and headsets — where every mW matters.

DSPs are usually an **invisible** layer: you don't program them
directly, you ship a model through an SDK and the compiler picks the
DSP if it's the best fit.

[↑ Back to Top](#-edgeai--hardware--theory)

---

## 6. Edge vs Cloud GPU — full comparison

This is the most-photographed table in all of Edge AI. Memorise it.

```
 ┌──────────────────┬──────────────────────────┬────────────────────────────┐
 │ Feature          │ EDGE GPU                 │ CLOUD / DATA-CENTER GPU    │
 ├──────────────────┼──────────────────────────┼────────────────────────────┤
 │ Primary goal     │ Real-time inference      │ Massive training,          │
 │                  │ + power efficiency       │ batched throughput         │
 │ Location         │ On-site: robot, camera,  │ Remote data-center         │
 │                  │ vehicle, factory         │ (us-east-1, europe-west-4) │
 │ Typical power    │ Low: watts to tens of W  │ High: hundreds of W per GPU│
 │                  │ (5 W Nano → 70 W T4)     │ (H100 = 700 W, B200 =1000W)│
 │ Latency          │ Extremely low (<50 ms)   │ Variable (50–500 ms round) │
 │ Memory           │ 4 – 24 GB shared/LPDDR5  │ 40 – 192 GB HBM3/HBM3e     │
 │ Memory bandwidth │ 50 – 320 GB/s            │ 2,000 – 8,000 GB/s         │
 │ Cost per unit    │ $100 – $3,000            │ $15,000 – $40,000          │
 │ Form factor      │ SoM / PCIe SFF / iGPU    │ Full-height PCIe / SXM     │
 │ Cooling          │ Passive → small fan      │ Heavy liquid / airflow     │
 │ Ecosystem        │ JetPack / OpenVINO /     │ CUDA, cuDNN, NCCL,         │
 │                  │ TFLite / Core ML         │ full datacenter stack      │
 │ Scale            │ Millions of devices      │ Thousands of GPUs per pod  │
 │ Best workload    │ Inference at the edge    │ Training giant models,     │
 │                  │                          │ serving massive LLMs       │
 │ Who buys?        │ OEMs, robotics, auto,    │ Hyperscalers, AI labs,     │
 │                  │ industrial               │ cloud vendors              │
 └──────────────────┴──────────────────────────┴────────────────────────────┘
```

### The 4 key differences in one line each
1. **Power:** edge measures in **watts**, cloud in **hundreds of watts**.
2. **Latency:** edge is **<50 ms**; cloud is **50–500 ms** round-trip.
3. **Memory:** edge uses **LPDDR (shared)**; cloud uses **HBM (private
   to GPU)**.
4. **Role:** edge **infers**; cloud **trains + serves-at-scale**.

> **The cloud makes the model. The edge runs the model.** That's the
> whole split in one sentence.

[↑ Back to Top](#-edgeai--hardware--theory)

---

## 7. Power & thermal budgets

### 👶 Easy Story
Every chip turns electricity into **heat**. If you can't get the heat
out of the box, the chip **slows itself down** to stop melting. That's
**thermal throttling**. Happy GPUs run cool; sad GPUs run slow.

### The rules

| Enclosure | Max sustained chip TDP |
|---|---|
| **Tiny fanless wearable** | 1 – 3 W |
| **Smartphone (fanless)** | 3 – 7 W (burst to 10 W) |
| **Fanless industrial PC** | 15 – 25 W |
| **Small fan (1U edge box)** | 30 – 75 W |
| **Rack 1U edge server** | 150 – 300 W |

### Why fanless matters
- **No moving parts** = 10× the reliability over 5 years.
- **No dust ingress** = works in factories and restaurants.
- **Silent** = works in offices and hospitals.
- But **fanless = ~15 W ceiling**. Above that you need airflow.

### Jetson power modes
```
Orin NX 16 GB:
  10W MAXN  — 10 W, ~20 TOPS
  15W       — 15 W, ~50 TOPS  ← sweet spot
  25W       — 25 W, ~100 TOPS
  MAX (+25W) — 40 W, burst

Switch with:
  sudo nvpmodel -m <id>
  sudo jetson_clocks   (lock clocks to max)
```

### Thermal throttling in practice
1. **Boot** — cold chip, full clocks, 100 FPS.
2. **3 minutes in** — silicon hits 85 °C, frequency drops 20 %, 80 FPS.
3. **10 minutes in** — thermal steady state at ~65 FPS.
Always benchmark **after** 10 minutes of warm-up. Day-one numbers lie.

[↑ Back to Top](#-edgeai--hardware--theory)

---

## 8. Benchmarks — MLPerf Tiny & Edge

### MLPerf Tiny (for MCUs)
- **Four tasks**: keyword spotting, visual wake words, anomaly
  detection, image classification.
- **Measured**: inference latency (ms), energy per inference (µJ).
- **Hardware range**: Cortex-M, RISC-V, Ethos-U, Coral, Syntiant NDP.

### MLPerf Edge / MLPerf Inference (edge suite)
- **Models**: ResNet-50, RetinaNet (800×800), BERT, 3D U-Net, DLRM.
- **Scenarios**:
  - **SingleStream** — 1 query at a time, measure P99 latency.
  - **Offline** — max throughput with unlimited batch.
- **Hardware range**: Jetson Orin, T4, L4, RTX 4000 Ada, Hailo, Ryzen AI.

### How to read an MLPerf result
1. Always note the **scenario** and **model** — numbers don't compare
   across.
2. Look at **power** (some categories are power-constrained).
3. Look for **"closed" vs "open"** division — closed uses unchanged
   models (apples-to-apples), open allows model tweaks.
4. MLPerf tables don't include **cost** — you add that yourself.

[↑ Back to Top](#-edgeai--hardware--theory)

---

## 9. Cheat sheet & red flags

### Cheat sheet
```
 ZOO            GPU · NPU · MCU · FPGA · DSP
 NPU STARS      Apple ANE · Hexagon · Hailo-8 · Coral · Intel NPU ·
                AMD XDNA
 TINYML STARS   Cortex-M55 + Ethos-U55 · ESP32-S3 · Alif Ensemble
 FPGA STARS     Xilinx Zynq · Intel Agilex · Lattice CrossLink-NX
 BIG TABLE      Edge vs Cloud GPU — (power, latency, memory, cost,
                role) — commit to memory
 POWER RULES    Fanless ≈ 15 W ceiling; always benchmark warm
 BENCHMARKS     MLPerf Tiny (MCU class) · MLPerf Edge (Jetson class)
```

### Red flags
- 🚩 Vendor quotes peak TOPS without saying which data type.
- 🚩 No sustained-vs-burst breakdown.
- 🚩 SDK only works on Linux x86, but your target is ARM.
- 🚩 NPU doesn't support a key op in your model (check the supported-op
  list **before** you commit).

### Green flags
- ✅ Chip has an MLPerf submission.
- ✅ Runtime supports ONNX as input (portable).
- ✅ Toolchain has a quantization-aware training path.
- ✅ 5+ year supply commitment.

---

## 🔭 Next up

You've now seen every non-GPU edge chip and the big comparison table.
The last folder [`CUDA_for_Edge/`](../CUDA_for_Edge/README.md) closes
the loop by showing how **CUDA programming** changes when you move
from desktop to Jetson — unified memory, JetPack, TensorRT, and all
the tricks specific to edge CUDA.

---

> *GPU Programming · EdgeAI · Hardware · THEORY · github.com/rpaut03l/TS-02*
