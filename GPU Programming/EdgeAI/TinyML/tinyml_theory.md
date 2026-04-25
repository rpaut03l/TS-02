# 📖 EdgeAI · TinyML — THEORY

### *Microcontrollers · TFLite Micro · CMSIS-NN · Ethos-U55 · Edge Impulse*

> **Nav:** [← TinyML README](README.md) | **THEORY** | [💻 CODE](tinyml_code.md) | [🎯 PRACTICE](tinyml_practice.md)

---

## 🧠 MNEMONIC: **"M-A-C-E"**

> **M**CU · **A**rena · **C**MSIS-NN · **E**thos-U

The four pieces you compose to ship a TinyML model.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | What makes TinyML different | [§1](#1-what-makes-tinyml-different) |
| 2 | The 4 canonical tasks | [§2](#2-the-4-canonical-tasks) |
| 3 | The MCU zoo | [§3](#3-the-mcu-zoo) |
| 4 | **TensorFlow Lite Micro** | [§4](#4-tensorflow-lite-micro) |
| 5 | **CMSIS-NN** | [§5](#5-cmsis-nn) |
| 6 | **Ethos-U55 / U65** and the Vela compiler | [§6](#6-ethos-u55--u65-and-the-vela-compiler) |
| 7 | Edge Impulse — end-to-end studio | [§7](#7-edge-impulse--end-to-end-studio) |
| 8 | Benchmarks — MLPerf Tiny | [§8](#8-benchmarks--mlperf-tiny) |
| 9 | The full TinyML workflow | [§9](#9-the-full-tinyml-workflow) |
| 10 | Cheat sheet | [§10](#10-cheat-sheet--red-flags) |

---

## 1. What makes TinyML different

### 👶 Easy Story
Imagine a **whole computer** that's smaller than your fingernail,
runs on a coin battery for **a year**, and still detects smoke, hears
"Hey, fridge!", or knows a cow fell over. TinyML fits AI into that
world.

### The 3 numbers that define TinyML

```
 RAM           64 KB – 1 MB        (your phone has 8 GB)
 FLASH         128 KB – 4 MB       (your phone has 256 GB)
 POWER         0.1 mW – 100 mW     (your laptop draws 30 W)
```

### Everything gets scaled down by ~1,000×

| Dimension | Mobile | TinyML | Scale |
|---|---|---|---|
| Model size | ~10 MB | ~30 KB | 300× |
| Inference latency | 10 ms | 10 ms (slower clocks!) | same |
| Power per inference | 100 mJ | 100 µJ | 1,000× |
| Dataset size | GBs | MBs | 1,000× |
| Cost per chip | $5–$30 | $1–$5 | 5× |

### Core idea
> The same **quantize → prune → compile** pipeline you've already
> learned, but the quantization is **mandatory INT8**, the pruning is
> **aggressive**, and the compile step targets a fixed-size **memory
> arena**.

[↑ Back to Top](#-edgeai--tinyml--theory)

---

## 2. The 4 canonical tasks

These are the four tasks MLPerf Tiny benchmarks — and every commercial
TinyML product is a flavour of one of them.

### Task 1 — Keyword Spotting (KWS)

- **👶 Story:** "Hey Siri" / "Ok Google" — the chip listens for a
  single word, wakes up the big chip.
- **Model:** **DS-CNN** (depthwise-separable CNN), ~30 KB INT8.
- **Input:** ~1 s audio → mel-spectrogram (49 × 10 frames, INT8).
- **Output:** 12 classes (10 keywords + silence + unknown).
- **Target:** < 10 ms per inference.
- **Reference dataset:** Google Speech Commands V2.

### Task 2 — Visual Wake Words (VWW)

- **👶 Story:** "Is there a person in this frame?" binary decision.
  Doorbells, smart cameras, energy-saving lights.
- **Model:** tiny MobileNet (96 × 96 grayscale input), ~250 KB.
- **Input:** 96 × 96 × 1 INT8.
- **Output:** 2 classes.
- **Target:** < 100 ms per inference.
- **Dataset:** COCO re-labelled for VWW.

### Task 3 — Anomaly Detection

- **👶 Story:** a vibration sensor on a pump says "normal" or
  "broken".
- **Model:** fully-connected **autoencoder**, ~5 KB.
- **Input:** ~6 s of log-mel from the target machine.
- **Output:** reconstruction error; threshold to alert.
- **Dataset:** ToyADMOS, MIMII.

### Task 4 — Image Classification (IC)

- **👶 Story:** classic CIFAR-10 at TinyML size.
- **Model:** ResNet-8-like, ~100 KB INT8.
- **Input:** 32 × 32 × 3 INT8.
- **Output:** 10 classes.

### The common shape
Input is **quantized INT8**, model is **INT8 weights + activations**,
output is **INT8** (then dequantized in a one-liner if needed).

[↑ Back to Top](#-edgeai--tinyml--theory)

---

## 3. The MCU zoo

```
 ┌──────────────────────────────────┬─────────┬────────────┬──────────┐
 │ Chip                              │ Core    │ RAM / Flash│ AI extra │
 ├──────────────────────────────────┼─────────┼────────────┼──────────┤
 │ STM32F4 (e.g. F411)               │ M4F     │ 128K/512K  │ DSP ISA  │
 │ STM32H7 (e.g. H743)               │ M7      │ 1M /2M     │ DSP ISA  │
 │ Nordic nRF52840                   │ M4F     │ 256K/1M    │ BLE + DSP│
 │ ESP32-S3 (Espressif)              │ Xtensa  │ 512K/flash │ Vector DSP│
 │ Raspberry Pi Pico (RP2040)        │ M0+ ×2  │ 264K/2M    │ PIO      │
 │ Arduino Nano 33 BLE Sense         │ M4F     │ 256K/1M    │ mic+IMU  │
 │ Alif Ensemble E7                  │ M55 ×2 +│ 13.5M SRAM │ Ethos-U55│
 │                                   │ A32 ×2  │            │ 0.5 TOPS │
 │ Himax WE2                         │ M55     │ 2.4M SRAM  │ Ethos-U55│
 │ Ambiq Apollo4 Blue Plus           │ M4F     │ 2.75M flash│ ultra-LP │
 │ Syntiant NDP120                   │ tiny core│ 640 KB    │ "always- │
 │                                   │          │            │ on" NDP  │
 └──────────────────────────────────┴─────────┴────────────┴──────────┘
```

### What to look at when picking
1. **Flash** — must be > model size + app code (typically 100 KB–1 MB).
2. **RAM** — must hold the **memory arena** (see §4). For VWW you
   need ~100 KB RAM.
3. **AI extras** — Helium SIMD (M55), Ethos-U55 NPU, or vector DSP
   can be 5–20× faster than M4 scalar code.
4. **Peripherals** — microphone, IMU, camera interface availability.
5. **Ecosystem** — is there a TFLM port? Does Edge Impulse support
   it? Is there a CMSIS-NN BSP?

[↑ Back to Top](#-edgeai--tinyml--theory)

---

## 4. TensorFlow Lite Micro

### 👶 Easy Story
TFLM is **TFLite with everything removed that needs an OS**. No
malloc. No files. No threads. No Python. Just a C++ interpreter that
reads a `.tflite` byte array and runs ops using a **pre-sized memory
arena**.

### The 5 lines every TFLM program has

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_data.h"        // the byte array: "alignas(8) const unsigned char g_model[]"

constexpr int kArenaSize = 100 * 1024;     // 100 KB, sized empirically
alignas(16) uint8_t tensor_arena[kArenaSize];

tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(g_model);
tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);
interp.AllocateTensors();
```

After that you just fill the input tensor, call `interp.Invoke()`, and
read the output tensor — same mental model as desktop TFLite, just in
C++ with no heap.

### The **memory arena**
- **What it is:** a flat contiguous block of RAM you hand to TFLM.
- **Why it exists:** no `malloc` is available (or desirable).
- **How to size it:** start with 2× the sum of all tensor sizes;
  after building, call `interp.arena_used_bytes()` and shrink to the
  actual number.

### Op resolvers
- `AllOpsResolver` — all ops compiled in (big binary).
- `MicroMutableOpResolver<N>` — register exactly the ops your model
  uses. Typical TinyML binaries end up 100–200 KB smaller this way.

### Example custom resolver:

```cpp
using OpsResolver = tflite::MicroMutableOpResolver<6>;
static OpsResolver resolver;
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
resolver.AddFullyConnected();
resolver.AddMaxPool2D();
resolver.AddSoftmax();
resolver.AddReshape();
```

### Kernel selection
- Default kernels = plain C++. Portable but slow.
- **CMSIS-NN kernels** (if available) = hand-tuned for Cortex-M.
  Enable with `-DTF_LITE_USE_CMSIS_NN=1` or use the CMSIS-NN-aware
  op resolver from the Arm pack.
- **Ethos-U55 kernel** (if available) = op is delegated to the NPU.

[↑ Back to Top](#-edgeai--tinyml--theory)

---

## 5. CMSIS-NN

### 👶 Easy Story
Cortex-M cores have tiny **DSP** and **Helium (MVE)** instructions
that do 4–8 multiplies at once. CMSIS-NN is a **free library of
neural-net kernels** that actually uses them. Dropping it in under
TFLM gives a **free 2–5× speed-up** on the same model on the same
chip.

### What's inside
- `arm_convolve_*` — Conv2D (regular, depthwise, 1×1), SIMD-optimised.
- `arm_fully_connected_*` — dense layer, INT8 matmul with MAC-pair
  instructions.
- `arm_max_pool`, `arm_avg_pool`, `arm_softmax_*`, `arm_relu_*`.
- `arm_svdf_*` — statefull SVDF layer used in KWS models.

### Why it's fast
- **MAC pair** — Cortex-M4F can do one MUL and one ADD in the same
  cycle; CMSIS-NN schedules them as pairs.
- **Helium / MVE** (M55 / M85) — 8-way INT8 SIMD; hand-unrolled inner
  loops.
- **Memory-bound kernels** use **im2col**-free algorithms to avoid
  temporary buffers the arena can't afford.

### Licensing
Apache 2.0 — ship it freely.

[↑ Back to Top](#-edgeai--tinyml--theory)

---

## 6. Ethos-U55 / U65 and the Vela compiler

### 👶 Easy Story
The **Ethos-U55** is a **tiny AI chip** (~0.5 TOPS) that sits next to
a Cortex-M55. Your model runs on the NPU; the M55 just orchestrates.
You get **10–100×** speed-up on supported ops for ~10 mW.

### The flow

```
 .tflite (INT8)  ──►  Vela compiler  ──►  .tflite with Ethos-U ops
                                         (embedded command stream)
                                         │
                                         ▼
                    TFLM with Ethos-U kernel → runs on U55 NPU
```

### Vela compiler
- CLI: `vela model.tflite --accelerator-config ethos-u55-128`
- Takes a plain INT8 `.tflite` and turns it into one where **all
  supported ops** are replaced by a single `ethos-u` command-stream op.
- Reports any ops it **cannot** accelerate (they stay on the CPU).

### System configs
- `ethos-u55-32` / `-64` / `-128` / `-256` — the number = MACs per
  cycle (more = faster, more silicon).

### Pros / Cons
- **Pros:** 10–100× faster than pure M55, dedicated small footprint,
  Apache-licensed reference code.
- **Cons:** only supported on a few MCU vendors (Alif, Himax, NXP).
- **Cons:** some ops fall back to CPU. Always run Vela's op-report
  before you commit.

[↑ Back to Top](#-edgeai--tinyml--theory)

---

## 7. Edge Impulse — end-to-end studio

### 👶 Easy Story
Edge Impulse is the **"no-code" TinyML platform**. Web UI to:
1. Collect data from the device's sensors.
2. Design features (FFT, spectrogram, filter banks).
3. Train a model.
4. Test it on-device via WebUSB / Bluetooth.
5. Export as a firmware library for Arduino, STM32, Nordic, Sony
   Spresense, Raspberry Pi Pico, ESP32, Alif, …

### When to use it
- Fast prototyping.
- Sensor fusion (audio + IMU + image in one project).
- You want someone else to maintain the Cortex-M toolchain.

### When to outgrow it
- You need custom ops or a custom loss.
- You ship millions of devices and want fine-grained control of the
  binary.
- You're already deep in TFLM / PyTorch Mobile and just need the
  model — then use the [Model_Compression](../Model_Compression/README.md)
  + [Deployment_Frameworks](../Deployment_Frameworks/README.md) flow
  instead.

[↑ Back to Top](#-edgeai--tinyml--theory)

---

## 8. Benchmarks — MLPerf Tiny

### The rules
- **Closed division:** use one of the 4 reference models (DS-CNN,
  MobileNet VWW, autoencoder, ResNet-8). Compare apples to apples.
- **Open division:** any model is allowed. Shows the frontier.
- **Scenarios:** latency + energy per inference (microjoules).
- **Power measurement:** external reference monitor (EEMBC's ULPMark).

### Typical results (closed, keyword spotting)

| System | Latency (ms) | Energy (µJ) |
|---|---|---|
| **Syntiant NDP120** | 1.8 | 35 |
| **Alif Ensemble E7 + U55** | 3.5 | 90 |
| **Himax WE2 + U55** | 4.0 | 100 |
| **STM32H747 M7** | 18 | 800 |
| **Coral Dev Board (Edge TPU)** | 2.0 | 180 |
| **ESP32-S3** | 24 | 1200 |

### How to read it
- Dedicated "always-on" NPUs (Syntiant) crush both latency and energy.
- NPUs stapled to MCUs (Ethos-U55) are close behind.
- Pure Cortex-M is fine for latency (sub-10 ms is doable) but bleeds
  energy compared to specialised NPUs.

[↑ Back to Top](#-edgeai--tinyml--theory)

---

## 9. The full TinyML workflow

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                 TinyML end-to-end                                │
 │                                                                   │
 │  1. Collect data   (sensor logs; Edge Impulse CLI)                │
 │  2. Pick task      (KWS / VWW / anomaly / IC)                     │
 │  3. Design model   (DS-CNN / MobileNet-tiny / autoencoder)        │
 │  4. Train          (Keras / PyTorch; standard desktop)            │
 │  5. INT8 quant     (full-integer PTQ, representative dataset)     │
 │  6. Export tflite  (converter; verify size < flash)               │
 │  7. Compile        (Vela if Ethos-U55; `xxd -i` to C array)       │
 │  8. Integrate      (TFLM main.cc, define arena, ops resolver)     │
 │  9. Flash & test   (on-device accuracy ≈ desktop?)                │
 │ 10. Measure        (latency, energy via ULPMark or scope)         │
 │ 11. Iterate        (prune, smaller input, different arch)         │
 └─────────────────────────────────────────────────────────────────┘
```

Most TinyML teams spend 60 % of the time on **steps 1, 10, 11** —
data collection and physical measurement. The training is the easy
part.

[↑ Back to Top](#-edgeai--tinyml--theory)

---

## 10. Cheat sheet & red flags

### Cheat sheet
```
 WHAT             AI on MCUs — KB-scale models at mW-scale power
 4 TASKS          KWS · VWW · anomaly · IC
 RAM SCALE        64 KB – 1 MB arena
 FLASH SCALE      100 KB – 4 MB code + model
 QUANT            mandatory full-integer INT8
 LIBRARIES        TFLite Micro + CMSIS-NN (+ Ethos-U kernel)
 COMPILER         Vela (Ethos-U), xxd -i (TFLite → C array)
 PLATFORM         Edge Impulse for fast prototyping
 BENCH            MLPerf Tiny (closed + open)
 SHIPPING UNIT    1 × firmware image = app + TFLM + model
```

### Red flags 🚩
- 🚩 Using `malloc` anywhere in the inference path.
- 🚩 Float-fallback ops in a TinyML model — they blow the arena.
- 🚩 Arena size guessed, not measured.
- 🚩 CMSIS-NN installed but `TF_LITE_USE_CMSIS_NN` flag missing —
  you get 2–5× less speed for free.
- 🚩 Training data collected on a desktop microphone and deployed on
  a $0.50 MEMS mic with no re-calibration.

### Green flags ✅
- ✅ `arena_used_bytes()` logged in CI.
- ✅ Reference on-device dataset with 100+ real samples per class.
- ✅ Pre-recorded **golden test vectors** that fail the build if
  accuracy drifts by > X %.
- ✅ Vela op-report committed alongside the `.tflite`.

---

## 🔭 Next up

Next folder [`Federated_Learning/`](../Federated_Learning/README.md) —
keeping all this on-device data **private** while still learning from
it collectively.

---

> *GPU Programming · EdgeAI · TinyML · THEORY · github.com/rpaut03l/TS-02*
