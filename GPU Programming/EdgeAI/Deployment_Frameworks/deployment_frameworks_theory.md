# 📖 EdgeAI · Deployment Frameworks — THEORY

### *TFLite · ONNX Runtime · OpenVINO — how they work and when to use which*

> **Nav:** [← Deployment Frameworks README](README.md) | **THEORY** | [💻 CODE](deployment_frameworks_code.md) | [🎯 PRACTICE](deployment_frameworks_practice.md)

---

## 🧠 MNEMONIC: **"T-O-O"**

> **T**FLite · **O**NNX Runtime · **O**penVINO

Three runtimes. Each fits one slice of the market best. Learn all three
once and you can deploy to almost anything.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Why we need a separate runtime | [§1](#1-why-we-need-a-separate-runtime) |
| 2 | **ONNX** — the lingua franca | [§2](#2-onnx--the-lingua-franca) |
| 3 | **TensorFlow Lite** | [§3](#3-tensorflow-lite) |
| 4 | **ONNX Runtime** | [§4](#4-onnx-runtime) |
| 5 | **OpenVINO** | [§5](#5-openvino) |
| 6 | Side-by-side — coverage, speed, ecosystem | [§6](#6-side-by-side--coverage-speed-ecosystem) |
| 7 | The portable deployment pattern | [§7](#7-the-portable-deployment-pattern) |
| 8 | Common export gotchas | [§8](#8-common-export-gotchas) |
| 9 | Cheat sheet | [§9](#9-cheat-sheet--red-flags) |

---

## 1. Why we need a separate runtime

### 👶 Easy Story
PyTorch and TensorFlow are like **big kitchens** with every tool —
knives, blenders, ovens, spatulas. Perfect for cooking (training),
overkill for **serving** (inference). You wouldn't carry an entire
kitchen to a food truck. You'd carry a **portable grill** — just the
bits that heat food fast.

The inference runtime is that portable grill:

- **Smaller binary** (MBs vs GBs).
- **No training-only code** (autodiff, optimizers, data loaders).
- **Accelerator-aware** (knows how to dispatch to NNAPI / Hexagon /
  CoreML / TensorRT / OpenVINO).
- **Deterministic** — same bits in → same bits out, every time.

### The numbers (approx)
| Stack | Binary size | Cold start | Memory |
|---|---|---|---|
| PyTorch + Python | 1.5 GB | 2–5 s | 300 MB+ |
| TensorFlow full | 700 MB | 1–2 s | 250 MB+ |
| TFLite | **1–3 MB** | 50 ms | 10–30 MB |
| ONNX Runtime (base) | **~10 MB** | 100 ms | 20–40 MB |
| OpenVINO runtime | **~40 MB** | 150 ms | 30–60 MB |
| TFLite Micro | **~100 KB** | 1 ms | **< 1 MB** |

[↑ Back to Top](#-edgeai--deployment-frameworks--theory)

---

## 2. ONNX — the lingua franca

### 👶 Easy Story
Pretend every country speaks its own language (PyTorch, TF, JAX,
Keras). ONNX is **English** — a shared language every runtime
understands. You translate *once* from your home language to English;
then every device in the world can read it.

### Formal
**ONNX (Open Neural Network Exchange)** = protobuf-based file format
for neural-network graphs. It specifies:

- A fixed set of **operators** (conv, matmul, softmax, …) with a
  versioned **opset**.
- A standard **tensor layout** (NCHW is the default).
- Metadata (input/output shapes, model version, domain).

### The export flow

```
 PyTorch / TF / JAX  ──►  model.onnx  ──►  ANY runtime
                         (portable)
                         ├── ORT (CPU / CUDA / TRT / OpenVINO)
                         ├── TFLite (via onnx-tensorflow → TFLite)
                         ├── OpenVINO (via Model Optimizer)
                         ├── Core ML (via coremltools)
                         ├── NCNN / MNN / TNN (mobile)
                         └── Vendor SDKs (Hailo, SNPE, Hexagon)
```

### Why it matters
One training pipeline → one `.onnx` → many deployment targets.
This is the **single biggest leverage** you have in edge AI. If your
training code can't export to ONNX, your deployment story is
broken from day one.

[↑ Back to Top](#-edgeai--deployment-frameworks--theory)

---

## 3. TensorFlow Lite

### 👶 Easy Story
TFLite is Google's **tiny eater of TF models**. It takes a
`SavedModel`, crunches it down into a flat `.tflite` file, and runs
it anywhere — phones, microcontrollers, Raspberry Pi, even the
browser.

### The architecture

```
 ┌────────────────┐      ┌──────────────┐      ┌──────────────┐
 │  TF SavedModel │ ───► │  Converter    │ ───► │  .tflite     │
 │  (or Keras)    │      │  (+ optimize, │      │  flatbuffer  │
 │                │      │   quantize)   │      │  file        │
 └────────────────┘      └──────────────┘      └──────────────┘
                                                      │
                                                      ▼
                            ┌────────────────────────────────────────────┐
                            │               TFLite Interpreter            │
                            │  ┌──────┐ ┌──────┐ ┌──────┐ ┌───────────┐   │
                            │  │ CPU  │ │ GPU  │ │NNAPI │ │ Hexagon / │   │
                            │  │ XNNP │ │delg. │ │delg. │ │ Core ML / │   │
                            │  │ ACK  │ │      │ │      │ │ Edge TPU  │   │
                            │  └──────┘ └──────┘ └──────┘ └───────────┘   │
                            └────────────────────────────────────────────┘
```

### Delegates — TFLite's "plug-in accelerators"
A **delegate** is a module that takes over execution of (some) ops on
a specific chip. The converter marks which ops can be delegated; at
runtime the dispatcher sends them to the chosen accelerator and runs
the rest on CPU.

| Delegate | Target chip | Typical speed-up |
|---|---|---|
| **XNNPACK** (default CPU) | Any ARM / x86 CPU | 1.5–3× over unoptimised |
| **GPU delegate** | OpenCL / OpenGL / Metal | 2–10× |
| **NNAPI** | Android 8+ — routes to OEM HAL | 2–10× (depends on OEM) |
| **Hexagon** | Qualcomm DSP / NPU | 5–15× (if supported ops) |
| **Core ML** (via converter) | Apple Neural Engine | 5–20× |
| **Edge TPU** | Google Coral | 10–40× |

### Ops & "select TF ops"
TFLite's **built-in op set** covers ~90 % of vision and speech models,
but not everything. Two escape hatches:

- **Select TF ops** — allow the model to include full TF ops
  (larger binary, slower). Used when a CustomOp you need isn't in
  TFLite built-ins.
- **Custom ops** — register your own C++ op. Used in production when
  speed matters more than binary size.

### TFLite Micro
A stripped-down TFLite for microcontrollers — **no OS, no malloc**,
just a static **memory arena** you size at compile time. Covered in
[TinyML/](../TinyML/README.md).

### Pros / Cons

| ✅ Pros | ❌ Cons |
|---|---|
| Smallest mobile footprint | TF-centric (PyTorch → TF conversion is needed) |
| Best Android story (NNAPI, Play Services) | Some ops missing, select-TF ops is heavy |
| Full-integer INT8 is first-class | OpenCL GPU path is fussy vs Metal |
| Strong tools (benchmark tool, model analyzer) | Less popular outside mobile |

[↑ Back to Top](#-edgeai--deployment-frameworks--theory)

---

## 4. ONNX Runtime

### 👶 Easy Story
ONNX Runtime is the **universal adapter**. You give it an `.onnx`
file and tell it which chip to run on. It figures out the rest.

### The architecture

```
 ┌──────────────────────────────────────────────────────────────┐
 │                        ONNX Runtime                           │
 │                                                                │
 │  ┌────────────┐       ┌─────────────────────────────────┐    │
 │  │  Graph     │  ──►  │         Execution Providers      │    │
 │  │  Optimizer │       ├──────────────┬──────────────────┤    │
 │  └────────────┘       │  CUDA EP     │ TensorRT EP      │    │
 │         ▲             │  OpenVINO EP │ CoreML EP        │    │
 │  model.onnx           │  QNN EP      │ DirectML EP      │    │
 │                       │  ROCm EP     │ CPU EP (default) │    │
 │                       └──────────────┴──────────────────┘    │
 └──────────────────────────────────────────────────────────────┘
```

### Execution Providers (EPs)
An **EP** is a backend that handles a sub-graph of the ONNX model.
Specify one or many; ORT greedily assigns ops to the first EP in your
list that supports them, falls back to `CPUExecutionProvider`
for the rest.

| EP | Target | Highlight |
|---|---|---|
| **CPUExecutionProvider** | Any CPU | Default, always present |
| **CUDAExecutionProvider** | NVIDIA GPU | Same as PyTorch target |
| **TensorRTExecutionProvider** | NVIDIA GPU | 2–4× faster than CUDA EP |
| **OpenVINOExecutionProvider** | Intel CPU / iGPU / NPU | Intel sweet spot |
| **CoreMLExecutionProvider** | Apple (macOS/iOS) | ANE dispatch |
| **QNNExecutionProvider** | Qualcomm NPU / DSP | Hexagon acceleration |
| **DirectMLExecutionProvider** | Any DirectX 12 GPU | Windows / cross-vendor |
| **ROCmExecutionProvider** | AMD GPU | HIP backend |
| **DmlExecutionProvider** | WebNN / browser | Edge-in-the-browser |

### Session options that matter
- `SessionOptions.graph_optimization_level` = `BASIC / EXTENDED / ALL`.
- `SessionOptions.intra_op_num_threads` — parallelism within an op.
- `SessionOptions.inter_op_num_threads` — parallelism across ops.
- `SessionOptions.enable_mem_pattern` — reuse buffers across runs.
- **IOBinding** — pre-allocate GPU tensors and avoid host↔device copies
  on every call.

### Built-in quantization
ORT ships a **quantization API** that can produce INT8 / INT4 ONNX
models directly, with per-channel, static, dynamic, and QDQ modes.
Nicely interchangeable with PyTorch AO quant output.

### Pros / Cons

| ✅ Pros | ❌ Cons |
|---|---|
| Truly cross-platform | Slightly larger binary than TFLite |
| Broadest hardware coverage via EPs | Some EPs need vendor driver stacks |
| Strong CPU performance on x86 & ARM | Mobile binary bigger than TFLite |
| Same API on Python / C++ / C# / JS | Not all ops cover every EP |

[↑ Back to Top](#-edgeai--deployment-frameworks--theory)

---

## 5. OpenVINO

### 👶 Easy Story
OpenVINO is **Intel's kitchen-in-a-box**. If you're running on Intel
CPUs, Intel iGPUs, or the Intel NPU (Core Ultra), OpenVINO is almost
always faster and uses less power than anything else — because Intel
wrote it for their silicon.

### The architecture

```
 ┌─────────────┐    ┌───────────────────┐    ┌───────────────────┐
 │ onnx / TF / │──► │  Model Optimizer  │──► │       .xml +      │
 │ PyTorch     │    │  (convert + quant)│    │       .bin        │
 │ (Keras,     │    │                   │    │    (IR format)    │
 │ PaddlePaddle)│   └───────────────────┘    └───────────────────┘
 └─────────────┘                                         │
                                                         ▼
                                ┌───────────────────────────────────────────┐
                                │            OpenVINO Runtime                │
                                │                                            │
                                │  Devices: CPU · GPU · NPU · AUTO · MULTI · │
                                │           HETERO                           │
                                └───────────────────────────────────────────┘
```

### IR format
**Intermediate Representation** — two files:
- `.xml` — topology (op graph)
- `.bin` — weights

Much smaller than ONNX because Intel applies aggressive pre-processing.

### Device plugins
- **CPU** — all Intel CPUs, uses oneDNN under the hood.
- **GPU** — Intel iGPU / Arc dGPU, uses oneAPI L0.
- **NPU** — Intel AI Boost (Core Ultra).
- **AUTO** — runtime picks the best device.
- **MULTI** — split a network across multiple devices.
- **HETERO** — same as MULTI but per-op-granularity.

### NNCF — OpenVINO's quantization / pruning toolkit
Full QAT, PTQ (DefaultQuantization, AccuracyAwareQuantization),
pruning, filter pruning, sparsity. Similar in spirit to TFMOT for TF.

### Pros / Cons

| ✅ Pros | ❌ Cons |
|---|---|
| Best on Intel silicon (often 2× vs ORT CPU) | Intel-only (CPU/iGPU/NPU) |
| AUTO / MULTI / HETERO are genuinely useful | IR format is vendor-specific |
| NNCF is strong for INT8 / sparsity | Smaller community than TF/ORT |
| C++ / Python / REST API | Less documentation outside Intel sites |

[↑ Back to Top](#-edgeai--deployment-frameworks--theory)

---

## 6. Side-by-side — coverage, speed, ecosystem

```
 ┌───────────────────┬─────────┬─────────┬─────────┐
 │ Axis              │ TFLite  │   ORT   │ OpenVINO│
 ├───────────────────┼─────────┼─────────┼─────────┤
 │ Binary size       │  ★★★★★  │  ★★★☆☆  │  ★★☆☆☆  │
 │ Ops coverage      │  ★★★★☆  │  ★★★★★  │  ★★★☆☆  │
 │ Android support   │  ★★★★★  │  ★★★★☆  │  ★☆☆☆☆  │
 │ iOS / ANE         │  ★★★★☆  │  ★★★☆☆  │  ☆☆☆☆☆  │
 │ NVIDIA Jetson     │  ★★☆☆☆  │  ★★★★★  │  ☆☆☆☆☆  │
 │ Intel CPU / iGPU  │  ★★★☆☆  │  ★★★★☆  │  ★★★★★  │
 │ MCU / TinyML      │  ★★★★★  │  ★☆☆☆☆  │  ☆☆☆☆☆  │
 │ Quantization API  │  ★★★★★  │  ★★★★☆  │  ★★★★★  │
 │ Framework support │  TF+JAX │  ALL    │  ALL    │
 │ C++ / Python API  │  ★★★★☆  │  ★★★★★  │  ★★★★★  │
 │ Browser / WebNN   │  ★★★☆☆  │  ★★★★★  │  ☆☆☆☆☆  │
 └───────────────────┴─────────┴─────────┴─────────┘
```

### Typical "best on each target"

| Target | First pick | Why |
|---|---|---|
| Android phone | TFLite + NNAPI / Hexagon | Native OS path |
| iOS phone | Core ML (from ONNX via `coremltools`) | ANE dispatch |
| Jetson | ONNX Runtime + TensorRT EP (or TRT direct) | NVIDIA sweet spot |
| Intel iGPU / NPU laptop | OpenVINO | Intel-tuned |
| Generic x86 server | ORT CPU or OpenVINO CPU | Similar; benchmark both |
| Cortex-M MCU | TFLite Micro | Only viable choice |
| Browser / WebNN | ORT Web | Best cross-vendor |

[↑ Back to Top](#-edgeai--deployment-frameworks--theory)

---

## 7. The portable deployment pattern

### 👶 Easy Story
Train once. Export to **ONNX**. Then fan out — convert to TFLite,
OpenVINO IR, Core ML — one per target. CI builds all three from the
same ONNX. Devices download only the artifact they need.

### The pipeline

```
                    ┌──────────────────┐
                    │  Training code    │
                    │  (PyTorch / TF)   │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │    model.onnx     │   ← single source of truth
                    └──────┬──┬──┬─────┘
                           │  │  │
              ┌────────────┘  │  └────────────┐
              ▼               ▼               ▼
      ┌─────────────┐ ┌───────────────┐ ┌───────────────┐
      │  .tflite    │ │  .xml / .bin  │ │  .mlpackage   │
      │  (TFLite)   │ │  (OpenVINO IR)│ │  (Core ML)    │
      └─────────────┘ └───────────────┘ └───────────────┘
              │              │                │
              ▼              ▼                ▼
       Android phone   Intel edge PC     iPhone / Mac
```

### Rules that save your sanity
1. **Test the ONNX locally** before converting downstream.
2. Keep a **golden numeric check** — feed the same input through
   PyTorch and through every runtime, compare outputs to 1e-3.
3. Store the **opset version** in model metadata. Runtime mismatches
   are the #1 cause of weird bugs.
4. Automate conversion in **CI**, not on the developer's laptop.

[↑ Back to Top](#-edgeai--deployment-frameworks--theory)

---

## 8. Common export gotchas

- **Dynamic shapes** — set `dynamic_axes={"input": {0: "batch"}}` on
  `torch.onnx.export`. Without it, the model is fixed-batch.
- **Control flow** — `if`/`while` on tensors need `torch.onnx.export`
  with `opset >= 14`. Sometimes rewrite in tensor terms.
- **Unsupported ops** — `nn.functional.grid_sample`, custom autograd
  functions, or exotic losses may need manual substitution.
- **BatchNorm in eval mode** — always `model.eval()` before export.
- **Python lambdas** — inline `lambda x: x * 2` inside a forward will
  silently break tracing. Use real modules.
- **NHWC vs NCHW** — TFLite prefers NHWC; ONNX is NCHW. TFLite
  converter handles transposition, but latency spikes if it fails to
  fuse the transpose.
- **Opset version mismatch** — pin opset in export, pin opset in
  runtime. Don't mix 13 + 17.
- **FP16 only on some layers** — mixed-precision export can lead to
  half-precision BatchNorm epsilon → NaN. Export FP32 then let the
  runtime cast.

[↑ Back to Top](#-edgeai--deployment-frameworks--theory)

---

## 9. Cheat sheet & red flags

### Cheat sheet
```
 LINGUA FRANCA   ONNX (opset 17+ recommended)
 SIZE RANKING    TFLite < ORT < OpenVINO
 ANDROID         TFLite + NNAPI/Hexagon delegate
 iOS             Core ML (from ONNX via coremltools)
 JETSON          ORT + TensorRT EP  (or TensorRT direct)
 INTEL           OpenVINO
 BROWSER         ORT Web / WebNN / WebGPU backends
 MCU             TFLite Micro (see TinyML/)
 GOLDEN CHECK    max |torch(x) - runtime(x)| < 1e-3
 OPSET HYGIENE   pin on export, pin at runtime
```

### Red flags 🚩
- 🚩 Training code that can't export cleanly to ONNX → rewrite before
  it gets worse.
- 🚩 One-off conversion scripts that only run on "Alice's laptop".
  Bake conversion into CI.
- 🚩 Using **select-TF-ops** for one layer and forgetting that your
  binary just doubled in size.
- 🚩 Mixing runtimes on the same device without measuring memory
  overhead.
- 🚩 Skipping the **numeric golden check** — you'll debug "99 % on
  desktop, 40 % on device" for weeks.

### Green flags ✅
- ✅ One `model.onnx` in the release artifact list.
- ✅ `convert_all.sh` that spits out `.tflite`, `.xml/.bin`, `.mlpackage`.
- ✅ Per-runtime CI job that compares outputs against a saved
  reference.
- ✅ A **supported-ops smoke test** that fails loudly when a new op
  is introduced.

---

## 🔭 Next up

You now know the three big runtimes. Next folder [`TinyML/`](../TinyML/README.md)
zooms into the **smallest** end of the deployment spectrum —
microcontrollers running `.tflite` files in kilobytes.

---

> *GPU Programming · EdgeAI · Deployment Frameworks · THEORY · github.com/rpaut03l/TS-02*
