# 💻 EdgeAI · Hardware — CODE

### *Enumerate accelerators, benchmark across them, estimate energy per inference*

> **Nav:** [← Hardware README](README.md) | [📖 THEORY](edge_ai_hardware_theory.md) | **CODE** | [🎯 PRACTICE →](edge_ai_hardware_practice.md)

---

## Ex 1 — Enumerate ONNX Runtime providers

### 👶 What this does
**ONNX Runtime** is the "universal adapter" of Edge AI. A single model
file, one API, many accelerators. The *available providers* tell you
which chips the host exposes.

```python
# pip install onnxruntime onnxruntime-gpu  (pre-installed on Colab)
import onnxruntime as ort
print("ONNX Runtime version   :", ort.__version__)
print("Available providers    :")
for p in ort.get_available_providers():
    print(" -", p)
```

Typical Colab (T4) output:
```
ONNX Runtime version   : 1.17.1
Available providers    :
 - TensorrtExecutionProvider
 - CUDAExecutionProvider
 - CPUExecutionProvider
```

On a Jetson you'd also see `TensorrtExecutionProvider`. On a laptop
with Intel iGPU + NPU and OpenVINO installed, you'd see
`OpenVINOExecutionProvider`. On a Qualcomm SoC: `QNNExecutionProvider`.

---

## Ex 2 — Enumerate OpenVINO devices (CPU / iGPU / NPU)

### 👶 What this does
On Intel hardware, **OpenVINO** decides whether your model runs on
CPU, iGPU, or NPU. `core.available_devices` tells you the menu.

```bash
pip install openvino
```

```python
from openvino import Core
core = Core()
print("OpenVINO devices:", core.available_devices)
for d in core.available_devices:
    print(" -", d, core.get_property(d, "FULL_DEVICE_NAME"))
```

Typical Core Ultra (Meteor Lake) output:
```
OpenVINO devices: ['CPU', 'GPU', 'NPU']
 - CPU   Intel(R) Core(TM) Ultra 7 155H
 - GPU   Intel(R) Arc(TM) Graphics (iGPU)
 - NPU   Intel(R) AI Boost
```

Colab will just show `['CPU']`. That's fine — the shape of the API is
what matters.

---

## Ex 3 — Run the same model across all available providers

### 👶 What this does
Benchmark **one model** on **every** backend the host exposes. This
is how you decide, in production, *where* to route inference.

```python
import onnxruntime as ort, numpy as np, time, urllib.request, os

# Download a small ONNX model (MobileNetV2 from the ONNX Model Zoo)
URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12-int8.onnx"
if not os.path.exists("mobilenetv2.onnx"):
    urllib.request.urlretrieve(URL, "mobilenetv2.onnx")

x = np.random.rand(1, 3, 224, 224).astype("float32")

def bench(provider, iters=100):
    sess = ort.InferenceSession("mobilenetv2.onnx",
                                providers=[provider, "CPUExecutionProvider"])
    # warm-up
    for _ in range(10):
        sess.run(None, {sess.get_inputs()[0].name: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {sess.get_inputs()[0].name: x})
    return (time.perf_counter() - t0) / iters * 1000

for p in ort.get_available_providers():
    try:
        ms = bench(p)
        print(f"{p:30s} {ms:7.2f} ms/inference")
    except Exception as e:
        print(f"{p:30s} skipped ({e.__class__.__name__})")
```

Typical Colab output:
```
TensorrtExecutionProvider       2.10 ms/inference
CUDAExecutionProvider           3.40 ms/inference
CPUExecutionProvider           15.00 ms/inference
```

> The **same model**, same process, chose between the CUDA path (GPU)
> and the CPU path. On real Jetson / iGPU / NPU hardware, the same
> code just routes to a different provider.

---

## Ex 4 — Simulate an NPU: INT8 path on CPU

### 👶 What this does
We don't have a real Coral / Hailo on Colab. But we can **simulate**
an NPU-class workload by running the fully-INT8 TFLite model on the
CPU, which is how Hexagon / ANE typically execute.

```python
import tensorflow as tf, numpy as np, time, os

# Use the INT8 MobileNetV2 you built in the Fundamentals notebook
#   (catdog_int8.tflite) OR quantize this one.
# Quick re-quant from the ONNX you just downloaded → out of scope here;
# for this exercise, reuse the earlier catdog_int8.tflite file.

path = "catdog_int8.tflite"
if os.path.exists(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=2)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]

    shape = in_det["shape"]
    x_q = np.random.randint(-128, 128, shape, dtype=np.int8)

    for _ in range(10):
        interp.set_tensor(in_det["index"], x_q)
        interp.invoke()

    t0 = time.perf_counter()
    for _ in range(200):
        interp.set_tensor(in_det["index"], x_q)
        interp.invoke()
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"Simulated NPU (INT8 TFLite CPU): {ms:.2f} ms/inference")
else:
    print("Run the Fundamentals notebook first to build catdog_int8.tflite")
```

This gives you a number you can compare against Cell 3's providers.
On a real NPU, the INT8 path is usually **2–5×** faster than what CPU
TFLite shows here.

---

## Ex 5 — Energy per inference (rough estimate)

### 👶 What this does
Edge chips are judged by **energy per inference**, not just latency.
We multiply chip TDP (in watts) by measured latency (in seconds) to
get **joules** per inference.

```python
# Rough TDPs while busy for each provider / family
CHIP_TDP_W = {
    "CPU (laptop)"          : 25,
    "Tesla T4"              : 70,
    "Jetson Orin Nano 15 W" : 15,
    "Jetson AGX Orin 60 W"  : 60,
    "Qualcomm Hexagon (NPU)": 3,
    "Apple Neural Engine"   : 2,
    "Google Coral USB"      : 2,
}

# Assumed latency per chip (ms/inference) for the same MobileNetV2
LATENCY_MS = {
    "CPU (laptop)"          : 15.0,
    "Tesla T4"              : 3.4,
    "Jetson Orin Nano 15 W" : 8.0,
    "Jetson AGX Orin 60 W"  : 1.4,
    "Qualcomm Hexagon (NPU)": 2.5,
    "Apple Neural Engine"   : 2.0,
    "Google Coral USB"      : 12.0,
}

print(f"{'Chip':28s} {'ms':>8s} {'W':>5s} {'mJ/inf':>10s} {'inf/J':>10s}")
for chip in LATENCY_MS:
    ms = LATENCY_MS[chip]
    W  = CHIP_TDP_W[chip]
    mJ = (ms / 1000) * W * 1000          # mJ per inference
    per_joule = 1000 / mJ                # inferences per joule
    print(f"{chip:28s} {ms:8.2f} {W:5.1f} {mJ:10.2f} {per_joule:10.1f}")
```

Typical output:
```
Chip                               ms     W     mJ/inf      inf/J
CPU (laptop)                    15.00  25.0     375.00        2.7
Tesla T4                         3.40  70.0     238.00        4.2
Jetson Orin Nano 15 W            8.00  15.0     120.00        8.3
Jetson AGX Orin 60 W             1.40  60.0      84.00       11.9
Qualcomm Hexagon (NPU)           2.50   3.0       7.50      133.3
Apple Neural Engine              2.00   2.0       4.00      250.0
Google Coral USB                12.00   2.0      24.00       41.7
```

### The lesson
A **mobile NPU** produces **10–100× more inferences per joule** than a
discrete GPU — even though the GPU is "faster" on raw ms. This is why
phones are ruthless about dispatching to the NPU first.

---

## Ex 6 — Parse MLPerf Tiny results

### 👶 What this does
MLPerf publishes closed results as CSV. Parsing them is how you do
quantitative chip comparisons in a design review.

```python
# Minimal example: hand-built pseudo-MLPerf rows for the keyword-
# spotting task (DS-CNN). In real life you'd download the CSV from
# mlcommons.org/benchmarks/inference-tiny
import pandas as pd
rows = [
    dict(system="Syntiant NDP120",   latency_ms=1.80, energy_uJ=35),
    dict(system="Alif Ensemble E7",  latency_ms=3.50, energy_uJ=90),
    dict(system="STM32H747 M7",      latency_ms=18.2, energy_uJ=800),
    dict(system="ESP32-S3",          latency_ms=24.0, energy_uJ=1200),
    dict(system="Coral Dev Board",   latency_ms=2.00, energy_uJ=180),
]
df = pd.DataFrame(rows)
df["inf_per_joule"] = 1_000_000 / df["energy_uJ"]
df.sort_values("inf_per_joule", ascending=False)
```

Run this — the tiny, dedicated NPUs dominate "inferences per joule",
and that's the shape of every MLPerf Tiny leaderboard you will read.

---

## Ex 7 — Estimate thermal throttle over time

### 👶 What this does
We simulate a fanless Jetson that boots at 100 FPS and throttles by
20 % as it warms up. This is what *happens* to an edge device after
a few minutes of load.

```python
import numpy as np, matplotlib.pyplot as plt

t = np.arange(0, 600, 1)                 # 10 minutes, 1-second ticks
cold_fps = 100
# Exponential approach to 80 % of peak with 120 s time-constant
fps = cold_fps * (0.80 + 0.20 * np.exp(-t / 120))

plt.figure()
plt.plot(t, fps)
plt.xlabel("Time (s)"); plt.ylabel("FPS")
plt.title("Fanless edge device — thermal throttling")
plt.grid(True); plt.show()
```

Two takeaways:
1. If you benchmark during the **first 30 seconds**, you overstate
   the sustained FPS by 25 %.
2. Specs should always report **sustained** FPS, not peak.

---

## 📝 Summary — what you've done

| Step | Result |
|---|---|
| Listed ONNX Runtime providers | Knew which chips were available |
| Listed OpenVINO devices | CPU / iGPU / NPU on Intel hosts |
| Benchmarked across providers | Real ms/inference per backend |
| Simulated NPU-class INT8 path | Rough stand-in for Coral / Hexagon |
| Computed energy per inference | Put **inf/joule** on the map |
| Parsed MLPerf-style rows | Practiced quantitative chip compare |
| Simulated thermal throttle | Learned why sustained ≠ peak |

Now go to [practice.md](edge_ai_hardware_practice.md) and build the
full "hardware fit chart" that ranks every edge chip under your
workload + thermal cap.

---

> *GPU Programming · EdgeAI · Hardware · CODE · github.com/rpaut03l/TS-02*
