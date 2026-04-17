# 💻 EdgeAI · GPU Types — CODE

### *Detect the GPU you're on, read the spec sheet programmatically, benchmark across families*

> **Nav:** [← GPU Types README](README.md) | [📖 THEORY](edge_gpu_varieties_theory.md) | **CODE** | [🎯 PRACTICE →](edge_gpu_varieties_practice.md)

---

## 🏗️ Setup

All snippets run on **Google Colab (T4 GPU)**. Where Jetson-only
commands are shown, the output is included so you see what it looks
like — you don't need a real Jetson to read along.

---

## Ex 1 — What family am I on?

### 👶 What this does
Before anything else, identify the GPU. The **family** (Jetson /
discrete / iGPU / mobile) determines every tool you'll use.

```python
import subprocess, platform, os, re

def detect_gpu_family():
    # 1. Jetson-specific marker
    if os.path.exists("/etc/nv_tegra_release"):
        with open("/etc/nv_tegra_release") as f:
            return "JETSON", f.readline().strip()

    # 2. Any NVIDIA discrete GPU?
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name",
                                       "--format=csv,noheader"]).decode()
        name = out.strip().splitlines()[0]
        # crude heuristic: Tesla / T4 / L4 / RTX A / RTX x000 Ada → discrete edge
        if re.search(r"(Tesla|T4|L4|RTX A\d000|RTX \d000 Ada|RTX \d000)", name):
            return "DISCRETE_EDGE", name
        return "DISCRETE_NVIDIA", name
    except Exception:
        pass

    # 3. Intel / AMD iGPU
    try:
        out = subprocess.check_output(["lspci", "-nn"]).decode()
        if "Intel" in out and "VGA" in out:
            return "iGPU_INTEL", "Intel integrated GPU"
        if "AMD" in out and "VGA" in out:
            return "iGPU_AMD", "AMD integrated GPU"
    except Exception:
        pass

    return "UNKNOWN", platform.platform()

family, detail = detect_gpu_family()
print(f"Family : {family}")
print(f"Detail : {detail}")
```

Typical Colab output:
```
Family : DISCRETE_EDGE
Detail : Tesla T4
```

---

## Ex 2 — Read the NVIDIA spec sheet

### 👶 What this does
Ask the GPU what it is — cores, memory, power, driver — using NVML
(the library behind `nvidia-smi`).

```python
# pynvml comes pre-installed with the NVIDIA driver
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)

name     = pynvml.nvmlDeviceGetName(h)
mem      = pynvml.nvmlDeviceGetMemoryInfo(h)
cap      = pynvml.nvmlDeviceGetCudaComputeCapability(h)
tdp_mW   = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h)

print(f"Name            : {name}")
print(f"Compute Cap.    : {cap[0]}.{cap[1]}")
print(f"Memory total    : {mem.total / 1e9:.2f} GB")
print(f"Memory free     : {mem.free / 1e9:.2f} GB")
print(f"Default TDP     : {tdp_mW / 1000:.1f} W")
pynvml.nvmlShutdown()
```

Typical Colab output:
```
Name            : Tesla T4
Compute Cap.    : 7.5
Memory total    : 15.84 GB
Memory free     : 15.17 GB
Default TDP     : 70.0 W
```

### 👶 How to read "Compute Capability"
- **7.5** = Turing (T4).
- **8.0** = Ampere (A100).
- **8.6** = Ampere (RTX A2000 / consumer 30-series).
- **8.7** = Jetson Orin.
- **8.9** = Ada Lovelace (RTX 4000 Ada SFF / RTX 40-series).
- **9.0** = Hopper (H100).

Higher number → newer feature set (TF32, FP8, INT4, transformer engine).

---

## Ex 3 — PyTorch / TensorFlow view

```python
import torch, tensorflow as tf

print("PyTorch view")
print("  CUDA available:", torch.cuda.is_available())
print("  Device name   :", torch.cuda.get_device_name(0))
print("  Capability    :", torch.cuda.get_device_capability(0))
print("  Multi-processors:", torch.cuda.get_device_properties(0).multi_processor_count)

print("\nTensorFlow view")
for d in tf.config.list_physical_devices("GPU"):
    print(" ", d, tf.config.experimental.get_device_details(d))
```

On Jetson, `multi_processor_count` is much smaller (8–16 SMs) than on a
T4 (40 SMs) or H100 (132 SMs). That number × 128 ≈ CUDA core count for
Ampere-class parts.

---

## Ex 4 — Benchmark the same model across "simulated" edge GPUs

### 👶 What this does
We can't rent every edge GPU, but we can **simulate** them by limiting
Colab's T4 to different power/compute budgets using
`torch.cuda.set_per_process_memory_fraction` and manual throttling —
and we can scale the batch size so the arithmetic intensity matches
what each family would see.

```python
import torch, time
device = torch.device("cuda")

# A MobileNetV2-sized model
model = torch.hub.load("pytorch/vision", "mobilenet_v2",
                       weights=None).to(device).eval()

def bench(batch, iters=100):
    x = torch.randn(batch, 3, 224, 224, device=device)
    # warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

# Approximate the batch size each family can cope with in memory
scenarios = [
    ("Jetson Orin Nano  (simulated)", 1),
    ("Jetson Orin NX    (simulated)", 2),
    ("Jetson AGX Orin   (simulated)", 8),
    ("Tesla T4          (real)",      16),
    ("RTX 4000 Ada SFF  (simulated)", 24),
]

print(f"{'Scenario':32s} {'Batch':>6s} {'ms/batch':>10s} {'img/s':>8s}")
for label, bs in scenarios:
    ms = bench(bs)
    print(f"{label:32s} {bs:>6d} {ms:10.2f} {1000/ms*bs:8.0f}")
```

The **shape** of the output teaches the lesson even if the absolute
numbers are Colab-T4 dependent: bigger batches use the GPU better, but
an Orin Nano with only 8 GB RAM can't hold that batch — so in real life
you'd run smaller batches there.

---

## Ex 5 — Using `tegrastats` (Jetson only)

### 👶 What this does
On a real Jetson, **`tegrastats`** is the one tool you will live in —
it prints real-time CPU/GPU/RAM/power every second.

Run on a Jetson:
```bash
sudo tegrastats --interval 1000
```

Typical single line of output (Orin NX, 15 W mode, under load):
```
RAM 4518/15872MB  SWAP 0/7936MB (cached 0MB)
CPU [12%@1984,8%@1984,2%@1984,0%@1984,0%@1984,0%@1984]
GR3D_FREQ 89%@[828,828]
VDD_CPU_CV 1520mW   VDD_SOC 2040mW
POM_5V_IN 7120mW
```

### 👶 How to read it
- **RAM** — used / total (unified between CPU and GPU!).
- **CPU [12%@1984]** — each ARM core's load % and frequency.
- **GR3D_FREQ 89%** — GPU utilisation (higher = better).
- **POM_5V_IN** — **total board power** in mW. 7.1 W total while busy
  on a 15 W part = good thermal headroom.

### Automating on Jetson

```python
# Python-side parser you'd use on the device
import subprocess, re

def read_tegrastats_once():
    out = subprocess.check_output(["tegrastats", "--interval", "500",
                                   "--count", "1"]).decode()
    m = re.search(r"GR3D_FREQ (\d+)%", out)
    p = re.search(r"POM_5V_IN (\d+)mW", out)
    return int(m.group(1)) if m else None, int(p.group(1)) if p else None

# gpu_util_pct, board_mW = read_tegrastats_once()
```

`jtop` (`sudo pip install jetson-stats`) is the prettier alternative.

---

## Ex 6 — iGPU path (Intel OpenVINO sketch)

### 👶 What this does
If the only edge GPU you have is an **Intel iGPU**, you use
**OpenVINO** instead of CUDA. Here's the minimal flow.

```bash
pip install openvino openvino-dev[onnx]
```

```python
# Convert an ONNX model to OpenVINO IR (Intermediate Representation)
# then run it on CPU, GPU (iGPU), or NPU (if present).
from openvino import Core, Tensor
import numpy as np

core = Core()
print("OpenVINO devices:", core.available_devices)   # ["CPU", "GPU", "NPU"?]

# Example — load an ONNX file that you exported from PyTorch
# model = core.read_model("mobilenet_v2.onnx")
# compiled = core.compile_model(model, "GPU")      # iGPU path
# out = compiled([np.random.rand(1, 3, 224, 224).astype("float32")])
# print(out)
```

> Colab's free VM has **no Intel iGPU**, so `"GPU"` won't appear — run
> this on any recent Intel laptop (Core Ultra or 12th gen+ Iris Xe) and
> it will.

---

## Ex 7 — Mobile SoC GPU path (TFLite GPU delegate)

### 👶 What this does
On Android, TFLite's **GPU delegate** ships the kernel to the phone's
Adreno / Mali / Apple GPU via OpenCL or Metal.

```python
# Create a GPU-delegated interpreter (shape only — full path is Android-side)
import tensorflow as tf

try:
    gpu_delegate = tf.lite.experimental.load_delegate(
        "libtensorflowlite_gpu_delegate.so")    # Android / Linux GPU
    interp = tf.lite.Interpreter(
        model_path="catdog_int8.tflite",
        experimental_delegates=[gpu_delegate])
    interp.allocate_tensors()
    print("GPU delegate ready.")
except ValueError as e:
    print("No GPU delegate available on this host:", e)
```

On Colab this usually falls back to CPU — we're only here to see the
**shape** of the API. On an Android device the exact same Python-like
call runs on the Adreno GPU.

---

## 📝 Summary — what you've done

| Step | Result |
|---|---|
| Detected the GPU family | Jetson / Discrete / iGPU / Mobile |
| Read spec sheet via NVML | Cores, memory, TDP, compute capability |
| Benchmarked across simulated families | Watched img/s scale with batch |
| Peeked at `tegrastats` | Learned to read real Jetson telemetry |
| Sketched iGPU and mobile paths | OpenVINO + TFLite GPU delegate |

Now go into [practice.md](edge_gpu_varieties_practice.md) to build your
own "which edge GPU should I buy?" decision notebook.

---

> *GPU Programming · EdgeAI · GPU Types · CODE · github.com/rpaut03l/TS-02*
