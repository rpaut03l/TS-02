# 💻 EdgeAI · CUDA for Edge — CODE

### *pycuda hello world · ONNX → TensorRT engine · INT8 calibration · CUDA Graphs*

> **Nav:** [← CUDA for Edge README](README.md) | [📖 THEORY](edge_cuda_theory.md) | **CODE** | [🎯 PRACTICE →](edge_cuda_practice.md)

---

## 🏗️ Setup

Most snippets run on **Colab (T4 GPU)** — Colab's T4 is a fine proxy
for a Jetson because both speak CUDA and TensorRT with the same
Python APIs. A few Jetson-only commands have their expected output
shown so you can follow along without the hardware.

Colab install, first cell of the notebook:

```python
!pip install -q pycuda tensorrt onnx onnxruntime-gpu
```

---

## Ex 1 — Hello-world CUDA kernel with pycuda

### 👶 What this does
Write a raw CUDA kernel in Python, launch it on the GPU, verify the
result. Same idea on a desktop GPU and on a Jetson — the code is
identical.

```python
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

KERNEL = r"""
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

mod = SourceModule(KERNEL)
vec_add = mod.get_function("vec_add")

N = 1 << 20                         # 1,048,576 elements
a = np.random.rand(N).astype("float32")
b = np.random.rand(N).astype("float32")
c = np.zeros_like(a)

# Allocate GPU memory and copy
d_a = cuda.mem_alloc(a.nbytes); cuda.memcpy_htod(d_a, a)
d_b = cuda.mem_alloc(b.nbytes); cuda.memcpy_htod(d_b, b)
d_c = cuda.mem_alloc(c.nbytes)

# Launch
block = (256, 1, 1)
grid  = ((N + 255) // 256, 1, 1)
vec_add(d_a, d_b, d_c, np.int32(N), block=block, grid=grid)

# Copy back + verify
cuda.memcpy_dtoh(c, d_c)
print("Max error:", np.max(np.abs(c - (a + b))))
```

Expect `Max error: 0.0`. This same code works on Jetson — only the
memory story changes (see Ex 5 below).

---

## Ex 2 — Build a TensorRT engine from ONNX (FP16)

### 👶 What this does
Take a model already in ONNX form, pass it through the TensorRT
builder, save a `.engine` file, and run inference. This is the
default production path on any Jetson.

```python
import tensorrt as trt
import numpy as np
import os, urllib.request

# Download a ready-made MobileNetV2 ONNX
URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
ONNX = "mobilenetv2.onnx"
if not os.path.exists(ONNX):
    urllib.request.urlretrieve(URL, ONNX)

logger = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, fp16=True):
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read()), "ONNX parse failed"

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)   # 1 GB
    if fp16 and builder.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, cfg)
    return serialized

plan = build_engine(ONNX, fp16=True)
with open("mobilenetv2_fp16.engine", "wb") as f:
    f.write(plan)
print("Saved engine:", os.path.getsize("mobilenetv2_fp16.engine") / 1024, "KB")
```

### 👶 What's happening
The builder reads the ONNX graph, picks the best CUDA kernel for **this
specific GPU** (Compute Capability 7.5 on T4, 8.7 on Orin), and writes
a serialized plan. The plan is **not portable** — rebuild it on each
target GPU family.

---

## Ex 3 — Full INT8 calibration with TensorRT

### 👶 What this does
INT8 is the **fastest, most power-efficient** TensorRT precision. The
builder needs real data to calibrate. Here's a minimal calibrator.

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class MyInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_iter, cache_file="calib.cache"):
        super().__init__()
        self.data_iter = iter(data_iter)
        self.cache_file = cache_file
        # Pre-allocate one batch's worth of GPU memory
        self.batch = next(self.data_iter)
        self.device_input = cuda.mem_alloc(self.batch.nbytes)
        cuda.memcpy_htod(self.device_input, self.batch)
        self.exhausted = False

    def get_batch_size(self):
        return self.batch.shape[0]

    def get_batch(self, names):
        if self.exhausted:
            return None
        try:
            self.batch = next(self.data_iter)
            cuda.memcpy_htod(self.device_input, self.batch)
            return [int(self.device_input)]
        except StopIteration:
            self.exhausted = True
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f: return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f: f.write(cache)


def fake_calibration_batches(n_batches=20, batch=8):
    """Replace this with real image batches for production."""
    for _ in range(n_batches):
        yield np.random.rand(batch, 3, 224, 224).astype("float32")

def build_int8(onnx_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.int8_calibrator = MyInt8Calibrator(fake_calibration_batches())

    plan = builder.build_serialized_network(network, cfg)
    return plan

plan_int8 = build_int8(ONNX)
with open("mobilenetv2_int8.engine", "wb") as f:
    f.write(plan_int8)
print("INT8 engine:", os.path.getsize('mobilenetv2_int8.engine')/1024, "KB")
```

Replace `fake_calibration_batches` with **real** images for anything
other than a smoke test.

---

## Ex 4 — Run the engine and benchmark FP32 / FP16 / INT8

```python
def run_engine(plan_bytes, x_host, iters=200):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(plan_bytes)
    ctx     = engine.create_execution_context()

    # (TensorRT 8.6+: context.set_input_shape / get_tensor_*)
    in_name  = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    ctx.set_input_shape(in_name, x_host.shape)

    d_in  = cuda.mem_alloc(x_host.nbytes)
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    y_host = np.empty(out_shape, dtype=np.float32)
    d_out = cuda.mem_alloc(y_host.nbytes)

    ctx.set_tensor_address(in_name, int(d_in))
    ctx.set_tensor_address(out_name, int(d_out))

    stream = cuda.Stream()
    import time
    for _ in range(20):
        cuda.memcpy_htod_async(d_in, x_host, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(y_host, d_out, stream)
    stream.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        cuda.memcpy_htod_async(d_in, x_host, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(y_host, d_out, stream)
    stream.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

x = np.random.rand(1, 3, 224, 224).astype("float32")
ms_fp16 = run_engine(plan,       x)
ms_int8 = run_engine(plan_int8,  x)
print(f"FP16 engine : {ms_fp16:6.3f} ms")
print(f"INT8 engine : {ms_int8:6.3f} ms")
```

Typical Colab T4 output:
```
FP16 engine :  0.75 ms
INT8 engine :  0.42 ms
```

Same model, **~2× faster** going FP16 → INT8.

---

## Ex 5 — Unified memory on Jetson (what to write)

Colab's T4 has separate device memory, so unified memory on Jetson is
shown as Python *pseudocode* — the API is the same, only the hardware
semantics change.

```python
# pycuda equivalent on Jetson — the allocation comes out of the shared
# Tegra memory pool, so no real copy happens.
import pycuda.driver as cuda
managed = cuda.managed_empty(shape=(1, 3, 224, 224), dtype=np.float32,
                             mem_flags=cuda.mem_attach_flags.GLOBAL)

# CPU fills it directly…
managed[:] = np.random.rand(1, 3, 224, 224).astype("float32")

# …and then the GPU reads it without an explicit cudaMemcpy.
# With TensorRT you'd pass int(managed.base) as the device pointer.
```

### The big difference vs desktop
- On desktop: `managed` is a **host-shadowed** buffer; the runtime
  pages it to VRAM on first access.
- On Jetson: `managed` **lives in the same LPDDR** the CPU uses — no
  migration, no PCIe. Pointer magic, zero copy cost.

---

## Ex 6 — CUDA Graphs for low-latency inference

### 👶 What this does
For tiny models (< 1 ms / inference), the **kernel-launch overhead**
starts to dominate. A **CUDA Graph** captures a whole sequence of
launches once, then replays it with a single API call — latency drops
noticeably.

```python
# Requires pycuda with graph support
import pycuda.driver as cuda
stream = cuda.Stream()

# (capture not yet exposed in all pycuda releases — conceptually:)
# 1. stream.begin_capture(cuda.stream_capture_mode.GLOBAL)
# 2. kernel1(..., stream=stream)
# 3. kernel2(..., stream=stream)
# 4. graph = stream.end_capture()
# 5. graph_exec = graph.instantiate()
# 6. for _ in range(N): graph_exec.launch(stream)

# In TensorRT 10+, enable graph capture via:
#   engine.create_execution_context().set_optimization_profile_async(...)
#   plus the IExecutionContext capture APIs.
```

On Jetson, CUDA Graphs are the single biggest trick for sub-millisecond
inference. It works well **when input/output shapes are fixed** —
exactly the case for a fixed-camera pipeline.

---

## Ex 7 — Simulate `nvpmodel` / `jetson_clocks`

These are Jetson-only, but you can write a Python wrapper so your
benchmarking script is portable.

```python
import subprocess, shutil, platform

def setup_max_perf(mode=0):
    """Lock Jetson to MAXN + max clocks.  No-op elsewhere."""
    if shutil.which("nvpmodel") is None:
        print("Not on a Jetson — skipping")
        return
    subprocess.run(["sudo", "nvpmodel", "-m", str(mode)], check=True)
    subprocess.run(["sudo", "jetson_clocks"], check=True)
    print(f"nvpmodel set to {mode}, jetson_clocks locked")

def restore_defaults():
    if shutil.which("jetson_clocks"):
        subprocess.run(["sudo", "jetson_clocks", "--restore"], check=False)
```

Call `setup_max_perf(0)` at the top of a benchmark, `restore_defaults()`
at the end. Print the state in your result logs.

---

## Ex 8 — Automate `tegrastats`

```python
import subprocess, re, time

def average_gpu_util(duration_s=5):
    """Parse tegrastats for `duration_s` and return avg GPU util %.
       Run on Jetson. Returns None on non-Jetson systems."""
    if subprocess.run(["which", "tegrastats"],
                      capture_output=True).returncode != 0:
        return None
    p = subprocess.Popen(["tegrastats", "--interval", "500"],
                         stdout=subprocess.PIPE, text=True)
    utils = []
    t_end = time.time() + duration_s
    try:
        for line in p.stdout:
            m = re.search(r"GR3D_FREQ (\d+)%", line)
            if m: utils.append(int(m.group(1)))
            if time.time() > t_end: break
    finally:
        p.terminate()
    return sum(utils) / max(1, len(utils))
```

Use alongside your inference loop — you'll see whether the GPU is
actually being utilised or whether preprocessing is the bottleneck.

---

## 📝 Summary — what you've built

| Step | File produced |
|---|---|
| pycuda hello-world vector add | (in-memory) |
| FP16 TensorRT engine | `mobilenetv2_fp16.engine` |
| INT8 calibration + engine | `mobilenetv2_int8.engine` + `calib.cache` |
| Benchmarked FP16 vs INT8 | stdout numbers |
| Wrote Jetson-ready perf helpers | `setup_max_perf()` etc. |

Next: open [practice.md](edge_cuda_practice.md) and put it together in
one end-to-end cat-vs-dog → TensorRT notebook.

---

> *GPU Programming · EdgeAI · CUDA for Edge · CODE · github.com/rpaut03l/TS-02*
