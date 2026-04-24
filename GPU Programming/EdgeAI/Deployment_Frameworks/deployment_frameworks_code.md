# 💻 EdgeAI · Deployment Frameworks — CODE

### *Export once to ONNX — then fan out to TFLite, ORT, and OpenVINO*

> **Nav:** [← Deployment Frameworks README](README.md) | [📖 THEORY](deployment_frameworks_theory.md) | **CODE** | [🎯 PRACTICE →](deployment_frameworks_practice.md)

---

## 🏗️ Setup

```python
!pip install -q torch torchvision tensorflow onnx onnx-tf onnxruntime \
    onnxruntime-gpu openvino openvino-dev tf2onnx coremltools
```

> Some installs are large. Stick to a Colab T4 with plenty of disk.

---

## Ex 1 — Export a PyTorch model to ONNX

### 👶 What this does
One call converts a traced model into the portable `.onnx` format.

```python
import torch, torchvision

m = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
x = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    m, x, "mobilenetv2.onnx",
    opset_version=17,
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    do_constant_folding=True,
)
print("Saved mobilenetv2.onnx")
```

### Validate the export
```python
import onnx
model = onnx.load("mobilenetv2.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid.")
print("Inputs:",  [(i.name, [d.dim_value or d.dim_param
                             for d in i.type.tensor_type.shape.dim])
                   for i in model.graph.input])
print("Outputs:", [(o.name, [d.dim_value or d.dim_param
                             for d in o.type.tensor_type.shape.dim])
                   for o in model.graph.output])
```

---

## Ex 2 — Golden numeric check (PyTorch vs ORT)

### 👶 What this does
Before trusting any conversion, prove the math matches.

```python
import onnxruntime as ort, numpy as np

x_np = x.numpy()
with torch.no_grad():
    torch_out = m(x).numpy()

sess = ort.InferenceSession("mobilenetv2.onnx",
                            providers=["CPUExecutionProvider"])
onnx_out = sess.run(None, {"input": x_np})[0]

err = np.max(np.abs(torch_out - onnx_out))
print(f"Max absolute diff: {err:.6f}  (should be < 1e-3)")
```

---

## Ex 3 — ONNX Runtime: enumerate and benchmark providers

### 👶 What this does
Same `.onnx`, several backends. Choose one per deployment target.

```python
import time
print("Available providers:", ort.get_available_providers())

def bench(provider, iters=200):
    sess = ort.InferenceSession("mobilenetv2.onnx",
                                providers=[provider, "CPUExecutionProvider"])
    for _ in range(10): sess.run(None, {"input": x_np})
    t0 = time.perf_counter()
    for _ in range(iters): sess.run(None, {"input": x_np})
    return (time.perf_counter() - t0) / iters * 1000

for p in ort.get_available_providers():
    try:
        print(f"{p:30s} {bench(p):6.2f} ms")
    except Exception as e:
        print(f"{p:30s} skipped ({e.__class__.__name__})")
```

On Colab T4 typical output:
```
TensorrtExecutionProvider        1.30 ms
CUDAExecutionProvider            3.50 ms
CPUExecutionProvider            14.20 ms
```

---

## Ex 4 — ORT with IOBinding (avoid per-call host↔device copies)

### 👶 What this does
On GPU, copying a new tensor to VRAM every call wastes time. IOBinding
lets you preload once and reuse.

```python
sess = ort.InferenceSession("mobilenetv2.onnx",
                            providers=["CUDAExecutionProvider"])
binding = sess.io_binding()

x_gpu = ort.OrtValue.ortvalue_from_numpy(x_np, "cuda", 0)
binding.bind_input(name="input", device_type="cuda", device_id=0,
                   element_type=np.float32, shape=x_np.shape,
                   buffer_ptr=x_gpu.data_ptr())
binding.bind_output("logits", "cuda")

# Benchmark
for _ in range(10): sess.run_with_iobinding(binding)
t0 = time.perf_counter()
for _ in range(200): sess.run_with_iobinding(binding)
print(f"IOBinding run: {(time.perf_counter()-t0)/200*1000:.2f} ms")
```

Typical speed-up over naive `sess.run(...)`: **1.3–2×**.

---

## Ex 5 — ORT quantization (dynamic INT8)

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("mobilenetv2.onnx", "mobilenetv2_int8_dyn.onnx",
                 weight_type=QuantType.QInt8)

import os
print("FP32 :", os.path.getsize("mobilenetv2.onnx")/1024**2, "MB")
print("INT8 :", os.path.getsize("mobilenetv2_int8_dyn.onnx")/1024**2, "MB")
```

---

## Ex 6 — Convert ONNX → TFLite (via onnx-tf)

### 👶 What this does
TFLite doesn't accept ONNX directly. Go ONNX → TF SavedModel → TFLite.

```python
!pip install -q onnx-tf
from onnx_tf.backend import prepare
import onnx, tensorflow as tf

onnx_model = onnx.load("mobilenetv2.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("mobilenetv2_tf")         # SavedModel dir

conv = tf.lite.TFLiteConverter.from_saved_model("mobilenetv2_tf")
conv.optimizations = [tf.lite.Optimize.DEFAULT]
open("mobilenetv2_dyn.tflite", "wb").write(conv.convert())

print("TFLite FP+INT8-dyn:", os.path.getsize("mobilenetv2_dyn.tflite")/1024**2, "MB")
```

### Add the GPU / NNAPI delegate at runtime (shape only on Colab)

```python
# On Android / Linux with OpenCL:
# import tensorflow as tf
# gpu_delegate = tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
# interp = tf.lite.Interpreter(model_path="mobilenetv2_dyn.tflite",
#                              experimental_delegates=[gpu_delegate])
```

---

## Ex 7 — Convert ONNX → OpenVINO IR

```python
# OpenVINO's Model Optimizer (invoked via the openvino-dev wheel)
!mo --input_model mobilenetv2.onnx --output_dir ov_ir --compress_to_fp16
print(os.listdir("ov_ir"))    # mobilenetv2.xml + mobilenetv2.bin
```

### Run it with OpenVINO Python API

```python
from openvino import Core, Tensor
core = Core()
print("OpenVINO devices:", core.available_devices)

m_ov = core.read_model("ov_ir/mobilenetv2.xml")
compiled = core.compile_model(m_ov, "CPU")
infer    = compiled.create_infer_request()

for _ in range(10): infer.infer({0: x_np})
t0 = time.perf_counter()
for _ in range(200): infer.infer({0: x_np})
print(f"OpenVINO CPU: {(time.perf_counter()-t0)/200*1000:.2f} ms")
```

### AUTO / MULTI / HETERO

```python
# "AUTO" lets OpenVINO pick; "MULTI:GPU,CPU" runs in parallel.
# compiled = core.compile_model(m_ov, "AUTO")
# compiled = core.compile_model(m_ov, "MULTI:GPU,CPU")
# compiled = core.compile_model(m_ov, "HETERO:GPU,CPU")
```

---

## Ex 8 — NNCF: OpenVINO's INT8 post-training quantization

```python
# pip install nncf
# import nncf, openvino as ov
# data = [[np.random.rand(1, 3, 224, 224).astype("float32")] for _ in range(100)]
# calibration_dataset = nncf.Dataset(data)
# model = ov.Core().read_model("ov_ir/mobilenetv2.xml")
# int8 = nncf.quantize(model, calibration_dataset)
# ov.save_model(int8, "ov_ir_int8/mobilenetv2.xml")
```

NNCF also supports **QAT, pruning, filter pruning, sparsity** — see
the [Model_Compression/](../Model_Compression/README.md) chapter.

---

## Ex 9 — Convert ONNX → Core ML (iOS / Mac)

```python
# pip install coremltools onnx
# import coremltools as ct
# cml_model = ct.convert("mobilenetv2.onnx",
#                        inputs=[ct.TensorType(shape=(1, 3, 224, 224))])
# cml_model.save("MobileNetV2.mlpackage")
```

Open the `.mlpackage` in Xcode → Instruments → **Core ML** template
to see ANE dispatch in real time.

---

## Ex 10 — Side-by-side benchmark harness

### 👶 What this does
One Python dict → one chart. The harness runs every installed
runtime and records latency + output hash.

```python
import hashlib
def h(arr): return hashlib.sha1(arr.tobytes()).hexdigest()[:10]

def run_torch():
    with torch.no_grad():
        for _ in range(10): m(x)
        t0 = time.perf_counter()
        for _ in range(200): y = m(x)
        return (time.perf_counter()-t0)/200*1000, h(y.numpy())

def run_ort(provider, path):
    sess = ort.InferenceSession(path,
                                providers=[provider, "CPUExecutionProvider"])
    for _ in range(10): sess.run(None, {"input": x_np})
    t0 = time.perf_counter()
    for _ in range(200): y = sess.run(None, {"input": x_np})[0]
    return (time.perf_counter()-t0)/200*1000, h(y)

def run_ov(device):
    core = Core()
    m_ov = core.read_model("ov_ir/mobilenetv2.xml")
    compiled = core.compile_model(m_ov, device)
    ir = compiled.create_infer_request()
    for _ in range(10): ir.infer({0: x_np})
    t0 = time.perf_counter()
    for _ in range(200): y = ir.infer({0: x_np})[0]
    return (time.perf_counter()-t0)/200*1000, h(np.asarray(y))

def run_tflite(path):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_i = interp.get_input_details()[0]["index"]
    out_i = interp.get_output_details()[0]["index"]
    x_nhwc = np.transpose(x_np, (0, 2, 3, 1)).astype(np.float32)
    for _ in range(10): interp.set_tensor(in_i, x_nhwc); interp.invoke()
    t0 = time.perf_counter()
    for _ in range(200): interp.set_tensor(in_i, x_nhwc); interp.invoke()
    return (time.perf_counter()-t0)/200*1000, h(interp.get_tensor(out_i))

rows = [("PyTorch CPU (GPU)",) + run_torch(),
        ("ORT CPU",)   + run_ort("CPUExecutionProvider", "mobilenetv2.onnx"),
        ("ORT CUDA",)  + run_ort("CUDAExecutionProvider", "mobilenetv2.onnx"),
        ("ORT TRT",)   + run_ort("TensorrtExecutionProvider", "mobilenetv2.onnx"),
        ("OpenVINO CPU",) + run_ov("CPU"),
        ("TFLite FP",) + run_tflite("mobilenetv2_dyn.tflite"),
        ]
print(f"{'Backend':20s} {'ms':>8s}  {'hash':>10s}")
for r in rows:
    name, ms, hh = r
    print(f"{name:20s} {ms:8.2f}  {hh:>10s}")
```

Output hashes won't match byte-for-byte (different kernels) but should
be very close in the decoded predictions.

---

## 📝 Summary

| Exercise | Artifact produced |
|---|---|
| 1 | `mobilenetv2.onnx` |
| 2 | Numeric golden check PyTorch vs ORT |
| 3 | Latency table across ORT providers |
| 4 | IOBinding speed-up |
| 5 | `mobilenetv2_int8_dyn.onnx` |
| 6 | `mobilenetv2_dyn.tflite` |
| 7 | OpenVINO IR + CPU benchmark |
| 8 | NNCF INT8 quantized IR |
| 9 | Core ML `.mlpackage` |
| 10 | Side-by-side harness table |

Now run everything in one notebook → [practice.md](deployment_frameworks_practice.md).

---

> *GPU Programming · EdgeAI · Deployment Frameworks · CODE · github.com/rpaut03l/TS-02*
