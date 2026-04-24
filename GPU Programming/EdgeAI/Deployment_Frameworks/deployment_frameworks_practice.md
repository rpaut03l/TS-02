# 🎯 EdgeAI · Deployment Frameworks — PRACTICE

### *One training job → one ONNX → three runtimes, one portability report*

> **Nav:** [← Deployment Frameworks README](README.md) | [📖 THEORY](deployment_frameworks_theory.md) | [💻 CODE](deployment_frameworks_code.md) | **PRACTICE**

---

## 🎯 What you'll build

A Colab notebook that takes **one PyTorch model** and produces a
**portability report**: for every runtime (ORT CPU/CUDA/TRT, TFLite,
OpenVINO CPU, optionally Core ML), it records:

- **Does it load?** (yes / op-coverage error)
- **Does it agree numerically with PyTorch?** (|Δ| < 1e-3)
- **Latency per inference**
- **Binary / artifact size**

The notebook ends with a single recommendation line per target device.

---

## Cell 1 — Setup

```python
!pip install -q torch torchvision tensorflow onnx onnx-tf onnxruntime \
    onnxruntime-gpu openvino openvino-dev tf2onnx

import os, time, numpy as np, torch, torchvision as tv
import onnx, onnxruntime as ort, tensorflow as tf
from openvino import Core
print("Torch :", torch.__version__)
print("ORT   :", ort.__version__)
print("TF    :", tf.__version__)
```

---

## Cell 2 — Train or load a small cat-vs-dog model

Reuse the model from [Fundamentals/practice](../Fundamentals/edge_ai_fundamentals_practice.md)
— or quickly retrain below.

```python
m = tv.models.mobilenet_v2(weights="DEFAULT")
m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
m.eval()   # assume pretrained for speed; refit if needed
torch.save(m.state_dict(), "model_fp32.pt")
print("FP32 size:", os.path.getsize("model_fp32.pt")/1024**2, "MB")
```

---

## Cell 3 — Export to ONNX

```python
x = torch.randn(1, 3, 160, 160)
torch.onnx.export(m, x, "cat_dog.onnx", opset_version=17,
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"},
                                "logits": {0: "batch"}})
onnx.checker.check_model(onnx.load("cat_dog.onnx"))
print("ONNX size:", os.path.getsize("cat_dog.onnx")/1024**2, "MB")
```

---

## Cell 4 — Save the reference output (PyTorch)

```python
with torch.no_grad():
    ref_out = m(x).numpy()
np.save("ref_out.npy", ref_out)
print("Ref logits:", ref_out.flatten()[:5])
```

---

## Cell 5 — Convert ONNX → TFLite

```python
from onnx_tf.backend import prepare
prepare(onnx.load("cat_dog.onnx")).export_graph("tf_saved")

conv = tf.lite.TFLiteConverter.from_saved_model("tf_saved")
conv.optimizations = [tf.lite.Optimize.DEFAULT]
open("cat_dog.tflite", "wb").write(conv.convert())
print("TFLite size:", os.path.getsize("cat_dog.tflite")/1024**2, "MB")
```

---

## Cell 6 — Convert ONNX → OpenVINO IR

```python
!mo --input_model cat_dog.onnx --output_dir ir --compress_to_fp16
print("IR files:", os.listdir("ir"))
```

---

## Cell 7 — Portability harness

```python
ref = np.load("ref_out.npy")
x_np = x.numpy()

def check_ort(provider):
    try:
        sess = ort.InferenceSession("cat_dog.onnx",
                                    providers=[provider, "CPUExecutionProvider"])
        for _ in range(10): sess.run(None, {"input": x_np})
        t0 = time.perf_counter()
        for _ in range(200): y = sess.run(None, {"input": x_np})[0]
        ms = (time.perf_counter()-t0)/200*1000
        diff = float(np.max(np.abs(y - ref)))
        return dict(loads=True, max_abs_diff=diff, ms=ms)
    except Exception as e:
        return dict(loads=False, error=str(e))

def check_ov(device):
    try:
        core = Core()
        mdl = core.read_model("ir/cat_dog.xml")
        compiled = core.compile_model(mdl, device)
        ir = compiled.create_infer_request()
        for _ in range(10): ir.infer({0: x_np})
        t0 = time.perf_counter()
        for _ in range(200): y = ir.infer({0: x_np})[0]
        ms = (time.perf_counter()-t0)/200*1000
        diff = float(np.max(np.abs(np.asarray(y) - ref)))
        return dict(loads=True, max_abs_diff=diff, ms=ms)
    except Exception as e:
        return dict(loads=False, error=str(e))

def check_tflite():
    try:
        interp = tf.lite.Interpreter(model_path="cat_dog.tflite")
        interp.allocate_tensors()
        in_i = interp.get_input_details()[0]["index"]
        out_i = interp.get_output_details()[0]["index"]
        x_nhwc = np.transpose(x_np, (0, 2, 3, 1)).astype(np.float32)
        for _ in range(10):
            interp.set_tensor(in_i, x_nhwc); interp.invoke()
        t0 = time.perf_counter()
        for _ in range(200):
            interp.set_tensor(in_i, x_nhwc); interp.invoke()
        ms = (time.perf_counter()-t0)/200*1000
        y = interp.get_tensor(out_i)
        diff = float(np.max(np.abs(y - ref)))
        return dict(loads=True, max_abs_diff=diff, ms=ms)
    except Exception as e:
        return dict(loads=False, error=str(e))

report = {
    "PyTorch (reference)"      : dict(loads=True, max_abs_diff=0.0, ms=None),
    "ORT CPU"                  : check_ort("CPUExecutionProvider"),
    "ORT CUDA"                 : check_ort("CUDAExecutionProvider"),
    "ORT TensorRT"             : check_ort("TensorrtExecutionProvider"),
    "OpenVINO CPU"             : check_ov("CPU"),
    "TFLite (CPU)"             : check_tflite(),
}
```

---

## Cell 8 — Pretty-print the report

```python
print(f"{'Runtime':<24}{'Loads':>7}{'Δmax':>12}{'Latency (ms)':>15}")
for name, r in report.items():
    if r.get("loads"):
        ms = r.get("ms")
        ms_str = f"{ms:10.2f}" if ms is not None else "     n/a"
        print(f"{name:<24}{'✅':>7}{r['max_abs_diff']:12.2e}{ms_str:>15}")
    else:
        print(f"{name:<24}{'❌':>7}  {r['error'][:50]}")
```

Typical output:
```
Runtime                     Loads        Δmax   Latency (ms)
PyTorch (reference)              ✅    0.00e+00        n/a
ORT CPU                          ✅    2.38e-06      14.02
ORT CUDA                         ✅    5.96e-06       3.41
ORT TensorRT                     ✅    3.81e-05       1.22
OpenVINO CPU                     ✅    6.68e-06       8.90
TFLite (CPU)                     ✅    7.15e-06      19.40
```

---

## Cell 9 — "Which runtime for which device?" recommender

```python
TARGETS = {
    "Android phone":            ["TFLite (CPU)"],
    "iPhone":                   ["TFLite (CPU)"],         # stand-in for CoreML
    "NVIDIA Jetson":            ["ORT TensorRT", "ORT CUDA"],
    "Intel laptop / NUC":       ["OpenVINO CPU"],
    "Generic x86 server":       ["ORT CPU", "OpenVINO CPU"],
}

def pick(target):
    options = [name for name in TARGETS[target]
               if report.get(name, {}).get("loads")]
    if not options: return "— no working runtime"
    # pick fastest
    best = min(options, key=lambda n: report[n].get("ms") or 1e9)
    return f"{best} ({report[best]['ms']:.2f} ms)"

print("\n🏷️  Recommended runtime per target:")
for t in TARGETS:
    print(f"  {t:<28} {pick(t)}")
```

---

## Cell 10 — Artifact sizes

```python
print("\n📦  Artifact sizes (on disk):")
for path in ["model_fp32.pt", "cat_dog.onnx", "cat_dog.tflite",
             "ir/cat_dog.xml", "ir/cat_dog.bin"]:
    if os.path.exists(path):
        mb = os.path.getsize(path)/1024**2
        print(f"  {path:<28} {mb:7.2f} MB")
```

---

## Cell 11 — Stretch goals

1. Rebuild **`cat_dog.tflite`** with full-integer INT8 (representative
   dataset). Re-run the harness — which backend benefits most?
2. Add a **CoreML** branch (`coremltools`) — skipped on Colab because
   CoreML runtime is macOS only; add the *conversion* step anyway.
3. Enable **ORT IOBinding** on CUDA. Does latency drop?
4. Re-run with **dynamic batch = 32** — which runtime scales best?
5. Add a **numeric-tolerance** gate: fail the harness if any runtime
   has `Δmax > 1e-3`. You've just built the kernel of a deployment CI
   job.

---

## 🎓 What you should take away

- **ONNX is the universal starter artifact.** Every other format
  flows from it.
- The **portability harness** + **golden check** is the reason your
  device-side bugs disappear.
- One chart is all you need to justify your runtime choice to a
  team.
- **Runtimes beat frameworks on the edge**, every time. Never ship
  PyTorch / TF to a device.

Next: [**TinyML →**](../TinyML/README.md) — running these ideas on a
chip with kilobytes of RAM.

---

> *GPU Programming · EdgeAI · Deployment Frameworks · PRACTICE · github.com/rpaut03l/TS-02*
