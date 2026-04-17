# 🎯 EdgeAI · CUDA for Edge — PRACTICE

### *End-to-end: cat-vs-dog model → ONNX → TensorRT FP16 → TensorRT INT8 → benchmark*

> **Nav:** [← CUDA for Edge README](README.md) | [📖 THEORY](edge_cuda_theory.md) | [💻 CODE](edge_cuda_code.md) | **PRACTICE**

---

## 🎯 What you'll build

Take the **same cat-vs-dog model** you trained in
[Fundamentals/practice](../Fundamentals/edge_ai_fundamentals_practice.md)
and push it all the way to TensorRT:

1. Export it to **ONNX**.
2. Build a **TensorRT FP16** engine.
3. Build a **TensorRT INT8** engine with real calibration data.
4. Benchmark three versions (PyTorch FP32, TRT FP16, TRT INT8) on the
   same Colab T4.
5. Compare size, latency, accuracy, and **inferences / joule**.

You'll see **3–5× speed-ups** that are the whole reason TensorRT
exists.

---

## 📋 Before you start

- **Runtime → T4 GPU**.
- Either re-train the cat-vs-dog model in this notebook (Cells 2–5
  are almost the same as the Fundamentals notebook) or mount it from
  Drive.

```python
!pip install -q pycuda tensorrt onnx onnxruntime-gpu
```

---

## Cell 1 — Environment check

```python
import tensorflow as tf, torch, tensorrt as trt, numpy as np, time, os
print("TensorFlow :", tf.__version__)
print("PyTorch    :", torch.__version__)
print("TensorRT   :", trt.__version__)
print("CUDA       :", torch.version.cuda)
print("GPU        :", torch.cuda.get_device_name(0))
```

Expect something like:
```
TensorFlow : 2.15.0
PyTorch    : 2.2.1+cu121
TensorRT   : 10.0.x
CUDA       : 12.1
GPU        : Tesla T4
```

---

## Cell 2 — Train or reload the cat-vs-dog model

The fastest path is to **re-train** in this session (5 epochs, ~2 min).
If you'd rather reload, upload your `catdog_fp32.keras` first.

```python
import os, numpy as np, tensorflow as tf

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/cats_and_dogs_filtered.zip"
zp  = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", URL, extract=True)
root = os.path.join(os.path.dirname(zp), "cats_and_dogs_filtered")

IMG = (160, 160); BATCH = 32
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{root}/train", image_size=IMG, batch_size=BATCH, label_mode="binary")
val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{root}/validation", image_size=IMG, batch_size=BATCH, label_mode="binary")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV2(input_shape=IMG + (3,),
                                         include_top=False, weights="imagenet")
base.trainable = False
inputs = tf.keras.Input(IMG + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, out)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=2)
model.save("catdog_fp32.keras")
```

---

## Cell 3 — Export the model to ONNX

Converting Keras → ONNX on Colab is simplest via **`tf2onnx`**.

```python
!pip install -q tf2onnx
!python -m tf2onnx.convert --saved-model /tmp/saved_model --output catdog.onnx --opset 17 --signature_def serving_default
```

If that errors (Keras 3 quirks), fall back to a SavedModel flow:

```python
model.export("/tmp/saved_model")
!python -m tf2onnx.convert --saved-model /tmp/saved_model --output catdog.onnx --opset 17
```

Inspect the ONNX file's I/O:

```python
import onnx
m = onnx.load("catdog.onnx")
for i in m.graph.input:  print("INPUT :", i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output: print("OUTPUT:", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
```

---

## Cell 4 — Build a TensorRT FP16 engine

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)

def build(onnx_path, int8=False, calibrator=None, workspace_bytes=1 << 30):
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    net = builder.create_network(flag)
    parser = trt.OnnxParser(net, logger)
    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read()), "ONNX parse failed"

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if builder.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = calibrator
    return builder.build_serialized_network(net, cfg)

plan_fp16 = build("catdog.onnx")
with open("catdog_fp16.engine", "wb") as f:
    f.write(plan_fp16)
print("FP16 engine:", os.path.getsize("catdog_fp16.engine")/1024, "KB")
```

---

## Cell 5 — Build a TensorRT INT8 engine with **real** calibration data

```python
import pycuda.driver as cuda, pycuda.autoinit

class RealInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, tfds, cache="catdog_calib.cache", n_batches=10):
        super().__init__()
        self.it = iter(tfds.take(n_batches))
        self.cache = cache
        self.device_input = None
        self.done = False

    def get_batch_size(self):
        return 32

    def get_batch(self, names):
        if self.done: return None
        try:
            imgs, _ = next(self.it)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(imgs).numpy()
            arr = arr.astype("float32")
            if self.device_input is None:
                self.device_input = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(self.device_input, arr)
            return [int(self.device_input)]
        except StopIteration:
            self.done = True
            return None

    def read_calibration_cache(self):
        return open(self.cache, "rb").read() if os.path.exists(self.cache) else None

    def write_calibration_cache(self, cache):
        with open(self.cache, "wb") as f: f.write(cache)

calibrator = RealInt8Calibrator(train_ds)
plan_int8 = build("catdog.onnx", int8=True, calibrator=calibrator)
with open("catdog_int8.engine", "wb") as f:
    f.write(plan_int8)
print("INT8 engine:", os.path.getsize("catdog_int8.engine")/1024, "KB")
```

---

## Cell 6 — Benchmark PyTorch-FP32 vs TRT-FP16 vs TRT-INT8

```python
import time, tensorrt as trt, pycuda.driver as cuda, numpy as np

def trt_latency(plan_bytes, shape=(1, 160, 160, 3), iters=300):
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    eng = rt.deserialize_cuda_engine(plan_bytes)
    ctx = eng.create_execution_context()

    in_name  = eng.get_tensor_name(0)
    out_name = eng.get_tensor_name(1)
    ctx.set_input_shape(in_name, shape)
    out_shape = tuple(ctx.get_tensor_shape(out_name))

    x = np.random.rand(*shape).astype("float32")
    y = np.empty(out_shape, dtype="float32")

    d_in  = cuda.mem_alloc(x.nbytes)
    d_out = cuda.mem_alloc(y.nbytes)
    ctx.set_tensor_address(in_name, int(d_in))
    ctx.set_tensor_address(out_name, int(d_out))
    stream = cuda.Stream()

    for _ in range(30):
        cuda.memcpy_htod_async(d_in, x, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(y, d_out, stream)
    stream.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        cuda.memcpy_htod_async(d_in, x, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(y, d_out, stream)
    stream.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def keras_latency(iters=300):
    x = np.random.rand(1, 160, 160, 3).astype("float32")
    for _ in range(30): model(x, training=False)
    t0 = time.perf_counter()
    for _ in range(iters): model(x, training=False)
    return (time.perf_counter() - t0) / iters * 1000

ms_fp32 = keras_latency()
ms_fp16 = trt_latency(plan_fp16)
ms_int8 = trt_latency(plan_int8)

print(f"{'Backend':<16}{'Size KB':>12}{'Latency ms':>14}{'Speed-up':>12}")
print(f"{'Keras FP32':<16}{os.path.getsize('catdog_fp32.keras')/1024:12.1f}{ms_fp32:14.2f}{1.0:>11.2f}x")
print(f"{'TRT FP16':<16}{os.path.getsize('catdog_fp16.engine')/1024:12.1f}{ms_fp16:14.2f}{ms_fp32/ms_fp16:>11.2f}x")
print(f"{'TRT INT8':<16}{os.path.getsize('catdog_int8.engine')/1024:12.1f}{ms_int8:14.2f}{ms_fp32/ms_int8:>11.2f}x")
```

Typical Colab T4 result:
```
Backend          Size KB   Latency ms   Speed-up
Keras FP32        9128.7       11.30       1.00x
TRT FP16          4502.9        0.95      11.89x
TRT INT8          1174.0        0.52      21.73x
```

Even on a T4, INT8 TensorRT is **~20×** faster than Keras FP32 on the
same model. On a Jetson Orin you'd see the same shape with different
absolute numbers.

---

## Cell 7 — Accuracy check (does INT8 still predict correctly?)

```python
# Take 200 validation images and compare labels
imgs, lbls = next(iter(val_ds.unbatch().batch(200)))
imgs = imgs.numpy().astype("float32")
lbls = lbls.numpy().astype("int").flatten()

# Reference (Keras FP32)
preds_fp32 = (model.predict(imgs, verbose=0) > 0.5).astype("int").flatten()

def predict_trt(plan_bytes, imgs):
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    eng = rt.deserialize_cuda_engine(plan_bytes)
    ctx = eng.create_execution_context()
    in_name, out_name = eng.get_tensor_name(0), eng.get_tensor_name(1)
    preds = []
    for img in imgs:
        x = np.expand_dims(img, 0).astype("float32")
        ctx.set_input_shape(in_name, x.shape)
        out_shape = tuple(ctx.get_tensor_shape(out_name))
        y = np.empty(out_shape, dtype="float32")
        d_in  = cuda.mem_alloc(x.nbytes); cuda.memcpy_htod(d_in, x)
        d_out = cuda.mem_alloc(y.nbytes)
        ctx.set_tensor_address(in_name, int(d_in))
        ctx.set_tensor_address(out_name, int(d_out))
        stream = cuda.Stream()
        ctx.execute_async_v3(stream.handle)
        stream.synchronize()
        cuda.memcpy_dtoh(y, d_out)
        preds.append(int(y.flatten()[0] > 0.5))
    return np.array(preds)

preds_fp16 = predict_trt(plan_fp16, imgs)
preds_int8 = predict_trt(plan_int8, imgs)

print(f"FP32 Keras acc : {(preds_fp32 == lbls).mean()*100:5.2f}%")
print(f"TRT  FP16 acc  : {(preds_fp16 == lbls).mean()*100:5.2f}%")
print(f"TRT  INT8 acc  : {(preds_int8 == lbls).mean()*100:5.2f}%")
```

Typical result:
```
FP32 Keras acc : 95.50%
TRT  FP16 acc  : 95.00%
TRT  INT8 acc  : 94.50%
```

**1 % accuracy for 20× speed** is an amazing trade on any edge chip.

---

## Cell 8 — Energy per inference (simulated)

```python
# Approximate TDPs while busy
TDPs = {"Keras FP32 on T4": 70, "TRT FP16 on T4": 70, "TRT INT8 on T4": 60,
        "TRT INT8 on Orin NX (15W)": 15, "TRT INT8 on Orin Nano (10W)": 10}

# Measured (or typical) latencies in ms
LATENCIES = {"Keras FP32 on T4": ms_fp32, "TRT FP16 on T4": ms_fp16,
             "TRT INT8 on T4": ms_int8,
             "TRT INT8 on Orin NX (15W)": ms_int8 * 3,
             "TRT INT8 on Orin Nano (10W)": ms_int8 * 5}

print(f"{'Config':<30}{'ms':>8}{'W':>5}{'mJ/inf':>10}{'inf/J':>10}")
for k, w in TDPs.items():
    ms = LATENCIES[k]
    mJ = ms / 1000 * w * 1000
    per_joule = 1000 / mJ
    print(f"{k:<30}{ms:8.2f}{w:5.0f}{mJ:10.2f}{per_joule:10.1f}")
```

Typical output:
```
Config                              ms    W    mJ/inf     inf/J
Keras FP32 on T4                 11.30   70    791.00       1.3
TRT FP16 on T4                    0.95   70     66.50      15.0
TRT INT8 on T4                    0.52   60     31.20      32.1
TRT INT8 on Orin NX (15W)         1.56   15     23.40      42.7
TRT INT8 on Orin Nano (10W)       2.60   10     26.00      38.5
```

### The punchline
A Jetson Orin NX running the **TensorRT INT8** engine gets **~40
inferences per joule** — more than the T4. Not because Orin is a
faster GPU (it isn't), but because it has a **lower TDP** while still
having strong INT8 throughput. This is the whole value proposition
of edge CUDA.

---

## Cell 9 — Stretch goals

1. Add a **QAT** (Quantization-Aware Training) path: insert fake-quant
   layers during training, export QDQ ONNX, compare accuracy vs PTQ.
2. Flip to **CUDA Graph** capture mode — measure the additional
   latency drop on batch-1 inference.
3. Change the workspace pool limit (64 MB, 256 MB, 1 GB, 4 GB). Plot
   engine build time vs final latency.
4. Export to a **dynamic-batch** ONNX (batch=-1) and rebuild the
   engine. Benchmark throughput at batch 1, 8, 32.
5. **Jetson-only:** copy `catdog_int8.engine` + the inference script
   to an Orin Nano dev kit. Re-run Cell 6 there. The numbers should
   be within 10 % of your simulated "Orin NX" estimate in Cell 8.

---

## 🎓 What you should take away

- The pipeline **PyTorch/TF → ONNX → TensorRT FP16/INT8 → runtime** is
  the canonical edge path on any NVIDIA GPU.
- Speed-ups of **5× – 20×** are normal, with accuracy drop < 1 %.
- **Unified memory** on Tegra removes most `cudaMemcpy` costs — free
  performance if you refactor.
- **Always measure inferences per joule**, not just ms — the edge
  optimises **energy**, not wall-clock.
- **`nvpmodel`** + **`jetson_clocks`** are mandatory before every
  benchmark on real hardware.

---

## 🎉 You've finished the EdgeAI track

You now have a full, consistent, hands-on understanding of Edge AI:

- **Why** it exists → [Fundamentals/](../Fundamentals/README.md)
- **Which GPUs** power it → [GPU_Types/](../GPU_Types/README.md)
- **Which non-GPU chips** also power it → [Hardware/](../Hardware/README.md)
- **How CUDA** actually lands on a Jetson → this folder

Planned 🔭 future sub-tracks (model compression, TinyML, federated
learning, Edge MLOps, security) will build directly on top of what
you've done here.

---

> *GPU Programming · EdgeAI · CUDA for Edge · PRACTICE · github.com/rpaut03l/TS-02*
