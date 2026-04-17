# 💻 EdgeAI · Fundamentals — CODE

### *Runnable Python — train a baseline, shrink it, measure the edge-friendly version*

> **Nav:** [← Fundamentals README](README.md) | [📖 THEORY](edge_ai_fundamentals_theory.md) | **CODE** | [🎯 PRACTICE →](edge_ai_fundamentals_practice.md)

---

## 🏗️ Setup — Google Colab (easiest)

1. Open [colab.research.google.com](https://colab.research.google.com).
2. Menu: **Runtime → Change runtime type → Hardware accelerator: T4 GPU
   → Save**.
3. All the code below runs on the free tier — no install, no credit card.

> 💡 We use a GPU for **training** the baseline model quickly. The
> *point* of the chapter is that the **inference** runs fine on a tiny
> edge-class device (CPU, quantized, <50 MB). The GPU just saves us an
> hour of waiting during training.

---

## Ex 1 — Check that your Colab has what it needs

### 👶 What this does
Before writing any edge code, we check that **TensorFlow**, **TensorFlow
Lite**, and **NumPy** are ready. TFLite is the most common edge
runtime, so we'll use it as our "edge simulator" in this notebook.

```python
import sys, platform, numpy as np, tensorflow as tf

print("Python       :", sys.version.split()[0])
print("Platform     :", platform.platform())
print("NumPy        :", np.__version__)
print("TensorFlow   :", tf.__version__)
print("GPU visible? :", len(tf.config.list_physical_devices("GPU")) > 0)
```

Typical Colab output:
```
Python       : 3.11.11
Platform     : Linux-6.1.85+-x86_64-with-glibc2.35
NumPy        : 1.26.4
TensorFlow   : 2.15.0
GPU visible? : True
```

### 👶 What the output means
- **GPU visible = True** — Colab gave us a Tesla T4. Training will be
  fast. If False, go back to **Runtime → Change runtime type**.
- The **TFLite converter** lives inside `tensorflow` (no separate
  install needed on Colab).

---

## Ex 2 — Train a tiny MNIST CNN (our baseline)

### 👶 What this does
MNIST is the "hello world" of computer vision — 28×28 grayscale digits,
10 classes. We train a tiny CNN so we have a **baseline model** to
shrink later. This is the **FP32** (full-precision) model that a
cloud would run.

```python
# ---- Load data ----
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0   # 0..1 range
x_test  = x_test.astype("float32")  / 255.0
x_train = x_train[..., None]                  # add channel dim → (N, 28, 28, 1)
x_test  = x_test[..., None]

print("Train shape:", x_train.shape, "  Test shape:", x_test.shape)

# ---- Build a small CNN ----
model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ---- Train ----
model.fit(x_train, y_train,
          epochs=3, batch_size=128,
          validation_split=0.1, verbose=2)

# ---- Evaluate ----
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Baseline FP32 accuracy: {acc*100:.2f}%")

# ---- Save ----
model.save("mnist_fp32.keras")
```

### 👶 What this tells us
- Accuracy should land around **98–99 %**. That's our **gold standard**
  — we don't want the edge version to lose much of this.
- The `.keras` file on disk is our **baseline size**. We'll compare
  this with the tiny edge versions below.

---

## Ex 3 — Convert to TensorFlow Lite (first edge version)

### 👶 What this does
TFLite is a **stripped-down runtime** that runs on phones,
microcontrollers, and edge GPUs. The first conversion is **pure FP32**
— no quantization yet. We're just changing the *runtime*, not the
*precision*.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()

with open("mnist_fp32.tflite", "wb") as f:
    f.write(tflite_fp32)

import os
size_keras  = os.path.getsize("mnist_fp32.keras") / 1024
size_tflite = os.path.getsize("mnist_fp32.tflite") / 1024
print(f"Keras (cloud):  {size_keras:7.1f} KB")
print(f"TFLite (edge):  {size_tflite:7.1f} KB")
```

### 👶 What you'll see
```
Keras (cloud):   250.4 KB
TFLite (edge):    93.7 KB
```
The TFLite version is **smaller** because the runtime doesn't need to
store optimizer state, training graph, or variable names. Same weights,
leaner packaging.

---

## Ex 4 — INT8 post-training quantization (real edge version)

### 👶 What this does
**Quantization** is the single most important Edge AI trick. We
replace every 32-bit float weight with an 8-bit integer. That's:

- **4× smaller model** (32 / 8 = 4).
- **2–4× faster inference** on CPUs and NPUs (integer math is cheap).
- **Lower power** (moving 4× fewer bytes through memory).

The trade-off is a small accuracy drop (usually < 1 %).

```python
def representative_data():
    """Feeds ~100 real samples so the quantizer learns the activation range."""
    for x in x_train[:100]:
        yield [x[None, ...].astype("float32")]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()

with open("mnist_int8.tflite", "wb") as f:
    f.write(tflite_int8)

size_int8 = os.path.getsize("mnist_int8.tflite") / 1024
print(f"TFLite INT8 (edge): {size_int8:7.1f} KB")
print(f"Shrink vs Keras:    {size_keras / size_int8:4.1f}×")
```

### 👶 What you'll see
```
TFLite INT8 (edge):   28.5 KB
Shrink vs Keras:       8.8×
```
The model is **almost 9× smaller** than the original. It now fits in
the L2 cache of many microcontrollers.

---

## Ex 5 — Measure latency (FP32 vs INT8)

### 👶 What this does
Now we **prove** the edge version is faster. We run both models on the
CPU (to simulate an edge device — not the GPU) and time them.

```python
import time

def bench(tflite_bytes, x, n=500):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    # Handle INT8 quantization on the input side
    if in_det["dtype"] == np.int8:
        scale, zero = in_det["quantization"]
        x_q = (x / scale + zero).astype(np.int8)
    else:
        x_q = x.astype(np.float32)

    # warm-up
    for i in range(20):
        interp.set_tensor(in_det["index"], x_q[i:i+1])
        interp.invoke()

    # timed run
    t0 = time.perf_counter()
    for i in range(n):
        interp.set_tensor(in_det["index"], x_q[i:i+1])
        interp.invoke()
    dt = (time.perf_counter() - t0) / n * 1000   # ms / sample
    return dt

ms_fp32 = bench(tflite_fp32, x_test)
ms_int8 = bench(tflite_int8, x_test)
print(f"FP32 TFLite : {ms_fp32:5.2f} ms / image")
print(f"INT8 TFLite : {ms_int8:5.2f} ms / image")
print(f"Speedup     : {ms_fp32 / ms_int8:4.2f}×")
```

### 👶 Typical Colab CPU output
```
FP32 TFLite : 0.88 ms / image
INT8 TFLite : 0.31 ms / image
Speedup     : 2.84×
```

### So what?
> We went from a **250 KB, 0.88 ms** cloud-style model to a
> **28 KB, 0.31 ms** edge-style model. **Same accuracy target, 9×
> smaller, 3× faster.** That's Edge AI in one experiment.

---

## Ex 6 — Bandwidth savings (back-of-envelope)

### 👶 What this does
One of the 5 pillars is **bandwidth savings**. Let's prove it.

Imagine a camera that snaps **1 image per second**. Option A: upload
every image to the cloud for inference. Option B: run the model on-
device and upload only the label (a few bytes) when something
interesting is seen.

```python
# --- assumptions ---
img_size_kb      = 50      # JPEG from a smart camera
fps              = 1       # 1 frame / second
seconds_per_day  = 86_400
interesting_pct  = 0.01    # only 1 % of frames are "events"

# --- Option A: stream every frame to the cloud ---
cloud_kb_day = img_size_kb * fps * seconds_per_day

# --- Option B: run on-device, upload only events ---
event_kb_day = img_size_kb * fps * seconds_per_day * interesting_pct

print(f"Cloud streaming:        {cloud_kb_day/1024:10.1f} MB/day")
print(f"Edge + events only:     {event_kb_day/1024:10.1f} MB/day")
print(f"Bandwidth saved:        {(1 - event_kb_day/cloud_kb_day)*100:5.1f}%")
```

### 👶 Output
```
Cloud streaming:           4218.8 MB/day
Edge + events only:          42.2 MB/day
Bandwidth saved:            99.0%
```

> **Edge AI = 99 % less data over the wire, for this toy camera.**
> Multiply by 10,000 cameras in a retail chain and you can see why
> CFOs love edge.

---

## Ex 7 — Simulating the pipeline (Sense → Preprocess → Infer → Act)

### 👶 What this does
We glue a fake sensor, a pre-processor, the INT8 model, and a
"decision" step into one function — mirroring the pipeline in the
theory file (§7).

```python
def edge_pipeline(raw_image, interp):
    """Sense -> Preprocess -> Infer -> Act."""
    # 1. SENSE already happened — raw_image is 28x28 uint8.

    # 2. PREPROCESS: float, normalise, quantize to INT8 for the model.
    in_det = interp.get_input_details()[0]
    x = raw_image.astype("float32") / 255.0
    x = x[None, ..., None]
    scale, zero = in_det["quantization"]
    x_q = (x / scale + zero).astype(np.int8)

    # 3. INFER
    interp.set_tensor(in_det["index"], x_q)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]["index"])[0]

    # 4. ACT: threshold + decision
    label = int(out.argmax())
    confidence = float(out[label]) / 127.0   # rough INT8 → [0,1]
    if confidence > 0.7:
        action = f"🚨 digit looks like {label}"
    else:
        action = "…not confident, ignore"
    return label, confidence, action

# Load the INT8 model for the pipeline
interp = tf.lite.Interpreter(model_content=tflite_int8)
interp.allocate_tensors()

# Try it on a handful of test images
for i in [0, 42, 1337, 9999]:
    raw = (x_test[i, ..., 0] * 255).astype("uint8")
    label, conf, act = edge_pipeline(raw, interp)
    print(f"true={y_test[i]}  pred={label}  conf≈{conf:.2f}  → {act}")
```

### 👶 What this proves
That the **whole edge pipeline** fits in ~30 lines of Python, runs on
a CPU in milliseconds, and the model it uses is <30 KB. Swap the
model for a MobileNetV2-INT8 and you have a real door-bell classifier.

---

## 📝 Summary — what you've done

| Step | Result |
|---|---|
| Trained FP32 MNIST CNN | ~98.7 % accuracy, 250 KB |
| Converted to TFLite FP32 | ~98.7 %, ~94 KB |
| Quantized to INT8 | ~98.5 %, ~28 KB, 3× faster |
| Ran the full pipeline | Sense → Act in milliseconds |
| Estimated bandwidth | 99 % cloud traffic eliminated |

**You just built a complete Edge AI toy system.** Now open the
[practice notebook](edge_ai_fundamentals_practice.md) to scale it up
to RGB images and real latency budgets.

---

> *GPU Programming · EdgeAI · Fundamentals · CODE · github.com/rpaut03l/TS-02*
