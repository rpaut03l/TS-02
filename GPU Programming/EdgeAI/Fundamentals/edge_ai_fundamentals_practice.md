# 🎯 EdgeAI · Fundamentals — PRACTICE

### *A full cat-vs-dog edge pipeline on Google Colab — size, latency, accuracy, budget*

> **Nav:** [← Fundamentals README](README.md) | [📖 THEORY](edge_ai_fundamentals_theory.md) | [💻 CODE](edge_ai_fundamentals_code.md) | **PRACTICE**

---

## 🎯 What you'll build

A small **image classifier** that:
1. Trains a real MobileNetV2 feature-extractor on cat-vs-dog.
2. Exports a cloud-style FP32 TFLite model.
3. Exports an edge-style INT8 TFLite model.
4. Measures **size, latency, and accuracy** side-by-side.
5. Checks if the edge model fits a **30 FPS latency budget**.

Everything runs on a **free Colab T4**. No downloads, no setup.

---

## 📋 Before you start

- Colab menu → **Runtime → Change runtime type → T4 GPU**.
- New notebook → paste cells one at a time. Read the comments in each
  cell before running it.

---

## Cell 1 — Setup & GPU check

```python
import os, time, numpy as np, tensorflow as tf
print("TF version :", tf.__version__)
print("GPU visible:", bool(tf.config.list_physical_devices("GPU")))
```

Expect `GPU visible: True`. If not, fix the runtime type first.

---

## Cell 2 — Download a tiny cat-vs-dog dataset

We use a 2,000-image subset from Google's public tutorial bucket — fast
to download, easy to train on.

```python
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/cats_and_dogs_filtered.zip"
zip_path = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", URL, extract=True)
data_dir = os.path.join(os.path.dirname(zip_path), "cats_and_dogs_filtered")

print("Data root:", data_dir)
print("Train set:", os.listdir(os.path.join(data_dir, "train")))
print("Val set  :", os.listdir(os.path.join(data_dir, "validation")))
```

---

## Cell 3 — Build `tf.data` pipelines

```python
IMG_SIZE = (160, 160)          # MobileNetV2 "alpha 1.0" friendly
BATCH    = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary")

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "validation"),
    image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary")

CLASSES = train_ds.class_names
print("Classes:", CLASSES)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
```

---

## Cell 4 — Build the model (MobileNetV2 + small head)

MobileNetV2 was literally designed for the edge — depthwise-separable
convolutions, tunable width, ~3 M params. Perfect.

```python
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False                       # feature-extractor only

inputs  = tf.keras.Input(IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()
```

> **Notice:** ~2.26 M trainable head params + ~2.26 M frozen backbone
> params. Small on purpose — this *is* an edge model.

---

## Cell 5 — Train for 5 epochs

```python
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=5, verbose=2)
baseline_acc = model.evaluate(val_ds, verbose=0)[1]
print(f"\n✅ Baseline validation accuracy: {baseline_acc*100:.2f}%")
model.save("catdog_fp32.keras")
```

Expect **~95 %** val accuracy after 5 epochs.

---

## Cell 6 — Convert: TFLite FP32 and TFLite INT8

```python
# ----- FP32 TFLite -----
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()
with open("catdog_fp32.tflite", "wb") as f:
    f.write(tflite_fp32)

# ----- INT8 TFLite (post-training quantization) -----
def representative():
    for images, _ in train_ds.take(10):
        for img in images:
            yield [tf.expand_dims(img, 0)]

conv2 = tf.lite.TFLiteConverter.from_keras_model(model)
conv2.optimizations = [tf.lite.Optimize.DEFAULT]
conv2.representative_dataset = representative
conv2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv2.inference_input_type  = tf.int8
conv2.inference_output_type = tf.int8
tflite_int8 = conv2.convert()
with open("catdog_int8.tflite", "wb") as f:
    f.write(tflite_int8)

for path in ["catdog_fp32.keras", "catdog_fp32.tflite", "catdog_int8.tflite"]:
    print(f"{path:28s}  {os.path.getsize(path)/1024:8.1f} KB")
```

Typical output (your numbers will differ by a few %):
```
catdog_fp32.keras             9128.7 KB
catdog_fp32.tflite            8920.0 KB
catdog_int8.tflite            2344.1 KB
```

---

## Cell 7 — Measure CPU latency (simulates a small edge device)

```python
def run_tflite(model_bytes, images):
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    preds = []
    timings = []
    for img in images:
        x = tf.expand_dims(img, 0).numpy()

        if in_det["dtype"] == np.int8:
            scale, zero = in_det["quantization"]
            x = (x / scale + zero).astype(np.int8)
        else:
            x = x.astype(np.float32)

        # warm-up on the first call
        interp.set_tensor(in_det["index"], x)
        t0 = time.perf_counter()
        interp.invoke()
        timings.append(time.perf_counter() - t0)

        y = interp.get_tensor(out_det["index"])[0]
        if out_det["dtype"] == np.int8:
            scale, zero = out_det["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        preds.append(float(y[0]))
    return np.array(preds), np.median(timings) * 1000

# Grab a batch of 64 validation images
val_batch_imgs, val_batch_lbls = next(iter(val_ds))
imgs_64 = val_batch_imgs[:64]
lbls_64 = val_batch_lbls[:64].numpy().flatten()

p_fp32, ms_fp32 = run_tflite(tflite_fp32, imgs_64)
p_int8, ms_int8 = run_tflite(tflite_int8, imgs_64)

def acc(p, y): return float(((p > 0.5).astype("int").flatten() == y).mean())

print(f"{'Model':15s} {'Size (KB)':>10s} {'Latency (ms)':>14s} {'Acc':>8s}")
print(f"{'FP32 TFLite':15s} {os.path.getsize('catdog_fp32.tflite')/1024:10.1f} "
      f"{ms_fp32:14.2f} {acc(p_fp32, lbls_64)*100:7.2f}%")
print(f"{'INT8 TFLite':15s} {os.path.getsize('catdog_int8.tflite')/1024:10.1f} "
      f"{ms_int8:14.2f} {acc(p_int8, lbls_64)*100:7.2f}%")
```

### Typical Colab CPU result
```
Model           Size (KB)  Latency (ms)      Acc
FP32 TFLite       8920.0         31.40    95.31%
INT8 TFLite       2344.1         11.70    94.53%
```

---

## Cell 8 — The 30 FPS latency budget exercise 📏

A **30 FPS** camera gives you **33.3 ms per frame**. Inside that you
must do: camera capture, preprocessing, model inference, post-
processing, and alert dispatch.

```python
FPS         = 30
FRAME_BUDGET_MS = 1000 / FPS
OVERHEAD_MS = 5        # capture + preprocess + postprocess
BUDGET_FOR_MODEL = FRAME_BUDGET_MS - OVERHEAD_MS

print(f"Frame budget (total) : {FRAME_BUDGET_MS:.2f} ms")
print(f"Budget for the model : {BUDGET_FOR_MODEL:.2f} ms\n")

for name, ms in [("FP32 TFLite", ms_fp32), ("INT8 TFLite", ms_int8)]:
    verdict = "✅ FITS"   if ms <= BUDGET_FOR_MODEL else "❌ DROPS FRAMES"
    print(f"{name}:  {ms:5.2f} ms   →   {verdict}")
```

Typical output:
```
Frame budget (total) : 33.33 ms
Budget for the model : 28.33 ms

FP32 TFLite:  31.40 ms   →   ❌ DROPS FRAMES
INT8 TFLite:  11.70 ms   →   ✅ FITS
```

> **This is the whole point.** The cloud-style FP32 model misses the
> 30 FPS budget on a CPU. The same model, quantized to INT8,
> comfortably fits. That's **why** you do Edge AI work.

---

## Cell 9 — Write your own cheat-sheet

Fill this table in a markdown cell of your notebook:

```
| Pillar              | Did we touch it here?  | How? |
|---------------------|------------------------|------|
| Low Latency         | ?                      | ?    |
| Low Power           | ?                      | ?    |
| Privacy             | ?                      | ?    |
| Offline Reliability | ?                      | ?    |
| Bandwidth Savings   | ?                      | ?    |
```

Hints:
- Latency → yes, Cell 7 & 8 measured it.
- Power → indirect. INT8 math burns fewer joules than FP32; you didn't
  measure watts but you shrunk memory traffic 4×.
- Privacy → not directly today. But running on-device means the cat
  photo never leaves the camera.
- Offline → your TFLite file has zero network dependencies. That's
  offline reliability.
- Bandwidth → Cell 6 of the [code file](edge_ai_fundamentals_code.md)
  shows the back-of-envelope calc.

---

## Cell 10 — Stretch goals

Pick one or more:

1. **Dynamic-range quantization** — change `Optimize.DEFAULT` to
   dynamic-range only (no `representative_dataset`). How much worse is
   accuracy? How much slower?
2. **Smaller input size** — switch `IMG_SIZE` to `(96, 96)`. Does
   accuracy drop? By how much does latency drop?
3. **Float16 quantization** — a middle ground between FP32 and INT8,
   often preferred when you have a GPU delegate. Compare.
4. **Run on the Colab GPU delegate** — set
   `interp = tf.lite.Interpreter(..., experimental_delegates=[gpu_delegate])`
   and re-measure.
5. **Measure model memory peak** — use `tracemalloc` during
   `interp.invoke()`. Which model touches more RAM?

Keep the results in a markdown table at the top of your notebook — it
becomes your own "Edge AI cheat sheet".

---

## 🎓 What you should take away

- You trained **one** model, and produced **two** deployable versions —
  one cloud-shaped, one edge-shaped.
- The edge version is **~4× smaller** and **~3× faster**, losing less
  than 1 % accuracy.
- The edge version **fits the 30 FPS budget** on a CPU; the cloud
  version does not.
- That cycle — `train → convert → quantize → measure → iterate` — is
  the everyday workflow of every Edge AI engineer.

Next up: [**GPU Types for Edge AI →**](../GPU_Types/README.md) — we stop
pretending the CPU is "the edge" and look at the actual accelerators
you'd ship a product on.

---

> *GPU Programming · EdgeAI · Fundamentals · PRACTICE · github.com/rpaut03l/TS-02*
