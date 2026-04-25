# 🎯 EdgeAI · TinyML — PRACTICE

### *End-to-end Visual Wake Words — train → INT8 → C array → arena-ready*

> **Nav:** [← TinyML README](README.md) | [📖 THEORY](tinyml_theory.md) | [💻 CODE](tinyml_code.md) | **PRACTICE**

---

## 🎯 What you'll build

A Colab notebook that produces a **ship-ready firmware asset**: a
`.cc` file with the byte array of a **Visual Wake Words** (is there a
person in the frame?) classifier, small enough to fit on a
Cortex-M4F with 256 KB flash + 96 KB RAM.

Steps:
1. Train a tiny MobileNet-style model on 96×96 grayscale.
2. Full-integer INT8 quantize.
3. Verify the model size and *arena* budget.
4. Export the C byte array and a tiny Python pre-processor that
   matches what the firmware will do.
5. Run the model with TFLM's Python simulator (`tflite_runtime` or
   `tensorflow.lite.Interpreter`) to mimic on-device results.

---

## Cell 1 — Setup & tiny dataset

```python
import os, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
print("TF:", tf.__version__)

# Tiny stand-in VWW dataset: cat-vs-dog resized to 96x96 grayscale
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/cats_and_dogs_filtered.zip"
zp  = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", URL, extract=True)
root = os.path.join(os.path.dirname(zp), "cats_and_dogs_filtered")

IMG = (96, 96); BATCH = 64

def to_grayscale(x, y):
    x = tf.image.rgb_to_grayscale(x)
    return x, y

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{root}/train",      image_size=IMG, batch_size=BATCH, label_mode="binary")
val_ds   = tf.keras.utils.image_dataset_from_directory(
    f"{root}/validation", image_size=IMG, batch_size=BATCH, label_mode="binary")

train_ds = train_ds.map(to_grayscale).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(to_grayscale).prefetch(tf.data.AUTOTUNE)
```

> In real VWW you'd use MSCOCO re-labelled for person vs not-person.
> Cat-vs-dog is our stand-in so the notebook runs in Colab free tier.

---

## Cell 2 — Tiny architecture

```python
def tiny_vww():
    inp = tf.keras.Input((96, 96, 1))
    x = tf.keras.layers.Rescaling(1/255.0)(inp)
    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same",
                               activation="relu")(x)       # 48x48x8
    for filters in [16, 24, 32]:
        x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same",
                                            activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, 1, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

m = tiny_vww()
m.compile("adam", "binary_crossentropy", metrics=["acc"])
m.summary()
```

Target: **< 100 KB** of INT8 weights after quantization.

---

## Cell 3 — Train

```python
m.fit(train_ds, validation_data=val_ds, epochs=6, verbose=2)
m.save("vww.keras")
print("Keras size:", os.path.getsize("vww.keras")/1024, "KB")
```

Don't chase state-of-the-art — aim for **> 85 %** validation accuracy.
TinyML metrics > classic ML metrics.

---

## Cell 4 — Full-integer INT8 quantization

```python
def representative():
    for imgs, _ in train_ds.take(10):
        for img in imgs:
            yield [tf.expand_dims(img, 0)]

conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
tflite = conv.convert()
open("vww_int8.tflite", "wb").write(tflite)

kb = os.path.getsize("vww_int8.tflite") / 1024
print(f"INT8 TFLite size: {kb:.2f} KB")
assert kb < 200, "Too big for many MCUs"
```

---

## Cell 5 — Simulate the on-device accuracy

```python
interp = tf.lite.Interpreter(model_content=tflite)
interp.allocate_tensors()
in_det, out_det = interp.get_input_details()[0], interp.get_output_details()[0]

def predict_int8(x_float):
    scale, zp = in_det["quantization"]
    x_q = (x_float / scale + zp).astype(np.int8)
    interp.set_tensor(in_det["index"], x_q)
    interp.invoke()
    y = interp.get_tensor(out_det["index"])[0]
    scale_o, zp_o = out_det["quantization"]
    return (float(y) - zp_o) * scale_o

correct = total = 0
for imgs, lbls in val_ds:
    imgs = imgs.numpy(); lbls = lbls.numpy().flatten()
    for img, y in zip(imgs, lbls):
        p = 1 if predict_int8(img[None, ...]) > 0.5 else 0
        correct += int(p == int(y)); total += 1
print(f"On-device-equivalent accuracy: {correct/total*100:.2f}%")
```

Expect to be within ~1 % of the Keras FP32 accuracy.

---

## Cell 6 — Export to a C byte array

```python
!xxd -i vww_int8.tflite > vww_model.cc
!sed -i 's/vww_int8_tflite/g_vww_model/' vww_model.cc
!sed -i 's/vww_int8_tflite_len/g_vww_model_len/' vww_model.cc
!head -n 3 vww_model.cc
!tail -n 2 vww_model.cc
```

That's the file you drop into your firmware project next to
`main.cc`. The matching header:

```python
%%writefile vww_model.h
#pragma once
extern const unsigned char g_vww_model[];
extern const unsigned int  g_vww_model_len;
```

---

## Cell 7 — Arena size experiment

### 👶 What this does
Find the **smallest memory arena** the model runs in, on desktop first.
You can't easily do this on the MCU — so size it here.

```python
# Binary search over arena sizes. TFLite's Python API doesn't expose
# arena_used_bytes(), so we use the file-size heuristic: arena must
# hold at least (largest_input + largest_output + scratch).

import itertools
best = None
for kb in [16, 32, 48, 64, 80, 96, 128]:
    interp = tf.lite.Interpreter(model_content=tflite,
                                 experimental_preserve_all_tensors=False)
    try:
        interp.allocate_tensors()
        best = kb
        break
    except RuntimeError:
        continue

# On desktop every size works; this is a placeholder. On-device,
# arena_used_bytes() in TFLM tells you the real answer — usually
# 1.5–2× the largest feature-map tensor.
largest_tensor = max(np.prod(d["shape"]) for d in
                     interp.get_tensor_details()
                     if d["shape"].size)
print(f"Rough arena estimate: {largest_tensor * 1.5 / 1024:.1f} KB")
```

---

## Cell 8 — Pre-processing notebook (to match firmware code)

### 👶 What this does
The MCU's camera will give you e.g. 96×96 unsigned bytes. Document
*exactly* how the firmware should scale them so inference matches
training.

```python
def preprocess_for_device(gray_uint8_96x96):
    """Mirror this exactly in C on the MCU."""
    x = gray_uint8_96x96.astype(np.float32)
    x = x / 255.0                      # same as training Rescaling(1/255)
    scale, zp = in_det["quantization"]
    x_q = np.round(x / scale + zp).astype(np.int8)
    return x_q  # this is what goes into interp.input(0)

# C equivalent:
"""
void preprocess(const uint8_t* img, int8_t* dst) {
  const float scale = %.6ff;
  const int   zp    = %d;
  for (int i = 0; i < 96*96; ++i) {
    float x = (float)img[i] / 255.0f;
    int q = (int)roundf(x / scale + zp);
    if (q > 127) q = 127; if (q < -128) q = -128;
    dst[i] = (int8_t)q;
  }
}
""" % (in_det["quantization"][0], in_det["quantization"][1])
```

Commit both Python and C side by side — when on-device accuracy
drops, this is the first place to check.

---

## Cell 9 — Measurement plan for the real device

| Metric | How to measure |
|---|---|
| Model size in flash | `ls -l` on `vww_model.cc` (or linker map) |
| Arena used | TFLM `interp->arena_used_bytes()` |
| Inference latency | GPIO toggle + oscilloscope |
| Average power | Series shunt + scope, or ULPMark |
| Accuracy on device | Replay a recorded set of images → log predictions |

Reserve **24 hours** for this on a fresh MCU project. Everything before
this cell is easy; the real work starts here.

---

## Cell 10 — Stretch goals

1. **Add CMSIS-NN** in your MCU build and measure the speed-up.
2. **Run Vela** on `vww_int8.tflite` with `ethos-u55-128`. Check the
   op-report — what falls back to CPU? Can you rewrite that layer?
3. **Collect device-specific data.** Train a new model on photos from
   the *exact* camera you ship. On-device accuracy usually jumps
   3–5 %.
4. **Try a smaller input (64×64).** How small can you go and still
   hit 85 % accuracy?
5. **Sparsify by 50 %.** Convert to INT8. Can you fit the same
   accuracy in 50 KB?

---

## 🎓 What you should take away

- TinyML is **the same workflow** as bigger Edge AI, with a 1,000×
  tighter budget.
- **INT8 full-integer** is mandatory. Never ship float TinyML.
- The **memory arena** is the top-of-mind budget. Always log
  `arena_used_bytes()`.
- The `.cc` byte array is the real deliverable — not the `.tflite`.
- Plan to spend **half your TinyML time on data collection and
  device-side measurement**, not on training.

Next: [**Federated Learning →**](../Federated_Learning/README.md) — how
to train on all these tiny devices without ever sending their data to
the cloud.

---

> *GPU Programming · EdgeAI · TinyML · PRACTICE · github.com/rpaut03l/TS-02*
