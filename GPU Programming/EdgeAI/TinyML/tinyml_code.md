# 💻 EdgeAI · TinyML — CODE

### *Train → INT8 quant → `.tflite` → C byte array → TFLM main.cc*

> **Nav:** [← TinyML README](README.md) | [📖 THEORY](tinyml_theory.md) | **CODE** | [🎯 PRACTICE →](tinyml_practice.md)

---

## 🏗️ Setup

Most training runs on **Colab (CPU is fine — models are tiny)**.
Device-side C / C++ code is shown but runs on a real MCU (Alif,
STM32, Nano 33 BLE Sense, ESP32-S3, Pico, …).

---

## Ex 1 — Train a keyword-spotting DS-CNN in Colab

### 👶 What this does
The classic TinyML "hello world": recognise 10 short words from the
Google Speech Commands dataset.

```python
import tensorflow as tf, numpy as np, os, pathlib, tensorflow_datasets as tfds

# --- small subset for speed ---
!pip install -q tensorflow-datasets tensorflow-model-optimization

ds = tfds.load("speech_commands", split="train[:10%]", as_supervised=True)
CLASSES = 12   # 10 words + silence + unknown; simplified

def to_spec(audio, label):
    audio = tf.cast(audio, tf.float32) / 32768.0
    stft  = tf.signal.stft(audio, frame_length=640, frame_step=320, fft_length=640)
    mag   = tf.abs(stft)
    mel_w = tf.signal.linear_to_mel_weight_matrix(40, mag.shape[-1], 16000, 20, 4000)
    mel   = tf.math.log(tf.tensordot(mag, mel_w, 1) + 1e-6)
    mel   = mel[:49, :]                                 # fixed 49×40 crop
    return mel[..., None], tf.one_hot(label % CLASSES, CLASSES)

ds = ds.map(to_spec).batch(32).prefetch(tf.data.AUTOTUNE)

def ds_cnn():
    x = tf.keras.Input((49, 40, 1))
    y = tf.keras.layers.Conv2D(64, (10, 4), strides=(2, 2),
                               padding="same", activation="relu")(x)
    for _ in range(4):
        y = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same",
                                            activation="relu")(y)
        y = tf.keras.layers.Conv2D(64, 1, activation="relu")(y)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    return tf.keras.Model(x, tf.keras.layers.Dense(CLASSES, activation="softmax")(y))

m = ds_cnn()
m.compile("adam", "categorical_crossentropy", metrics=["acc"])
m.summary()
m.fit(ds, epochs=5)
m.save("kws.keras")
```

---

## Ex 2 — Full-integer INT8 quantization

### 👶 What this does
MCUs need **everything INT8** — weights, activations, even inputs and
outputs. This is the only quant mode that fits.

```python
def representative():
    for mel, _ in ds.take(10):
        for i in range(mel.shape[0]):
            yield [mel[i:i+1].numpy()]

conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
open("kws_int8.tflite", "wb").write(conv.convert())

print("Model size:", os.path.getsize("kws_int8.tflite") / 1024, "KB")
```

Expect **~30 KB** — small enough to fit on any Cortex-M4 with 256 KB
flash.

---

## Ex 3 — Convert `.tflite` to a C byte array

### 👶 What this does
MCUs can't read files. You compile the model *into* the firmware.

```bash
# On Colab, xxd is pre-installed:
!xxd -i kws_int8.tflite > model_data.cc
!head -n 5 model_data.cc
```

Output:
```c
unsigned char kws_int8_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  …
};
unsigned int kws_int8_tflite_len = 31184;
```

Include it in the firmware build. Rename the identifier to
`g_model[]` and it's ready for TFLM.

---

## Ex 4 — The minimal TFLM `main.cc` (C++)

### 👶 What this does
The 50-line C++ skeleton that every TinyML project starts from. Runs
on any MCU supported by TFLite Micro.

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"              // byte array g_model[] from Ex 3

namespace {
  constexpr int kArenaSize = 32 * 1024;          // 32 KB; tune via arena_used_bytes()
  alignas(16) uint8_t tensor_arena[kArenaSize];

  tflite::MicroInterpreter* interp = nullptr;
}

void setup() {
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Schema mismatch"); while (true);
  }

  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddMean();                 // GAP decomposes to Mean
  resolver.AddReshape();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interp(
      model, resolver, tensor_arena, kArenaSize);
  interp = &static_interp;

  if (interp->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed"); while (true);
  }
  MicroPrintf("Arena used: %u bytes", interp->arena_used_bytes());
}

void loop() {
  // Fill interp->input(0)->data.int8 with your 49x40 mel-spectrogram...

  if (interp->Invoke() != kTfLiteOk) return;

  int8_t* out = interp->output(0)->data.int8;
  int best = 0;
  for (int i = 1; i < 12; ++i) if (out[i] > out[best]) best = i;
  MicroPrintf("Predicted class: %d", best);
}
```

### The 5 essentials you can't skip
1. `alignas(16) uint8_t tensor_arena[...]`
2. `tflite::MicroMutableOpResolver<N>` registering exactly the ops
   the model uses.
3. `MicroInterpreter` constructed with model + resolver + arena.
4. `AllocateTensors()` → check return code.
5. Fill `interp->input(0)->data.int8`, call `Invoke()`, read
   `output(0)`.

---

## Ex 5 — Enable CMSIS-NN kernels

### 👶 What this does
Drop-in 2–5× speed-up on Cortex-M by delegating kernels to
**CMSIS-NN** — Arm's hand-tuned library.

### Option A: via CMake / Zephyr / PlatformIO
Pass this define when compiling TFLM:
```
-DTF_LITE_USE_CMSIS_NN=1
```
and link `libCMSISNN.a` (from the CMSIS_5 / ARM-software/CMSIS-NN repo).

### Option B: via the Arm pack
Use `ARM::CMSIS-NN` component in Keil or STM32CubeIDE and the
`tensorflow-lite-micro-cmsis-nn` module — kernels are swapped
automatically.

### What to expect after enabling
On a Cortex-M4F @ 80 MHz running KWS:
```
Default TFLM kernels : 38 ms / inference
CMSIS-NN kernels     :  9 ms / inference
```

---

## Ex 6 — Ethos-U55 path via the Vela compiler

### 👶 What this does
Take the plain INT8 TFLite and rewrite it so that all compatible ops
run on the **Ethos-U55 NPU** — often 10–100× faster.

```bash
!pip install -q ethos-u-vela
!vela --accelerator-config ethos-u55-128 \
      --output-dir vela_out \
      kws_int8.tflite
!cat vela_out/*.csv | head -n 5    # op report
```

### What you get
- `kws_int8_vela.tflite` — a new `.tflite` with `ethos-u` command-
  stream ops replacing the supported layers.
- A CSV report showing which ops were delegated and which fell back
  to the CPU.

TFLM with the Ethos-U kernel compiled in will pick up the NPU
automatically.

---

## Ex 7 — Arduino sketch stub (Nano 33 BLE Sense)

```cpp
// For the Arduino Nano 33 BLE Sense (nRF52840, Cortex-M4F, 256 KB RAM).
// Install the "Harvard_TinyMLx" or "Arduino_TensorFlowLite" library.
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_data.h"

constexpr int kArenaSize = 70 * 1024;
uint8_t tensor_arena[kArenaSize];

tflite::MicroMutableOpResolver<6> resolver;
tflite::MicroInterpreter* interp;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddSoftmax();

  auto* model = tflite::GetModel(g_model);
  static tflite::MicroInterpreter i(model, resolver, tensor_arena, kArenaSize);
  interp = &i;
  interp->AllocateTensors();
}

void loop() {
  // Read PDM microphone → FFT → mel-spectrogram → interp->input(0)
  // (use the PDM library; omitted for brevity)
  interp->Invoke();
  int8_t* o = interp->output(0)->data.int8;
  Serial.println(o[0]);
  delay(500);
}
```

### How to flash
Arduino IDE → select the Nano 33 BLE Sense board → Upload. The first
inference runs about 1 second after boot.

---

## Ex 8 — Power measurement stub

You can't do this on Colab — you need a **scope** or **ULPMark
reference monitor**. The code shape you'd wrap the benchmark in on
the device:

```cpp
// Toggle a GPIO around the Invoke() call; measure the pulse on an
// oscilloscope to get exact inference time and average power.
digitalWrite(PIN_BENCH, HIGH);
interp->Invoke();
digitalWrite(PIN_BENCH, LOW);
```

With a 10 Ω shunt + scope, you can calculate **energy per inference**
in microjoules — the real TinyML metric.

---

## Ex 9 — ESP32-S3 flow (Espressif's `esp-tflite-micro`)

```c
// idf_component.yml
/*
dependencies:
  espressif/esp-tflite-micro: "*"
*/
```

```c
#include "tensorflow/lite/micro/micro_interpreter.h"
// ...same as the generic TFLM main, but the build picks up the
// optimised ESP-NN kernels (vector DSP acceleration).
```

Espressif maintains **ESP-NN** — the ESP32-S3 equivalent of CMSIS-NN,
targeting the Xtensa vector DSP. Typical speed-up: ~10× over plain C
kernels.

---

## Ex 10 — Edge Impulse one-liner export

Assume you've trained a VWW model in [Edge Impulse Studio](https://studio.edgeimpulse.com/):

- **Deployment → Arduino Library** → zip download.
- Unzip, drop into Arduino `libraries/`.
- The zip contains: `model-parameters/`, `tflite-model/`,
  `edge-impulse-sdk/` with TFLM + CMSIS-NN + drivers pre-wired.
- In your sketch:

```cpp
#include <your_project_inferencing.h>
ei_impulse_result_t result;
signal_t signal; numpy::signal_from_buffer(buf, buf_len, &signal);
run_classifier(&signal, &result, /*debug*/ false);
for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; ++i)
  Serial.println(result.classification[i].value);
```

Done — no CMake, no memory arena math, no Vela CLI. Use this for
prototypes and graduate to hand-rolled TFLM when you need fine
control.

---

## 📝 Summary

| Exercise | Artifact | Where it runs |
|---|---|---|
| 1 | `kws.keras` | Colab |
| 2 | `kws_int8.tflite` | Desktop → MCU |
| 3 | `model_data.cc` | Firmware build tree |
| 4 | Minimal TFLM `main.cc` | Any Cortex-M with TFLM |
| 5 | CMSIS-NN enabled | Cortex-M4/M7/M33/M55 |
| 6 | Vela-compiled model | Ethos-U55 MCUs |
| 7 | Arduino sketch | Nano 33 BLE Sense |
| 8 | GPIO pulse benchmark | Scope / ULPMark |
| 9 | `esp-tflite-micro` | ESP32-S3 |
| 10 | Edge Impulse zip | Any supported MCU |

Now glue them together in the [practice notebook](tinyml_practice.md).

---

> *GPU Programming · EdgeAI · TinyML · CODE · github.com/rpaut03l/TS-02*
