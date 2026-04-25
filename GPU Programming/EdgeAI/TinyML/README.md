# 🔬 EdgeAI · TinyML

### *AI that runs on a button-cell battery — microcontrollers, kilobytes, milliwatts*

> **Nav:** [← EdgeAI](../README.md) | [← Model Compression](../Model_Compression/README.md) | [← Deployment Frameworks](../Deployment_Frameworks/README.md) | **TinyML** | [Federated Learning →](../Federated_Learning/README.md) | [Edge MLOps →](../Edge_MLOps/README.md) | [Security & Privacy →](../Security_Privacy/README.md)

---

## 👶 30-second story

TinyML is **AI that runs on a grain of rice**.

- **Memory:** kilobytes, not gigabytes (256 KB is huge here).
- **Power:** milliwatts, not watts (a coin cell lasts *years*).
- **Price:** cents per chip.
- **OS:** usually none — bare metal, or a tiny RTOS.
- **Examples:** smoke detectors, fridge door alarms, hearing aids,
  Arduino boards, ESP32, lightbulbs that know you've sat down.

It's a whole different planet from Jetson — but the core ideas
(quantize, shrink, compile) are identical, just pushed to their limit.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [tinyml_theory.md](tinyml_theory.md) | The full theory — what makes TinyML different, the MCU zoo (Cortex-M4/M7/M33/M55, ESP32-S3, Alif Ensemble, RP2040), **TensorFlow Lite Micro** internals (memory arena, op resolver), **CMSIS-NN** hand-tuned kernels, **Ethos-U55/U65 NPU** flow, Edge Impulse studio pipeline, benchmarks (MLPerf Tiny — KWS, VWW, anomaly, IC), the 4 canonical TinyML tasks |
| [tinyml_code.md](tinyml_code.md) | Runnable code — train a keyword-spotting DS-CNN in Colab, export to TFLite full-integer INT8, convert to a C byte array with `xxd`, show the **TFLM main.cc skeleton** that runs it, CMSIS-NN kernel snippets, `vela` compiler CLI for Ethos-U55, Arduino Nano 33 BLE Sense sketch stub |
| [tinyml_practice.md](tinyml_practice.md) | **Colab-ready notebook** — end-to-end TinyML workflow for **visual wake words** (is a person in this 96×96 grayscale image?). Produces a 250 KB `.cc` file ready to flash onto any Cortex-M with at least 256 KB flash + 128 KB RAM |

---

## 🎯 After reading this you should be able to…

- Name the **4 canonical TinyML tasks** and the MLPerf Tiny target
  metric for each
- Tell when a **Cortex-M4F** is enough and when you need an **M55 +
  Ethos-U55**
- Explain the **TFLite Micro memory arena** and how to size it
- Write the 5 mandatory lines every TFLM C++ program needs
- Use **`xxd -i`** to turn a `.tflite` into a C byte array
- Understand why **CMSIS-NN** gives 2–5× on Cortex-M over the default
  TFLM kernels
- Explain what **Ethos-U55 + Vela compiler** buys you
- Outline an **Edge Impulse** project end-to-end

---

## ⚡ The one rule of TinyML

> **Measure twice — in bytes and microjoules.**
> Everything you learned on desktop / Jetson still applies, but the
> *units* are a thousand times smaller.

---

> *GPU Programming · EdgeAI · TinyML · github.com/rpaut03l/TS-02*
