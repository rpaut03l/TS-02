# 🚢 EdgeAI · Deployment Frameworks

### *TensorFlow Lite · ONNX Runtime · OpenVINO — the three runtimes that matter*

> **Nav:** [← EdgeAI](../README.md) | [← Model Compression](../Model_Compression/README.md) | **Deployment Frameworks** | [TinyML →](../TinyML/README.md) | [Federated Learning →](../Federated_Learning/README.md) | [Edge MLOps →](../Edge_MLOps/README.md) | [Security & Privacy →](../Security_Privacy/README.md)

---

## 👶 30-second story

You baked a tiny model in the kitchen (training in PyTorch / TF). Now
you need to **serve it to customers** — in their phone, their camera,
their car. You can't just ship the kitchen. You ship a small **take-
away box**: the **runtime**.

There are three take-away boxes every edge engineer should know:

1. **TensorFlow Lite (TFLite)** — Google's default for Android + MCUs.
2. **ONNX Runtime (ORT)** — vendor-neutral, "run anywhere" box.
3. **OpenVINO** — Intel's box, king of CPU + iGPU + Intel NPU.

This folder compares them, shows when to use which, and walks through
real code for each.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [deployment_frameworks_theory.md](deployment_frameworks_theory.md) | The full tour — TFLite (converter, delegates, ops, select-TF-ops, micro), ONNX Runtime (ExecutionProviders, IOBinding, optimizer), OpenVINO (IR, Model Optimizer, NNCF, AUTO/MULTI/HETERO plugins), side-by-side comparison, cross-platform matrix, model portability & the "ONNX as lingua-franca" strategy |
| [deployment_frameworks_code.md](deployment_frameworks_code.md) | Runnable Python — export a PyTorch model to ONNX; run it through ORT with CPU, CUDA, TensorRT, OpenVINO, CoreML providers; convert to TFLite with delegates; convert to OpenVINO IR; benchmark all backends side-by-side |
| [deployment_frameworks_practice.md](deployment_frameworks_practice.md) | **Colab notebook** — take MobileNetV2, round-trip it through all three runtimes, produce a "portability report" showing which ops are supported where and what latency you get on each |

---

## 🎯 After reading this you should be able to…

- Pick the right runtime for a given target chip in < 30 seconds
- Export **any** PyTorch / TF model to ONNX and troubleshoot common
  export errors
- Use ONNX Runtime's **Execution Providers** to target CUDA, TensorRT,
  OpenVINO, CoreML, QNN
- Use TFLite's **delegates** (GPU, NNAPI, Hexagon, CoreML) and
  understand what "select-TF-ops" costs
- Use OpenVINO's **AUTO / MULTI / HETERO** device plugins
- Read a framework's **supported-ops matrix** and spot the dealbreakers
- Write a portable CI pipeline that produces all three artifacts from
  one training run

---

## ⚡ One-liner picker

| Your target | First runtime to try |
|---|---|
| **Android phone / tablet** | TFLite (with NNAPI / Hexagon delegate) |
| **iPhone / iPad** | Core ML (via `coremltools` from ONNX) |
| **NVIDIA Jetson** | ONNX Runtime + **TensorRT EP** (or TensorRT directly) |
| **Intel laptop / industrial PC** | OpenVINO |
| **MCU (Cortex-M, ESP32)** | TFLite Micro (see [TinyML/](../TinyML/README.md)) |
| **Vendor-neutral / multi-chip** | ONNX Runtime |
| **Cloud-to-edge pipeline** | ONNX first, then convert per target |

---

> *GPU Programming · EdgeAI · Deployment Frameworks · github.com/rpaut03l/TS-02*
