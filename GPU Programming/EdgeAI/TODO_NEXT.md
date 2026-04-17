# 📝 EdgeAI · TODO — next sub-tracks to add

> Captured for the **next session**. These extend the EdgeAI track
> and match the ✅ / 🔭 roadmap in the top-level
> [EdgeAI/README.md](README.md). Same folder style as the existing
> four sub-tracks (README + `_theory.md` + `_code.md` + `_practice.md`).

---

## 🔭 1. Model Compression

- Quantization — **INT8, INT4**, mixed precision, per-channel vs per-tensor
- Post-Training Quantization (PTQ) vs **Quantization-Aware Training (QAT)**
- Pruning — unstructured, structured, magnitude-based, lottery ticket
- **Knowledge distillation** — teacher → student, feature vs logit distillation
- Low-rank factorization, tensor decomposition
- Trade-off curves: size vs accuracy vs latency

Folder: `GPU Programming/EdgeAI/Model_Compression/`
Files: `README.md`, `model_compression_theory.md`,
`model_compression_code.md`, `model_compression_practice.md`

---

## 🔭 2. Deployment Frameworks

- **TensorFlow Lite** — delegates (GPU/NNAPI/Hexagon/Core ML), ops set,
  select-TF-ops, benchmarking tool
- **ONNX Runtime** — Execution Providers (CUDA, TensorRT, OpenVINO,
  QNN, CoreML, DirectML), session options, IOBinding, quantization API
- **Intel OpenVINO** — IR format, Model Optimizer, POT/NNCF, device
  plugins (CPU/GPU/NPU/AUTO/MULTI/HETERO)
- Framework comparison: coverage, speed, ops, ecosystem

Folder: `GPU Programming/EdgeAI/Deployment_Frameworks/`

---

## 🔭 3. TinyML

- **TF Lite Micro** — runtime, memory arena, op resolver
- **CMSIS-NN** — hand-tuned kernels for Cortex-M + Helium
- Cortex-M4 / M7 / M55 + **Ethos-U55 / U65** NPU flow
- ESP32-S3, STM32, Alif Ensemble, Arduino Nano 33 BLE Sense
- Edge Impulse end-to-end studio walkthrough
- MLPerf Tiny benchmarks (keyword spotting, anomaly, visual wake)

Folder: `GPU Programming/EdgeAI/TinyML/`

---

## 🔭 4. Federated Learning & On-Device Training

- Federated Averaging (FedAvg), FedProx, FedSGD
- Cross-device vs cross-silo FL
- On-device fine-tuning — LoRA, adapters on edge devices
- Differential privacy + secure aggregation
- Frameworks: **TensorFlow Federated**, **Flower**, **PySyft**
- Communication efficiency, dropouts, stragglers

Folder: `GPU Programming/EdgeAI/Federated_Learning/`

---

## 🔭 5. Edge MLOps

- **OTA (Over-The-Air) model updates** — signing, A/B rollouts,
  staged deployment, rollback, dual-partition updates
- On-device **telemetry** — latency, accuracy proxies, drift signals
- **Data drift** and **concept drift** at the edge (limited labels)
- Shadow mode, canary, champion-challenger on devices
- Device fleet management — Balena, AWS Greengrass, Azure IoT Edge,
  NVIDIA Fleet Command, Google Cloud IoT (EOL → alternatives)
- Observability stacks — OpenTelemetry for edge, Prometheus node-exporter,
  Grafana Cloud edge

Folder: `GPU Programming/EdgeAI/Edge_MLOps/`

---

## 🔭 6. Security & Privacy on Edge Devices

- **Secure boot** chain of trust, HSM / TEE (ARM TrustZone, Intel SGX,
  Apple Secure Enclave, NVIDIA PSC)
- **Model encryption at rest** and in RAM; encrypted weights
- **Adversarial attacks** on edge models — evasion, extraction,
  membership inference
- **Model extraction** defences, watermarking, fingerprinting
- **Side-channel attacks** — power analysis, timing, EM
- Privacy — on-device PII redaction, federated aggregation,
  differential privacy budget
- Regulations — GDPR, HIPAA, India DPDPA 2023, EU AI Act obligations

Folder: `GPU Programming/EdgeAI/Security_Privacy/`

---

## ✅ Style reminders for tomorrow

- Keep **"👶 easy story first, then technical depth"** voice
- Topic-named folders, **no "Lec_XX" naming**
- Text-only ASCII diagrams inside code fences
- Mnemonic → TOC → numbered sections → cheat sheet → red-flag block
- Every section ends with a "so what?" takeaway
- Nav links at the top of each file
- Footer: `> *GPU Programming · EdgeAI · <Topic> · github.com/rpaut03l/TS-02*`
- After finishing, update the **top-level `EdgeAI/README.md`** so the
  6 🔭 entries flip to ✅ and add rows in the contents table

---

> *Captured at the end of the previous session. Continue here next time.*
