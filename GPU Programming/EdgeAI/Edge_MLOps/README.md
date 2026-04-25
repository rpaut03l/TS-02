# 🔁 EdgeAI · Edge MLOps

### *Ship · Update · Monitor · Roll back — models that live in millions of devices*

> **Nav:** [← EdgeAI](../README.md) | [← Model Compression](../Model_Compression/README.md) | [← Deployment Frameworks](../Deployment_Frameworks/README.md) | [← TinyML](../TinyML/README.md) | [← Federated Learning](../Federated_Learning/README.md) | **Edge MLOps** | [Security & Privacy →](../Security_Privacy/README.md)

---

## 👶 30-second story

You baked a cake. The cake is now **in ten million kitchens**. One
morning you realise the recipe had a typo. How do you:

1. **Update** the recipe in all ten million kitchens?
2. **Roll back** if the new recipe is worse?
3. **Know** if some kitchens are burning the cake?
4. **Not break** the kitchens whose internet is down?
5. **Monitor** whether the cake still tastes right three months later?

That whole problem set is **Edge MLOps**. It's MLOps minus central
infrastructure and plus **fleet, network, and power** constraints.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [edge_mlops_theory.md](edge_mlops_theory.md) | Full theory — CI/CD for edge (build + sign + publish), **OTA (Over-The-Air) model updates** (full vs delta, dual-partition A/B, staged rollouts, rollback), on-device **telemetry** (latency, accuracy proxies, drift), **data drift vs concept drift** with limited labels, shadow mode & canary, champion/challenger on-device, device-fleet managers (Balena, AWS Greengrass, Azure IoT Edge, NVIDIA Fleet Command, Google Distributed Cloud Edge), observability stack (OTel for edge, Prometheus) |
| [edge_mlops_code.md](edge_mlops_code.md) | Runnable code — CI pipeline YAML for the triple artifact (.tflite + .xml/.bin + .onnx), a **signed OTA** in Python (SHA-256 + ed25519), a **delta update** with bsdiff/bspatch, a tiny staged-rollout controller, a drift detector (KS test on feature stats), a structured telemetry emitter (JSON lines), an A/B test scorer |
| [edge_mlops_practice.md](edge_mlops_practice.md) | **Colab notebook** — simulate a fleet of 1,000 devices, push an OTA update to 10 %, watch drift + accuracy on the canary, auto-promote or rollback, produce a human-readable release report |

---

## 🎯 After reading this you should be able to…

- Describe the **6 stages** of an Edge MLOps release pipeline
- Tell **full-image OTA** from **model-only OTA** from **delta OTA**
- Design a **dual-partition A/B rollback** scheme for a $2 MCU
- Write a **signed model artifact** flow (hash + signature)
- Emit **structured telemetry** that survives intermittent connectivity
- Detect **data drift** without ground-truth labels on the device
- Pick between Balena / Greengrass / IoT Edge / Fleet Command for a
  concrete scenario

---

## ⚡ Three lines to memorise

> 1. **Every deployment is reversible.** Design rollback first, forward
>    second.
> 2. **Trust nothing from the wire.** Signed artifacts only.
> 3. **The cloud can't see everything.** Emit telemetry that survives
>    offline windows and batches on reconnect.

---

> *GPU Programming · EdgeAI · Edge MLOps · github.com/rpaut03l/TS-02*
