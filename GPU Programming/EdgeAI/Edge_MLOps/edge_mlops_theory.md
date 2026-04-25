# 📖 EdgeAI · Edge MLOps — THEORY

### *CI · OTA · Monitoring · Drift · Rollback — for fleets in the wild*

> **Nav:** [← Edge MLOps README](README.md) | **THEORY** | [💻 CODE](edge_mlops_code.md) | [🎯 PRACTICE](edge_mlops_practice.md)

---

## 🧠 MNEMONIC: **"B-S-D-M-R"**

> **B**uild · **S**ign · **D**eploy · **M**onitor · **R**ollback

Every release on an edge fleet goes through these five gates. Skip
one and you will be paged at 3 a.m.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | What's different about Edge MLOps | [§1](#1-whats-different-about-edge-mlops) |
| 2 | The 6-stage release pipeline | [§2](#2-the-6-stage-release-pipeline) |
| 3 | **OTA update strategies** | [§3](#3-ota-update-strategies) |
| 4 | Staged rollouts & canaries | [§4](#4-staged-rollouts--canaries) |
| 5 | Device telemetry — what to emit, how, when | [§5](#5-device-telemetry--what-to-emit-how-when) |
| 6 | **Drift detection** without labels | [§6](#6-drift-detection-without-labels) |
| 7 | Shadow mode & champion/challenger | [§7](#7-shadow-mode--championchallenger) |
| 8 | Fleet management platforms | [§8](#8-fleet-management-platforms) |
| 9 | Observability at the edge | [§9](#9-observability-at-the-edge) |
| 10 | Cheat sheet | [§10](#10-cheat-sheet--red-flags) |

---

## 1. What's different about Edge MLOps

### 👶 Easy Story
Cloud MLOps is **one server**. You redeploy, everyone sees it
instantly. Edge MLOps is **ten million** servers, each with its own
battery, Wi-Fi, firmware version, and mood. You can't ssh into them.
They reboot on their own. They go offline for days.

### The edge-specific pains

| Pain | Cloud | Edge |
|---|---|---|
| Environment | 1–N homogeneous servers | N heterogeneous devices |
| Update speed | seconds | hours → days (OTA) |
| Failure recovery | restart the pod | **physical return** (RMA) |
| Bandwidth | unlimited | metered, spotty, paid |
| Observability | full logs everywhere | tiny metrics only |
| Rollback | git revert + redeploy | dual partitions + rollback flag |
| Labels for drift | plentiful | scarce, noisy, biased |
| Security surface | secure DC | physical access by users |

### So what?
> **Every single MLOps practice you've learned in the cloud costs
> 10× more on the edge.** Which is why the core practices — CI,
> signing, staged rollouts, drift detection — are *more* important on
> the edge, not less.

[↑ Back to Top](#-edgeai--edge-mlops--theory)

---

## 2. The 6-stage release pipeline

```
 ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
 │ 1. TRAIN   │→ │ 2. CONVERT │→ │ 3. PACKAGE │→ │ 4. SIGN    │→ │ 5. DEPLOY  │→ │ 6. OBSERVE │
 │            │  │ ONNX/TFLite│  │ manifest + │  │ ed25519 /  │  │ staged OTA │  │ telemetry, │
 │  PyTorch/TF│  │ OpenVINO   │  │ artifacts  │  │ HSM        │  │ A/B, canary│  │ drift,     │
 │  + eval    │  │ + quant    │  │ + hashes   │  │            │  │            │  │ rollback   │
 └────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

### What's produced at each stage
1. **Train** → `.pt` / `.h5` + evaluation report.
2. **Convert** → `.onnx` (always) + target artifacts (`.tflite`,
   `.xml/.bin`, `.mlpackage`, TensorRT `.engine`).
3. **Package** → `manifest.json` with versions, hashes, target
   platform, minimum runtime version, rollback target.
4. **Sign** → `ed25519` or RSA-PSS signature over the manifest. Device
   verifies with a baked-in public key.
5. **Deploy** → pushed to the fleet manager, rolled out in stages.
6. **Observe** → telemetry (latency, errors, drift) fed back into the
   training loop.

### The 2 mandatory rules
- **Every stage is reproducible from CI** (no laptop-only conversion).
- **Every artifact is addressable by hash** (no "latest" tags on the
  wire).

[↑ Back to Top](#-edgeai--edge-mlops--theory)

---

## 3. OTA update strategies

### Three flavours

```
 ┌────────────────────┬──────────────────────────────────────────────┐
 │ Strategy           │ Description                                   │
 ├────────────────────┼──────────────────────────────────────────────┤
 │ Full-image OTA     │ Replace the entire firmware image.             │
 │                    │ Simple, big download (~MB–GB).                 │
 │                    │ Used when OS + app + model ship together.      │
 │                    │                                                │
 │ Model-only OTA     │ Swap just the model file. App stays the same.  │
 │                    │ Small download (~KB–MB). Needs versioning      │
 │                    │ between app and model.                         │
 │                    │                                                │
 │ Delta / binary-    │ Ship only the binary diff against the previous │
 │ diff OTA           │ version (bsdiff, xdelta, Courgette).           │
 │                    │ ~5–20× smaller payloads. More complex.          │
 └────────────────────┴──────────────────────────────────────────────┘
```

### Dual-partition A/B

```
  ┌───────────────┐   ┌───────────────┐
  │  Partition A  │   │  Partition B  │
  │  (active)     │   │  (staging)    │
  └───────────────┘   └───────────────┘
         ▲                    │
         │  if B boots OK,    │ new OTA flashes into B
         │  swap pointers     │ reboot into B
```

### Rules of the road
1. **Never overwrite the running partition.** Flash into the other
   one; reboot; run a self-test.
2. **Bootloader fallback** — if the new image doesn't mark itself
   "healthy" within N boots, boot back into the old partition.
3. **Signed manifest first, payload second.** Verify the signature
   *before* erasing anything.
4. **Version skew tolerance** — the model must be compatible with
   the runtime version on the device. Encode it in the manifest.
5. **Resumable downloads** — the device may go offline mid-OTA.

[↑ Back to Top](#-edgeai--edge-mlops--theory)

---

## 4. Staged rollouts & canaries

### The canonical ramp

```
 Day 0:    1 %  internal devices      (1,000 of 100k)
 Day 1:    5 %  beta testers
 Day 3:   10 %  early adopters
 Day 7:   25 %  general rollout
 Day 14: 100 %  everyone
```

### What to watch at each stage
- **Crash rate** — must not rise.
- **Latency P50 / P95 / P99** — must not rise.
- **Model metric proxy** (e.g. confidence drop, anomaly-score mean) —
  must not shift.
- **Battery impact** — must not shift more than X %.
- **Business KPIs** — whatever "the model is useful" looks like.

### Automatic halt criteria
- P99 latency > 1.3× baseline for 2 consecutive hours.
- Crash rate > 2× baseline for 30 minutes.
- Drift score > threshold across ≥ 10 % of canary devices.

### Rollback trigger
Either (a) automated criteria fire, or (b) human on-call flips a
"promotion" flag. The fleet manager instantly stops pushing the new
version and (optionally) forces devices that already received it to
boot their old partition.

[↑ Back to Top](#-edgeai--edge-mlops--theory)

---

## 5. Device telemetry — what to emit, how, when

### The "thin" telemetry philosophy
Bandwidth is expensive and batteries are finite. **Emit small
structured records, not logs.** Batch them, compress them, ship on a
schedule.

### The minimum useful record (per inference, sampled)

```json
{
  "device_id": "abcd1234",
  "fw_version": "2.3.1",
  "model_version": "vww.2024.04.12.int8",
  "ts_ms": 1714100000000,
  "latency_ms": 18.2,
  "pred_class": 1,
  "confidence": 0.87,
  "input_stats": {"mean": 102.4, "std": 41.3}
}
```

### Sampling rules
- **Always** send errors and crashes.
- **1:100** sampling for successful inferences.
- **Flush** when buffer > 4 KB or every 10 minutes.
- **Back off** on failed uploads (exponential).
- **Clock:** ship `ts_ms` from device but don't trust it — servers
  correct for skew.

### Transport
- **MQTT** (common for IoT fleets), **HTTP/2 or gRPC** (phones),
  **LoRaWAN** (ultra-low-bandwidth sensors).
- Always over **TLS** with a **pinned CA** or pinned server public
  key.

### Crash signatures
For C / C++ firmware: **CFI** + a tiny crash-dump buffer that survives
reboots. Upload on next boot. The single highest-leverage thing you
can do for fleet reliability.

[↑ Back to Top](#-edgeai--edge-mlops--theory)

---

## 6. Drift detection without labels

### 👶 Easy Story
You don't have labels from the device (nobody tells you which prediction
was right). So you can't measure accuracy directly. But you **can**
measure whether **the world looks different** than it did when you
trained — that's **drift**.

### Two kinds of drift
- **Data drift** — P(x) changed. The model's inputs look different.
- **Concept drift** — P(y|x) changed. The right answer for the same
  input is now different. Much harder to detect.

### Signals that work without labels
1. **Input feature stats** — mean / std / quantile of each feature.
   Compare to training-set values using KS test, PSI, or JS-divergence.
2. **Prediction distribution** — fraction of each class. If it drifts
   vs training, something changed.
3. **Confidence distribution** — shift to lower mean confidence often
   precedes accuracy drop.
4. **Embedding norm** — the L2 norm of an intermediate feature vector.
5. **Reconstruction error** (if using an autoencoder side-head).

### The PSI metric (simple, cheap, popular)
```
 PSI = Σ_i (p_i − q_i) · log(p_i / q_i)
 where p_i, q_i are bin probabilities of feature f at train vs now.
  < 0.1      → stable
  0.1 – 0.25 → moderate drift, investigate
  > 0.25     → significant drift, consider retraining
```

### A device-friendly drift monitor
- Compute a tiny **histogram** (e.g. 16 bins) of the input feature
  mean.
- Ship one histogram per hour per device.
- Server aggregates and raises alerts when PSI > 0.25 over 3
  consecutive hours.

[↑ Back to Top](#-edgeai--edge-mlops--theory)

---

## 7. Shadow mode & champion/challenger

### Shadow mode
Run the **new model side-by-side** with the old one, but only the old
one's predictions affect the user. Record both. Compare. Deploy the
new one only if the diff looks good.

### Champion / challenger
Deploy **two models** in production to a random split of devices.
Compare long-term KPIs (not just loss). Promote the winner.

### On-device twist
Both patterns are **expensive** — you're running two models on a
battery. Two mitigations:
- **Split the fleet** — half the devices run the champion, half the
  challenger. No per-device double cost.
- **Shadow only on plug / charge** — only run the second model when
  the device is charging, so no user-visible battery impact.

[↑ Back to Top](#-edgeai--edge-mlops--theory)

---

## 8. Fleet management platforms

### The main options

| Platform | Sweet spot | Model deploy story |
|---|---|---|
| **Balena** | Small/medium IoT fleets | Docker-based OTA per container |
| **AWS IoT Greengrass** | AWS-native fleets | Lambda + model artifacts via AWS IoT Jobs |
| **Azure IoT Edge** | Azure-native fleets | Docker modules with model env-vars / mounts |
| **Google Distributed Cloud Edge** | Google-native fleets | Kubernetes-based; models as GKE workloads |
| **NVIDIA Fleet Command** | NVIDIA-centric (Jetson + EGX) | Container registry + signed manifests |
| **BalenaEtcher / Mender** | Firmware OTA (non-Docker) | Full-image OTA, A/B, robust offline |
| **Home-rolled MQTT + S3** | When you need full control | Manifests + signed tarballs over MQTT |

### Matching platform to device
```
 Docker-capable Linux device (Jetson, iMX, RPi) → Balena / Greengrass / Azure IoT Edge
 Full-image firmware device (Yocto, buildroot)  → Mender / RAUC / balenaOS
 MCU (no OS, < 1 MB RAM)                        → Home-rolled OTA protocol,
                                                   custom bootloader
```

### What every platform needs to do
1. Authenticate the device.
2. Signal "new manifest available".
3. Stream signed artifact.
4. Verify signature, hash, target platform.
5. Flash into a staging partition / container.
6. Report success / failure.
7. Flip pointers on success. Roll back on failure.
8. Stream telemetry back.

[↑ Back to Top](#-edgeai--edge-mlops--theory)

---

## 9. Observability at the edge

### OpenTelemetry at the edge
- **Traces** — tag each inference with a span; correlate to server
  requests (for hybrid architectures).
- **Metrics** — counters (inferences_total), histograms
  (latency_ms), gauges (model_version, drift_score).
- **Logs** — structured, rate-limited, error-level only.

### The "5 golden signals" for Edge AI

1. **Inference latency** P50 / P95 / P99.
2. **Error rate** — failed invokes, exception counts.
3. **Model version distribution** across the fleet (should converge
   to latest).
4. **Drift score** — per feature, per device.
5. **Business KPI proxy** — confidence distribution, prediction
   distribution.

Dashboard these five and you've caught 90 % of problems before users do.

### Secondary but high-value
- **Battery / power** — mW while busy, hours on a charge.
- **Thermal** — temperature over time (throttling predictor).
- **Storage** — flash wear-level estimate.
- **Network** — upload retries, mean upload latency.

[↑ Back to Top](#-edgeai--edge-mlops--theory)

---

## 10. Cheat sheet & red flags

### Cheat sheet
```
 B-S-D-M-R       Build · Sign · Deploy · Monitor · Rollback
 OTA FLAVOURS    Full-image · Model-only · Delta
 DUAL-PARTITION  Always. Bootloader fallback included.
 ROLLOUT RAMP    1 % → 5 % → 10 % → 25 % → 100 %, days apart
 TELEMETRY       Tiny, structured, TLS, pinned, batched
 DRIFT           PSI / KS over feature histograms (no labels needed)
 SHADOW MODE     Double-run new model, decide, then switch
 PLATFORMS       Balena · Greengrass · IoT Edge · Fleet Command · custom
 GOLDEN SIGNALS  latency · errors · model-version dist · drift · KPI
```

### Red flags 🚩
- 🚩 OTA without signatures. Don't do it. Ever.
- 🚩 "Promote-to-prod" button that skips the canary.
- 🚩 Dashboards showing 100 % of the fleet as "healthy" — probably the
  offline ones are just not reporting.
- 🚩 No drift monitor. You're flying blind.
- 🚩 Rollback untested. If you haven't exercised it in the last 30
  days, it doesn't work.

### Green flags ✅
- ✅ CI produces **all** artifact formats from a single ONNX.
- ✅ Every release has a **manifest + signature + staged plan**.
- ✅ Dual-partition A/B with 3-boot healthy window.
- ✅ Drift dashboard with PSI alerts.
- ✅ Monthly **drill** that rolls back a release on purpose.

---

## 🔭 Next up

Final folder [`Security_Privacy/`](../Security_Privacy/README.md) —
how to keep the model, the data, and the weights safe from attackers
who have **physical access** to the device.

---

> *GPU Programming · EdgeAI · Edge MLOps · THEORY · github.com/rpaut03l/TS-02*
