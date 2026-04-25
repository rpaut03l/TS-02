# 📖 EdgeAI · Fundamentals — THEORY

### *Concepts only — what Edge AI is, why it exists, how it fits*

> **Nav:** [← Fundamentals README](README.md) | **THEORY** | [💻 CODE](edge_ai_fundamentals_code.md) | [🎯 PRACTICE](edge_ai_fundamentals_practice.md)

---

## 🧠 MNEMONIC: **"LIVE-PACE"**

> **L**atency · **I**nference · **V**olume of data · **E**nergy · **P**rivacy · **A**utonomy · **C**ost · **E**dge-cloud split

The 8 reasons Edge AI exists. If you can explain each letter in one line,
you already know half the field.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Why Edge AI matters | [§1](#1-why-edge-ai-matters) |
| 2 | The definition, in 3 sentences | [§2](#2-the-definition-in-3-sentences) |
| 3 | Cloud AI vs Edge AI (full comparison) | [§3](#3-cloud-ai-vs-edge-ai-full-comparison) |
| 4 | The 5 pillars of Edge AI | [§4](#4-the-5-pillars-of-edge-ai) |
| 5 | The edge continuum (4 tiers) | [§5](#5-the-edge-continuum-4-tiers) |
| 6 | Real-world use cases | [§6](#6-real-world-use-cases) |
| 7 | The Edge AI pipeline (Sense → Act) | [§7](#7-the-edge-ai-pipeline-sense--act) |
| 8 | Challenges and trade-offs | [§8](#8-challenges-and-trade-offs) |
| 9 | How Edge AI fits inside the bigger AI world | [§9](#9-how-edge-ai-fits-inside-the-bigger-ai-world) |
| 10 | Cheat sheet & red flags | [§10](#10-cheat-sheet--red-flags) |

---

## 1. Why Edge AI matters

### 👶 Easy Story
Imagine you're playing **catch** with a friend. She throws the ball.
You catch it. That whole thing takes maybe half a second. Your **eyes**
see the ball, your **brain** does the math, your **hand** moves — and it
all happens **inside your body**. You did not pause, take a photo, mail
it to a science lab, wait for the lab to tell you "the ball is 3.2 m
away, moving at 7 m/s," and *then* move your hand. You'd never catch
the ball.

**Edge AI is the body-and-brain version of AI**. The thinking happens
*right next to* the sensor that sees the data.

### The formal story
- **By 2025, there are ~**75 **billion connected devices** in the world
  — cameras, phones, cars, watches, sensors.
- Each of them generates data every second.
- Sending all of it to the cloud is **too slow, too expensive, too
  insecure, and sometimes plain impossible** (the drone in a forest has
  no 5G tower).
- So we push the AI *out to the devices themselves*. That's Edge AI.

### So what?
> **Edge AI exists because "send everything to the cloud" does not
> scale and does not work for things that must react in milliseconds.**

[↑ Back to Top](#-edgeai--fundamentals--theory)

---

## 2. The definition, in 3 sentences

1. **Edge AI** = running AI inference (and sometimes training) **on the
   device that generated the data**, instead of in a remote data-center.
2. "The device" can be anything from a **$2 microcontroller** in a
   light bulb to a **$2,000 industrial PC** next to a CNC machine.
3. The goal is always the same: **lower latency, lower energy, lower
   bandwidth, higher privacy, and offline reliability**.

---

## 3. Cloud AI vs Edge AI (full comparison)

### 👶 Easy Story
- **Cloud AI** = mailing your homework to a genius teacher who lives in
  another city. She's smart, but the post takes days.
- **Edge AI** = a small answer-key you keep in your desk drawer. Not as
  smart as the genius, but the answer is right there in one second.

### The full table

```
 ┌─────────────────┬───────────────────────────┬─────────────────────────────┐
 │ Dimension       │ ☁️  Cloud AI              │ 📱 Edge AI                 │
 ├─────────────────┼───────────────────────────┼─────────────────────────────┤
 │ Where it runs   │ Data-center, far away     │ On the device itself        │
 │ Typical latency │ 50 – 500 ms (round-trip)  │ 1 – 50 ms                   │
 │ Needs internet? │ YES                       │ NO (or optional)            │
 │ Power budget    │ ~500 – 1000 W per GPU     │ 0.1 – 30 W per device       │
 │ Model size      │ Unlimited (100s of GB OK) │ Usually < 100 MB            │
 │ Who sees data?  │ Cloud + vendor            │ Only the device             │
 │ Cost model      │ Pay per call / per hour   │ One-time chip cost          │
 │ Scales to…      │ Millions of calls / sec   │ Millions of devices         │
 │ Update speed    │ Instant (redeploy cloud)  │ Slow (OTA to each device)   │
 │ Best for        │ Huge models, batch jobs   │ Real-time, private, offline │
 └─────────────────┴───────────────────────────┴─────────────────────────────┘
```

### The important trade-off
**Cloud is smarter, Edge is faster.** You rarely pick one forever —
most real products use **both** (edge for the fast reaction, cloud for
the heavy lifting). That hybrid pattern is called the **edge-cloud
split** and you'll meet it again in §5.

---

## 4. The 5 pillars of Edge AI

These are the five reasons a team chooses Edge AI over Cloud AI.

### Pillar 1 — **Low Latency**
- **👶 Story:** catching the ball. Your brain can't wait for the mail.
- **Formal:** inference round-trip under **50 ms** (often under 10 ms).
- **Why it matters:** self-driving cars need to stop in time; AR glasses
  need to overlay graphics in one frame.
- **Numbers to remember:**
  - 🧑 Human reaction time: ~**250 ms**
  - 🚗 Car at 60 km/h moves **1.7 m** in 100 ms — that's the difference
    between a safe stop and a crash.

### Pillar 2 — **Low Power**
- **👶 Story:** a cell-phone that dies in 2 hours is a brick. Edge AI has
  to sip power, not gulp it.
- **Formal:** inference under a **10–15 W** ceiling for wall-powered
  edge devices, and **< 1 W** for battery-powered ones.
- **Why it matters:** battery life, heat, fan noise, case size, cost of
  power-supply components.
- **Unit to remember:** **TOPS / Watt** = Trillion Operations Per Second
  per Watt. Higher is better. Edge NPUs often hit **5–10 TOPS/W**; an
  H100 GPU is only ~**1.4 TOPS/W** (but absolute TOPS is huge).

### Pillar 3 — **Privacy**
- **👶 Story:** your diary never leaves your bedroom.
- **Formal:** raw sensor data (face, voice, medical readings) is
  processed **on-device and never uploaded**. Only anonymised summaries
  or aggregates leave, if anything.
- **Why it matters:** regulations (GDPR, India's DPDPA, HIPAA), user
  trust, competitive moat.
- **Real example:** Apple's on-device "Hey Siri" wake-word detection —
  audio **never leaves your iPhone** until you've confirmed you meant to
  talk to Siri.

### Pillar 4 — **Offline Reliability**
- **👶 Story:** the elevator must still work when the internet is down.
- **Formal:** the edge device keeps doing its job even if the network
  is gone (airplane, forest, tunnel, power-cut router).
- **Why it matters:** life-critical systems (medical devices, industrial
  safety), remote deployments (oil rigs, farms, drones), or simply
  better user experience when Wi-Fi is flaky.

### Pillar 5 — **Bandwidth Savings**
- **👶 Story:** don't FedEx every grain of rice. Send the shopping list.
- **Formal:** a 4K camera produces ~25 Mbps of video. Streaming 1,000
  cameras to the cloud = **25 Gbps** of permanent uplink. Processing
  at the edge and only sending **alerts** (a few bytes) cuts the bill
  by 99 %+.
- **Rule of thumb:** sending a 1080p JPEG over LTE costs ~**100×** more
  energy than running a small CNN on-device.

> **🎯 Mental model:** every edge product is built on 1–3 of these
> pillars. When you read about a new device, ask: **which pillars does
> it lean on?**

---

## 5. The edge continuum (4 tiers)

Edge AI is not one thing — it's a **spectrum** from tiny sensors to
small data-centers sitting near the users.

```
    🔬 TIER 1            🏠 TIER 2           🏢 TIER 3              ☁️  TIER 4
    ──────────           ──────────          ──────────             ──────────
    Device Edge          Gateway Edge        Edge Server / MEC       Cloud
    (TinyML)             (Home / Hub)        (Near user / Telco)     (Data-center)

    💡 Light bulb        🏠 Smart hub        🏢 Tower / store         ☁️  AWS / GCP
    ⌚ Smartwatch        📡 5G gateway       🏭 Factory floor PC       (us-east-1)
    📷 Doorbell cam      🚗 Car head-unit    🚅 Train-side box
    🌱 Soil sensor       🎚️ Camera NVR       🏥 Hospital edge rack

    Power: mW – 1 W      Power: 5 – 30 W     Power: 50 – 1000 W     Power: MW
    Compute: MCU/NPU     Compute: small GPU  Compute: big GPU / TPU  Compute: rack
    Model: < 1 MB        Model: < 100 MB     Model: < 10 GB          Model: any size
    Latency: 1–20 ms     Latency: 5–50 ms    Latency: 10–80 ms       Latency: 50–500 ms
```

### How to read the continuum
- Moving **left** = tinier, cheaper, more private, but dumber.
- Moving **right** = smarter, more flexible, but further and slower.
- Most real products **use several tiers at once**. A doorbell camera
  does *person detection* on Tier 1, *face clustering* on Tier 2, and
  *biometric search* on Tier 4 — the classic **edge-cloud split**.

### So what?
> **"Edge" is a spectrum, not a single box.** Ask which tier your
> problem actually lives in.

[↑ Back to Top](#-edgeai--fundamentals--theory)

---

## 6. Real-world use cases

A short tour. Each one calls out the **dominant pillar(s)** (§4).

### 1. 🚗 Self-driving & ADAS
- **What:** lane keep, pedestrian detection, adaptive cruise.
- **Why edge:** **latency + offline reliability**. A 100 ms cloud
  round-trip means the car has already moved 1.7 m.
- **Typical chip:** NVIDIA DRIVE Orin, Tesla HW4, Mobileye EyeQ.

### 2. 📷 Smart cameras & video analytics
- **What:** person detection, licence-plate reading, loitering alerts.
- **Why edge:** **bandwidth savings + privacy**. 25 Mbps × 1,000 cameras
  × 24 h is infeasible to stream.
- **Typical chip:** Ambarella CV5, Hailo-8, Jetson Xavier NX.

### 3. 🏭 Industrial / manufacturing
- **What:** defect detection on a moving belt, predictive maintenance.
- **Why edge:** **latency + reliability**. The belt won't pause for
  Wi-Fi.
- **Typical chip:** Jetson AGX Orin, RTX 4000 Ada SFF on a rugged PC.

### 4. ⌚ Wearables & mobile AI
- **What:** heart-rate anomaly detection, fall detection, translation,
  on-device photo edits.
- **Why edge:** **privacy + low power + offline**.
- **Typical chip:** Apple Neural Engine, Qualcomm Hexagon NPU.

### 5. 🏥 Medical devices
- **What:** hearing aids (noise suppression), glucose monitors, bedside
  monitors.
- **Why edge:** **privacy (HIPAA) + reliability + latency**.
- **Typical chip:** tiny Cortex-M NPUs, custom ASICs.

### 6. 🚁 Drones & robotics
- **What:** obstacle avoidance, visual SLAM, package recognition.
- **Why edge:** **offline + latency**. Drones fly beyond cell towers.
- **Typical chip:** Jetson Orin Nano / NX.

### 7. 🌾 Smart agriculture & IoT
- **What:** disease detection on leaves, livestock tracking.
- **Why edge:** **bandwidth + power** (fields have no 5G, only solar).
- **Typical chip:** ESP32-S3, Cortex-M55 + Ethos-U55.

### 8. 🔊 Voice assistants
- **What:** wake-word detection ("Hey Siri", "Ok Google"), command
  classification.
- **Why edge:** **privacy + latency**. Audio stays local until the
  wake-word fires.
- **Typical chip:** NPU inside every modern phone SoC.

[↑ Back to Top](#-edgeai--fundamentals--theory)

---

## 7. The Edge AI pipeline (Sense → Act)

Every single edge product in §6 follows the **same 4-stage pipeline**.
Learn it once and you can reason about any edge system.

```
   ┌────────┐     ┌────────────┐     ┌───────────┐     ┌────────┐
   │ SENSE  │ ──► │ PRE-       │ ──► │  INFER    │ ──► │  ACT   │
   │ raw    │     │ PROCESS    │     │  (model)  │     │ result │
   │ data   │     │ resize,    │     │  forward  │     │ alert, │
   │        │     │ normalise  │     │  pass     │     │ motor, │
   │ (µs)   │     │ (ms)       │     │  (ms)     │     │ display│
   └────────┘     └────────────┘     └───────────┘     └────────┘
```

### Example — Smart doorbell
1. **Sense** — camera grabs a 1080p frame (33 ms at 30 FPS).
2. **Preprocess** — resize to 224×224, convert BGR→RGB, normalise to
   [-1, 1] (~2 ms on a Jetson Nano).
3. **Infer** — run MobileNetV2 quantized to INT8 (~8 ms).
4. **Act** — if `score("person") > 0.7`, ring the chime and push a
   notification.

Total on-device time: **~45 ms** — under 1 frame. Cloud round-trip
would be 300 ms — **7× slower**, and you'd pay for 30 uploads/second.

### The numbers you need to hit
- **30 FPS camera** → you have **33 ms per frame** for *everything*.
- Subtract ~5 ms for I/O and preprocessing.
- That leaves **~28 ms** for the model forward pass. **Anything slower
  drops frames.**

### So what?
> **Design the pipeline backwards from the latency budget**, not
> forwards from the model accuracy.

[↑ Back to Top](#-edgeai--fundamentals--theory)

---

## 8. Challenges and trade-offs

Edge AI is **hard**. Every pillar in §4 fights you at some point.

### Challenge 1 — **Tiny memory**
A Cortex-M4 has **~256 KB** of SRAM. A MobileNetV2 weighs ~14 MB in
FP32. You can't even *load* it, let alone run it. Solutions:
**quantization** (FP32 → INT8 shrinks 4×), **pruning**, **distillation**
into a smaller student model, or **architectural changes**
(MobileNet, SqueezeNet, EfficientNet-Lite).

### Challenge 2 — **Compute budget**
An H100 does **~1,000 TFLOPS**. A Jetson Orin Nano does **~40 TOPS**
(INT8). A Cortex-M55 + Ethos-U55 does **~0.5 TOPS**. You must match the
model to the chip — a 7 B-param LLM will **not** run on an Ethos-U55.

### Challenge 3 — **Thermal budget**
Sustained power dissipates as heat. Passive-cooled (fanless) edge
devices usually cap at **~10 W**. Exceed it and the chip **throttles**
(slows itself down) — your 30 FPS drops to 10 FPS in 2 minutes.

### Challenge 4 — **OTA (Over-The-Air) updates**
Pushing a new model to 10 million devices is **not** the same as pushing
to one cloud endpoint. You need: signed binaries, rollback, staged
rollouts, network tolerance, sometimes dual-partition A/B updates.

### Challenge 5 — **Model staleness**
A model on a device might be 6 months old while the cloud retrains
daily. Drift (data changing over time) is worse at the edge because
updates are slow.

### Challenge 6 — **Fragmentation**
Every vendor has its own runtime (TFLite, ONNX Runtime, TensorRT,
OpenVINO, Core ML, Hailo SDK, SNPE, …). Moving the same model to a
new chip is rarely "just recompile".

### Challenge 7 — **Security**
Edge devices live in the wild. Attackers can **physically** access
the chip, dump firmware, or try to extract weights. Secure boot,
encrypted weights, and attestation are not optional.

> **Takeaway:** Edge AI is 30 % model, 70 % systems engineering.

[↑ Back to Top](#-edgeai--fundamentals--theory)

---

## 9. How Edge AI fits inside the bigger AI world

```
 ┌────────────────────────────────────────────────────────────────┐
 │                       MODERN AI STACK                          │
 ├────────────────────────────────────────────────────────────────┤ 
 │                                                                │
 │   ☁️  CLOUD AI           (training, batch, giant LLMs)         │
 │     │                                                           │
 │     ▼  pushes a compressed model                                │
 │   📡 MEC / EDGE SERVER  (multi-camera, store-level, 5G)         │
 │     │                                                           │
 │     ▼                                                           │
 │   🏠 GATEWAY / HUB       (home hub, car head-unit)              │
 │     │                                                           │
 │     ▼                                                           │
 │   📱 DEVICE EDGE         (phones, Jetson, Coral)                │
 │     │                                                           │
 │     ▼                                                           │
 │   💡 TINY-ML             (MCUs, doorbells, lightbulbs)          │
 │                                                                 │
 └─────────────────────────────────────────────────────────────────┘
```

- **Cloud trains, edge serves.** That's the default.
- **On-device fine-tuning** and **federated learning** are making edge
  training real, but it's still small-scale.
- **LLMs** are slowly moving to the edge — a 4-bit quantized 7 B
  Llama runs on an Orin NX today; a 1 B model runs on a flagship phone.

### So what?
> **"Cloud vs Edge" is a false fight.** Real systems are built on the
> *whole* stack. Edge makes the cloud cheaper, and the cloud makes the
> edge smarter.

[↑ Back to Top](#-edgeai--fundamentals--theory)

---

## 10. Cheat sheet & red flags

### Cheat sheet
```
 WHAT          Edge AI = inference (± training) on-device
 WHY           L-I-V-E-P-A-C-E  (latency, inference, volume,
               energy, privacy, autonomy, cost, edge-cloud split)
 PILLARS       Low Latency · Low Power · Privacy · Reliability · Bandwidth
 TIERS         Tiny-ML → Device-Edge → Gateway → Edge-Server → Cloud
 PIPELINE      Sense → Preprocess → Infer → Act
 BUDGET        30 FPS ⇒ 33 ms/frame ⇒ ~28 ms for model forward
 KEY UNIT      TOPS/Watt   (higher = more edge-friendly)
 KEY TOOL      Quantization  (FP32 → INT8, 4× smaller, 2–4× faster)
```

### Red flags (when NOT to use Edge AI)
- 🚩 Your model is **50 GB**. That's not an edge model — that's cloud.
- 🚩 You need **batched, offline** analysis of yesterday's data.
  Cloud wins.
- 🚩 Your product has **guaranteed** gigabit internet and latency
  doesn't matter. Keep it simple, use cloud.
- 🚩 Your team has **no embedded/systems experience**. Edge AI is
  more systems than ML — budget for that.

### Green flags (when Edge AI is the answer)
- ✅ You must react **under 50 ms**.
- ✅ Your users' data is **private / regulated**.
- ✅ Your device goes **offline** some of the time.
- ✅ You have **lots of sensors** and bandwidth is expensive.
- ✅ You ship **millions of units** and cloud bills would kill you.

---

## 🔭 Next up

Now that you know *what* Edge AI is and *why* it exists, the next
folder [`GPU_Types/`](../GPU_Types/README.md) zooms in on the **GPU
Chips** that actually make it happen — the Jetson family, discrete
edge GPUs, integrated GPUs, and mobile SoC GPUs.

---

> *GPU Programming · EdgeAI · Fundamentals · THEORY · github.com/rpaut03l/TS-02*
