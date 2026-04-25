# 📖 EdgeAI · Federated Learning — THEORY

### *FedAvg · non-IID data · Secure Aggregation · Differential Privacy*

> **Nav:** [← Federated Learning README](README.md) | **THEORY** | [💻 CODE](federated_learning_code.md) | [🎯 PRACTICE](federated_learning_practice.md)

---

## 🧠 MNEMONIC: **"FACES"**

> **F**edAvg · **A**ggregation · **C**lients · **E**ncrypted · **S**tragglers

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Why Federated Learning exists | [§1](#1-why-federated-learning-exists) |
| 2 | The FedAvg round, end to end | [§2](#2-the-fedavg-round-end-to-end) |
| 3 | Cross-device vs cross-silo | [§3](#3-cross-device-vs-cross-silo) |
| 4 | Heterogeneity — non-IID, stragglers, dropouts | [§4](#4-heterogeneity--non-iid-stragglers-dropouts) |
| 5 | Algorithms — FedAvg, FedSGD, FedProx | [§5](#5-algorithms--fedavg-fedsgd-fedprox) |
| 6 | Communication efficiency | [§6](#6-communication-efficiency) |
| 7 | **Privacy** — DP + Secure Aggregation | [§7](#7-privacy--dp--secure-aggregation) |
| 8 | On-device training & LoRA on the edge | [§8](#8-on-device-training--lora-on-the-edge) |
| 9 | When NOT to use FL | [§9](#9-when-not-to-use-fl) |
| 10 | Cheat sheet | [§10](#10-cheat-sheet--red-flags) |

---

## 1. Why Federated Learning exists

### 👶 Easy Story
You want to teach a model to recognise a **local dialect** — but
people won't upload their voices. Or diagnose a **rare disease** —
but hospitals can't share patient data. Or improve your keyboard's
autocorrect — but what I type is private.

FL is the answer: **learn from everyone without moving the data**.

### Formal motivations
- **Privacy / regulation** — GDPR, HIPAA, India DPDPA 2023 often
  forbid raw data export.
- **Bandwidth** — streaming raw data to the cloud is expensive.
- **Liability** — centralised data stores are targets for breaches.
- **Non-shareable data** — sometimes the client *literally* cannot
  share (e.g. cross-hospital medical data).

### When FL is the natural fit
- **Keyboards / autocorrect** (Google Gboard) — learn language
  patterns across phones without leaking individual typing.
- **Medical imaging** across hospitals — each hospital's data stays
  behind its firewall.
- **Recommendation** across banks / insurers — competitive data
  can't be pooled.
- **Industrial IoT** — factories train a shared anomaly model
  without sharing trade-secret sensor data.

[↑ Back to Top](#-edgeai--federated-learning--theory)

---

## 2. The FedAvg round, end to end

**FedAvg** (McMahan et al., 2017) is the algorithm that started the
field. One *round* looks like this:

```
 ┌────────────────────────────────────────────────────────────────────┐
 │                      ONE FedAvg ROUND                               │
 └────────────────────────────────────────────────────────────────────┘

   (server holds the global model w_t)

   1. SELECT  — pick K clients (usually K ≪ total)
   2. BROADCAST — send w_t to each selected client
   3. LOCAL TRAIN — each client k trains on its OWN data for
                    E local epochs → ends with w_t^(k)
   4. UPLOAD  — clients send only  Δ_k = w_t^(k) − w_t  (or w_t^(k))
   5. AGGREGATE — server averages:
                     w_{t+1} = Σ_k (n_k / n) · w_t^(k)
                  where n_k = |dataset_k|, n = Σ n_k
   6. REPEAT  until convergence
```

### The picture
```
          server (model w_t)
        ┌───────────────────────┐
        │                       │
   send │                       │ receive Δ_k
        ▼                       ▲
 ┌──────┐   ┌──────┐   ┌──────┐ │
 │ c_1  │   │ c_2  │   │ c_K  │ │
 │ local│   │ local│   │ local│ │
 │ train│   │ train│   │ train│ │
 └──────┘   └──────┘   └──────┘ │
     │         │         │       │
     └─────────┼─────────┴───────┘
               ▼
       AGGREGATE (weighted mean)
```

### Key hyperparameters
- **K** — clients per round (typ. 10–1000).
- **E** — local epochs (typ. 1–5).
- **B** — local batch size (typ. 10–50).
- **C** — fraction of clients selected per round (typ. 0.1–1.0).

### The main win vs SGD
FedAvg does **E epochs** of local work before syncing — communication
is 10× less than if every local step was a global step (that's
**FedSGD**).

[↑ Back to Top](#-edgeai--federated-learning--theory)

---

## 3. Cross-device vs cross-silo

```
 ┌──────────────────────┬──────────────────────┬──────────────────────┐
 │                      │ Cross-DEVICE         │ Cross-SILO            │
 ├──────────────────────┼──────────────────────┼──────────────────────┤
 │ # clients            │ thousands–millions   │ 2–100                 │
 │ Reliability          │ unreliable (Wi-Fi,   │ reliable (datacenters)│
 │                      │ battery, sleep)      │                       │
 │ Data per client      │ small (MB–GB)        │ large (TB+)            │
 │ Selection per round  │ ~100–1000 sampled    │ all (or most)          │
 │ Trust model          │ untrusted clients    │ mutually-suspicious    │
 │                      │ + trusted server     │ peers                  │
 │ Examples             │ phones, smartwatches │ hospitals, banks       │
 │ Framework            │ TensorFlow Federated,│ Flower, NVIDIA FLARE,  │
 │                      │ Gboard production    │ OpenFL                 │
 └──────────────────────┴──────────────────────┴──────────────────────┘
```

Different problems → different design choices. Gboard's cross-device
setup is fundamentally different from a hospital consortium.

[↑ Back to Top](#-edgeai--federated-learning--theory)

---

## 4. Heterogeneity — non-IID, stragglers, dropouts

### Why it's hard

In standard ML you assume all data is **IID** (independent and
identically distributed). In FL, **nothing** is IID:

- **Statistical heterogeneity (non-IID).** Child Alice only plays
  with puppies; child Bob only with cats. Alice's local gradients
  pull the model toward puppies; Bob's toward cats. Mixing them via
  naive average can make the global model *oscillate*.
- **System heterogeneity.** Alice has an iPhone 15; Bob has an
  entry-level Android. Alice finishes local training in 10 s; Bob
  in 5 minutes. If we wait for everyone, rounds are dictated by the
  slowest device (**straggler**).
- **Participation bias.** Only charged + idle + Wi-Fi-connected
  devices check in. If those skew certain demographics, so does the
  model.
- **Dropouts.** Clients disappear mid-round (user switches off Wi-Fi,
  battery dies).

### Fixes
- **Client sampling** — random subset per round (reduces straggler
  impact).
- **Secure aggregation with dropout tolerance** — protocol tolerates
  up to X % dropouts (Bonawitz et al., 2017).
- **FedProx** — proximal term `μ/2 · ||w - w_t||²` in local loss to
  keep local updates close to the global model → helps non-IID
  convergence.
- **Personalization layers** — freeze the backbone globally, fine-tune
  the last layer per-client. Best of both worlds.

[↑ Back to Top](#-edgeai--federated-learning--theory)

---

## 5. Algorithms — FedAvg, FedSGD, FedProx

| Algo | Local work per round | Comms | Handles non-IID? |
|---|---|---|---|
| **FedSGD** | 1 SGD step | most | poorly |
| **FedAvg** | E epochs of SGD | typical | OK on mildly non-IID |
| **FedProx** | E epochs with proximal term | typical | better on non-IID |
| **SCAFFOLD** | tracks control variates | 2× | strong on non-IID |
| **FedOpt / FedAdam** | server uses adaptive optimizer | typical | faster convergence |
| **FedAvgM** | server momentum | typical | faster convergence |

### FedProx in one equation
```
 local objective:   min  L_k(w)  +  (μ/2) · ||w − w_t||²
 "don't drift too far from the global model I was given"
```

`μ = 0` recovers FedAvg. `μ > 0` = stabilised training under non-IID.

[↑ Back to Top](#-edgeai--federated-learning--theory)

---

## 6. Communication efficiency

**Uploading a full model update every round is expensive.** A 100 MB
model × 1,000 clients × 1,000 rounds = 100 TB of uplink traffic.
Tricks:

- **Quantized updates** — INT8 or lower for the gradient / delta.
  4× smaller uplink.
- **Sparsification** — send only the top-k% of gradient entries by
  magnitude.
- **Sketching** — compressed random projection; server reconstructs.
- **Structured updates** — low-rank deltas (similar to LoRA).
- **Federated dropout / width reduction** — each client trains a
  subset of the network, reducing uplink size and compute.

Combined, these routinely cut communication **10–100×** at small
accuracy cost.

[↑ Back to Top](#-edgeai--federated-learning--theory)

---

## 7. Privacy — DP + Secure Aggregation

### Why just "not sharing data" is not enough
Even a **gradient** leaks information. Inversion attacks can sometimes
reconstruct the original training sample from the gradient alone. So
FL by itself is **not** a privacy guarantee — it's a step in the right
direction, and must be combined with more.

### Layer 1 — **Secure Aggregation**
Cryptographic protocol where **the server only sees the sum** of
clients' updates, never individual ones. If 100 clients upload Δ_i,
the server sees Σ Δ_i but cannot isolate any single Δ_i.

Implementation sketch (Bonawitz et al., 2017):
1. Each pair of clients agrees on a random **mask** using
   Diffie-Hellman key exchange.
2. Client i uploads  `Δ_i + Σ_j>i mask_{ij} − Σ_j<i mask_{ij}`.
3. When summed, all masks cancel out — server sees `Σ Δ_i` only.
4. Handles **dropouts** with a secret-sharing scheme.

### Layer 2 — **Differential Privacy (DP)**
Every individual's data contributes **a bounded amount** to the
output, formalised as an `(ε, δ)` budget. In FL:
- **DP-SGD** — each local gradient is **clipped** to norm `C`, then
  Gaussian noise with variance `σ² C²` is added.
- Over many rounds, a **privacy accountant** (RDP) tracks the total
  `(ε, δ)` spent.
- Smaller `ε` = more privacy = more noise = usually lower accuracy.
  Production budgets: `ε = 1–10`, `δ = 1e-5`.

### Stack them
**FL + Secure Aggregation + DP-SGD** is the gold-standard recipe
used by Gboard — no raw data leaves the device, no individual
gradient is seen by the server, and each contribution is noisy
enough to satisfy a formal privacy guarantee.

[↑ Back to Top](#-edgeai--federated-learning--theory)

---

## 8. On-device training & LoRA on the edge

### The on-device training challenge
Full back-prop on a phone is **10–100× more expensive** than
inference. Practical compromises:

- **Head-only fine-tuning** — train the classifier head, freeze the
  backbone.
- **LoRA / adapters** — train two small low-rank matrices injected
  next to each frozen big matrix. Common in LLM fine-tuning.
- **BitFit** — train only biases. Surprisingly effective.
- **Federated embedding** — only embedding tables are updated
  per-device, great for personalised language / suggestion models.

### LoRA in one equation
Replace a big weight `W` with `W + α (B A)`, where `A ∈ R^(r×d)`,
`B ∈ R^(d×r)` and `r ≪ d`. Train only `A, B` (a few megabytes),
leave `W` frozen on ROM.

LoRA adapters turn a 7 B LLM fine-tune into a 10 MB update — perfect
for federated rounds.

[↑ Back to Top](#-edgeai--federated-learning--theory)

---

## 9. When NOT to use FL

FL is powerful but expensive. Skip it when:

- **Your data is already centralised and legal to use.** Just train
  centrally. FL gives you nothing here.
- **Data is stable and small.** Aggregate it once, retrain yearly,
  done.
- **You can't run a reliable FL infrastructure.** Certificate
  pinning, secure boot, rollout, monitoring — all required.
- **Clients have no compute budget.** FL requires local training;
  coin-battery MCUs can barely do inference.
- **You need fine-grained labels** that only exist on the server.

### Typical decision rule
```
 Can we legally & technically centralise the data?
   YES → centralise & train normally
   NO  → FL
     │
     Clients reliable + few?  →  cross-silo FL
     Clients unreliable + many? → cross-device FL
       │
       High privacy bar? → + Secure Aggregation + DP-SGD
       Limited uplink?   → + compressed / sparsified updates
```

[↑ Back to Top](#-edgeai--federated-learning--theory)

---

## 10. Cheat sheet & red flags

### Cheat sheet
```
 WHAT              Train on distributed data without moving it
 CORE ALGO         FedAvg (weighted mean of local models)
 VARIANTS          FedSGD · FedProx · SCAFFOLD · FedOpt
 CROSS-DEVICE      many · unreliable · phone-style
 CROSS-SILO        few · reliable · hospital-style
 NON-IID           use FedProx or personalisation layers
 COMMS             quant + top-k + structured updates (10–100× cut)
 PRIVACY           Secure Aggregation + DP-SGD (ε, δ)
 ON-DEVICE TRAIN   head-only · LoRA · BitFit
 FRAMEWORKS        Flower · TF Federated · NVIDIA FLARE · OpenFL · PySyft
```

### Red flags 🚩
- 🚩 Claiming "FL = privacy" without DP or Secure Aggregation. Gradients
  can leak.
- 🚩 Naively averaging updates from wildly non-IID clients. Use FedProx.
- 🚩 Waiting for all clients per round. One bad phone stalls
  everything. Use client sampling + timeouts.
- 🚩 Shipping the **server code** to the device (full FL runtime) when
  a **stripped client** would have sufficed.

### Green flags ✅
- ✅ Secure Aggregation on all updates.
- ✅ DP-SGD with a logged `(ε, δ)` budget.
- ✅ Client-sampling strategy with timeout + retry policy.
- ✅ Monitoring of client participation skew by demographic.
- ✅ Adapter / LoRA-sized updates to respect mobile data plans.

---

## 🔭 Next up

Now that you can train across devices privately, the next folder
[`Edge_MLOps/`](../Edge_MLOps/README.md) is how you actually **ship,
update, and monitor** models that live on millions of devices in the
wild.

---

> *GPU Programming · EdgeAI · Federated Learning · THEORY · github.com/rpaut03l/TS-02*
