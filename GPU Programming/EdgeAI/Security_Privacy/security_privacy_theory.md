# 📖 EdgeAI · Security & Privacy — THEORY

### *Secure boot · TEE · Adversarial attacks · Side channels · Privacy · Regulations*

> **Nav:** [← Security & Privacy README](README.md) | **THEORY** | [💻 CODE](security_privacy_code.md) | [🎯 PRACTICE](security_privacy_practice.md)

---

## 🧠 MNEMONIC: **"BASTION"**

> **B**oot · **A**ttestation · **S**torage · **T**EE · **I**nference · **O**bservation · **N**orms (laws)

Seven layers, outside to inside, that every edge device should have.
Attackers pick the weakest layer — your defence is only as strong as
the weakest one.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Threat model for the edge | [§1](#1-threat-model-for-the-edge) |
| 2 | **Secure boot** — chain of trust | [§2](#2-secure-boot--chain-of-trust) |
| 3 | **TEE / HSM** — trusted execution | [§3](#3-tee--hsm--trusted-execution) |
| 4 | **Encrypted model** at rest and in RAM | [§4](#4-encrypted-model-at-rest-and-in-ram) |
| 5 | **Adversarial attacks** on the model | [§5](#5-adversarial-attacks-on-the-model) |
| 6 | Defences against adversarial attacks | [§6](#6-defences-against-adversarial-attacks) |
| 7 | **Side-channel** attacks | [§7](#7-side-channel-attacks) |
| 8 | **Privacy** — on-device PII & DP recap | [§8](#8-privacy--on-device-pii--dp-recap) |
| 9 | **Regulations** — GDPR, HIPAA, DPDPA, EU AI Act | [§9](#9-regulations--gdpr-hipaa-dpdpa-eu-ai-act) |
| 10 | Cheat sheet | [§10](#10-cheat-sheet--red-flags) |

---

## 1. Threat model for the edge

### 👶 Easy Story
On the cloud, your enemy is **far away, typing on a keyboard**. At
the edge, your enemy can be **sitting at your kitchen table with a
screwdriver, a multimeter, and an oscilloscope**. You have to defend
against both the keyboard and the screwdriver.

### The 4 attacker categories

1. **Network attacker.** MITM, replay, downgrade.
2. **User attacker.** Jailbreak, root, side-load, spoof sensors.
3. **Firmware attacker.** Dump flash, patch binaries, swap images.
4. **Physical attacker.** Scope traces, EM probe, fault injection
   (laser, glitch), chip decap.

Your security goals break into:
- **Confidentiality** — model weights and user data stay secret.
- **Integrity** — firmware and model are what you shipped.
- **Availability** — device works when the user needs it.
- **Authenticity** — "this message came from my device" is provable.

### Standard refusal: **"we trust the hardware"**
That's a fine stance if "hardware" = a **secure element** with
tamper resistance and a TEE. It's **not** fine if "hardware" = a
plain MCU with `JTAG` enabled.

[↑ Back to Top](#-edgeai--security--privacy--theory)

---

## 2. Secure boot — chain of trust

### 👶 Easy Story
Starting the device is like opening a set of Russian dolls, each one
signed by the previous one. If any doll's signature fails, the device
refuses to run.

### The chain

```
 ┌───────────────┐  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐
 │ 0. Immutable  │─▶│ 1. BootROM  │─▶│ 2. BL1   │─▶│ 3. BL2   │─▶│ 4. App +  │
 │    ROM / fuse │  │ verifies    │  │ verifies │  │ verifies │  │    model  │
 │    public key │  │ next stage  │  │ next     │  │ next     │  │ verified  │
 └───────────────┘  └─────────────┘  └──────────┘  └──────────┘  └───────────┘
```

### The two rules of secure boot
1. **The top of the chain is immutable** — fused into silicon at
   manufacture, not writable. A ROM key (or its hash) burnt into
   e-fuses.
2. **Every link verifies the next** — if any signature fails, halt
   or fall back to a known-good image.

### What actually gets signed
- **BL1 / BL2 / U-Boot / kernel** — signed by vendor key.
- **User-space app** — signed by product key.
- **Model file** — signed by MLOps release key.
- **Manifest** (the "this bundle belongs together" file) — signed by
  release key.

### Combined with dual-partition OTA
The update process writes the new image into a staging partition
**without** erasing the running one; bootloader verifies signatures
on boot; falls back to the old partition if the new one fails health
checks. This is how **rollback-safe secure boot** works in practice.

[↑ Back to Top](#-edgeai--security--privacy--theory)

---

## 3. TEE / HSM — trusted execution

### The idea
**Trusted Execution Environment (TEE)** = a region of the CPU that
runs code and holds keys **isolated from the normal OS**. Even a
rooted Android kernel cannot read TEE memory.

### The popular TEEs

| TEE | Platform | Used for |
|---|---|---|
| **ARM TrustZone** | Most phones, Cortex-A + Cortex-M33/M85 | fingerprints, DRM, payment, attestation |
| **Apple Secure Enclave** | iPhone, iPad, Mac | keys, Face ID, Touch ID |
| **Intel SGX / TDX** | Intel data-centre + some laptops | confidential computing |
| **AMD SEV / SEV-SNP** | AMD servers | confidential VMs |
| **NVIDIA Platform Security Controller (PSC)** | Jetson Orin / DRIVE | secure boot + attestation |
| **NXP EdgeLock Secure Enclave** | i.MX 8/9 | secure boot, keys, DRM |
| **Google Titan / Tensor Security Core** | Pixel | same as TrustZone for Pixel |

### What you do inside a TEE
1. **Verify firmware** — enforce secure boot.
2. **Unwrap model weights** — AES-GCM decrypt keys live in TEE.
3. **Attestation** — device proves to the cloud "I'm running the
   exact firmware you signed."
4. **Key derivation / signing** — for telemetry, OTA ack, etc.

### TEE for inference itself?
Running the **whole** model inference in a TEE is rare — TEE memory
is small and TEE CPUs are slow. The usual split:
- **TEE** holds the key, verifies firmware, decrypts weights into
  secure shared memory.
- **Normal world** runs the (decrypted) model on GPU / NPU with
  periodic attestation checks.

### HSM vs TEE
- **HSM (Hardware Security Module)** — dedicated discrete chip;
  tamper-responsive; used in servers and payment terminals.
- **TEE** — a region of the main CPU; cheaper; less tamper-resistant.

[↑ Back to Top](#-edgeai--security--privacy--theory)

---

## 4. Encrypted model at rest and in RAM

### 👶 Easy Story
Don't ship the raw model. Ship it **encrypted**, and decrypt only
inside a trusted region with a key derived from the device's unique
ID.

### At rest

- Encrypt each `.tflite` / `.engine` / `.xml+.bin` with **AES-GCM-256**
  before shipping. Nonce + tag stored alongside.
- Key is either:
  - **Bound to the device** (derived from device UID inside the TEE),
    or
  - **Bound to the release** (key blob wrapped by the TEE's key
    wrapping key).
- Decryption happens **only** inside the TEE or right before the
  accelerator consumes weights.

### In RAM

- Most edge platforms map model weights into **shared GPU/CPU
  memory** that the Linux kernel can see. Even on Jetson with unified
  memory, weights in plaintext are reachable from a privileged
  attacker.
- On Android, the NNAPI driver can keep weights in **per-process
  memory** via the NNHAL.
- Apple's Core ML runs compiled programs inside a sandboxed process
  and keeps ANE weights outside the kernel-visible range.

### ChaCha20-Poly1305 as alternative
On chips without AES-NI, **ChaCha20-Poly1305** is usually faster and
simpler. Same security level. Used by WireGuard and TLS 1.3 for this
reason.

### Key escrow
Never bake a **plaintext key** into firmware. Use:
- **Device-bound** keys derived from e-fuse values inside the TEE.
- **Per-release** keys wrapped by the device's Key Wrapping Key.
- **Rotatable** keys — the release includes new wrap-key material.

[↑ Back to Top](#-edgeai--security--privacy--theory)

---

## 5. Adversarial attacks on the model

The model itself is an attack surface.

### The attack taxonomy

```
 ┌──────────────────────┬──────────────────────────────────────────────┐
 │ Attack               │ Goal                                           │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ Evasion              │ Make the deployed model mis-classify a chosen  │
 │                      │ input. FGSM, PGD, Carlini & Wagner.            │
 │                      │                                                │
 │ Poisoning            │ Inject bad samples during training so the      │
 │                      │ model mis-classifies at deploy time.           │
 │                      │                                                │
 │ Backdoor (trigger)   │ Poisoning with a specific "trigger pattern"    │
 │                      │ that flips predictions when present.           │
 │                      │                                                │
 │ Model extraction     │ Query the model many times and train a stolen  │
 │                      │ copy from input/output pairs.                  │
 │                      │                                                │
 │ Model inversion      │ Reconstruct a training sample from the model    │
 │                      │ (e.g. the face of user X).                     │
 │                      │                                                │
 │ Membership inference │ Decide whether a given sample was in training. │
 │                      │                                                │
 │ Property inference   │ Learn dataset-level properties (average age,   │
 │                      │ class imbalance) from the model.               │
 └──────────────────────┴──────────────────────────────────────────────┘
```

### FGSM and PGD in 2 lines each
- **FGSM (Fast Gradient Sign Method):**
  `x_adv = x + ε · sign(∇_x L(f(x), y))`
  One step. Cheap. Easy to defend against.
- **PGD (Projected Gradient Descent):**
  iterate `x ← project_{||·||_∞ ≤ ε}(x + α · sign(∇_x L))`
  Many steps. Projection keeps perturbation small. Much stronger.

### Why it matters for the edge
Edge models are white-box to the attacker — they own the binary.
White-box attacks can craft adversarial inputs that **reliably** fool
the model. This is not theoretical: researchers have fooled stop-sign
detectors, skin-cancer detectors, and face-auth at the edge.

[↑ Back to Top](#-edgeai--security--privacy--theory)

---

## 6. Defences against adversarial attacks

No defence is perfect. The ones that work best are **layered**.

### Defence 1 — **Adversarial training**
Include adversarial examples in the training set. The model learns
to classify them correctly. Robustness at a known ε budget improves
substantially. Slower training, mild accuracy drop.

### Defence 2 — **Defensive distillation**
Train a student at high temperature on the teacher's softened
outputs. Makes the gradient the attacker relies on **smoother**,
raising the cost of finding adversarial examples. Broken by strong
attackers but helps in the wild.

### Defence 3 — **Randomized smoothing**
Add Gaussian noise `N(0, σ²)` to inputs at inference; average many
predictions. Provides a **certified** robustness radius `R = σ · Φ^{-1}(p)`.
Slow but mathematically principled.

### Defence 4 — **Input preprocessing**
Compression, JPEG re-encoding, or bit-depth reduction can destroy
small perturbations. Low-tech, surprisingly effective against FGSM;
bypassed by adaptive attackers.

### Defence 5 — **Detection**
A second model / auxiliary head that flags "this input looks
adversarial." Fallback to a safe decision.

### Defence 6 — **Rate limiting + anomaly detection**
Limit query rate per session; watch for burst patterns suggesting
model-extraction attacks; require attestation for high-rate clients.

### The honest take
If an attacker has white-box access + unbounded time, they will
eventually fool you. Defence is **raising the cost** so that the
reward no longer justifies the effort.

[↑ Back to Top](#-edgeai--security--privacy--theory)

---

## 7. Side-channel attacks

### 👶 Easy Story
Even if you never tell me the key, I can **listen to the chip humming**
and guess it. Power, timing, and electromagnetic emissions leak
information about what the chip is doing.

### The three classes

| Channel | Attack | Mitigation |
|---|---|---|
| **Power analysis** (SPA, DPA) | Scope on the power rail watches multiplies | constant-time, power-balanced circuits, blinding |
| **Timing** | Different inputs cause different-latency paths | constant-time code; no data-dependent branches |
| **EM emission** | Antenna near the chip sees RF leakage | shielding; same as power mitigations |
| **Cache timing** | Measure LLC access times to infer branches | page-coloring, cache partitioning |
| **Fault injection** | Glitch voltage / clock / laser to skip checks | redundant checks, clock monitors, shielding |

### Why ML is extra-exposed
- Multiplies dominate inference — their power profile varies with
  weight values → **model extraction via DPA** is real.
- Some edge NPUs are not hardened against DPA because "the model is
  public anyway" — but your fine-tuned head often isn't.

### Practical mitigations
- Use **silicon** (TEEs / HSMs) with DPA-resistant crypto.
- Constant-time kernels for sensitive preprocessing (e.g. biometric
  enrolment).
- Physical shielding (conductive enclosure, filtered power rails) for
  hostile deployments.

[↑ Back to Top](#-edgeai--security--privacy--theory)

---

## 8. Privacy — on-device PII & DP recap

Security keeps attackers out. **Privacy** makes sure that *even you,
the product owner, never see data you shouldn't*.

### The 4 practical levers

1. **Keep data on-device.** Edge AI's whole premise — maximise this.
2. **Redact PII before any upload** — on-device regex and ML-based
   redactors for names, phone numbers, faces, plate numbers.
3. **Aggregate or sketch.** Ship histograms, counts, sketched
   embeddings, not raw records.
4. **Differential Privacy.** Noise all telemetry under a formal
   `(ε, δ)` budget.

### On-device PII redaction
Two tools:
- **Regex + allow/deny lists** — cheap; good for phone numbers,
  emails, postal codes.
- **NER model (e.g. small SpaCy / distilled BERT)** — handles names,
  addresses, uncommon patterns.
- **Faces / plates** — run an on-device detector, blur before upload.

### Privacy budgets in practice
- ε < 1: very strong privacy, usually hurts utility.
- 1 ≤ ε ≤ 5: practical range; Apple publishes per-query ε budgets
  here.
- ε > 10: weak privacy, mostly "we added some noise."
- δ: should be ≤ 1/|dataset|.

[↑ Back to Top](#-edgeai--security--privacy--theory)

---

## 9. Regulations — GDPR, HIPAA, DPDPA, EU AI Act

### Why you care as an engineer
- They dictate **what data can leave the device** and **what logs
  must exist**.
- They require **data-subject rights** (deletion, access, rectification).
- They mandate **risk classifications** for AI systems and
  corresponding documentation.

### The big five

| Law | Who | What it cares about |
|---|---|---|
| **GDPR** (EU, 2018) | EU residents | consent, purpose, data-subject rights, data-processing records |
| **CCPA / CPRA** (California) | CA residents | disclosure, opt-out, sensitive-info handling |
| **HIPAA** (US, healthcare) | US patients | PHI handling, breach notification, BAA contracts |
| **DPDPA 2023** (India) | Indian citizens | consent, data fiduciaries, breach notification |
| **EU AI Act** (2024+) | EU AI deployers | risk tiers (prohibited / high-risk / limited / minimal), documentation, human oversight |

### How it lands in code
- **Data minimisation.** Only emit what is strictly necessary —
  don't log full images if a hash or a class label will do.
- **Purpose limitation.** Tag every log field with a purpose; drop
  fields in flight if purpose no longer applies.
- **Retention.** Automatic expiry on raw telemetry.
- **Audit trail.** Who saw what, when.
- **Model cards / datasheets.** Document training data, known
  limitations, intended use. Required under the EU AI Act for
  high-risk systems.
- **DPIA (Data Protection Impact Assessment).** Required for
  high-risk ML. Templates exist.

### If you only remember one thing
> **Think of every sensor byte as a legal liability by default.**
> Prove it's not before uploading it anywhere.

[↑ Back to Top](#-edgeai--security--privacy--theory)

---

## 10. Cheat sheet & red flags

### Cheat sheet
```
 THREATS         Network · User · Firmware · Physical
 BOOT            Immutable fuse key → BL1 → BL2 → App → Model (all signed)
 TEE             TrustZone · Secure Enclave · SGX · PSC · EdgeLock
 STORAGE         AES-GCM-256 at rest; keys in TEE; rotate on release
 ATTACKS         evasion · poisoning · extraction · inversion · MI · side-ch
 DEFENCES        adv. training · distillation · randomized smoothing ·
                 preprocessing · detectors · rate limit · attestation
 SIDE CH.        power · timing · EM · cache · fault-injection
 PRIVACY         keep on-device · redact PII · aggregate · DP-SGD
 LAWS            GDPR · CCPA · HIPAA · DPDPA · EU AI Act
```

### Red flags 🚩
- 🚩 Plaintext weights on a removable SD card.
- 🚩 `JTAG` enabled on production firmware.
- 🚩 Single symmetric key baked into every device.
- 🚩 No secure boot — bootloader accepts any firmware.
- 🚩 Model with no adversarial robustness evaluation.
- 🚩 Telemetry with raw PII and no retention policy.
- 🚩 "We comply with GDPR" claim without a DPIA.

### Green flags ✅
- ✅ Every model release is signed end-to-end (see [Edge MLOps](../Edge_MLOps/README.md)).
- ✅ Weights encrypted with per-device key; key in TEE.
- ✅ Adversarial robustness numbers published in the model card.
- ✅ On-device PII redactor with 99.x % precision on test set.
- ✅ DP-SGD for any telemetry that's individual-level.
- ✅ Privacy budget tracked and exposed in the dashboard.

---

## 🎉 Track complete

You've walked the whole EdgeAI path:

- Fundamentals → what Edge AI is
- GPU Types → which GPUs power it
- Hardware → the non-GPU chips
- CUDA for Edge → CUDA on Jetson
- Model Compression → shrinking the model
- Deployment Frameworks → the runtimes
- TinyML → AI on MCUs
- Federated Learning → private distributed training
- Edge MLOps → shipping and monitoring
- **Security & Privacy** → keeping it all safe (you are here)

Next step: build something real. Pick a tiny product idea, take it
through every chapter end-to-end, and publish the result.

---

> *GPU Programming · EdgeAI · Security & Privacy · THEORY · github.com/rpaut03l/TS-02*
