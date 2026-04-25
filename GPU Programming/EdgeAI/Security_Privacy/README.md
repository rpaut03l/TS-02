# 🔒 EdgeAI · Security & Privacy

### *Secure boot · Encrypted weights · Adversarial attacks · Side channels · Regulations*

> **Nav:** [← EdgeAI](../README.md) | [← Model Compression](../Model_Compression/README.md) | [← Deployment Frameworks](../Deployment_Frameworks/README.md) | [← TinyML](../TinyML/README.md) | [← Federated Learning](../Federated_Learning/README.md) | [← Edge MLOps](../Edge_MLOps/README.md) | **Security & Privacy**

---

## 👶 30-second story

On the cloud, attackers are far away. At the edge, the attacker may
be **holding your device in their hand**. They can open it,
probe the chip with a scope, dump the flash, swap the firmware, feed
strange inputs until the model misbehaves, or measure power to
extract secrets.

This folder is the **last line of defence** — what stops a clever
teenager or a skilled adversary from stealing your model, fooling
it, or leaking your users' data.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [security_privacy_theory.md](security_privacy_theory.md) | Full theory — **secure boot chain of trust**, **TEE / HSM** (ARM TrustZone, Intel SGX/TDX, Apple Secure Enclave, NVIDIA PSC), **encrypted model at rest and in RAM**, **adversarial attacks** (evasion, poisoning, extraction, membership inference), **watermarking / fingerprinting** of weights, **side-channel** attacks (power, timing, EM), **privacy** (on-device PII redaction, DP budget), **regulations** (GDPR, HIPAA, India DPDPA 2023, EU AI Act) |
| [security_privacy_code.md](security_privacy_code.md) | Runnable code — hash + signature verify on device, AES-GCM encrypt/decrypt model weights, adversarial examples with FGSM / PGD, membership-inference attack, simple defensive distillation, Opacus DP-SGD recap, PII redactor regex + entity tagger, a "secure-boot rehearsal" flow in Python |
| [security_privacy_practice.md](security_privacy_practice.md) | **Colab notebook** — attack a MobileNetV2 with FGSM / PGD, measure success rate and human perceptibility; then add adversarial training + defensive distillation, measure robustness recovery; bolt on DP-SGD and produce a 3-axis (accuracy / robustness / privacy ε) Pareto plot |

---

## 🎯 After reading this you should be able to…

- Draw the **secure boot chain** from the immutable ROM to the model
- Pick between **AES-GCM** and **ChaCha20-Poly1305** for weight
  encryption
- Describe **FGSM** and **PGD** in one line each and write the attack
- Tell **evasion** from **extraction** from **poisoning** from
  **membership inference**
- Explain when to use a **TEE** (TrustZone / SGX / Secure Enclave)
- List the 3 main **side-channel** attack types and one mitigation
  per type
- Map a data-handling pattern to **GDPR / HIPAA / DPDPA / EU AI Act**
  obligations

---

## ⚡ The shortest possible summary

> **On the edge, you must defend against an attacker who owns the
> hardware.** Encrypt, sign, attest, compartmentalise, and keep
> data on-device by design — not by hope.

---

> *GPU Programming · EdgeAI · Security & Privacy · github.com/rpaut03l/TS-02*
