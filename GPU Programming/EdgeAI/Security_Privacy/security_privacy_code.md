# 💻 EdgeAI · Security & Privacy — CODE

### *Sig verify · AES-GCM · FGSM / PGD · membership inference · PII redaction*

> **Nav:** [← Security & Privacy README](README.md) | [📖 THEORY](security_privacy_theory.md) | **CODE** | [🎯 PRACTICE →](security_privacy_practice.md)

---

## 🏗️ Setup

```python
!pip install -q cryptography pynacl torch torchvision opacus presidio-analyzer \
    presidio-anonymizer
```

---

## Ex 1 — Verify a signed firmware + model bundle

### 👶 What this does
The device-side check that must happen **before** any update is
applied.

```python
import nacl.signing, base64

sk = nacl.signing.SigningKey.generate()
pk = sk.verify_key
pk_b64 = base64.b64encode(pk.encode()).decode()
print("Device baked-in public key:", pk_b64[:16], "…")

manifest = b'{"model_version":"v2","sha256":"abc..."}'
sig = sk.sign(manifest).signature

# Device side:
verifier = nacl.signing.VerifyKey(base64.b64decode(pk_b64))
try:
    verifier.verify(manifest, sig)
    print("✅ signature OK — accept update")
except Exception as e:
    print("❌ reject:", e)
```

In C on a Cortex-M, use **Monocypher** or **libsodium-static** (ed25519,
tiny, MIT-licensed).

---

## Ex 2 — AES-GCM encrypt & decrypt model weights

### 👶 What this does
Encrypt a `.tflite` or `.engine` so the file on disk is useless without
the per-device key.

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

key = AESGCM.generate_key(bit_length=256)     # 32 bytes
nonce = os.urandom(12)                         # 96-bit nonce
aesgcm = AESGCM(key)

plain = open("kws_int8.tflite", "rb").read() if os.path.exists("kws_int8.tflite") \
        else os.urandom(30_000)
cipher = aesgcm.encrypt(nonce, plain, associated_data=b"kws_int8.tflite")

# Ship (nonce || cipher) alongside the bundle; keep key in TEE.
open("model.enc", "wb").write(nonce + cipher)

# Device side
blob = open("model.enc", "rb").read()
n, c = blob[:12], blob[12:]
recovered = AESGCM(key).decrypt(n, c, associated_data=b"kws_int8.tflite")
assert recovered == plain
print("✅ decrypted", len(recovered), "bytes")
```

### Why **AES-GCM** (not plain AES-CTR)
GCM gives you **confidentiality + integrity** in one pass. If the
ciphertext is tampered with, decrypt **fails** — the attacker can't
silently mutate weights to backdoor your model.

### ChaCha20-Poly1305 alternative

```python
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
chacha = ChaCha20Poly1305(ChaCha20Poly1305.generate_key())
c2 = chacha.encrypt(os.urandom(12), plain, b"aad")
```

Faster on chips without AES-NI (most Cortex-M, older ARM-A).

---

## Ex 3 — FGSM attack (1-step adversarial example)

### 👶 What this does
The simplest gradient-based attack. Perturb the input in the
direction that **maximises** the loss.

```python
import torch, torchvision, torch.nn.functional as F

m = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
x = torch.randn(1, 3, 224, 224, requires_grad=True)
y = torch.tensor([281])     # true label (tabby cat in ImageNet)

def fgsm(x, y, eps=0.01):
    x = x.clone().detach().requires_grad_(True)
    out = m(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    return (x + eps * x.grad.sign()).detach()

x_adv = fgsm(x, y, eps=0.02)
print("clean pred :", m(x).argmax().item())
print("adv pred   :", m(x_adv).argmax().item())
print("max |Δx|   :", (x_adv - x).abs().max().item())
```

Flip in one step without visible difference — hence the problem.

---

## Ex 4 — PGD (multi-step, stronger)

```python
def pgd(x, y, eps=0.02, alpha=0.002, steps=20):
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(m(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = x_adv.clamp(-3, 3)  # keep in normalised range
    return x_adv

x_pgd = pgd(x, y)
print("PGD pred   :", m(x_pgd).argmax().item())
```

PGD is usually ~10× more effective than FGSM at the same ε and is
the standard benchmark for robustness claims.

---

## Ex 5 — Adversarial training (Madry-style)

```python
def train_step(model, x, y, eps=0.02, alpha=0.005, steps=7):
    model.train()
    x_adv = pgd(x, y, eps=eps, alpha=alpha, steps=steps)
    out = model(x_adv)
    loss = F.cross_entropy(out, y)
    loss.backward()
    return loss.item()

# Plug into your normal training loop — replace CE(x,y) with CE(pgd(x,y),y).
# Expect 5–15 % clean accuracy drop but much higher robust accuracy.
```

---

## Ex 6 — Defensive distillation (temperature > 1)

```python
def distill_loss(student_logits, teacher_logits, T=20.0):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits  / T, dim=-1),
        reduction="batchmean") * (T * T)

# Use this as the training loss. Smoother logits → weaker gradients
# → higher cost for gradient-based attackers.
```

Originally shown to be broken by stronger attacks — keep it as a
**layer** in your defence, not the only defence.

---

## Ex 7 — Membership inference (shadow-model style)

### 👶 What this does
Decide whether a sample was **in** the training set. If your model
gives much higher confidence on seen vs unseen samples, members are
distinguishable.

```python
import numpy as np

# Confidence for a member (higher) vs non-member (lower)
conf_member   = np.random.beta(8, 1, 500)   # model very confident
conf_nonmem   = np.random.beta(3, 2, 500)   # less confident

# Simple threshold attacker:
thresh = 0.8
acc = (
    (conf_member  >= thresh).mean() +
    (conf_nonmem < thresh).mean()
) / 2.0
print(f"Attacker advantage over random: {acc - 0.5:+.2f}")
```

An advantage close to 0 means you're safe; close to 0.5 means the
model is leaking membership. **DP-SGD** reduces this advantage.

---

## Ex 8 — DP-SGD quick recap (Opacus)

### 👶 What this does
Clip each per-sample gradient to `C` and add Gaussian noise with std
`σC`. Track the `(ε, δ)` budget.

```python
# from opacus import PrivacyEngine
# pe = PrivacyEngine()
# model, optim, loader = pe.make_private(
#     module=model, optimizer=optim, data_loader=loader,
#     noise_multiplier=1.0, max_grad_norm=1.0)
# for epoch: train...
# print(pe.get_epsilon(delta=1e-5))
```

See [Federated_Learning/code Ex 4](../Federated_Learning/federated_learning_code.md#ex-4--differential-privacy-with-opacus-dp-sgd-locally)
for the full version.

---

## Ex 9 — PII redaction with Presidio

### 👶 What this does
Scan text (e.g. a voice-to-text transcript) for PII and replace with
tokens **before** any upload.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer  = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = "Hi, I'm Rohit Patel, my phone is +91 9811-234-567 and email rp@example.com"
results = analyzer.analyze(text=text, language="en")
clean = anonymizer.anonymize(text=text, analyzer_results=results).text
print("in :", text)
print("out:", clean)
```

Typical output:
```
in : Hi, I'm Rohit Patel, my phone is +91 9811-234-567 and email rp@example.com
out: Hi, I'm <PERSON>, my phone is <PHONE_NUMBER> and email <EMAIL_ADDRESS>
```

---

## Ex 10 — Randomized smoothing (certified robustness)

```python
def smoothed_predict(model, x, sigma=0.25, N=100, classes=1000):
    # Averaged prediction under Gaussian noise
    preds = torch.zeros(classes)
    with torch.no_grad():
        for _ in range(N):
            noise = torch.randn_like(x) * sigma
            out = model(x + noise).softmax(-1)
            preds += out[0]
    return preds / N

avg = smoothed_predict(m, x)
print("Top-1 under smoothing:", int(avg.argmax()))
```

Formally: certified robust radius `R = σ · Φ^{-1}(p_A)` where `p_A`
is the smoothed probability of the top class.

---

## Ex 11 — Secure boot rehearsal (Python mock)

```python
import hashlib

def stage_digest(name, data):
    return hashlib.sha256(data).hexdigest()

# "Fuse"-burned known-good digests for each stage
FUSES = {
    "BL1": stage_digest("BL1", b"bl1 binary v1.0"),
    "BL2": stage_digest("BL2", b"bl2 binary v1.2"),
    "APP": stage_digest("APP", b"app binary v2.3.1"),
}

def boot(stage, blob):
    d = stage_digest(stage, blob)
    if d != FUSES[stage]:
        raise RuntimeError(f"❌ {stage} digest mismatch — halt")
    print(f"✅ {stage} verified")

boot("BL1", b"bl1 binary v1.0")
boot("BL2", b"bl2 binary v1.2")
boot("APP", b"app binary v2.3.1")
```

A tampered stage raises — exactly what a real bootloader would do in
silicon.

---

## Ex 12 — Constant-time comparison (side-channel safe)

```python
def constant_time_eq(a: bytes, b: bytes) -> bool:
    """Don't early-return; keep running time independent of input."""
    if len(a) != len(b): return False
    r = 0
    for x, y in zip(a, b):
        r |= x ^ y
    return r == 0

# Use this to compare MACs / tokens / hashes — never `==`.
```

In Python, use `hmac.compare_digest`. In C, `memcmp` is *not* safe —
use `CRYPTO_memcmp` from OpenSSL or a custom constant-time routine.

---

## 📝 Summary

| Exercise | What you built |
|---|---|
| 1 | ed25519 signature verify for OTA |
| 2 | AES-GCM encrypt/decrypt for model weights |
| 3 | FGSM attack |
| 4 | PGD attack |
| 5 | Adversarial training loop |
| 6 | Defensive distillation loss |
| 7 | Membership inference threshold attack |
| 8 | DP-SGD recap |
| 9 | Presidio PII redactor |
| 10 | Randomized smoothing |
| 11 | Secure boot rehearsal |
| 12 | Constant-time compare |

Now put attack + defence + privacy into one notebook →
[practice.md](security_privacy_practice.md).

---

> *GPU Programming · EdgeAI · Security & Privacy · CODE · github.com/rpaut03l/TS-02*
