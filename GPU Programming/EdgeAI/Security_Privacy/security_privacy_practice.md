# 🎯 EdgeAI · Security & Privacy — PRACTICE

### *Attack · Defend · Privatise · Plot the 3-axis Pareto*

> **Nav:** [← Security & Privacy README](README.md) | [📖 THEORY](security_privacy_theory.md) | [💻 CODE](security_privacy_code.md) | **PRACTICE**

---

## 🎯 What you'll build

One Colab notebook that:
1. Trains a tiny CIFAR-10 model.
2. Attacks it with **FGSM** and **PGD** at several ε values.
3. Rebuilds it with **adversarial training**.
4. Rebuilds it again with **DP-SGD** (privacy).
5. Plots a **three-axis Pareto** — clean accuracy vs robust accuracy
   vs privacy ε.

---

## Cell 1 — Setup & data

```python
!pip install -q torch torchvision opacus
import os, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv, torchvision.transforms as T, numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = T.Compose([T.ToTensor(),
                 T.Normalize((0.5,)*3, (0.5,)*3)])
train = tv.datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
test  = tv.datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
from torch.utils.data import DataLoader
train_dl = DataLoader(train, 128, shuffle=True, num_workers=2)
test_dl  = DataLoader(test, 256, shuffle=False, num_workers=2)
```

---

## Cell 2 — Small model

```python
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, 10))
    def forward(self, x): return self.net(x)
```

---

## Cell 3 — Baseline training

```python
def train_loop(model, epochs=3, attack=None, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    for _ in range(epochs):
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            if attack is not None:
                x = attack(model, x, y)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def eval_clean(model):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total

base = train_loop(TinyCNN())
print(f"Baseline clean acc: {eval_clean(base)*100:.2f}%")
```

---

## Cell 4 — FGSM and PGD attacks

```python
def fgsm(model, x, y, eps=8/255):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + eps * x.grad.sign()).detach().clamp(-1, 1)

def pgd(model, x, y, eps=8/255, alpha=2/255, steps=10):
    orig = x.clone().detach()
    x_adv = orig + (torch.rand_like(orig) * 2 - 1) * eps
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, orig - eps), orig + eps)
        x_adv = x_adv.clamp(-1, 1)
    return x_adv

def eval_robust(model, attack, eps=8/255):
    model.eval(); correct = total = 0
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, eps=eps)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total

print(f"FGSM  ε=8/255: {eval_robust(base, fgsm)*100:.2f}%")
print(f"PGD   ε=8/255: {eval_robust(base, pgd)*100:.2f}%")
```

Baseline robustness typically drops from ~75 % clean to **5–15 %**
under PGD — the whole problem in one line.

---

## Cell 5 — Adversarial training

```python
def pgd_for_training(model, x, y, eps=8/255, alpha=2/255, steps=5):
    return pgd(model, x, y, eps=eps, alpha=alpha, steps=steps)

adv = train_loop(TinyCNN(), epochs=3, attack=pgd_for_training)
clean_adv = eval_clean(adv)
robust_adv = eval_robust(adv, pgd)
print(f"Adv-trained: clean {clean_adv*100:.2f}% | PGD {robust_adv*100:.2f}%")
```

Expect clean accuracy ~5 % lower but robust accuracy **much** higher.

---

## Cell 6 — DP-SGD model (trades clean acc for formal privacy)

```python
from opacus import PrivacyEngine
dp = TinyCNN().to(device)
opt = torch.optim.Adam(dp.parameters(), lr=1e-3)
dp_loader = DataLoader(train, batch_size=64, shuffle=True)

pe = PrivacyEngine()
dp, opt, dp_loader = pe.make_private(
    module=dp, optimizer=opt, data_loader=dp_loader,
    noise_multiplier=1.1, max_grad_norm=1.0)

dp.train()
for _ in range(2):
    for x, y in dp_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(dp(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

eps = pe.get_epsilon(delta=1e-5)
clean_dp = eval_clean(dp)
print(f"DP model : clean {clean_dp*100:.2f}%  ε={eps:.2f}")
```

---

## Cell 7 — 3-axis Pareto plot

```python
rows = [
    ("Baseline",         eval_clean(base), eval_robust(base, pgd), None),
    ("Adv-trained",      clean_adv,        robust_adv,              None),
    ("DP-SGD",           clean_dp,         eval_robust(dp, pgd),    eps),
]

import pandas as pd
df = pd.DataFrame(rows, columns=["name","clean","robust","eps"])
print(df)

plt.figure(figsize=(8, 5))
for _, r in df.iterrows():
    size = 100 if r["eps"] is None else max(40, 400 / r["eps"])
    plt.scatter(r["clean"]*100, r["robust"]*100, s=size)
    plt.annotate(f"{r['name']}" + (f"\nε={r['eps']:.1f}" if r["eps"] else ""),
                 (r["clean"]*100, r["robust"]*100),
                 xytext=(6, 6), textcoords="offset points")
plt.xlabel("Clean accuracy (%)"); plt.ylabel("Robust accuracy (%)")
plt.title("3-axis trade-off — bubble size ∝ privacy strength (smaller ε)")
plt.grid(alpha=0.3); plt.show()
```

No model dominates all three axes — the **Pareto frontier** is the
honest answer to "which defence?". You pick based on the deployment
constraints (regulation, threat model, battery).

---

## Cell 8 — Membership inference attack (simple)

```python
def conf_dist(model, loader, n=1000):
    model.eval(); confs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x).softmax(-1)
            confs.extend(p.gather(1, y.unsqueeze(1)).cpu().numpy().flatten())
            if len(confs) >= n: break
    return np.array(confs[:n])

member_conf    = conf_dist(base, DataLoader(train, 256))
nonmember_conf = conf_dist(base, test_dl)
print(f"Baseline — mem mean {member_conf.mean():.3f} vs nonmem {nonmember_conf.mean():.3f}")

dp_member_conf    = conf_dist(dp, DataLoader(train, 256))
dp_nonmember_conf = conf_dist(dp, test_dl)
print(f"DP       — mem mean {dp_member_conf.mean():.3f} vs nonmem {dp_nonmember_conf.mean():.3f}")
```

The gap `mem_mean − nonmem_mean` is the attacker's signal. DP-SGD
should shrink it noticeably.

---

## Cell 9 — Randomized smoothing (certified)

```python
def smoothed_predict(model, x, sigma=0.25, N=50):
    preds = torch.zeros(10, device=device)
    with torch.no_grad():
        for _ in range(N):
            preds += model(x + torch.randn_like(x) * sigma).softmax(-1)[0]
    return (preds / N).argmax().item()

def smoothed_acc(model, N=30, sigma=0.25):
    model.eval(); correct = total = 0
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        for xi, yi in zip(x, y):
            p = smoothed_predict(model, xi.unsqueeze(0), sigma=sigma, N=N)
            correct += int(p == yi.item()); total += 1
        if total >= 200: break
    return correct / total

print(f"Smoothed (σ=0.25) acc: {smoothed_acc(base)*100:.2f}%")
```

---

## Cell 10 — Stretch goals

1. Sweep the **PGD ε grid** (1/255, 4/255, 8/255, 16/255) for
   baseline vs adv-trained. Plot a **robustness curve**.
2. Try **MixUp + adv-training**. Often a free accuracy boost with
   no extra robustness cost.
3. Sweep `noise_multiplier` in DP-SGD (0.5, 1.0, 2.0) and plot the
   **utility-vs-ε** curve.
4. Implement **LiRA** — a stronger membership inference attack than
   the simple threshold above. Compare advantages.
5. Combine: DP-SGD **inside** federated learning (cross-device) —
   see the [Federated_Learning](../Federated_Learning/README.md)
   chapter. Measure ε and robust accuracy.

---

## 🎓 What you should take away

- **Attacks work.** Every edge model you ship is white-box to a
  motivated adversary.
- **Defences help.** Adversarial training + distillation +
  smoothing + detectors + rate-limits are all useful — **layer them**.
- **Privacy costs accuracy.** Formalise the trade-off with DP
  `(ε, δ)`.
- The **Pareto frontier** is again the deliverable — clean acc vs
  robust acc vs privacy. Pick a point; justify it; ship it.
- **Security without secure boot + signed OTA is theatre.** Start
  from the boot chain.

---

## 🎉 EdgeAI track — done.

You've gone from "what is Edge AI?" all the way to "how do I keep
my deployed models safe against a physical attacker with an
oscilloscope?". If you build something real at each chapter, you'll
finish the track with:

- a trained model, INT8-quantized, pruned, distilled,
- ONNX / TFLite / OpenVINO IR artifacts,
- a TinyML `.cc` file for a 256 KB MCU,
- a federated training simulation,
- a signed OTA pipeline with rollback + drift detection,
- an adversarial robustness + privacy evaluation.

Every one of those is a portfolio-grade artifact.

---

> *GPU Programming · EdgeAI · Security & Privacy · PRACTICE · github.com/rpaut03l/TS-02*
