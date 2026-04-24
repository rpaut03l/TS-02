# 🎯 EdgeAI · Federated Learning — PRACTICE

### *Simulate 20 non-IID clients · Run 20 rounds of FedAvg · Plot global accuracy*

> **Nav:** [← Federated Learning README](README.md) | [📖 THEORY](federated_learning_theory.md) | [💻 CODE](federated_learning_code.md) | **PRACTICE**

---

## 🎯 What you'll build

A Colab notebook that:
1. Partitions MNIST into **20 non-IID clients** (each sees only 2
   digits, on average).
2. Runs **20 rounds** of FedAvg (fraction of clients sampled per
   round).
3. Plots **global test accuracy** vs round.
4. Runs an equivalent **centralised training** baseline and compares.
5. Adds DP-SGD and re-runs to see the privacy vs utility trade-off.

---

## Cell 1 — Setup

```python
import os, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train = datasets.MNIST("./data", train=True, download=True,
                       transform=transforms.ToTensor())
test  = datasets.MNIST("./data", train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test, batch_size=500)
```

---

## Cell 2 — Partition into 20 non-IID clients

```python
NUM_CLIENTS = 20
SHARDS_PER_CLIENT = 2

# Sort indices by label
by_label = [[] for _ in range(10)]
for i, (_, y) in enumerate(train):
    by_label[y].append(i)

# Build shards (each shard is a slice of one label)
shards = []
per_label_shards = 4            # 10 * 4 = 40 shards
for y in range(10):
    ids = by_label[y]
    chunk = len(ids) // per_label_shards
    for k in range(per_label_shards):
        shards.append(ids[k*chunk:(k+1)*chunk])

random.shuffle(shards)
client_indices = []
for c in range(NUM_CLIENTS):
    s = shards[c * SHARDS_PER_CLIENT:(c+1) * SHARDS_PER_CLIENT]
    client_indices.append([i for sh in s for i in sh])

for c in range(5):
    labels = set(train[i][1] for i in client_indices[c][:200])
    print(f"client {c}: {len(client_indices[c])} samples, labels={sorted(labels)}")
```

You should see each client has only ~2 distinct labels — a textbook
non-IID split.

---

## Cell 3 — Model + local-train + FedAvg

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 10))
    def forward(self, x): return self.net(x)

def get_loader(idx, bs=64):
    return DataLoader(Subset(train, idx), batch_size=bs, shuffle=True)

def local_train(state, idx, epochs=1, lr=0.01):
    m = MLP().to(device); m.load_state_dict(state); m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in get_loader(idx):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}

def fed_avg(states, weights):
    total = float(sum(weights))
    avg = {}
    for k in states[0]:
        stack = torch.stack([s[k].float() for s in states], 0)
        w = torch.tensor(weights, dtype=torch.float32).view(-1, *([1]*(stack.dim()-1)))
        avg[k] = (stack * w).sum(0) / total
        avg[k] = avg[k].to(states[0][k].dtype)
    return avg

def evaluate(state):
    m = MLP().to(device); m.load_state_dict(state); m.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (m(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total
```

---

## Cell 4 — Run FedAvg for 20 rounds

```python
ROUNDS = 20; FRAC_C = 0.5
global_state = MLP().state_dict()
fed_acc = []

for r in range(ROUNDS):
    sampled = random.sample(range(NUM_CLIENTS),
                            int(NUM_CLIENTS * FRAC_C))
    updates, weights = [], []
    for c in sampled:
        u = local_train(global_state, client_indices[c], epochs=1)
        updates.append(u); weights.append(len(client_indices[c]))
    global_state = fed_avg(updates, weights)
    acc = evaluate(global_state)
    fed_acc.append(acc)
    print(f"round {r+1:2d}: acc {acc*100:5.2f}%")
```

Expect global accuracy to climb from ~20 % to **~85–90 %** over 20
rounds despite the hard non-IID split.

---

## Cell 5 — Centralised baseline

```python
m = MLP().to(device)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
central_acc = []

full_loader = DataLoader(train, batch_size=64, shuffle=True)
for epoch in range(5):
    m.train()
    for x, y in full_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    central_acc.append(evaluate(m.state_dict()))
    print(f"central epoch {epoch+1}: acc {central_acc[-1]*100:.2f}%")
```

Centralised with the same model typically reaches **~97 %** — a
reminder that FL isn't *free*; you pay some accuracy for privacy.

---

## Cell 6 — Plot the comparison

```python
plt.figure(figsize=(8, 5))
plt.plot(range(1, ROUNDS+1), [a*100 for a in fed_acc], "o-", label="FedAvg (20 non-IID clients)")
plt.axhline(max(central_acc)*100, color="gray", linestyle="--",
            label=f"Centralised best ({max(central_acc)*100:.1f}%)")
plt.xlabel("Round"); plt.ylabel("Global test accuracy (%)")
plt.title("FedAvg on non-IID MNIST")
plt.grid(alpha=0.3); plt.legend(); plt.show()
```

---

## Cell 7 — FedProx — softer convergence on non-IID

### 👶 What this does
Add a **proximal term** to the local objective so updates don't drift
too far from the global model.

```python
def local_train_prox(state, idx, epochs=1, lr=0.01, mu=0.01):
    m = MLP().to(device); m.load_state_dict(state); m.train()
    global_tensors = {k: v.to(device) for k, v in state.items()}
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in get_loader(idx):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(m(x), y)
            # proximal term
            prox = 0.0
            for name, p in m.named_parameters():
                prox += ((p - global_tensors[name]) ** 2).sum()
            loss = loss + (mu / 2.0) * prox
            opt.zero_grad(); loss.backward(); opt.step()
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}

prox_state = MLP().state_dict(); prox_acc = []
for r in range(ROUNDS):
    sampled = random.sample(range(NUM_CLIENTS), int(NUM_CLIENTS * FRAC_C))
    updates, weights = [], []
    for c in sampled:
        u = local_train_prox(prox_state, client_indices[c], mu=0.01)
        updates.append(u); weights.append(len(client_indices[c]))
    prox_state = fed_avg(updates, weights)
    prox_acc.append(evaluate(prox_state))
    print(f"FedProx round {r+1}: {prox_acc[-1]*100:.2f}%")
```

---

## Cell 8 — Compressed updates: top-10% + INT8

### 👶 What this does
Drop all but the top-10% biggest deltas, quantize to INT8. Keep
reconstructing on the server. Compare final accuracy.

(Use the `top_k_int8` / `decompress` pair from [code.md Ex 7](federated_learning_code.md#ex-7--compressed-updates-top-k--int8)
inside your `fed_avg` loop before aggregating. Expect ~1 % accuracy
drop for ~40× less uplink.)

---

## Cell 9 — DP-SGD local training (Opacus)

### 👶 What this does
Replace the plain local SGD with **clipped + noised** SGD. Show the
privacy budget you've spent.

```python
!pip install -q opacus
from opacus import PrivacyEngine

def local_train_dp(state, idx, noise=1.0, max_norm=1.0):
    m = MLP().to(device); m.load_state_dict(state); m.train()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    loader = get_loader(idx, bs=64)
    pe = PrivacyEngine()
    m, opt, loader = pe.make_private(
        module=m, optimizer=opt, data_loader=loader,
        noise_multiplier=noise, max_grad_norm=max_norm)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

    eps = pe.get_epsilon(delta=1e-5)
    sd = {k.replace("_module.", ""): v.detach().cpu()
          for k, v in m.state_dict().items() if not k.startswith("_module_gsm")}
    return sd, eps

# 5 DP rounds for a smoke test
dp_state = MLP().state_dict(); dp_acc = []; eps_log = []
for r in range(5):
    sampled = random.sample(range(NUM_CLIENTS), int(NUM_CLIENTS * FRAC_C))
    updates, weights, round_eps = [], [], 0
    for c in sampled:
        u, eps = local_train_dp(dp_state, client_indices[c])
        updates.append(u); weights.append(len(client_indices[c])); round_eps = eps
    dp_state = fed_avg(updates, weights)
    acc = evaluate(dp_state)
    dp_acc.append(acc); eps_log.append(round_eps)
    print(f"DP round {r+1}: acc {acc*100:.2f}%  ε={round_eps:.2f}")
```

DP-SGD hurts utility — usually **5–15 %** accuracy drop — but the
`(ε, δ)` guarantee is often worth it for regulated data.

---

## Cell 10 — Stretch goals

1. Increase heterogeneity — 1 shard per client. How does FedAvg vs
   FedProx diverge?
2. Add **client dropout** — on each round, randomly skip 30 % of
   sampled clients. Does the curve still climb?
3. Replace `fed_avg` with **server momentum** (FedAvgM) — track a
   running `m_t = β m_{t-1} + Δ_t` and apply it instead. Does
   convergence speed up?
4. Switch the model to **LoRA-style**: freeze the first linear
   layer, train only the second. Upload only the smaller tensor.
5. Measure **total uplink bytes** per experiment and plot a
   "bytes-per-% accuracy" chart.

---

## 🎓 What you should take away

- FedAvg **works**, even on hard non-IID splits — just slower than
  centralised.
- **Non-IID** is always the hardest part. Reach for FedProx / server
  momentum first.
- **Privacy is a separate ingredient** — Secure Aggregation +
  DP-SGD. FL alone is not privacy.
- **Communication is the first thing you'll actually optimise** in
  production — compressed + top-k + LoRA adapters.
- Frameworks like **Flower** and **TFF** give you all of this
  without rewriting the loop.

Next: [**Edge MLOps →**](../Edge_MLOps/README.md) — shipping, updating,
and monitoring all of this in the wild.

---

> *GPU Programming · EdgeAI · Federated Learning · PRACTICE · github.com/rpaut03l/TS-02*
