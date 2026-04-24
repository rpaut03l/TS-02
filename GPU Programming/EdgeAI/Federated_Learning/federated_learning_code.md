# 💻 EdgeAI · Federated Learning — CODE

### *Hand-rolled FedAvg · Flower framework · LoRA on the edge · DP-SGD*

> **Nav:** [← Federated Learning README](README.md) | [📖 THEORY](federated_learning_theory.md) | **CODE** | [🎯 PRACTICE →](federated_learning_practice.md)

---

## 🏗️ Setup

```python
!pip install -q torch torchvision flwr opacus peft transformers
```

---

## Ex 1 — Hand-rolled FedAvg simulator in ~60 lines

### 👶 What this does
No frameworks. Just a function that simulates K clients on one
process and implements the FedAvg round from the theory file.

```python
import torch, torch.nn as nn, torch.nn.functional as F, random, copy

# --- 1. Tiny model used by everyone ---
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

# --- 2. Fake K client datasets (non-IID by class) ---
from torchvision import datasets, transforms
mnist = datasets.MNIST("./data", train=True, download=True,
                       transform=transforms.ToTensor())
K = 10
client_data = [[] for _ in range(K)]
for x, y in mnist:
    # each client mostly sees 2 digits → non-IID
    target_client = (y * 3) % K
    client_data[target_client].append((x, y))
for i in range(K):
    print(f"client {i}: {len(client_data[i])} samples")

# --- 3. Local training ---
def local_train(state_dict, data, epochs=1, lr=0.01, bs=32):
    m = SmallCNN(); m.load_state_dict(state_dict); m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    data = [data[i:i+bs] for i in range(0, len(data), bs)]
    for _ in range(epochs):
        random.shuffle(data)
        for batch in data:
            x = torch.stack([b[0] for b in batch])
            y = torch.tensor([b[1] for b in batch])
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return {k: v.detach() for k, v in m.state_dict().items()}

# --- 4. FedAvg ---
def fed_avg(states, weights):
    total = sum(weights)
    avg = {}
    for k in states[0]:
        avg[k] = sum(w * s[k] for w, s in zip(weights, states)) / total
    return avg

# --- 5. Run 10 rounds on half the clients each round ---
global_m = SmallCNN()
for rnd in range(10):
    picked = random.sample(range(K), K // 2)
    local_states, weights = [], []
    for c in picked:
        ls = local_train(global_m.state_dict(), client_data[c])
        local_states.append(ls)
        weights.append(len(client_data[c]))
    global_m.load_state_dict(fed_avg(local_states, weights))

    # tiny eval
    x_ev, y_ev = zip(*mnist[:1000])
    x_ev = torch.stack(x_ev); y_ev = torch.tensor(y_ev)
    with torch.no_grad():
        acc = (global_m(x_ev).argmax(1) == y_ev).float().mean().item()
    print(f"round {rnd}: acc {acc*100:.1f}%")
```

This is the **entire core** of federated learning. Every framework
you'll meet is just features layered on top of this loop.

---

## Ex 2 — Same thing in **Flower** (production-style)

### 👶 What this does
Flower is the most popular cross-language FL framework. The client ↔
server contract is tiny — just `get_parameters`, `fit`, `evaluate`.

```python
import flwr as fl
from collections import OrderedDict

# Client definition
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, cid, data):
        self.cid, self.data = cid, data
        self.model = SmallCNN()
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]
    def set_parameters(self, params):
        keys = self.model.state_dict().keys()
        sd = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
        self.model.load_state_dict(sd, strict=True)
    def fit(self, params, config):
        self.set_parameters(params)
        new_sd = local_train(self.model.state_dict(), self.data)
        self.model.load_state_dict(new_sd)
        return self.get_parameters(config={}), len(self.data), {}
    def evaluate(self, params, config):
        self.set_parameters(params)
        # tiny eval on client data
        x = torch.stack([d[0] for d in self.data[:200]])
        y = torch.tensor([d[1] for d in self.data[:200]])
        with torch.no_grad():
            acc = (self.model(x).argmax(1) == y).float().mean().item()
        return float(1 - acc), 200, {"accuracy": float(acc)}

# Server side — default FedAvg strategy, 3 rounds, simulated
def client_fn(cid: str):
    return MNISTClient(int(cid), client_data[int(cid)])

# Run in-process simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=K,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(fraction_fit=0.5,
                                       fraction_evaluate=0.5),
)
```

Flower handles client selection, aggregation, failure handling, logs,
and scaling out to real distributed runs — all with the same
`NumPyClient` you just wrote.

---

## Ex 3 — TensorFlow Federated (cross-device flavour)

### 👶 What this does
TFF is Google's FL stack. Its superpower is **compiling** the
federated algorithm to a graph that can run unchanged in simulation
or in production with real devices.

```python
# !pip install -q tensorflow-federated==0.80.0
# import tensorflow as tf
# import tensorflow_federated as tff
#
# def create_keras_model():
#     return tf.keras.Sequential([
#         tf.keras.layers.Reshape((784,), input_shape=(28,28,1)),
#         tf.keras.layers.Dense(64, activation="relu"),
#         tf.keras.layers.Dense(10, activation="softmax")])
#
# def model_fn():
#     return tff.learning.models.from_keras_model(
#         create_keras_model(),
#         input_spec=(tf.TensorSpec([None,28,28,1]), tf.TensorSpec([None], tf.int32)),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#
# iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
#     model_fn=model_fn,
#     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01),
#     server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0))
#
# state = iterative_process.initialize()
# # Then call iterative_process.next(state, sampled_client_datasets) per round
```

TFF's learning curve is steep but the production story on Android /
Google Cloud is unmatched.

---

## Ex 4 — Differential Privacy with Opacus (DP-SGD locally)

### 👶 What this does
Add DP noise to the *local* gradient inside each client. Combined
with Secure Aggregation on the server, you get the gold-standard
private FL.

```python
import torch
from opacus import PrivacyEngine

m = SmallCNN()
opt = torch.optim.SGD(m.parameters(), lr=0.01)

# Fake local dataset loader
from torch.utils.data import DataLoader, TensorDataset
xs = torch.stack([d[0] for d in client_data[0][:512]])
ys = torch.tensor([d[1] for d in client_data[0][:512]])
loader = DataLoader(TensorDataset(xs, ys), batch_size=64)

pe = PrivacyEngine()
m, opt, loader = pe.make_private(
    module=m,
    optimizer=opt,
    data_loader=loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

for epoch in range(2):
    for xb, yb in loader:
        loss = F.cross_entropy(m(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()

print(f"ε = {pe.get_epsilon(delta=1e-5):.2f} at δ=1e-5")
```

Opacus **clips** each per-sample gradient to `max_grad_norm` and adds
Gaussian noise with std `noise_multiplier × max_grad_norm`. The
privacy accountant tracks the total `ε` budget.

---

## Ex 5 — Secure Aggregation sketch (pairwise masks)

### 👶 What this does
A toy two-client secure aggregation — server sees the sum but cannot
isolate any client's update. Real protocols use Diffie-Hellman +
secret sharing for dropout tolerance; this is just the core idea.

```python
import torch, hashlib

def prng_tensor(seed, shape):
    # Deterministic "random" tensor derived from a seed
    g = torch.Generator().manual_seed(
        int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
    return torch.randn(*shape, generator=g)

def mask_update(delta, pair_keys, my_id):
    """Add +m for each peer_id > my_id, -m for each peer_id < my_id."""
    m = torch.zeros_like(delta)
    for other_id, k in pair_keys.items():
        if other_id > my_id:
            m += prng_tensor(k, delta.shape)
        else:
            m -= prng_tensor(k, delta.shape)
    return delta + m

# Both clients share the same random key per pair
pair_keys = {(0, 1): "pair_0_1"}
d0 = torch.randn(5); d1 = torch.randn(5)

masked_0 = mask_update(d0, {1: pair_keys[(0,1)]}, my_id=0)
masked_1 = mask_update(d1, {0: pair_keys[(0,1)]}, my_id=1)

server_sees = masked_0 + masked_1
real_sum    = d0 + d1
print("server sum ≈ real sum?", torch.allclose(server_sees, real_sum, atol=1e-6))
```

Output: **True**. Server sees `d0 + d1` without ever seeing `d0` or
`d1` individually.

---

## Ex 6 — LoRA adapters for on-device fine-tuning

### 👶 What this does
Freeze a big backbone. Train two small rank-r matrices per layer.
Client uploads only these ~1–5 MB adapters per round.

```python
# !pip install -q peft transformers
# from peft import LoraConfig, get_peft_model, TaskType
# from transformers import AutoModelForSequenceClassification
#
# base = AutoModelForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased", num_labels=2)
#
# cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
#                  target_modules=["q_lin", "v_lin"], lora_dropout=0.1)
# lora_model = get_peft_model(base, cfg)
# lora_model.print_trainable_parameters()
# # Output: "trainable params: 1,182,720 || all params: 67,xxx,xxx"
```

A phone ships the frozen DistilBERT once. Each round it trains only
the ~1 M LoRA params locally — a tiny upload.

---

## Ex 7 — Compressed updates: top-k + INT8

### 👶 What this does
Keep only the 10 % largest-magnitude entries in each update, quantize
them to INT8. Uplink shrinks ~40×.

```python
def top_k_int8(delta, k_frac=0.1):
    flat = delta.flatten()
    k = int(k_frac * flat.numel())
    values, idx = torch.topk(flat.abs(), k)
    # Quantize the kept values to INT8
    scale = values.max() / 127.0
    v_q = (flat[idx] / scale).round().clamp(-128, 127).to(torch.int8)
    return idx.to(torch.int32), v_q, scale

def decompress(idx, v_q, scale, shape):
    out = torch.zeros(int(torch.prod(torch.tensor(shape))))
    out[idx] = v_q.float() * scale
    return out.reshape(shape)

delta = torch.randn(1024)
idx, v_q, s = top_k_int8(delta, 0.1)
recon = decompress(idx, v_q, s, delta.shape)
uplink_bytes = idx.numel() * 4 + v_q.numel() * 1 + 4
full_bytes   = delta.numel() * 4
print(f"Uplink {uplink_bytes} B  vs full {full_bytes} B  → "
      f"{full_bytes/uplink_bytes:.1f}× cut")
print(f"Error L2: {((delta - recon)**2).mean().sqrt():.4f}")
```

---

## Ex 8 — Stragglers & client sampling

### 👶 What this does
Only wait a fixed time budget per round. Use whoever reported back.

```python
import time, random
def simulated_client_step(cid):
    time.sleep(random.uniform(0.05, 0.5))  # variable local latency
    return f"update_from_{cid}"

def round_with_timeout(clients, timeout=0.2):
    results = []
    t0 = time.time()
    for c in clients:
        if time.time() - t0 > timeout: break
        results.append(simulated_client_step(c))
    return results

print(round_with_timeout(list(range(10))))
```

A real system does this in parallel with asyncio / threads, plus
cryptographic dropout-handling inside Secure Aggregation.

---

## 📝 Summary

| Exercise | Idea |
|---|---|
| 1 | Hand-rolled FedAvg — the whole algorithm in ~60 lines |
| 2 | Same in Flower — production-shaped client/server |
| 3 | TFF — Google's federated learning compiler |
| 4 | Opacus DP-SGD — private local training |
| 5 | Pairwise masks — kernel idea of Secure Aggregation |
| 6 | LoRA adapters — cheap on-device fine-tuning |
| 7 | Top-k + INT8 compression — uplink shrink |
| 8 | Stragglers — timeout + partial-result rounds |

Now glue them together in the [practice notebook](federated_learning_practice.md).

---

> *GPU Programming · EdgeAI · Federated Learning · CODE · github.com/rpaut03l/TS-02*
