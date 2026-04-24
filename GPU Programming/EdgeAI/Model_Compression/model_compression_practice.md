# 🎯 EdgeAI · Model Compression — PRACTICE

### *Take MobileNetV2 all the way down — quant + prune + distill — plot the Pareto*

> **Nav:** [← Model Compression README](README.md) | [📖 THEORY](model_compression_theory.md) | [💻 CODE](model_compression_code.md) | **PRACTICE**

---

## 🎯 What you'll build

One Colab notebook that, on the same cat-vs-dog dataset from
[Fundamentals/practice](../Fundamentals/edge_ai_fundamentals_practice.md):

1. Trains an FP32 MobileNetV2 baseline.
2. Produces **6 compressed variants** (FP16, INT8 PTQ, INT8 QAT, INT8 +
   pruning, distilled student, INT8 + prune + distill).
3. Measures (size, latency, accuracy) for each.
4. Plots a **Pareto chart** and picks the best config for a fixed
   latency budget.

---

## Cell 1 — Setup

```python
!pip install -q torch torchvision tensorflow onnx onnxruntime

import os, time, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F, torchvision as tv
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```

---

## Cell 2 — Dataset

```python
import tensorflow as tf
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/cats_and_dogs_filtered.zip"
zp  = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", URL, extract=True)
root = os.path.join(os.path.dirname(zp), "cats_and_dogs_filtered")

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

tf_train = T.Compose([T.Resize((160, 160)), T.RandomHorizontalFlip(),
                      T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
tf_eval  = T.Compose([T.Resize((160, 160)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_ds = ImageFolder(f"{root}/train",      transform=tf_train)
val_ds   = ImageFolder(f"{root}/validation", transform=tf_eval)
train_dl = DataLoader(train_ds, 32, shuffle=True, num_workers=2)
val_dl   = DataLoader(val_ds,   64, shuffle=False, num_workers=2)
print("train:", len(train_ds), "val:", len(val_ds))
```

---

## Cell 3 — Train the FP32 baseline (teacher)

```python
teacher = tv.models.mobilenet_v2(weights="DEFAULT")
teacher.classifier[1] = nn.Linear(teacher.classifier[1].in_features, 2)
teacher = teacher.to(device)

opt = torch.optim.Adam(teacher.parameters(), lr=3e-4)
for epoch in range(3):
    teacher.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(teacher(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

def eval_acc(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x).argmax(1); correct += (p == y).sum().item(); total += y.size(0)
    return correct / total

base_acc = eval_acc(teacher, val_dl)
torch.save(teacher.state_dict(), "fp32.pt")
print(f"FP32 teacher acc: {base_acc*100:.2f}%  size: {os.path.getsize('fp32.pt')/1024**2:.2f} MB")
```

---

## Cell 4 — Benchmark helper (size, latency, accuracy, in one call)

```python
def measure(model, path, loader, iters=200, input_shape=(1, 3, 160, 160)):
    torch.save(model.state_dict(), path)
    size_MB = os.path.getsize(path) / 1024**2

    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(20): model(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(iters): model(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        ms = (time.perf_counter() - t0) / iters * 1000

    acc = eval_acc(model, loader)
    return size_MB, ms, acc

size0, ms0, acc0 = measure(teacher, "fp32.pt", val_dl)
results = [("FP32 baseline", size0, ms0, acc0)]
print(f"FP32:   {size0:6.2f} MB   {ms0:5.2f} ms   acc {acc0*100:.2f}%")
```

---

## Cell 5 — FP16 weights

```python
import copy
fp16 = copy.deepcopy(teacher)
for p in fp16.parameters():
    p.data = p.data.half()

# Keep activations in FP16 too for a fair latency number
x = torch.randn(1, 3, 160, 160, device=device).half()
fp16.eval()
with torch.no_grad():
    for _ in range(20): fp16(x)
    t0 = time.perf_counter()
    for _ in range(200): fp16(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 200 * 1000

torch.save(fp16.state_dict(), "fp16.pt")
size = os.path.getsize("fp16.pt") / 1024**2
# Accuracy: cast inputs to half during eval
def eval_acc_fp16():
    fp16.eval(); correct = total = 0
    with torch.no_grad():
        for xv, yv in val_dl:
            xv = xv.to(device).half(); yv = yv.to(device)
            p = fp16(xv).argmax(1); correct += (p == yv).sum().item(); total += yv.size(0)
    return correct / total
acc = eval_acc_fp16()
results.append(("FP16 weights", size, ms, acc))
print(f"FP16:   {size:6.2f} MB   {ms:5.2f} ms   acc {acc*100:.2f}%")
```

---

## Cell 6 — INT8 PTQ full-integer (via TFLite path)

The cleanest INT8 on CPU. We go through TFLite because PyTorch's native
quant runs only on CPU anyway.

```python
# Build a matching tf.keras model, load teacher weights via ONNX roundtrip,
# or re-train a small TF twin quickly:
# (Simplified: re-use your earlier catdog_int8.tflite from the
#  Fundamentals notebook. Just capture its size, latency, accuracy.)

path = "catdog_int8.tflite"
if os.path.exists(path):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    # accuracy
    correct = total = 0
    for xv, yv in val_dl:
        arr = xv.numpy().transpose(0, 2, 3, 1).astype("float32")
        scale, zp = in_det["quantization"]
        arr = (arr / scale + zp).astype("int8")
        for i in range(arr.shape[0]):
            interp.set_tensor(in_det["index"], arr[i:i+1])
            interp.invoke()
            logit = interp.get_tensor(out_det["index"])[0][0]
            pred = int(logit > 0)
            correct += int(pred == yv[i].item())
            total += 1
    acc = correct / total

    # latency
    x_dummy = np.zeros(in_det["shape"], dtype=in_det["dtype"])
    for _ in range(20): interp.set_tensor(in_det["index"], x_dummy); interp.invoke()
    t0 = time.perf_counter()
    for _ in range(200): interp.set_tensor(in_det["index"], x_dummy); interp.invoke()
    ms = (time.perf_counter() - t0) / 200 * 1000
    size = os.path.getsize(path) / 1024**2
    results.append(("INT8 PTQ (TFLite)", size, ms, acc))
    print(f"INT8 PTQ: {size:6.2f} MB   {ms:5.2f} ms   acc {acc*100:.2f}%")
```

---

## Cell 7 — Pruning (50 % unstructured magnitude) + fine-tune

```python
import torch.nn.utils.prune as P

pruned = copy.deepcopy(teacher).to(device)
for name, m in pruned.named_modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        P.l1_unstructured(m, "weight", amount=0.5)

# Fine-tune 1 epoch to recover
opt = torch.optim.Adam(pruned.parameters(), lr=1e-4)
pruned.train()
for x, y in train_dl:
    x, y = x.to(device), y.to(device)
    loss = F.cross_entropy(pruned(x), y)
    opt.zero_grad(); loss.backward(); opt.step()

# Remove masks to make storage savings real (sparse tensor would help more)
for name, m in pruned.named_modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        P.remove(m, "weight")

size, ms, acc = measure(pruned, "pruned50.pt", val_dl)
results.append(("Pruned 50% + FT", size, ms, acc))
print(f"Pruned: {size:6.2f} MB   {ms:5.2f} ms   acc {acc*100:.2f}%")
```

*Note:* on dense hardware (Colab T4), unstructured pruning doesn't
actually speed things up — you'd see similar ms. It shrinks *sparse*
storage formats; it's a stepping stone toward structured pruning or
2:4 on Ampere+.

---

## Cell 8 — Knowledge distillation (student = MobileNetV3-Small)

```python
student = tv.models.mobilenet_v3_small(weights=None)
student.classifier[3] = nn.Linear(student.classifier[3].in_features, 2)
student = student.to(device)

T, alpha = 4.0, 0.1
opt = torch.optim.Adam(student.parameters(), lr=3e-4)
teacher.eval()
for epoch in range(3):
    student.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        sl = student(x)
        with torch.no_grad(): tl = teacher(x)
        hard = F.cross_entropy(sl, y)
        soft = F.kl_div(F.log_softmax(sl/T, -1), F.softmax(tl/T, -1),
                        reduction="batchmean") * (T*T)
        loss = alpha * hard + (1 - alpha) * soft
        opt.zero_grad(); loss.backward(); opt.step()

size, ms, acc = measure(student, "student.pt", val_dl)
results.append(("Distilled V3-Small", size, ms, acc))
print(f"Student: {size:6.2f} MB   {ms:5.2f} ms   acc {acc*100:.2f}%")
```

---

## Cell 9 — Stack: distilled + pruned + (INT8 via TFLite handled above)

```python
stacked = copy.deepcopy(student).to(device)
for name, m in stacked.named_modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        P.l1_unstructured(m, "weight", amount=0.3)

opt = torch.optim.Adam(stacked.parameters(), lr=5e-5)
stacked.train()
for x, y in train_dl:
    x, y = x.to(device), y.to(device)
    loss = F.cross_entropy(stacked(x), y)
    opt.zero_grad(); loss.backward(); opt.step()

for name, m in stacked.named_modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        P.remove(m, "weight")

size, ms, acc = measure(stacked, "stacked.pt", val_dl)
results.append(("Student + 30% prune", size, ms, acc))
print(f"Stacked: {size:6.2f} MB   {ms:5.2f} ms   acc {acc*100:.2f}%")
```

---

## Cell 10 — Plot the Pareto

```python
names, sizes, mss, accs = zip(*results)
plt.figure(figsize=(8, 5))
plt.scatter(sizes, [a*100 for a in accs], s=[m*30 for m in mss], alpha=0.7)
for n, s, a in zip(names, sizes, accs):
    plt.annotate(n, (s, a*100), xytext=(5, 5),
                 textcoords="offset points", fontsize=9)
plt.xlabel("Size (MB)"); plt.ylabel("Accuracy (%)")
plt.title("Compression Pareto — bubble size = latency")
plt.grid(alpha=0.3); plt.xscale("log"); plt.show()

print("\nRanked by accuracy/MB:")
import pandas as pd
pd.DataFrame(results, columns=["variant","size_MB","ms","acc"])\
  .assign(acc_per_MB=lambda d: d.acc / d.size_MB)\
  .sort_values("acc_per_MB", ascending=False)
```

---

## Cell 11 — Pick a config for a 10 ms / 15 MB budget

```python
budget_ms, budget_mb = 10.0, 15.0
ok = [r for r in results if r[1] <= budget_mb and r[2] <= budget_ms]
ok = sorted(ok, key=lambda r: -r[3])   # highest accuracy first
if ok:
    n, s, m, a = ok[0]
    print(f"🏆 Best under 15 MB & 10 ms: {n} — {a*100:.2f}% acc, {s:.2f} MB, {m:.2f} ms")
else:
    print("No config fits both budgets — relax one.")
```

---

## Cell 12 — Stretch goals

1. Add **QAT** (Ex 5 in code.md). Does it beat PTQ here?
2. Apply **SVD low-rank** on `student.classifier[3]`. Any free shrink?
3. Try **2:4 structured sparsity** on the stacked model, export ONNX,
   build a TensorRT engine with `SPARSE_WEIGHTS`, re-measure.
4. Swap the teacher for **ResNet-50** and re-distill — bigger gap
   often means better student.
5. Add an **energy column** using the formula `mJ = ms * TDP_W`.
   Re-rank.

---

## 🎓 What you should take away

- Even on one notebook, compression routinely gets you **5–15× smaller
  models** with < 1 % accuracy cost.
- **No single technique is the winner.** Stacked techniques always
  dominate any single lever.
- The **Pareto chart** is the deliverable. "Best model" is a
  lie — there's only "best model *for this budget*".
- For edge **CPUs / NPUs**: INT8 PTQ wins first.
- For edge **GPUs** (Jetson): FP16 wins first, INT8 second.
- For **LLMs**: NF4 / GPTQ / AWQ are the only practical choices on an
  edge-class card.

Next: [**Deployment Frameworks →**](../Deployment_Frameworks/README.md)

---

> *GPU Programming · EdgeAI · Model Compression · PRACTICE · github.com/rpaut03l/TS-02*
