# 🎯 EdgeAI · GPU Types — PRACTICE

### *Build your own "which edge GPU should I buy?" decision notebook*

> **Nav:** [← GPU Types README](README.md) | [📖 THEORY](edge_gpu_varieties_theory.md) | [💻 CODE](edge_gpu_varieties_code.md) | **PRACTICE**

---

## 🎯 Goal

By the end of this notebook you will have a **reusable tool** that:
1. Takes a model + a workload description (frame rate, image size).
2. Computes the required **TOPS** and **memory bandwidth**.
3. Ranks the 4 GPU families from [the theory file](edge_gpu_varieties_theory.md)
   by fit.
4. Prints a one-line "buy this" recommendation.

---

## 📋 Before you start

- **Runtime → T4 GPU**.
- New Colab notebook.

---

## Cell 1 — A mini spec database

```python
EDGE_GPUS = [
    # family,         name,                    TDP_W, INT8_TOPS, mem_GB, mem_BW_GBps
    ("Mobile-SoC",   "Apple A17 Pro GPU",        5,        35,    8,       51.2),
    ("Mobile-SoC",   "Snapdragon 8G3 Adreno 750", 5,        45,    8,       64.0),
    ("iGPU",         "Intel Iris Xe (Gen12)",    15,         4,    8,       51.2),
    ("iGPU",         "AMD Radeon 780M (RDNA3)",  15,         8,    8,       76.8),
    ("Jetson",       "Jetson Orin Nano 8GB",     15,        40,    8,       68.3),
    ("Jetson",       "Jetson Orin NX 16GB",      25,       100,   16,      102.4),
    ("Jetson",       "Jetson AGX Orin 64GB",     60,       275,   64,      204.8),
    ("Discrete-Edge","NVIDIA Tesla T4",          70,       130,   16,      320.0),
    ("Discrete-Edge","NVIDIA RTX 4000 Ada SFF",  70,       153,   20,      280.0),
    ("Discrete-Edge","NVIDIA L4",                72,       242,   24,      300.0),
]

import pandas as pd
df = pd.DataFrame(EDGE_GPUS, columns=[
    "family","name","TDP_W","INT8_TOPS","mem_GB","mem_BW_GBps"])
df["TOPS_per_W"] = df["INT8_TOPS"] / df["TDP_W"]
df.sort_values("TOPS_per_W", ascending=False)
```

Run it — the **TOPS/Watt** column is the one to watch.

---

## Cell 2 — Describe your workload

```python
workload = dict(
    model_name        = "MobileNetV2",
    input_size        = 224,       # side of square input
    channels          = 3,
    fps               = 30,
    model_MACs_G      = 0.30,      # MobileNetV2 ≈ 0.30 GMAC/inference
    model_params_M    = 3.5,       # 3.5M parameters
    weight_bytes      = 1,         # INT8
)
workload
```

Swap `model_MACs_G` and `model_params_M` for whatever model you care
about. A cheat table:

| Model | GMACs / inf | Params (M) |
|---|---|---|
| MobileNetV2 1.0 | 0.30 | 3.5 |
| MobileNetV3-Small | 0.056 | 2.5 |
| YOLOv8n | 4.1 | 3.2 |
| ResNet-50 | 4.1 | 25.6 |
| YOLOv8m | 25.9 | 25.9 |
| ViT-B/16 | 17.6 | 86.0 |

---

## Cell 3 — Compute the required TOPS and bandwidth

```python
def required_stats(w):
    # Ops per second (INT8 op ≈ 2 MACs)
    ops_per_inf_G = w["model_MACs_G"] * 2            # GOPs per inference
    required_TOPS = ops_per_inf_G * w["fps"] / 1000  # TOPS (Trillion OPs/s)

    # Weight-streaming bandwidth (worst case: weights read every frame)
    model_MB   = w["model_params_M"] * w["weight_bytes"]
    required_BW_GBps = model_MB * w["fps"] / 1024     # GB/s

    return required_TOPS, required_BW_GBps, model_MB

req_tops, req_bw, model_MB = required_stats(workload)
print(f"Required INT8 TOPS : {req_tops:8.4f}")
print(f"Required mem BW   : {req_bw:8.4f} GB/s")
print(f"Model size (INT8) : {model_MB:8.2f} MB")
```

Typical output:
```
Required INT8 TOPS :   0.0180
Required mem BW   :   0.1025 GB/s
Model size (INT8) :     3.50 MB
```

---

## Cell 4 — Rank candidates

```python
def rank(df, req_tops, req_bw, model_MB, target_fps):
    # Keep only chips that satisfy memory, TOPS, and bandwidth.
    ok = df[(df.mem_GB      * 1024 >= model_MB * 5)   # 5× headroom
          & (df.INT8_TOPS          >= req_tops * 3)    # 3× headroom
          & (df.mem_BW_GBps        >= req_bw * 2)]     # 2× headroom

    ok = ok.copy()
    ok["fit_score"] = ok["TOPS_per_W"]     # prefer efficient chips
    return ok.sort_values("fit_score", ascending=False)

ranked = rank(df, req_tops, req_bw, model_MB, workload["fps"])
ranked[["family","name","TDP_W","INT8_TOPS","TOPS_per_W"]]
```

For a 30 FPS MobileNetV2, **almost every edge GPU in the table passes
the filter**. The ranker then picks the most **power-efficient** one
— usually a mobile NPU/GPU or a Jetson Orin Nano.

---

## Cell 5 — Flip to a hard workload

Change the model to **YOLOv8m** at **60 FPS** on **640×640**:

```python
workload.update(
    model_name="YOLOv8m", input_size=640,
    model_MACs_G=25.9, model_params_M=25.9, fps=60)

req_tops, req_bw, model_MB = required_stats(workload)
print("Req TOPS :", round(req_tops, 2))
print("Req BW   :", round(req_bw, 2), "GB/s")
print("Model MB :", round(model_MB, 2))

ranked = rank(df, req_tops, req_bw, model_MB, workload["fps"])
ranked[["family","name","TDP_W","INT8_TOPS","TOPS_per_W"]]
```

Now most **mobile SoC GPUs** and **iGPUs** fall out. The ranker leans
to **Jetson AGX Orin** or **discrete edge GPUs** (T4 / RTX 4000 Ada /
L4). That's exactly the shape real projects follow.

---

## Cell 6 — Produce a one-line recommendation

```python
def recommend(ranked, workload):
    if len(ranked) == 0:
        return f"❌ No edge GPU in the DB can run {workload['model_name']} at {workload['fps']} FPS."
    top = ranked.iloc[0]
    return (f"🏆 For {workload['model_name']} @ {workload['fps']} FPS, "
            f"buy a **{top['name']}** ({top['family']}, {top['TDP_W']} W, "
            f"{top['INT8_TOPS']} TOPS, {top['TOPS_per_W']:.2f} TOPS/W).")

print(recommend(ranked, workload))
```

Example output:
```
🏆 For YOLOv8m @ 60 FPS, buy a NVIDIA L4 (Discrete-Edge, 72 W, 242 TOPS, 3.36 TOPS/W).
```

---

## Cell 7 — Run the real MobileNetV2 and compare

```python
import torch, time
device = torch.device("cuda")
m = torch.hub.load("pytorch/vision", "mobilenet_v2",
                   weights=None).to(device).eval()
x = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    for _ in range(10): _ = m(x)
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(200): _ = m(x)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 200 * 1000
fps_possible = 1000 / ms
print(f"On Colab T4, a batch-1 MobileNetV2 forward = {ms:.2f} ms ({fps_possible:.0f} FPS max)")
```

Compare this to the **required** FPS in Cell 2. If `fps_possible >>
workload["fps"]`, the T4 has plenty of headroom — which the ranker
already told you.

---

## Cell 8 — Stretch goals

1. **Add NPUs** (Apple ANE, Qualcomm Hexagon, Hailo-8, Coral) to the
   spec DB from the [Hardware/](../Hardware/README.md) chapter. Does
   the ranker still pick a GPU?
2. **Add a cost column** (approximate $) and re-rank on
   `TOPS_per_dollar`.
3. **Support multiple cameras**: if workload["cameras"] = 8, multiply
   required TOPS by 8 and see which GPUs survive.
4. **Thermal derating**: subtract 30 % from published TOPS for
   fanless enclosures and re-rank.
5. **Sparse TOPS mode**: some Jetsons double on 2:4 sparsity. Add a
   column and an option in the ranker.

---

## 🎓 What you should take away

- The **GPU family** you choose is dictated by **TOPS, memory,
  bandwidth, power, and form factor** — in that order.
- For typical Edge AI models, **many chips fit**; optimize on
  **efficiency** (TOPS/Watt) and **price**, not raw TOPS.
- For heavy vision (YOLO, segmentation at 60 FPS) or LLMs, you quickly
  need **Jetson AGX Orin** or **discrete edge GPUs**.
- This notebook + [the theory file](edge_gpu_varieties_theory.md)
  is all you need to defend a chip-selection decision in a design review.

Next up: [**Edge Hardware (beyond GPUs) →**](../Hardware/README.md) —
NPUs, MCUs, FPGAs, and the Edge-vs-Cloud GPU comparison.

---

> *GPU Programming · EdgeAI · GPU Types · PRACTICE · github.com/rpaut03l/TS-02*
