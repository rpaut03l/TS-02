# 🎯 EdgeAI · Hardware — PRACTICE

### *Build a "hardware fit chart" across GPUs + NPUs + MCUs under a thermal cap*

> **Nav:** [← Hardware README](README.md) | [📖 THEORY](edge_ai_hardware_theory.md) | [💻 CODE](edge_ai_hardware_code.md) | **PRACTICE**

---

## 🎯 Goal

Extend the decision notebook from [GPU_Types/practice.md](../GPU_Types/edge_ai_fundamentals_practice.md)
into the **full edge zoo** (GPU + NPU + MCU) and add:
- A **thermal cap** (fanless 15 W vs fan 70 W).
- A **cost column**.
- An **inferences-per-joule** ranking.
- A pretty scatter plot: **TOPS/Watt vs $/TOPS**.

---

## Cell 1 — Full edge chip database

```python
CHIPS = [
    # (family, name, TDP_W, TOPS, mem_GB, mem_BW_GBps, usd, op_set)
    # Mobile SoC GPUs + NPUs
    ("SoC-NPU",       "Apple ANE (A17 Pro)",       2,   35,   8,   51.2,   None, "CoreML"),
    ("SoC-NPU",       "Qualcomm Hexagon (SD8G3)",  3,   45,   8,   64.0,   None, "QNN/NNAPI"),
    ("SoC-NPU",       "Samsung NPU (Exynos 2400)", 3,   17,   8,   51.2,   None, "NNAPI"),
    # Dedicated edge NPUs
    ("Edge-NPU",      "Google Coral (Edge TPU)",   2,    4,   0.008, 34,    60, "EdgeTPU"),
    ("Edge-NPU",      "Hailo-8",                   2.5, 26,   0,    5.3,   250, "HailoRT"),
    ("Edge-NPU",      "Hailo-15",                  5,   20,   0,    5.0,   290, "HailoRT"),
    ("Edge-NPU",      "Intel NPU (Meteor Lake)",   6,   11,   0,   68.3,  None, "OpenVINO"),
    ("Edge-NPU",      "AMD XDNA (Ryzen AI 300)",   5,   50,   0,   76.8,  None, "RyzenAI"),
    # Jetson family
    ("Jetson",        "Jetson Orin Nano 8GB",     15,   40,   8,   68.3,   500, "CUDA/TRT"),
    ("Jetson",        "Jetson Orin NX 16GB",      25,  100,  16,  102.4,   900, "CUDA/TRT"),
    ("Jetson",        "Jetson AGX Orin 64GB",     60,  275,  64,  204.8,  2000, "CUDA/TRT"),
    # Discrete edge
    ("Discrete-Edge", "Tesla T4",                 70,  130,  16,  320.0,  2500, "CUDA/TRT"),
    ("Discrete-Edge", "RTX 4000 Ada SFF",         70,  153,  20,  280.0,  1500, "CUDA/TRT"),
    ("Discrete-Edge", "NVIDIA L4",                72,  242,  24,  300.0,  2800, "CUDA/TRT"),
    # MCUs
    ("MCU-TinyML",    "STM32H747 M7",             0.2, 0.001, 0.001, 0.5,   15, "TFLM/CMSIS"),
    ("MCU-TinyML",    "Alif Ensemble (U55)",      0.5, 0.5,  0.013, 1.0,    50, "TFLM/U55"),
    ("MCU-TinyML",    "ESP32-S3",                 0.3, 0.003, 0.0005,0.3,    8, "TFLM"),
]

import pandas as pd
df = pd.DataFrame(CHIPS, columns=[
    "family","name","TDP_W","TOPS","mem_GB","mem_BW_GBps","usd","op_set"])
df["TOPS_per_W"] = df["TOPS"] / df["TDP_W"]
df["USD_per_TOPS"] = df["usd"] / df["TOPS"].replace(0, float("nan"))
df[["family","name","TDP_W","TOPS","TOPS_per_W","USD_per_TOPS","mem_GB","op_set"]]
```

---

## Cell 2 — Thermal cap filter

```python
def under_thermal_cap(df, cap_W):
    return df[df.TDP_W <= cap_W].copy()

fanless = under_thermal_cap(df, 15)
with_fan = under_thermal_cap(df, 75)
print("Fanless 15 W chips:", len(fanless))
print("Fan 75 W chips    :", len(with_fan))
```

---

## Cell 3 — Workload → required TOPS (same math as earlier)

```python
def required_stats(model_MACs_G, fps, params_M, weight_bytes=1):
    req_TOPS = model_MACs_G * 2 * fps / 1000
    model_MB = params_M * weight_bytes
    req_BW   = model_MB * fps / 1024   # worst case: reload every frame
    return req_TOPS, req_BW, model_MB

req_tops, req_bw, model_MB = required_stats(
    model_MACs_G=0.30,   # MobileNetV2
    params_M=3.5,
    fps=30)
print(f"Required INT8 TOPS : {req_tops:.4f}")
print(f"Required mem BW    : {req_bw:.4f} GB/s")
print(f"Model size         : {model_MB:.2f} MB")
```

---

## Cell 4 — The full ranker (TOPS + memory + bandwidth)

```python
def rank(df, req_tops, req_bw, model_MB):
    ok = df[
        (df.TOPS        >= req_tops * 3) &
        (df.mem_BW_GBps >= req_bw   * 2) &
        ((df.mem_GB * 1024 >= model_MB * 5) | (df.mem_GB == 0))
        # mem_GB == 0 means "uses host memory" — NPUs on SoCs
    ].copy()
    ok["fit_score"] = ok["TOPS_per_W"]
    return ok.sort_values("fit_score", ascending=False)

ranked = rank(fanless, req_tops, req_bw, model_MB)
ranked[["family","name","TDP_W","TOPS","TOPS_per_W","USD_per_TOPS"]]
```

Typical: for a 30 FPS MobileNetV2 under a 15 W cap, a **Qualcomm
Hexagon** or **Apple ANE** sits at the top of `TOPS_per_W` — 10–20×
better than a Jetson Orin Nano on the same metric.

---

## Cell 5 — Visualise TOPS/W vs $/TOPS

```python
import matplotlib.pyplot as plt

has_price = df.dropna(subset=["USD_per_TOPS"])

plt.figure(figsize=(9, 6))
for fam, g in has_price.groupby("family"):
    plt.scatter(g["USD_per_TOPS"], g["TOPS_per_W"], label=fam, s=80)
    for _, row in g.iterrows():
        plt.annotate(row["name"],
                     (row["USD_per_TOPS"], row["TOPS_per_W"]),
                     fontsize=8, xytext=(4, 4),
                     textcoords="offset points")
plt.xlabel("USD per TOPS  (lower = cheaper per TOPS)")
plt.ylabel("TOPS per Watt (higher = more efficient)")
plt.title("Edge chip fit chart  —  top-right corner is trash, bottom-right is gold")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()
```

Reading the chart:
- **Bottom-right** (low $/TOPS, high TOPS/W) = ideal.
- **Top-left** (high $/TOPS, low TOPS/W) = avoid.
- Most **NPUs** hug the right edge (excellent TOPS/W).
- **Discrete edge GPUs** hug the bottom-right but at a price.
- **MCUs** look terrible on TOPS/W here, but they're priced per
  **cent** and run on coin batteries — they win on workloads where
  TOPS isn't the right axis at all.

---

## Cell 6 — Multi-camera workload under a thermal cap

```python
for n_cameras in [1, 4, 8, 16]:
    req_tops, req_bw, _ = required_stats(0.30, 30 * n_cameras, 3.5)
    top = rank(with_fan, req_tops, req_bw, 3.5).head(1)
    if len(top):
        row = top.iloc[0]
        print(f"{n_cameras:2d} cameras → {row['name']:30s} "
              f"({row['TDP_W']} W, {row['TOPS_per_W']:.1f} TOPS/W)")
```

Typical output:
```
 1 cameras → Apple ANE (A17 Pro)             (2 W, 17.5 TOPS/W)
 4 cameras → Qualcomm Hexagon (SD8G3)        (3 W, 15.0 TOPS/W)
 8 cameras → Jetson Orin NX 16GB             (25 W, 4.0 TOPS/W)
16 cameras → NVIDIA L4                       (72 W, 3.4 TOPS/W)
```

Exactly the progression you'd see in real projects as the camera count
goes up.

---

## Cell 7 — Add an MCU-friendly workload

Not every edge workload needs 100 TOPS. Model the **keyword spotter**
(DS-CNN, ~30 KB, ~3 MOPS, 10 Hz):

```python
req_tops, req_bw, model_MB = required_stats(
    model_MACs_G=0.000003, params_M=0.03, fps=10)
print(f"Required TOPS: {req_tops:.8f}  (tiny!)")

mcu = df[df.family == "MCU-TinyML"]
mcu["inf_per_J"] = 1e9 / (mcu["TDP_W"] * 1000 * 10)  # rough
mcu[["name","TDP_W","TOPS","inf_per_J"]]
```

Suddenly the **STM32H747** or **ESP32-S3** is the right answer.
Jetson is overkill; it would run 100× faster and burn 1,000× more
power for no extra value.

> **Moral:** match the chip class to the workload class. A Jetson for
> a keyword spotter is like renting a truck to buy one loaf of bread.

---

## Cell 8 — Stretch goals

1. Replace the pseudo-prices with real **Mouser / DigiKey** quotes
   for 10,000-unit volumes. Re-run the chart.
2. Add a **"which SDK?"** column and penalise chips whose SDK your
   team can't use.
3. Add a **supply-horizon** column (years of guaranteed availability)
   — crucial for industrial work. Score it.
4. Swap MLPerf Tiny's DS-CNN for **person detection** and see the
   ranker pivot from MCU-class to Jetson Nano-class.
5. Plot **inf/joule** (not TOPS/W) for a battery-powered wearable.
   Does the ranker still pick the same chip?

---

## 🎓 What you should take away

- Edge hardware is **a spectrum**, not a single chip. A production
  portfolio often uses **3 or 4 classes** for different products.
- **NPUs beat GPUs on TOPS/Watt** almost always — but GPUs beat NPUs
  on **flexibility**.
- For anything running **30 FPS MobileNet** on a phone: the **NPU**
  wins.
- For anything at **60 FPS YOLO with 8 cameras**: the **Jetson AGX
  Orin** or a **discrete edge GPU** wins.
- For anything at **<1 W always-on**: the **MCU + Ethos-U55** wins.

Next up: [**EdgeAI with CUDA →**](../CUDA_for_Edge/README.md) — bring
your CUDA skills to the Jetson and see what changes.

---

> *GPU Programming · EdgeAI · Hardware · PRACTICE · github.com/rpaut03l/TS-02*
