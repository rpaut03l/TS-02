# 🎯 MLOps Lec 2 — ML Lifecycle: PRACTICE

### *A mini end-to-end lifecycle in Kaggle / Google Colab*

> **Nav:** [← Lec 2 README](README.md) | [📖 THEORY](mlops_lec02_lifecycle_theory.md) | [💻 CODE](mlops_lec02_lifecycle_code.md) | **PRACTICE**

---

## 🏗️ How to use this file

1. Open **[Kaggle Notebooks](https://www.kaggle.com/code)** or **[Google Colab](https://colab.research.google.com)**.
2. Create a new Python notebook.
3. For each cell block below, **paste into a fresh cell** and run.
4. By the end you will have gone through **all 3 phases** of the lifecycle on one real-ish dataset.

### Dataset
We'll use **California Housing** (regression) — built in to sklearn, no download, no Kaggle competition needed. Small enough to fit comfortably on the free tier, big enough to feel real (20,640 rows, 8 features).

---

## 🎬 Phase 1 — Model Development

### Cell 1 — Setup
```python
# Colab / Kaggle already have these; uncomment on a fresh machine
# !pip install scikit-learn pandas numpy matplotlib joblib

import time, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("✅ Imports ready")
```

---

### Cell 2 — Data Collection
```python
raw = fetch_california_housing(as_frame=True)
df = raw.frame
print("Shape:", df.shape)
df.head()
```

### 👶 What you're seeing
- **MedInc** = median income in the block
- **HouseAge** = median house age
- **AveRooms** = average rooms per household
- **AveBedrms** = average bedrooms per household
- **Population** = block population
- **AveOccup** = average occupancy
- **Latitude / Longitude** = coordinates
- **MedHouseVal** = median house price (the target we'll predict, in 100k USD)

---

### Cell 3 — Cleaning & Visualization (EDA)
```python
# Quick stats
df.describe().T[["mean", "std", "min", "max"]]
```
```python
# Look for missing
print("Missing values per column:")
print(df.isna().sum())
```
```python
# Distribution of the target
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df["MedHouseVal"], bins=50, color="steelblue")
ax[0].set_title("Target: MedHouseVal"); ax[0].set_xlabel("price (100k USD)")
ax[1].hist(np.log1p(df["MedHouseVal"]), bins=50, color="darkorange")
ax[1].set_title("log(1+target)")
plt.tight_layout(); plt.show()
```

### 👶 What EDA is telling us
- The target is skewed — many cheap houses, few expensive ones.
- `AveRooms`, `AveBedrms`, `AveOccup` have HUGE maxes (50+ rooms? probably outliers).
- No missing values → that's one less thing to fix.

---

### Cell 4 — Feature Engineering
```python
# Cross features / derived features
df["rooms_per_person"] = df["AveRooms"] / df["AveOccup"]
df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"]
df["log_population"] = np.log1p(df["Population"])

# High-end cross feature: rich area AND low occupancy (luxury indicator)
df["luxury_cross"] = (
    (df["MedInc"] > df["MedInc"].quantile(0.75)) &
    (df["AveOccup"] < df["AveOccup"].quantile(0.25))
).astype(int)

# How much does luxury_cross separate the target?
print(df.groupby("luxury_cross")["MedHouseVal"].mean().round(3))
```

### 👶 What this is
Four brand-new features, all computed from existing columns:
- `rooms_per_person` — how roomy is each person's space
- `bedrooms_per_room` — efficiency of room layout
- `log_population` — compresses the skew
- `luxury_cross` — boolean indicator of "rich + low-occupancy" area

The cross feature separates the target cleanly — confirm with the `groupby` output.

---

### Cell 5 — Train / Valid Split + a baseline model
```python
feature_cols = [c for c in df.columns if c != "MedHouseVal"]
X = df[feature_cols].values
y = df["MedHouseVal"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
baseline.fit(X_tr, y_tr)

pred = baseline.predict(X_te)
print(f"Baseline Ridge  —  MSE: {mean_squared_error(y_te, pred):.4f}")
print(f"Baseline Ridge  —  R² : {r2_score(y_te, pred):.4f}")
```

### 👶 What this is
A **baseline**. It's the "simplest thing that could work" — scale features, run ridge regression. If your fancy model can't beat this, something's wrong.

---

### Cell 6 — Upgrade to a Gradient Boosting model
```python
gbm = Pipeline([
    ("scale", StandardScaler()),     # gbm doesn't strictly need it, but pipelines are tidy
    ("model", GradientBoostingRegressor(random_state=42)),
])
gbm.fit(X_tr, y_tr)
pred_gbm = gbm.predict(X_te)

print(f"GBM  —  MSE: {mean_squared_error(y_te, pred_gbm):.4f}")
print(f"GBM  —  R² : {r2_score(y_te, pred_gbm):.4f}")
```

Compare to the baseline. Gradient boosting should be meaningfully better.

---

### Cell 7 — Hyperparameter tuning (Grid vs Random)
```python
# GRID — 3 learning rates × 3 depths × 3 n_estimators = 27 combos
param_grid = {
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_depth":     [3, 5, 7],
    "model__n_estimators":  [100, 200, 300],
}
t0 = time.perf_counter()
grid = GridSearchCV(gbm, param_grid, cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_tr, y_tr)
t_grid = time.perf_counter() - t0
print(f"GRID  — best R²: {grid.best_score_:.4f}   time: {t_grid:.1f}s")
print(f"       best params: {grid.best_params_}")
```
```python
# RANDOM — 20 trials on a wider space
from scipy.stats import loguniform, randint
param_dist = {
    "model__learning_rate": loguniform(0.01, 0.5),
    "model__max_depth":     randint(2, 10),
    "model__n_estimators":  randint(50, 500),
}
t0 = time.perf_counter()
rand = RandomizedSearchCV(
    gbm, param_dist, n_iter=20, cv=3, scoring="r2",
    random_state=0, n_jobs=-1,
)
rand.fit(X_tr, y_tr)
t_rand = time.perf_counter() - t0
print(f"RANDOM — best R²: {rand.best_score_:.4f}   time: {t_rand:.1f}s")
print(f"        best params: {rand.best_params_}")
```

### 👶 Which won?
Compare **best R²** and **time**. Random search often matches grid with fewer trials. Save both best models for the next step.

---

## 🎬 Phase 2 — Training Pipeline (CODE, not a binary)

### Cell 8 — Wrap all of Phase 1 in a pipeline function
```python
def training_pipeline(df, model_dir="/tmp"):
    """
    Full pipeline — re-run this whenever new data arrives.
    Returns a dict with the trained pipe, metrics, version, and save path.
    """
    # Feature eng
    df = df.copy()
    df["rooms_per_person"] = df["AveRooms"] / df["AveOccup"]
    df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"]
    df["log_population"] = np.log1p(df["Population"])
    df["luxury_cross"] = (
        (df["MedInc"] > df["MedInc"].quantile(0.75)) &
        (df["AveOccup"] < df["AveOccup"].quantile(0.25))
    ).astype(int)

    feature_cols = [c for c in df.columns if c != "MedHouseVal"]
    X, y = df[feature_cols].values, df["MedHouseVal"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", GradientBoostingRegressor(
            learning_rate=0.1, max_depth=5, n_estimators=300, random_state=42
        )),
    ])
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    mse = mean_squared_error(y_te, pred)
    r2  = r2_score(y_te, pred)

    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"{model_dir}/housing_model_{version}.joblib"
    joblib.dump(
        {"pipe": pipe, "mse": mse, "r2": r2, "version": version,
         "feature_cols": feature_cols},
        path,
    )
    print(f"Trained model v{version}  —  MSE {mse:.4f}  R² {r2:.4f}  →  {path}")
    return {"pipe": pipe, "mse": mse, "r2": r2, "version": version, "path": path}

result = training_pipeline(df)
```

---

### Cell 9 — Re-run the pipeline (= "retraining on new data")
Simulate a day later: the world looks slightly different, so we re-run the same function on a new snapshot of the data.

```python
# Pretend new data: same structure, different random subset
rng = np.random.default_rng(1)
new_df = df.sample(frac=0.9, random_state=1).reset_index(drop=True)

result2 = training_pipeline(new_df)
print(f"\nOld model R²:  {result['r2']:.4f}")
print(f"New model R²:  {result2['r2']:.4f}")
```

### 👶 Lesson
The pipeline is a **function**, not a static file. "Retraining" = calling the function again with new data. The new model is automatically versioned so you never lose the old one — **rollback** is just "load the previous path."

---

## 🎬 Phase 3 — Inference

### Cell 10 — Load a trained model and serve predictions
```python
# Load whichever version you want to serve
loaded = joblib.load(result["path"])
pipe_live = loaded["pipe"]
print(f"Serving model version: {loaded['version']}")
print(f"Its offline R² was:    {loaded['r2']:.4f}")

# Build a single "request"
sample = df.iloc[[0]][loaded["feature_cols"]].values
print(f"Prediction: {pipe_live.predict(sample)[0]:.3f} (100k USD)")
print(f"Actual:     {df.iloc[0]['MedHouseVal']:.3f}")
```

---

### Cell 11 — Latency under load
```python
sample = df.iloc[[0]][loaded["feature_cols"]].values

# Warmup
for _ in range(50):
    pipe_live.predict(sample)

N = 2000
t0 = time.perf_counter()
for _ in range(N):
    pipe_live.predict(sample)
t1 = time.perf_counter()
per_call_ms = (t1 - t0) / N * 1000

latencies = []
for _ in range(N):
    t0 = time.perf_counter()
    pipe_live.predict(sample)
    latencies.append((time.perf_counter() - t0) * 1000)

print(f"Mean per-call latency: {per_call_ms:.3f} ms")
print(f"p50: {np.percentile(latencies, 50):.3f} ms")
print(f"p95: {np.percentile(latencies, 95):.3f} ms")
print(f"p99: {np.percentile(latencies, 99):.3f} ms")
print(f"Within 10 ms budget?  {'✅ YES' if np.percentile(latencies, 99) < 10 else '❌ NO'}")
```

### 👶 Why p99 matters in production
Your users don't care about the average. They care about the **worst** experience. If p99 is 50 ms and your SLA is 10 ms, **1 in 100 requests fails** — that's a lot when you serve 10,000 req/s.

---

### Cell 12 — Composition — two-model inference
Let's chain two models: one that predicts house price, and a second that predicts "is this house expensive?" from the first model's output.

```python
from sklearn.linear_model import LogisticRegression

# Sub-model 1: the price regressor we already trained (pipe_live)
# Sub-model 2: classifier that maps predicted price → "expensive?" (0/1)

# Train sub-model 2 on Phase 1's predictions
X_tr_preds = pipe_live.predict(df[loaded["feature_cols"]].values).reshape(-1, 1)
y_expensive = (df["MedHouseVal"] > 2.5).astype(int).values  # threshold at 250k

classifier = LogisticRegression().fit(X_tr_preds, y_expensive)

def composed_predict(features_vec):
    price_pred = pipe_live.predict([features_vec])[0]
    is_expensive = classifier.predict([[price_pred]])[0]
    return {"price_100k": round(float(price_pred), 3),
            "expensive": bool(is_expensive)}

for i in [0, 100, 1000]:
    x = df.iloc[i][loaded["feature_cols"]].values
    print(f"Sample {i}:", composed_predict(x), f"(actual: {df.iloc[i]['MedHouseVal']:.3f})")
```

### 👶 Lesson
Same composition idea as the cuteness detector. Each sub-model can be tested on its own — **but what matters is the end-to-end accuracy**. Errors in the first model become inputs to the second, and errors compound.

---

### Cell 13 — Monitoring: watch accuracy over fake "days"
Simulate a production deployment where predictions come in daily. We'll fake 7 days of drift and watch the loss climb.

```python
rng = np.random.default_rng(0)
fake_days = []
for day in range(7):
    # Each day, distribution shifts a tiny bit
    day_df = df.sample(200, random_state=day).reset_index(drop=True)
    day_df["MedInc"] = day_df["MedInc"] + day * 0.05   # incomes inflate slightly
    pred = pipe_live.predict(day_df[loaded["feature_cols"]].values)
    mse_day = mean_squared_error(day_df["MedHouseVal"], pred)
    fake_days.append((day, mse_day))
    print(f"Day {day}:  MSE {mse_day:.4f}")

ds = pd.DataFrame(fake_days, columns=["day", "mse"])
plt.plot(ds.day, ds.mse, marker="o")
plt.xlabel("day"); plt.ylabel("MSE"); plt.title("Model health over 7 fake days")
plt.grid(alpha=0.3); plt.show()
```

### 👶 Lesson
This is a **minimal monitoring dashboard**. Every day you plot your model's error. If the line climbs too high, an alert fires and you retrain. That's the **feedback loop** that keeps models alive in production.

---

## 🏋️ Challenge Exercises

### Challenge 1 — Beat the baseline by a bigger margin
Try a `RandomForestRegressor` or a stacked model. Can you beat the GBM's R²?

### Challenge 2 — Add feature importance logging to `training_pipeline`
Gradient Boosting exposes `model.feature_importances_`. Save the top 5 feature names + importances alongside the model in the joblib file. Next time you retrain, print a "what changed?" diff.

### Challenge 3 — Batch inference mode
Write a `predict_batch(df_batch)` function that takes a whole DataFrame of new houses and returns a DataFrame with predictions. Measure how much faster per sample this is compared to one-at-a-time.

### Challenge 4 — Simulate composition error
Introduce a 10% random error to the price regressor's output BEFORE it feeds the expensive classifier. Measure how the end-to-end accuracy degrades. This is what **silent sub-model failure** looks like in real composed pipelines.

### Challenge 5 — Build a drift detector
Write `data_drifted(old_df, new_df, threshold=0.3)` that returns True if the mean of any feature in `new_df` differs from `old_df` by more than `threshold` standard deviations. Use Cell 13's simulated days to verify it fires on day 5 or 6.

### Challenge 6 — Version comparison
Train two versions of the pipeline — one with `max_depth=3`, one with `max_depth=7`. Save both. Then write a function `compare_versions(pathA, pathB)` that loads both, runs them on the test set, and prints which one wins by which margin. **This is how teams decide whether to promote a new model to production.**

---

## 📝 Wrap-up — reflect

Before closing the notebook, answer these in a markdown cell:

1. Which cell took the longest to run? Was it a **training** step or an **inference** step? What does that tell you about where to optimize for **offline throughput** vs **online latency**?
2. Which hyperparameter made the biggest difference in R² — `learning_rate`, `max_depth`, or `n_estimators`?
3. If you were deploying this to a real estate website, which of the 13 cells would need the most extra work before going to production? **Justify.** (Answer: probably Cell 12 and 13 — latency + composition + monitoring — the things the theory file warned about.)

---

> **Next:** [← Lec 2 THEORY](mlops_lec02_lifecycle_theory.md) · [← Lec 1](../Lec_01_Introduction/README.md) · [← MLOps](../README.md)
>
> *MLOps · Lec 2 · github.com/rpaut03l/TS-02*
