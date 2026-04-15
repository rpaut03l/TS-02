# 💻 MLOps Lec 1 — Introduction: CODE

### *Minimal, runnable sklearn code that makes each concept concrete*

> **Nav:** [← Lec 1 README](README.md) | [📖 THEORY](mlops_lec01_intro_theory.md) | **CODE** | [🎯 PRACTICE →](mlops_lec01_intro_practice.md)

---

## 🏗️ Setup — one-time install

```bash
pip install scikit-learn pandas numpy matplotlib joblib
```

That's literally all you need for Lecture 1.

---

## Ex 1. Train a baby model (the "easy" part of ML)

### 👶 What we're doing
We'll build a tiny classifier on the classic Iris dataset. This is the *entire* ML part of most jobs — maybe 5 minutes of actual work.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1) Load data — flowers with 4 measurements each
X, y = load_iris(return_X_y=True)

# 2) Split into train / test so we can check how good our model is
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

# 3) Train a Random Forest — this is "the ML"
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_tr, y_tr)

# 4) Check accuracy
pred = model.predict(X_te)
print(f"Test accuracy: {accuracy_score(y_te, pred):.3f}")
```

**Output:** `Test accuracy: 0.974` or similar.

**Point:** that's it. 10 lines of code and we're done with the "ML" part. The rest of MLOps is everything that comes next.

---

## Ex 2. Save and load the model (the dumbest possible "deployment")

### 👶 What we're doing
Your model lives in memory right now. If Python closes, it's gone. Let's save it to disk so we can use it later — like putting your homework in your backpack instead of leaving it on your desk.

```python
import joblib

# Save
joblib.dump(model, "iris_model.joblib")
print("Model saved!")

# Later (maybe in a different process) — load and predict
loaded = joblib.load("iris_model.joblib")
print("Loaded model prediction on one flower:", loaded.predict([[5.1, 3.5, 1.4, 0.2]]))
```

**Output:** `Loaded model prediction on one flower: [0]` (= "setosa")

> ⚠️ **This is "deployment is easy."** But is it **reliable**? No. What if the file gets corrupted? What if sklearn's version changes? What if someone uploads a file with 3 features instead of 4? All of that is Myth #1 in action — **reliable** deployment is the hard part.

---

## Ex 3. Measure latency (why inference speed matters in production)

### 👶 What we're doing
How long does one prediction take? If it's 100ms per flower, and we get 10,000 requests per second, we have a problem. Let's find out.

```python
import time
import numpy as np

# Pretend we have a batch of 1 flower — like a single request to a web API
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Warm up (first call is always slower due to caching)
for _ in range(10):
    model.predict(sample)

# Measure 1000 single-prediction calls
N = 1000
t0 = time.perf_counter()
for _ in range(N):
    model.predict(sample)
t1 = time.perf_counter()

per_call_ms = (t1 - t0) / N * 1000
qps = 1000 / per_call_ms  # queries per second
print(f"Per-prediction latency: {per_call_ms:.3f} ms")
print(f"Throughput (1 at a time): ~{qps:,.0f} QPS")
```

**Output (typical laptop):**
```
Per-prediction latency: 0.8 ms
Throughput (1 at a time): ~1,250 QPS
```

### 👶 What the numbers mean
- **Latency** = how long *one* request takes. 0.8 ms is very fast.
- **Throughput** = how many requests per second we can handle. 1,250 QPS is enough for a small app.
- For reference: Lec 2 says a prediction service should respond in **~10 ms**. We have a **12× budget** — we can afford to load features, do logging, call the model, and still stay within the SLA.

---

## Ex 4. Batch prediction is much faster than one-at-a-time

### 👶 What we're doing
If the bakery gets 100 orders, one baker handles them all in one big batch way faster than 100 bakers doing one each. Models are the same.

```python
# 1000 samples, all at once
batch = np.tile(sample, (1000, 1))        # shape (1000, 4)

t0 = time.perf_counter()
model.predict(batch)
t1 = time.perf_counter()

print(f"Batch of 1000 took: {(t1 - t0)*1000:.3f} ms total")
print(f"Per-prediction (amortized): {(t1 - t0) * 1000 / 1000:.4f} ms")
```

**Typical output:**
```
Batch of 1000 took: 4.0 ms total
Per-prediction (amortized): 0.0040 ms
```

> 💡 **Per-prediction time dropped 200×.** This is why **batch prediction** is the mode of choice when you don't need low latency — it amortizes the overhead across many examples. Online prediction is harder because you can't batch easily.

---

## Ex 5. The "data tangled with code" problem

### 👶 What we're doing
In traditional software, code is code and data is data. In ML, a change in the **data** changes what the **model** does without changing any code. Let's prove it.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

rng = np.random.default_rng(0)
X, y = load_iris(return_X_y=True)

# Version A — clean data
model_A = RandomForestClassifier(random_state=0).fit(X, y)
pred_A = model_A.predict([[5.1, 3.5, 1.4, 0.2]])

# Version B — SAME CODE, but 10% of labels flipped to random classes
y_dirty = y.copy()
flip_idx = rng.choice(len(y), size=int(0.10 * len(y)), replace=False)
y_dirty[flip_idx] = rng.integers(0, 3, size=flip_idx.shape)

model_B = RandomForestClassifier(random_state=0).fit(X, y_dirty)
pred_B = model_B.predict([[5.1, 3.5, 1.4, 0.2]])

print(f"Clean data prediction:  {pred_A}")
print(f"Dirty data prediction:  {pred_B}")
print("Code is IDENTICAL. Only the data changed.")
```

### 👶 Lesson
Same code. Same random seed. **Different behaviour** — because the *data* changed. In traditional software, only code can change behaviour. In ML, data is a hidden input that affects every prediction. **This is why MLOps needs to version data and models, not just code.**

---

## Ex 6. Simulate data drift (Adaptability — the "A" in RSMA)

### 👶 What we're doing
Cookie tastes change over time. The robot trained in 2020 still thinks chocolate-chip is most popular, but in 2026 everyone wants matcha. The world shifts; the model doesn't know yet. Let's *watch* this happen.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

rng = np.random.default_rng(42)

# "2020 world" — train data
X_2020, y_2020 = make_classification(
    n_samples=500, n_features=4, n_informative=3,
    n_redundant=0, random_state=1, flip_y=0.02
)
model = RandomForestClassifier(random_state=0).fit(X_2020, y_2020)

# "2026 world" — same structure but feature distribution has drifted
X_2026, y_2026 = make_classification(
    n_samples=500, n_features=4, n_informative=3,
    n_redundant=0, random_state=2, flip_y=0.02
)
X_2026 += 2.0  # shift all features by 2 units — "the world changed"

acc_2020 = accuracy_score(y_2020, model.predict(X_2020))
acc_2026 = accuracy_score(y_2026, model.predict(X_2026))

print(f"Accuracy on 2020 data (what it was trained on): {acc_2020:.3f}")
print(f"Accuracy on 2026 data (shifted):                {acc_2026:.3f}")
```

**Typical output:**
```
Accuracy on 2020 data: 0.954
Accuracy on 2026 data: 0.700
```

### 👶 Lesson
Accuracy dropped by ~25 points **without anyone touching the code**. The model didn't change. The world changed. This is **data drift** — it's why the **A (Adaptable)** in RSMA is non-negotiable in production. Without retraining, every model gets worse over time.

---

## Ex 7. A tiny "deployment as a function" (what MLOps would dress up)

### 👶 What we're doing
Let's make a predict function that takes a dictionary of features and returns a class name. This is the *interface* to the model — the contract between your code and whoever's calling it.

```python
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

def predict_flower(payload: dict) -> dict:
    """
    Tiny 'API' around our model.
    In production this would be an HTTP endpoint (FastAPI, Flask, etc.)
    """
    # 1) Validate input — real deployment would reject bad requests here
    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [k for k in required if k not in payload]
    if missing:
        return {"error": f"missing fields: {missing}"}

    # 2) Build the feature vector in the RIGHT ORDER
    x = [[payload[k] for k in required]]

    # 3) Predict
    cls_idx = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0].tolist()

    return {
        "class": CLASS_NAMES[cls_idx],
        "class_idx": cls_idx,
        "probabilities": dict(zip(CLASS_NAMES, [round(p, 4) for p in proba])),
    }

# Try it
print(predict_flower({
    "sepal_length": 5.1, "sepal_width": 3.5,
    "petal_length": 1.4, "petal_width": 0.2,
}))

# Try with a missing field
print(predict_flower({"sepal_length": 5.1}))
```

**Output:**
```
{'class': 'setosa', 'class_idx': 0, 'probabilities': {'setosa': 0.99, 'versicolor': 0.01, 'virginica': 0.0}}
{'error': "missing fields: ['sepal_width', 'petal_length', 'petal_width']"}
```

### 👶 What a "production" version of this would add
- ✅ HTTP wrapper (FastAPI) so it's reachable over the network
- ✅ Authentication so not anyone can call it
- ✅ Rate limiting so one user can't DDoS you
- ✅ Logging (every request logged for debugging and retraining)
- ✅ Monitoring (alert if latency > 100ms or accuracy drops)
- ✅ Versioning (which model is serving? v3 or v4?)
- ✅ Rollback (if v4 is bad, snap back to v3 in seconds)
- ✅ Feature validation (types, ranges, missing values)
- ✅ Schema versioning (what if we add a 5th feature tomorrow?)

That's the "**easy to deploy, hard to deploy reliably**" gap — **nine checkmarks** that the 15-line function above does not have.

---

## Ex 8. Count your dependencies (Reproducibility & Myth #1)

### 👶 What we're doing
"It works on my laptop" is famous last words. Let's capture *exactly* what versions your model needs, so your teammate can rebuild it on their machine.

```python
import sys, platform, sklearn, numpy

print(f"Python:       {sys.version.split()[0]}")
print(f"Platform:     {platform.platform()}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"numpy:        {numpy.__version__}")
```

```bash
# In a real project you'd freeze your environment
pip freeze > requirements.txt
```

> ⚠️ **If you don't pin these**, you'll train a model with sklearn 1.3, deploy with sklearn 1.5, and find out later that a method renamed or a default changed — and now your predictions are subtly different. That's a classic MLOps bug: the "bug" is in the environment, not the code.

---

## 🧭 What did we learn from this file?

1. **Training a baby model is fast and fun.** (Ex 1)
2. **"Saving and loading" is not the same as "deploying."** (Ex 2)
3. **Latency matters** — especially for online prediction. (Ex 3)
4. **Batch >> one-at-a-time** for throughput. (Ex 4)
5. **Code + data are tangled.** Same code can give very different results when data changes. (Ex 5)
6. **Data drift is real.** Models rot without retraining. (Ex 6)
7. **Real deployment needs 10× more glue code** than "just load and predict." (Ex 7)
8. **Reproducibility starts with pinning versions.** (Ex 8)

These 8 tiny examples touch every single concept from the theory file. Next up: the practice file takes what you've seen here and turns it into a real notebook you can open in Kaggle or Colab.

---

> **Next:** [🎯 PRACTICE →](mlops_lec01_intro_practice.md) · [← 📖 THEORY](mlops_lec01_intro_theory.md)
>
> *MLOps · Lec 1 · github.com/rpaut03l/TS-02*
