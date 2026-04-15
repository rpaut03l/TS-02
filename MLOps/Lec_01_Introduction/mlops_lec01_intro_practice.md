# 🎯 MLOps Lec 1 — Introduction: PRACTICE

### *Kaggle / Google Colab notebook-as-markdown · paste one cell at a time*

> **Nav:** [← Lec 1 README](README.md) | [📖 THEORY](mlops_lec01_intro_theory.md) | [💻 CODE](mlops_lec01_intro_code.md) | **PRACTICE**

---

## 🏗️ How to use this file

1. Open **[Kaggle Notebooks](https://www.kaggle.com/code)** or **[Google Colab](https://colab.research.google.com)**.
2. Create a new Python notebook.
3. For each section below, **paste the code into a fresh cell** and run it.
4. Read the 👶 "what this does" bits between cells.
5. At the end, try the **Challenge Exercises**.

Everything here runs on the **free tier** of both platforms — no GPU, no paid features.

---

## Cell 1 — Install & imports

### 👶 What this does
We install (if needed) and import the few libraries we'll use. Both Kaggle and Colab already have sklearn, numpy, pandas, matplotlib pre-installed, so this is a no-op there. On your laptop you might need the `pip install` line.

```python
# Uncomment on a fresh machine:
# !pip install scikit-learn pandas numpy matplotlib joblib

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("✅ Imports ready")
```

---

## Cell 2 — The "research" model: train, evaluate, celebrate

### 👶 What this does
This is the version of ML you do when you don't care about production. Load data, fit a model, print a score, feel smart. We'll be done in 30 seconds.

```python
# Load
X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names
class_names = load_iris().target_names

# Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_tr, y_tr)

# Evaluate
pred = model.predict(X_te)
print(f"Test accuracy: {accuracy_score(y_te, pred):.4f}")
print("\nClassification report:")
print(classification_report(y_te, pred, target_names=class_names))
```

---

## Cell 3 — The "production" question: latency under load

### 👶 What this does
Now we pretend this model is behind a web API and a customer is hitting it. How many predictions per second can we serve? The goal from Lec 2 slide 20 is to keep each prediction under **~10 ms**.

```python
# Simulate 10,000 single-request calls
N = 10_000
sample = X_te[:1]

# Warmup
for _ in range(100):
    model.predict(sample)

t0 = time.perf_counter()
for _ in range(N):
    model.predict(sample)
t1 = time.perf_counter()

per_call_us = (t1 - t0) / N * 1e6
qps = 1e6 / per_call_us
print(f"Per-prediction latency: {per_call_us:.1f} μs  ({per_call_us/1000:.3f} ms)")
print(f"Approximate throughput: {qps:,.0f} QPS")
print(f"Within 10ms SLA?        {'✅ YES' if per_call_us/1000 < 10 else '❌ NO'}")
```

### 👶 Why this matters
If latency were 15 ms, we would have **zero budget** left for:
- Loading features from a database
- Network round-trip
- Logging
- Post-processing

Production systems **design backwards from the SLA**: "I have 10 ms total — how much can I give to the model?"

---

## Cell 4 — Online vs batch: the throughput gap

### 👶 What this does
The same model can do 1,000 predictions one-at-a-time OR 1 prediction on a batch of 1,000 — but the *second way is much faster per sample*. Let's measure the gap.

```python
sizes = [1, 10, 100, 1000, 10_000]
results = []

for batch_size in sizes:
    batch = np.tile(sample, (batch_size, 1))
    t0 = time.perf_counter()
    _ = model.predict(batch)
    t1 = time.perf_counter()
    per_sample_us = (t1 - t0) / batch_size * 1e6
    results.append((batch_size, (t1 - t0) * 1000, per_sample_us))

df = pd.DataFrame(results, columns=["batch_size", "total_ms", "per_sample_us"])
print(df.to_string(index=False))

plt.figure(figsize=(6, 4))
plt.plot(df.batch_size, df.per_sample_us, marker="o")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Batch size"); plt.ylabel("Per-sample latency (μs)")
plt.title("Bigger batches = lower per-sample cost")
plt.grid(True, which="both", alpha=0.3); plt.show()
```

### 👶 Takeaway
**Online prediction** (batch size 1) gives low per-request latency but low throughput.
**Batch prediction** (batch size 1000+) gives amazing per-sample throughput but higher latency per request.

You pick based on what the user expects. "Show me fraud probability right now" → online. "Score all users overnight for tomorrow's email blast" → batch.

---

## Cell 5 — Myth #1 in action: write a "deployment" and look at what's missing

### 👶 What this does
We'll write a tiny function that pretends to be an HTTP endpoint. Then we'll make a checklist of everything a *real* production endpoint would need. You'll see the gap between "deploy" and "deploy reliably."

```python
def predict_endpoint(payload: dict) -> dict:
    """Mini fake HTTP endpoint."""
    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if not all(k in payload for k in required):
        return {"error": "missing fields"}
    x = [[payload[k] for k in required]]
    idx = int(model.predict(x)[0])
    return {"class": class_names[idx], "class_idx": idx}

# Good request
print(predict_endpoint({
    "sepal_length": 5.1, "sepal_width": 3.5,
    "petal_length": 1.4, "petal_width": 0.2,
}))

# Bad request
print(predict_endpoint({"sepal_length": 5.1}))
```

### 👶 Your checklist — what's missing for real production?
Tick whichever ones your function handles. For a real deployment **all** of them should be ticked:
- [ ] HTTP / gRPC interface reachable over network
- [ ] Authentication / authorization
- [ ] Rate limiting per user
- [ ] Request logging
- [ ] Prometheus-style metrics (latency p50/p95/p99, error rate)
- [ ] Input type/range validation (what if `sepal_length = "banana"`?)
- [ ] Graceful error messages (no stack traces to the user)
- [ ] Model version in the response
- [ ] Safe shutdown (finish in-flight requests)
- [ ] Rollback plan if the model misbehaves
- [ ] Automated tests (unit, integration, drift)
- [ ] Monitoring on model accuracy, not just latency

You probably ticked 0/12. Now you understand the slide **"Deploying is easy, deploying reliably is hard."**

---

## Cell 6 — Myth #2: training many models at once

### 👶 What this does
Lec 1 slide 28 says Booking.com runs 150+ models and Uber runs thousands. Let's train **five** different models on the same data and "deploy" them together. Imagine scaling this to 150.

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

zoo = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "LogisticReg":  LogisticRegression(max_iter=2000),
    "SVM":          SVC(probability=True, random_state=42),
    "KNN":          KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
}

results = {}
for name, m in zoo.items():
    m.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, m.predict(X_te))
    results[name] = acc
    joblib.dump(m, f"/tmp/model_{name}.joblib")

print("All 5 models trained & saved.")
pd.Series(results).sort_values(ascending=False).to_frame("accuracy")
```

### 👶 What happens at scale?
Now imagine the 5 models is 150 (Booking) or thousands (Uber). Each one needs:
- Its own training pipeline
- Its own monitoring
- Its own retraining schedule
- Its own rollback policy
- Its own data versioning

The "ML" part of training 150 models isn't 150× harder — but the **MLOps** part is **150× more plumbing**. That's the whole reason the field exists.

---

## Cell 7 — Myth #3: a "frozen" model gets stale

### 👶 What this does
We'll train a model, then fast-forward time by shifting the input distribution, and watch accuracy crumble without anyone touching the code. This simulates what happens if you don't retrain.

```python
rng = np.random.default_rng(0)

# "Today" — train on this
X_today, y_today = make_classification(
    n_samples=1000, n_features=10, n_informative=6,
    random_state=1, flip_y=0.02
)

model_frozen = RandomForestClassifier(n_estimators=200, random_state=0)
model_frozen.fit(X_today, y_today)

# Simulate 6 months of drift in small steps
print(f"{'drift amount':>14} | {'accuracy':>10}")
print("-" * 30)
for drift in [0.0, 0.2, 0.5, 1.0, 2.0, 3.0]:
    X_future, y_future = make_classification(
        n_samples=1000, n_features=10, n_informative=6,
        random_state=1, flip_y=0.02
    )
    X_future = X_future + drift  # each feature shifted by 'drift' units
    acc = accuracy_score(y_future, model_frozen.predict(X_future))
    print(f"{drift:>14.1f} | {acc:>10.3f}")
```

### 👶 The pattern you'll see
Accuracy starts near 1.0 and decays as drift grows. **The model is literally not learning the new world.** That's why Myth #3 ("you won't need to update your models much") is wrong — in reality you need a retraining cadence (daily, weekly, monthly, depending on how fast your data changes).

---

## Cell 8 — Research vs Production: a single picture

### 👶 What this does
We'll train the model only on the 50 easiest examples (research-style: perfect, curated, cherry-picked) and evaluate it on the real test set. Then we'll compare to training on the full, noisy set.

```python
# "Research" model: trained on only 30 clean, well-spaced samples
rng = np.random.default_rng(7)
idx = rng.choice(len(X_tr), size=30, replace=False)
model_research = RandomForestClassifier(n_estimators=200, random_state=0)
model_research.fit(X_tr[idx], y_tr[idx])
acc_research = accuracy_score(y_te, model_research.predict(X_te))

# "Production" model: trained on all, including noisy edge cases
model_prod = RandomForestClassifier(n_estimators=200, random_state=0)
model_prod.fit(X_tr, y_tr)
acc_prod = accuracy_score(y_te, model_prod.predict(X_te))

print(f"Research-style (30 cherry-picked samples):  acc = {acc_research:.3f}")
print(f"Production-style (full messy training set): acc = {acc_prod:.3f}")
```

### 👶 Lesson
Even with a fraction of the data, research accuracy can look *amazing* because the examples were curated. Production is messier and harder. **This is why research results don't always carry over to production** — the data distribution is very different.

---

## 🏋️ Challenge Exercises

### Challenge 1 — Build your own drift detector
Write a function that takes two batches of data (old and new) and returns a number between 0 and 1 indicating how much the distribution has drifted. Simplest version: compare the means of each feature.

```python
def drift_score(X_old, X_new):
    # Your code here. Bigger number = more drift.
    pass
```

### Challenge 2 — Make the fake endpoint "more production-ready"
Starting from `predict_endpoint` in Cell 5, add **at least 3** of the missing 12 checklist items. Ideas:
- Input validation with type checks
- A version string in the response (`"model_version": "iris-rf-v1"`)
- Latency measurement on each call
- A try/except with a graceful error

### Challenge 3 — A/B test two models
Train `RandomForestClassifier` and `LogisticRegression`. Randomly route **50%** of test samples to each. Compare their accuracies with a little diff. This is the simplest version of what real A/B testing infrastructure does.

### Challenge 4 — Monitor latency over time
Wrap `model.predict` in a loop that does 1000 calls and stores each latency. Plot a histogram. Find the **p50**, **p95**, **p99** latencies (the 50th, 95th, 99th percentiles). In production, p99 is often what matters — not average.

```python
latencies = []
for _ in range(1000):
    t0 = time.perf_counter()
    model.predict(sample)
    latencies.append((time.perf_counter() - t0) * 1000)

import numpy as np
print(f"p50: {np.percentile(latencies, 50):.3f} ms")
print(f"p95: {np.percentile(latencies, 95):.3f} ms")
print(f"p99: {np.percentile(latencies, 99):.3f} ms")
plt.hist(latencies, bins=50); plt.xlabel("ms"); plt.show()
```

### Challenge 5 — Reproducibility audit
Make two sibling cells. In the first, train a model and print accuracy. In the second, train the same model but with a **different `random_state`**. Compare the two accuracies. Discuss with a friend: **what changed?** (Answer: not much — but the decision boundary is slightly different, and this is why reproducibility requires pinning **every** random seed, not just the model's.)

---

## 📝 Wrap-up — reflect

Before you close this notebook, answer these in a markdown cell at the bottom:

1. Which myth surprised you the most?
2. What percentage of your time in this notebook was "ML" (lines of fit/predict) vs "everything else" (plotting, loading, measuring, writing checklists)?
3. What's **one thing** from this lecture that you will remember in 6 months, even if you forget everything else?

---

> **Next:** [← Lec 1 THEORY](mlops_lec01_intro_theory.md) · [Lec 2 →](../Lec_02_ML_Lifecycle/README.md)
>
> *MLOps · Lec 1 · github.com/rpaut03l/TS-02*
