# 💻 MLOps Lec 2 — ML Lifecycle: CODE

### *Hands-on sklearn for each lifecycle concept*

> **Nav:** [← Lec 2 README](README.md) | [📖 THEORY](mlops_lec02_lifecycle_theory.md) | **CODE** | [🎯 PRACTICE →](mlops_lec02_lifecycle_practice.md)

---

## 🏗️ Setup

```bash
pip install scikit-learn pandas numpy matplotlib joblib
```

---

## Ex 1. EDA — what's in this data?

### 👶 What this does
Before touching any model, **look** at your data. We'll load the classic breast cancer dataset and poke around.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df["target"] = bc.target    # 0 = malignant, 1 = benign

print("Shape:", df.shape)             # (569, 31)
print("\nFirst rows:")
print(df.head())

print("\nLabel balance:")
print(df["target"].value_counts())

print("\nSummary stats for 3 features:")
print(df[["mean radius", "mean texture", "mean area"]].describe())

print("\nAny missing values?")
print(df.isna().sum().sum(), "total missing cells")
```

### 👶 What EDA tells you before you train anything
- Is the dataset **balanced** or **imbalanced**?
- Are the features on **similar scales**? (probably not — you might need scaling)
- Are there **missing values** to handle?
- Are some features **wildly skewed**?

---

## Ex 2. Visual EDA — plot trends and anomalies

### 👶 What this does
EDA is easier with pictures than with numbers. We'll draw 3 plots that reveal structure.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(14, 4))

# Histogram of one feature
ax[0].hist(df["mean radius"], bins=30)
ax[0].set_title("Distribution of 'mean radius'")
ax[0].set_xlabel("mean radius"); ax[0].set_ylabel("count")

# Box plot per class
df.boxplot(column="mean area", by="target", ax=ax[1])
ax[1].set_title("'mean area' by class")

# Correlation heat-map (small version)
import numpy as np
corr = df[["mean radius", "mean texture", "mean area", "mean smoothness"]].corr()
im = ax[2].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax[2].set_xticks(range(4)); ax[2].set_yticks(range(4))
ax[2].set_xticklabels(corr.columns, rotation=45, ha="right"); ax[2].set_yticklabels(corr.columns)
ax[2].set_title("Feature correlations")
plt.colorbar(im, ax=ax[2])

plt.tight_layout(); plt.show()
```

---

## Ex 3. Build simple features

### 👶 What this does
A **feature** is any property of the input. Let's turn a raw pandas DataFrame into feature columns the model can eat.

```python
# Simple — just use raw columns
X_simple = df[["mean radius", "mean texture", "mean smoothness"]].values
print("Simple features shape:", X_simple.shape)

# Scaled features (zero mean, unit variance) — K-NN and logistic love this
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_simple)
print("Scaled features — mean:", X_scaled.mean(axis=0).round(3), "std:", X_scaled.std(axis=0).round(3))
```

---

## Ex 4. Build CROSS features

### 👶 What this does
A **cross feature** combines conditions. The theory file's example was `I(20 < age < 30, male, "xbox" in desc)`. Let's make one for the breast cancer data: "high radius AND high texture".

```python
# Cross feature: both mean radius and mean texture are above their medians
r_thr = df["mean radius"].median()
t_thr = df["mean texture"].median()

df["cross_big_and_rough"] = (
    (df["mean radius"] > r_thr) & (df["mean texture"] > t_thr)
).astype(int)

print("How many samples are BOTH big and rough:",
      df["cross_big_and_rough"].sum(), "/", len(df))

# Does the cross feature separate classes?
print("\nLabel proportion within each cross-feature value:")
print(df.groupby("cross_big_and_rough")["target"].mean().round(3))
```

### 👶 Lesson
If the "label proportion" differs a lot between the two groups, the cross feature is **informative**. Cross features are cheap to construct and can add a lot to a simple model.

---

## Ex 5. Train / validate with a simple split

### 👶 What this does
Model development's core loop — train, measure, iterate. We'll train a classifier on scaled features and check accuracy.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

X = bc.data
y = bc.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)

print(f"Test accuracy: {accuracy_score(y_te, pred):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_te, pred)}")
```

### 👶 What a `Pipeline` buys you
The `make_pipeline` call bundles the scaler and the model together. Benefits:
- Scaler fits on **train only** — no train/test leakage.
- At inference time you call **one `.predict()`** and all the preprocessing happens automatically.
- This is **tiny MLOps** — the preprocessing is **versioned together** with the model.

---

## Ex 6. Hyperparameter tuning — Grid Search

### 👶 What this does
Let's find a good combination of `C` (the logistic regression's regularization strength) and `solver` by trying all combinations.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "logisticregression__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "logisticregression__solver": ["lbfgs", "liblinear"],
}

grid = GridSearchCV(
    pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid.fit(X_tr, y_tr)

print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_te, grid.predict(X_te)):.4f}")

# Total trials tried:
print(f"\nTrials: {len(grid.cv_results_['params'])}")
```

### 👶 Grid math
5 values of `C` × 2 solvers = **10 combinations**. With 5-fold CV, that's **50 model fits** total.

---

## Ex 7. Hyperparameter tuning — Random Search

### 👶 What this does
Instead of trying every combination, **randomly sample** 20 from a larger space. Often finds a better point than grid because it explores unimportant dimensions less.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    "logisticregression__C": loguniform(1e-3, 1e3),    # continuous sampling
    "logisticregression__solver": ["lbfgs", "liblinear"],
}

random_search = RandomizedSearchCV(
    pipe, param_dist,
    n_iter=20,               # only 20 trials total
    cv=5, scoring="accuracy",
    random_state=0, n_jobs=-1
)
random_search.fit(X_tr, y_tr)

print(f"Best params: {random_search.best_params_}")
print(f"Best CV accuracy: {random_search.best_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_te, random_search.predict(X_te)):.4f}")
```

### 👶 Grid vs Random — which won?
Compare the **test accuracies** of Ex 6 and Ex 7 on your machine. Often random search matches or beats grid with **half the trials**.

---

## Ex 8. A training pipeline as a function (= code, not a binary)

### 👶 What this does
Per the theory, the *pipeline* is code and the *model* is the binary it produces. Let's write a pipeline function that takes data, runs the full recipe, and returns a fresh model every time.

```python
import joblib
from datetime import datetime

def training_pipeline(X_full, y_full, model_dir="/tmp"):
    """
    Full training recipe. Rerun this anytime new data arrives.
    Returns the path to the saved model.
    """
    # 1) Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=0.25, random_state=42, stratify=y_full
    )

    # 2) Build pipeline
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

    # 3) Fit
    pipe.fit(X_tr, y_tr)

    # 4) Validate
    acc = accuracy_score(y_te, pipe.predict(X_te))
    print(f"Validation accuracy: {acc:.4f}")

    # 5) Version & save
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"{model_dir}/bc_model_{version}.joblib"
    joblib.dump({"model": pipe, "accuracy": acc, "version": version}, path)
    print(f"Saved: {path}")

    return path

# Run the pipeline
path = training_pipeline(bc.data, bc.target)
print("Latest model path:", path)
```

### 👶 Why this matters
- **Re-run anytime** → retraining is just calling the function again.
- **Self-validates** → you see the accuracy at training time.
- **Versioned output** → timestamped filename; old models stay around for rollback.
- **Single artifact** → the `.joblib` file has the model AND its metadata.

This is a toy, but the shape is what **real training pipelines** look like — just bigger.

---

## Ex 9. Inference composition — Cuteness Detector

### 👶 What this does
We'll simulate the 2-model composition from the theory: a "puppy detector" and a "ball detector" feeding into a "cuteness detector." Each one is a mini scikit-learn classifier trained on fake data.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(0)

# Pretend each "image" is a vector of 8 features
X_fake, _ = make_classification(n_samples=1000, n_features=8, random_state=0)

# Build ground-truth labels for the sub-models
has_puppy = (rng.random(len(X_fake)) > 0.5).astype(int)
has_ball  = (rng.random(len(X_fake)) > 0.5).astype(int)
cute      = ((has_puppy == 1) & (has_ball == 1)).astype(int)

# Sub-model 1: puppy detector
puppy_model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_fake, has_puppy)
# Sub-model 2: ball detector
ball_model  = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_fake, has_ball)

# Composed inference: cute = puppy AND ball
def cuteness_detector(x):
    p = puppy_model.predict([x])[0]
    b = ball_model.predict([x])[0]
    return int(p and b), {"puppy": int(p), "ball": int(b)}

# Try it
for i in range(5):
    x = X_fake[i]
    cute_pred, components = cuteness_detector(x)
    print(f"Sample {i}: {components} → cute={cute_pred} (true cute={cute[i]})")
```

### 👶 End-to-end accuracy vs sub-model accuracy
```python
from sklearn.metrics import accuracy_score

puppy_acc = accuracy_score(has_puppy, puppy_model.predict(X_fake))
ball_acc  = accuracy_score(has_ball,  ball_model.predict(X_fake))
composed_preds = [cuteness_detector(x)[0] for x in X_fake]
cute_acc  = accuracy_score(cute, composed_preds)

print(f"Puppy sub-model accuracy: {puppy_acc:.3f}")
print(f"Ball  sub-model accuracy: {ball_acc:.3f}")
print(f"COMPOSED end-to-end acc:  {cute_acc:.3f}")
```

### 👶 Lesson
If puppy-acc = 0.95 and ball-acc = 0.95, the composed accuracy is **not** 0.95 — in the worst case it's 0.95 × 0.95 ≈ 0.90. Errors **compound**. **Always measure the composed pipeline**, not just the parts.

---

## Ex 10. Inference latency under load

### 👶 What this does
Mimic a real inference endpoint by timing many single-prediction calls. We already did this in Lec 1 — now measure it on our trained pipeline and see if we meet the **10 ms** goal.

```python
import time

x = X_te[:1]
for _ in range(50):
    pipe.predict(x)

N = 1000
t0 = time.perf_counter()
for _ in range(N):
    pipe.predict(x)
t1 = time.perf_counter()

per_call_ms = (t1 - t0) / N * 1000
print(f"Per-prediction latency: {per_call_ms:.3f} ms")
print(f"Within 10 ms budget?   {'✅ YES' if per_call_ms < 10 else '❌ NO'}")
```

You'll almost certainly see sub-millisecond latency for a tiny model like this. Big models would struggle — which is why they often need optimization (quantization, distillation, ONNX, TensorRT).

---

## Ex 11. Periodic vs feature-update retraining

### 👶 What this does
Simulate a week where the model stays the same but the **features** get updated every day because the user's click history changes. We'll see how the prediction moves.

```python
# Pretend the input is a click-history feature plus other stuff
base_sample = X_te[0].copy()

# "Monday" prediction
print("Monday:", pipe.predict_proba([base_sample])[0].round(3))

# "Tuesday" — user had a bunch of clicks → bump up one feature
tue = base_sample.copy()
tue[0] += 2.0    # pretend the click-history feature got bigger
print("Tuesday:", pipe.predict_proba([tue])[0].round(3))

# "Wednesday" — more clicks
wed = base_sample.copy()
wed[0] += 5.0
print("Wednesday:", pipe.predict_proba([wed])[0].round(3))
```

### 👶 Lesson
**Same model.** Different features → different predictions. This is the **feature update** pattern — you keep the model stable, but the user's inputs get fresher, and the predictions follow. Much more robust than online learning for most use cases.

---

## 🧭 What did we learn from this file?

1. **EDA first, always.** (Ex 1–2)
2. **Features are crafted, not just pulled out of the dataset.** Scaling and cross features matter. (Ex 3–4)
3. **Pipelines bundle preprocessing + model.** (Ex 5)
4. **Grid search = try all combos; random search = sample smart.** (Ex 6–7)
5. **Training pipelines are FUNCTIONS, not trained models.** Re-run them any time. (Ex 8)
6. **Composition needs end-to-end validation, not unit testing.** (Ex 9)
7. **Measure latency against the SLA before you deploy.** (Ex 10)
8. **Feature updates are often a better way to "refresh" a model than retraining.** (Ex 11)

---

> **Next:** [🎯 PRACTICE →](mlops_lec02_lifecycle_practice.md) · [← 📖 THEORY](mlops_lec02_lifecycle_theory.md)
>
> *MLOps · Lec 2 · github.com/rpaut03l/TS-02*
