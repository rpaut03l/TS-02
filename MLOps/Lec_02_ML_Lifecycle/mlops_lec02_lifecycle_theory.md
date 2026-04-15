# 📖 MLOps Lec 2 — ML Lifecycle: THEORY

### *Model Development · Training · Inference · Features · Hyperparameters · Tools*

> **Nav:** [← Lec 2 README](README.md) | **THEORY** | [💻 CODE](mlops_lec02_lifecycle_code.md) | [🎯 PRACTICE](mlops_lec02_lifecycle_practice.md)

---

## 🧠 MNEMONIC: **"DTI-FHTI-F"**

> **D**evelopment · **T**raining · **I**nference · **F**eatures · **H**yperparameters · **T**ools · **I**nference-composition · **F**eedback

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | The Three Phases of the Lifecycle | [§1](#1-the-three-phases-of-the-lifecycle) |
| 2 | Model Development — what it includes | [§2](#2-model-development) |
| 3 | Features — what they are | [§3](#3-features) |
| 4 | Feature Engineering (Joins, Reuse, Predictions, Caches, Dynamic) | [§4](#4-feature-engineering) |
| 5 | Model Architecture | [§5](#5-model-architecture) |
| 6 | Hyperparameters & Tuning | [§6](#6-hyperparameters--tuning) |
| 7 | Model Development Tools | [§7](#7-model-development-tools) |
| 8 | Output of Model Development (the "Bad Idea") | [§8](#8-output-of-model-development) |
| 9 | Training Pipelines | [§9](#9-training-pipelines) |
| 10 | Training Technologies | [§10](#10-training-technologies) |
| 11 | Inference — the 10ms Goal | [§11](#11-inference--the-10ms-goal) |
| 12 | Inference using Composition | [§12](#12-inference-using-composition) |
| 13 | Incorporating Feedback | [§13](#13-incorporating-feedback) |
| 14 | Cheat Sheet | [§14](#14-cheat-sheet--exam-hacks) |

---

## 1. The Three Phases of the Lifecycle

### 👶 Easy Story
Think of a **lemonade stand**:
1. **Recipe invention** at home — try different sugar amounts, taste, decide what works.
2. **Bulk prep every morning** — squeeze 100 lemons, mix the tested recipe at scale.
3. **Serving customers all day** — pour cups fast, listen to "too sour!" feedback, adjust tomorrow.

The three ML phases map exactly:

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ MODEL DEVELOPMENT   │  │     TRAINING        │  │     INFERENCE       │
│  (Data Scientist)   │→ │  (Data Engineer)    │→ │  (Data Engineer)    │
│                     │  │                     │  │                     │
│ • Data collection   │  │ • Train at scale    │  │ • Serve predictions │
│ • Cleaning & viz    │  │ • Live data         │  │ • Low latency       │
│ • Feature eng.      │  │ • Retraining        │  │ • Handle bursts     │
│ • Training & valid. │  │ • Validate outputs  │  │ • Collect feedback  │
│                     │  │ • Model versioning  │  │                     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
      recipe                 kitchen                    serving
```

- **Model Development** is owned by the **data scientist** — they experiment until they find a recipe that works.
- **Training** is owned by **data engineers** — they turn the recipe into a daily pipeline that re-trains on fresh data.
- **Inference** is also owned by **data engineers** — they serve the trained model to end-users and watch how it behaves.

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 2. Model Development

### Inside the loop

```
       Data
    Collection
        │
        ▼
   Cleaning &
  Visualization
        │
        ▼
   Feature Eng. &    ◄─────┐
   Model Design           │
        │                 │
        ▼                 │
    Training &    ────────┘
   Validation
   (iterate)
```

### What happens in each step
| Step | What you do |
|---|---|
| **Data Collection** | Identify potential sources of data. Join data from multiple sources. |
| **Cleaning & Visualization** | Address missing values and outliers. Plot trends to identify anomalies. |
| **Feature Engineering & Model Design** | Build informative feature functions. Design new model architectures. |
| **Training & Validation** | Tune hyperparameters. Validate prediction accuracy. |

This loop **iterates** many times before you settle on a model that's worth deploying. Often dozens of passes for a single "final" model.

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 3. Features

### 👶 Easy Story
A **feature** is a property of your input. If you're predicting whether someone will buy a toy, features could be their **age** (kids love toys), their **click history** (did they browse the toy page?), the **price** (cheap toys sell more). Each one is a feature. A good model is built on good features.

### Formal definition
> **Features = properties or characteristics of the input.**

### Example — Click Prediction
Predict whether a user will click an ad. The features might include:

| Category | Examples |
|---|---|
| **User features** | age, gender, click history |
| **Product features** | price, popularity, description |
| **Combined (Cross) features** | `I(20 < age < 30, male, "xbox" in description)` — a single indicator that fires only when ALL three conditions are true |

A **cross feature** is a boolean that combines multiple simpler conditions. It lets the model capture interactions the base features can't.

### Exploratory Data Analysis (EDA)
Before touching models, you **look at the data**:
> EDA = uncover patterns, trends, and hidden insights using data visualization techniques.

Scatter plots, histograms, box plots, correlation heatmaps. This is where you notice that "oh, 30% of rows have a missing `age` field" or "all the positive labels are from one month" — things you can't learn from a model but that will wreck it if you don't fix them.

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 4. Feature Engineering

Five ideas that turn raw data into useful model inputs.

### Feature Joins
Combine multiple data sources into one feature. Example: join a user table + product table + click log → one row per user-product pair with all their info.

### Feature Reuse
> Good features can aid in many tasks.

Example: a **product embedding** trained for recommendation can also be used for:
- Ad targeting
- Search ranking
- Fraud detection

Re-using features across teams saves effort and keeps semantics consistent.

### Predictions as Features
Predictions from **one model** can be **features for another model**.

> Example: a "what's in this image?" classifier's output (`puppy`, `ball`, `grass`) can feed into an ad-targeting model.

This is the same idea as **inference composition** (§12), just viewed from the features side.

### Feature Tables / Caches
Features are often **pre-computed** and cached — especially expensive ones (embeddings, aggregations over huge history).

> Requires **tracking data, compute, and feature versions**.

If your online model reads feature v3 from the cache but the offline training pipeline uses v4, your production predictions will be off. This is a classic "training/serving skew" bug.

### Dynamic Features
> Features can often be modified faster than models.

Useful for **fast-changing dynamics** — e.g., a user's click history changes every minute, but the model behind it doesn't need to retrain every minute; updating the feature is enough.

> **Issue:** the resulting **covariate shift** can be problematic — the feature distribution seen at inference differs from what the model was trained on.

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 5. Model Architecture

> Based on the input and task, various model architectures can be explored.

### The usual suspects
| Input type | Typical architecture |
|---|---|
| Tabular / numeric | Gradient Boosting (XGBoost, LightGBM), Random Forest |
| Images | CNN, ResNet, ViT |
| Text | Transformer (BERT, GPT) |
| Audio | 1-D CNN, Transformer |
| Graphs | GNN (GCN, GAT) |
| Time series | LSTM, Temporal Fusion Transformer |

Choosing an architecture is a mix of "what worked last time," "what fits the input type," and "what can I afford to train and serve." Rarely do you invent a new architecture from scratch — you pick from a menu.

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 6. Hyperparameters & Tuning

### Definition
> Hyperparameters = **configuration details** that are **not directly determined through training**. Set by hand or tuned using cross-validation.

"Not learned from data" is the key phrase. The model learns weights; the *human* sets the hyperparameters.

### The big seven
| Hyperparameter | What it controls |
|---|---|
| **Learning rate** | How big a step the optimizer takes |
| **Number of epochs** | How many passes through the training data |
| **Batch size** | How many samples per gradient update |
| **Optimization algorithm** | SGD, Adam, RMSProp, … |
| **Regularization parameters** | L1, L2, dropout rates |
| **Decay rate** | How LR / momentum decreases over time |
| **Stopping criteria** | When to stop training (early stopping) |

### How to find good values
Since hyperparameters are not learned, they must be found through **trial and error**, **expert knowledge**, or **optimization techniques (hyperparameter tuning)**.

#### Grid Search
Test a **predefined set of values** across all combinations. Simple. Expensive if the grid is big.

```
learning_rate ∈ {0.001, 0.01, 0.1}
batch_size    ∈ {32, 64, 128}
→ 3 × 3 = 9 experiments
```

#### Random Search
**Randomly sample** from a range. Surprisingly, often **better than grid** (Bergstra & Bengio 2012): in high dimensions, grid wastes trials on unimportant parameters; random doesn't.

```
learning_rate ∈ Uniform(0.0001, 0.1)   (log-uniform actually)
batch_size    ∈ {32, 64, 128}
→ 20 random trials beats a 4×4 grid
```

#### Beyond these
- **Bayesian optimization** (Optuna, Hyperopt) — models the landscape, samples smarter
- **Population-based training** — evolutionary approach, used in DeepMind's work

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 7. Model Development Tools

Two buckets of tools a data scientist uses.

### Frameworks & Libraries
The stuff that does the math:

```
 scikit-learn   TensorFlow    PyTorch
 Keras          Caffe2        XGBoost
 pandas         matplotlib
```

### Environments & Notebooks
The stuff you *write code in*:

```
 Jupyter        Google Colab    Kaggle
 Visual Studio Code
```

The course will use **Python + PyTorch** (per Lec 1 prerequisites).

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 8. Output of Model Development

### The ❌ Bad Idea
The data scientist's output is **just a trained model** (a single `.pkl` or `.pt` file).

### Why it's a bad idea
> With just a trained model we are **unable to**:
> - **Retrain** models with new data
> - Track **data and code** for debugging
> - Capture **dependencies** for deployment
> - **Audit** training for compliance (e.g., GDPR)

If all you ship is `model.pkl`:
1. Someone asks "what data was this trained on?" → you don't know.
2. The model gets bad → you can't retrain because you don't have the pipeline.
3. Deployment fails → you don't know which version of sklearn to install.
4. A regulator asks for an audit trail → you have nothing to show them.

### The ✅ Good Idea
The output of model development should be:
1. **Reports & Dashboards** — the insights the data scientist found along the way
2. **Training Pipelines** — code that **can re-train the model from scratch on new data**
3. **Trained model** — the actual artifact for serving

The pipeline is the important thing. The trained model is just one snapshot the pipeline produced.

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 9. Training Pipelines

### Definition
> A **description of how to train the model** from data sources.

### Software-engineering analogy
```
 Training Pipelines   →   CODE     (the recipe — how to build it)
 Trained Models       →   BINARIES (the compiled output — the cake)
```

You version control the **recipe** (pipeline). The binary (model weights) is something the recipe produces on demand. If you ever need to rebuild the model — for a new dataset, a new library version, a bug fix — you run the pipeline, not tweak the binary.

### What a pipeline does at each run
- Pulls the latest data from the source
- Cleans and transforms it
- Computes features (possibly using cached ones)
- Trains the model
- **Validates** it against a holdout set
- Tags it with a version and stores it in a **model registry**
- Optionally pushes it to production

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 10. Training Technologies

### Workflow Management
Tools that schedule and glue the pipeline steps together:

```
 Apache Airflow       MLflow
 Azkaban              Luigi
 Apache Oozie
```

These handle "run step A, when it succeeds run B and C in parallel, on failure retry 3 times, email me if all retries fail, on success tag the output." That's pipeline orchestration.

### Scalable Training
Tools that do the actual heavy lifting when data or model is too big for one machine:

```
 Apache Spark    MXNet        PyTorch     TensorFlow
 XGBoost (dmlc)  Horovod
```

**Spark** for huge structured data. **PyTorch + Horovod** for distributed GPU training. **XGBoost** for tabular at scale. **TensorFlow** with TF-Distributed for many use cases.

### What training adds on top of development
From slide 17:
- Training models **at scale** on **live data**
- **Retraining** on new data
- **Automatically validate** prediction accuracy
- Manage **model versioning**
- **Requires minimal expertise in ML** (this is a goal — the pipeline should be automation, not a data scientist's manual workflow)

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 11. Inference — the 10 ms Goal

### 👶 Easy Story
Imagine a customer walks in and wants a cookie NOW. If you take 5 seconds to hand them one, they're impatient. If you take 10 seconds, they leave. That's **latency** — you have about **10 ms** to make a prediction in production.

### The goal

```
┌──────────────────────────────┐
│  Goal: make predictions in   │
│  ~10 ms under bursty load    │
│                              │
│  Complicated by              │
│  Deep Neural Networks        │
│  or Large Language Models    │
└──────────────────────────────┘
```

### Why it's hard
| Thing | Problem |
|---|---|
| **Deep models are slow** | A 1B-param LLM can't do a forward pass in 10 ms without specialized hardware |
| **Bursty load** | Normal day: 100 req/s. Peak moment: 10,000 req/s. Your service must not crash. |
| **Feature loading** | Features live in databases; fetching them takes network time |
| **Cold start** | First request after a long idle period is slow (caches empty) |
| **Tail latency** | p99 is what matters, not average |

### The inference service

```
 ┌──────────────────┐                    ┌──────────────┐
 │ Prediction       │     Query →         │              │
 │ Service          │ ───────────────     │  End User    │
 │                  │                     │  Application │
 │  [feat store]────┤                     │              │
 │     ↓            │     Prediction →    │              │
 │  [logic]         │                     │              │
 │     ↓            │ ←───────────────    │              │
 │  [model]         │                     │              │
 └──────────────────┘                     └──────────────┘
        ▲
        │ Feedback
        │
   Data Engineer
```

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 12. Inference using Composition

### 👶 Easy Story — the Cuteness Detector
You want a **Cuteness Detector**. It's built from two smaller models:
1. **Puppy Detector** — is there a puppy in the image?
2. **Ball Detector** — is there a ball in the image?

Rule: "cute if puppy AND ball."

### First run — correct image
```
   🐕+🎾 image
   ├──→ Puppy Detector  → YES
   │
   └──→ Ball Detector   → YES
                  ↓
            Cuteness Detector → "Cute!" ✅
```

### Second run — a wrong sub-model, but still correct end-to-end
Now the Puppy Detector gets "upgraded" but has a bug — it says YES to a picture of a **kitten** with a ball.

```
   🐈+🎾 image
   ├──→ Puppy Detector  → YES   ❌ wrong, but...
   │
   └──→ Ball Detector   → YES
                  ↓
            Cuteness Detector → "Cute!" ✅ (still correct by luck!)
```

A data scientist looking only at Puppy Detector might say "great, your model is perfect." A data scientist looking end-to-end says "it's still right — but the next time it'll fail in a weird way."

### Third run — the wrong sub-model silently breaks the pipeline
The same "improved" Puppy Detector now says NO to a puppy.

```
   🐕+🎾 image
   ├──→ Puppy Detector  → NO    ❌ regression from 'improvement'
   │
   └──→ Ball Detector   → YES
                  ↓
            Cuteness Detector → "Not Cute!" ❌
```

From the slides:
> Need to **track composition** and validate **end-to-end accuracy**.
> Need **unit AND integration testing** for models.

### Takeaway
When your prediction involves **multiple models chained together**, you can't test them in isolation and call it a day. You have to test the **full pipeline**. This is one of the trickiest MLOps problems — unit tests are easy, **integration tests for ML** are hard.

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 13. Incorporating Feedback

Your model is live. It's making predictions. Customers sometimes complain, or the world shifts, or your business adds a new product line. How do you update?

### Model updates: retrained as new data arrives

#### Periodically (batch)
```
 Every Sunday at 2 AM → retrain on the past week's data → deploy on Monday
```
- **Pro:** simple, leverages batch processing, easy to validate before deployment.
- **Con:** model could be out of date for a week — or more.

#### Continuously (online learning)
```
 Each new example → update model weights immediately
```
- **Pro:** freshest possible model.
- **Con:** needs validation on every update, learning-rate choices are tricky, can destabilize fast. **Complicated.**

### Feature updates: new data may change features
Even without retraining the **model**, updating the **features** it sees can make it behave differently — because the feature values change.

> **Example:** update click history for a user → the next inference call sees new feature values → different prediction.

### Model update vs Feature update
Usually **feature updates are more robust** than online learning because you don't touch the model weights — you just feed it fresher inputs. The model's structure is stable, the features are fresh. Good compromise.

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

## 14. Cheat Sheet & Exam Hacks

```
╔══════════════════════════════════════════════════════════════╗
║  MLOps LEC 2 ONE-LINERS                                      ║
╠══════════════════════════════════════════════════════════════╣
║  3 phases: Development → Training → Inference                ║
║  Development owners: Data Scientist                          ║
║  Training + Inference owners: Data Engineer                  ║
║                                                              ║
║  Feature = property of the input                             ║
║  Cross feature = I(cond1, cond2, …) interaction indicator    ║
║  Feature cache = pre-computed, must track versions           ║
║  Dynamic features = fast-changing values (e.g. click hist)   ║
║                                                              ║
║  Just a trained model ≠ output of development                ║
║  Pipelines = CODE, Models = BINARIES                         ║
║                                                              ║
║  Training tech: Airflow, MLflow, Spark, Horovod, PyTorch     ║
║  Inference goal: ~10 ms under bursty load                    ║
║                                                              ║
║  Composition: need END-TO-END validation, not unit tests     ║
║                                                              ║
║  Feedback:                                                   ║
║    Periodic  (batch)   → easy, may be stale                  ║
║    Continuous (online) → fresh, complicated                  ║
║    Feature updates     → robust middle ground                ║
║                                                              ║
║  Hyperparameters: LR, epochs, batch, optimizer, reg, decay, ║
║     stopping. Tune via grid or random search.                ║
╚══════════════════════════════════════════════════════════════╝
```

### ⚡ Exam Red Flags
1. **"Name the 3 phases of the ML lifecycle and who owns each."** — Model Development (Data Scientist), Training (Data Engineer), Inference (Data Engineer).
2. **"What's a cross feature?"** — An indicator function combining multiple conditions, e.g. `I(20 < age < 30, male, "xbox" in desc)`.
3. **"Why is the output of model development NOT just a trained model?"** — With only a trained model you can't retrain on new data, debug, redeploy, or audit.
4. **"Give the software-engineering analogy for pipelines vs models."** — Pipelines = code (recipe), models = binaries (output of running the recipe).
5. **"Name 3 workflow managers."** — Apache Airflow, MLflow, Luigi. (Also Azkaban, Oozie.)
6. **"Name 3 scalable training frameworks."** — Spark, Horovod, PyTorch. (Also XGBoost, TensorFlow, MXNet.)
7. **"What's the inference latency goal?"** — About **10 ms** under **bursty** load. Complicated by DNNs / LLMs.
8. **"Why does inference composition need end-to-end testing?"** — Because an "improvement" to one sub-model can regress the composed pipeline; local unit tests won't catch it.
9. **"Grid vs Random search — why is random often better?"** — In high-dimensional spaces, grid wastes trials on unimportant dimensions; random samples unimportant and important dimensions equally and so finds a good point faster.
10. **"Periodic vs online retraining?"** — Periodic is simpler but staler; online is freshest but needs careful validation and can be unstable.

[↑ Back to Top](#-mlops-lec-2--ml-lifecycle-theory)

---

> **Next:** [💻 CODE →](mlops_lec02_lifecycle_code.md) · [🎯 PRACTICE →](mlops_lec02_lifecycle_practice.md)
>
> *MLOps · Lec 2 · github.com/rpaut03l/TS-02*
