# 🎓 Lecture 2 — Machine Learning Lifecycle

### *Model Development → Training → Inference · Features · Hyperparameters · Tools*

> **Nav:** [← Lec 1](../Lec_01_Introduction/README.md) | [← MLOps](../README.md) | **Lec 2**

---

## 👶 30-second story

Remember the cookie robot from Lec 1? Here's the full picture of how it gets built and kept running.

1. **Model Development** — invent the recipe: collect ingredients (data), taste-test (cleaning/visualization), decide which ingredients matter most (features), pick an oven (model architecture), tune the temperature (hyperparameters).
2. **Training** — use the recipe in the kitchen every day. Bake fresh cakes on new ingredients. Check each batch before sending it out. Keep old recipes around in case the new one fails (model versioning).
3. **Inference** — serve cakes to customers. Customers want their cake in **10 seconds**, not 10 minutes. Some days 100 customers show up, some days 10,000 (bursty load). Listen to complaints (feedback) so next week's batch is better.

All three phases, together, are the **Machine Learning Lifecycle**.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [mlops_lec02_lifecycle_theory.md](mlops_lec02_lifecycle_theory.md) | The full concepts — 3 phases, features, feature engineering, model architecture, hyperparameters, dev/training/inference tools, inference composition, incorporating feedback |
| [mlops_lec02_lifecycle_code.md](mlops_lec02_lifecycle_code.md) | **scikit-learn** code for every concept: EDA, cross features, grid/random search, training pipelines, inference composition, latency measurement |
| [mlops_lec02_lifecycle_practice.md](mlops_lec02_lifecycle_practice.md) | **Kaggle / Colab ready** notebook with an end-to-end mini lifecycle on a real-ish dataset |

---

## 🎯 After this lecture you should be able to…

- Draw the **3-phase ML lifecycle** with the people involved (Data Scientist → Data Engineer → Data Engineer)
- Define **feature, feature joining, feature reuse, prediction-as-feature, dynamic feature**
- Explain why "**just a trained model**" is a bad output (4 reasons)
- List **7 typical hyperparameters** and the 2 common tuning methods (grid, random)
- Name **3 workflow managers** and **3 scalable training frameworks**
- Explain the **10 ms inference goal** and why LLMs make it hard
- Explain **inference composition** and why end-to-end testing matters
- Compare **periodic vs continuous (online) retraining**

---

## 🔗 Related reading for this lecture

- **[Machine Learning Workflows & ML Models](https://www.rohitpatel.in/2025/11/machine-learning-workflows-ml-models.html)** — an end-to-end walkthrough of the ML workflow (load → clean → EDA → feature eng → train → validate → deploy → serve) with memory aids and pandas / scikit-learn snippets. Maps very cleanly onto the 3-phase lifecycle in this lecture's theory file.
- **[rptl_gn_mlops — `mlops-pipeline` branch](https://github.com/rpaut03l/rptl_gn_mlops/tree/mlops-pipeline)** — a live project implementing an **automated CI/CD pipeline that combines Kubernetes deployment with ML-based microservice performance tuning** on a multi-node cluster. This is exactly what the "Training Pipelines as CODE, Models as BINARIES" analogy in §9 of the theory file looks like when it's real.
- **[GitHub Actions run — CI/CD with MLOps Pipeline (passing, 4m 31s)](https://github.com/rpaut03l/rptl_gn_mlops/actions/runs/19781350058)** — a specific successful run of that pipeline, triggered by a pull request, producing an `ml-model` artifact and a `pipeline-results` artifact. A concrete example of "every commit re-runs the whole recipe and versions the output."
- **[TS-01 / ML — algorithm fundamentals](https://github.com/rpaut03l/TS-01/tree/main/ML)** — the companion repo's ML track. If you need a refresher on a specific algorithm (Regression, Random Forest, Neural Networks, etc.) while working through the feature-engineering / hyperparameter sections, that's your one-stop reference.

---

> *MLOps · Lec 2 · github.com/rpaut03l/TS-02*
