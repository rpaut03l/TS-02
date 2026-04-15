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

> *MLOps · Lec 2 · github.com/rpaut03l/TS-02*
