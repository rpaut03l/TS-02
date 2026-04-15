# 🎓 Lecture 1 — Introduction to MLOps

### *Why this course exists, what MLOps is, and four myths to unlearn*

> **Nav:** [← MLOps](../README.md) | **Lec 1** | [Lec 2 →](../Lec_02_ML_Lifecycle/README.md)

---

## 👶 30-second story

Your older cousin built a magic robot in her garage that guesses which cookie you'll like best. It works on her laptop. She's very proud of it.

Now her aunt wants to use it in her real cookie shop, for 10,000 customers a day, and she doesn't want the robot to ever guess wrong when someone has a nut allergy, and she wants it to learn new flavours when the shop gets new cookies, and she wants to know right away if the robot starts getting grumpy.

Going from "cool robot on my laptop" → "robot that a cookie shop can depend on" is **MLOps**.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [mlops_lec01_intro_theory.md](mlops_lec01_intro_theory.md) | The full concepts — what MLOps is, ML research vs ML production, ML systems vs traditional software, the 4 myths, and why ML engineering is more engineering than ML |
| [mlops_lec01_intro_code.md](mlops_lec01_intro_code.md) | **scikit-learn** code that makes the concepts concrete: train a baby model, measure latency, simulate drift, deploy as a function |
| [mlops_lec01_intro_practice.md](mlops_lec01_intro_practice.md) | **Kaggle / Colab** notebook-as-markdown: a longer hands-on session you can paste straight into a notebook |

---

## 🎯 After this lecture you should be able to…

- Explain **what MLOps is** in one sentence to a non-technical friend
- Name **5 components** of an ML system (interface, data, algorithms, infra, hardware)
- Name the **4 requirements** of an ML system design (Reliable, Scalable, Maintainable, Adaptable)
- List **3 concrete differences** between ML research and ML production
- Bust each of the **4 myths** without looking at the slides
- Answer the question: "Why do ML engineers spend so little time on ML?"

---

## 🔗 Related reading for this lecture

- **[AWS Certified Machine Learning Engineer — Study Guide](https://www.rohitpatel.in/2025/11/aws-certified-machine-learning-engineer.html)** — an exam-style map of the same MLOps ideas (data prep, model development, deployment, monitoring & security) onto AWS SageMaker / Model Monitor. Good for translating the vendor-neutral concepts here into a cloud-specific playbook.
- **[TS-01 / ML — algorithm fundamentals](https://github.com/rpaut03l/TS-01/tree/main/ML)** — if you want to brush up on the actual ML algorithms (Regression, Decision Trees, Random Forest, Neural Networks, etc.) that live inside the "Algorithms" box of the ML-system diagram, that's the companion repo. Recommended as a prerequisite if any algorithm name here is unfamiliar.

---

> *MLOps · Lec 1 · github.com/rpaut03l/TS-02*
