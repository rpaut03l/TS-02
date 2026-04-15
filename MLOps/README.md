#  MLOps

### *Machine Learning Operations — notes, code, and practice*

> 🔗 **Repo:** [github.com/rpaut03l/TS-02](https://github.com/rpaut03l/TS-02) · MLOps Track
>
> **Style:** Every topic explained like you're **5 years old first** (easy story + picture), then the technical depth, then the code.

---

##  What even is MLOps?

### 👶 Easy Story
You bake a cake at home for your birthday (that's **ML training** — cool, fun, once-in-a-while). Now imagine you open a **bakery**: you bake hundreds of cakes every day, new flavours, you deliver them on time, you throw out the bad batches, the oven breaks, a new helper joins, a customer complains about a raisin… That's **MLOps**: the *bakery operations* around the fun "making one cake" part.

```
 ML         = make one model work on your laptop         (one cake at home)
 MLOps      = make that model run every day for real
              users, retrain it, monitor it, fix it, and
              not wake up anyone at 3am                    (running the bakery)
```

---

## 📁 Contents of this folder

| # | Lecture | Folder |
|---|---|---|
| 1 | **Introduction to MLOps** — Why, What, Research vs Production, ML Systems vs Traditional Software, Myths | [Lec_01_Introduction/](Lec_01_Introduction/) |
| 2 | **Machine Learning Lifecycle** — Model Development · Training · Inference, Features, Hyperparameters, Technologies | [Lec_02_ML_Lifecycle/](Lec_02_ML_Lifecycle/) |

Each lecture folder has the same **trio** of files:

| File | Purpose |
|---|---|
| `*_theory.md` | **Concepts explained like a kid** — easy stories, analogies, boxed diagrams, then the technical details |
| `*_code.md` | **Scikit-learn (and friends) code** — minimal, runnable, well-commented examples of every concept from the theory |
| `*_practice.md` | **Kaggle / Google Colab ready** — longer practice exercises you can copy straight into a notebook |

---

## 🧭 How to use this folder

1. **Read `theory.md` first.** Understand the concept with zero code. If the easy story makes sense, you're ready.
2. **Go through `code.md`.** Short, runnable, commented snippets. Type them out yourself; don't just read.
3. **Open `practice.md` in Colab or Kaggle.** The exercises there are bigger — they put several concepts together into something that looks like real work.

---

## 📚 Topic roadmap

Planned topics in this track (more folders will be added over time):

- ✅ **Lec 1** — Introduction to MLOps
- ✅ **Lec 2** — Machine Learning Lifecycle
- 🔭 Data preprocessing and normalization
- 🔭 Data visualization
- 🔭 Versioning, environments & orchestration (Git, virtualenv, Docker, Kubernetes)
- 🔭 Quantization
- 🔭 Reproducibility & experiment tracking

---

## 🔗 External Resources

Extra reading and hands-on material that pair well with this folder.

### 📝 Blog posts
- **[AWS Certified Machine Learning Engineer — Study Guide](https://www.rohitpatel.in/2025/11/aws-certified-machine-learning-engineer.html)** — a structured study guide for the AWS ML Engineer exam covering data prep, model development, deployment, and monitoring/security. Useful if you want to map the MLOps concepts here to a cloud-vendor-specific exam track (SageMaker endpoints, Model Monitor drift types, deployment decision trees, hyperparameter tuning strategies).
- **[Machine Learning Workflows & ML Models](https://www.rohitpatel.in/2025/11/machine-learning-workflows-ml-models.html)** — walks through the end-to-end ML workflow (load → clean → EDA → feature eng → train → validate → deploy → serve) with memory aids and runnable pandas / scikit-learn examples. Pairs directly with the 3-phase lifecycle in [Lec 2](Lec_02_ML_Lifecycle/README.md).

### 🛠️ Hands-on project repo
- **[rptl_gn_mlops — `mlops-pipeline` branch](https://github.com/rpaut03l/rptl_gn_mlops/tree/mlops-pipeline)** — a live example project that implements an automated **CI/CD pipeline combining Kubernetes deployment with ML-based microservice performance tuning** on a multi-node cluster. This is the "Lec 2 theory → real code" bridge: training pipelines as code, model artifacts versioned in CI, deployment automated on merge.
- **[GitHub Actions run — CI/CD with MLOps Pipeline (passing)](https://github.com/rpaut03l/rptl_gn_mlops/actions/runs/19781350058)** — a specific successful workflow run (≈4m 31s) triggered by a pull request to the `mlops-pipeline` branch, producing an `ml-model` artifact and a `pipeline-results` artifact. Good example of what "automated training pipeline + artifact tracking" looks like in practice.

### 🧠 Prerequisite ML fundamentals
- **[TS-01 / ML — algorithm fundamentals](https://github.com/rpaut03l/TS-01/tree/main/ML)** — the companion repo's ML track with theory + worked numerical problems + runnable practice code for the core algorithms (Regression, Decision Trees, Random Forest, K-NN, LDA, SVM/Kernels, PCA, Clustering, Neural Networks, Deep Learning, Parameter Estimation, Bayesian Decision Theory, and more). If a concept in this MLOps folder references an ML technique you want to refresh, that's the place to go.

---

> *github.com/rpaut03l/TS-02*
