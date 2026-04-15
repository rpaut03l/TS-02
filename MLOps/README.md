# 📚 MLOps — CSL7040

### *Machine Learning Operations · IIT Jodhpur · Trimester 2 · Prof. Dr. Hardik Jain*

> 🔗 **Repo:** [github.com/rpaut03l/TS-02](https://github.com/rpaut03l/TS-02) · MLOps Track
>
> **Style:** Every topic explained like you're **5 years old first** (easy story + picture), then the technical depth, then the code.

---

## 🧠 What even is MLOps?

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
| `*_theory.md` | **Concepts explained like to a 5-year-old** — easy stories, analogies, boxed diagrams, then the technical details |
| `*_code.md` | **Scikit-learn (and friends) code** — minimal, runnable, well-commented examples of every concept from the theory |
| `*_practice.md` | **Kaggle / Google Colab ready** — longer practice exercises you can copy straight into a notebook |

---

## 🧭 How to use this folder

1. **Read `theory.md` first.** Understand the concept with zero code. If the easy story makes sense, you're ready.
2. **Go through `code.md`.** Short, runnable, commented snippets. Type them out yourself; don't just read.
3. **Open `practice.md` in Colab or Kaggle.** The exercises there are bigger — they put several concepts together into something that looks like real work.

---

## 🔗 Reference

- **Slides:** Dr. Hardik Jain, IIT Jodhpur — CSL7040 MLOps
- Lecture 1 reference: slides of Dr. Yashaswi Verma
- Lecture 2 reference: slides of Dr. Joseph E. Gonzalez, CS294 (UC Berkeley)
- **Course schedule** (from Lec 2 slide 26):

```
 Lec  Date          Topic                                         Lab
 L1   11-04-2026    Introduction to the course                    —
 L2   12-04-2026    Machine Learning Lifecycle / Technologies     —
 L3   18-04-2026    Data preprocessing and normalization          Colab
 L4   19-04-2026    Data Visualization                            Colab
 L5   25-04-2026    Versioning, Environments & Orchestration      Lab
 L6   26-04-2026    Quantization                                  Colab
 A1   30-04-2026    Bird Migration Problem (EDA)                  15 marks
 L7   02-05-2026    Reproducibility & Experiment tracking         Kaggle
 Q1   10-05-2026    Quiz: MCQ                                     20 marks
 A2   15-05-2026    Versioning / Reproducibility                  15 marks
 Q2   17-05-2026    Quiz: MSQ                                     20 marks
 Q3   24-05-2026    Lab Exam                                      30 marks
```

---

> *Trimester 2 · MTech AI · IIT Jodhpur · github.com/rpaut03l/TS-02*
