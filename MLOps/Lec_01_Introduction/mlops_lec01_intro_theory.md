# 📖 MLOps Lec 1 — Introduction: THEORY

### *What, Why, Research vs Production, Systems vs Software, 4 Myths*

> **Nav:** [← Lec 1 README](README.md) | **THEORY** | [💻 CODE](mlops_lec01_intro_code.md) | [🎯 PRACTICE](mlops_lec01_intro_practice.md)

---

## 🧠 MNEMONIC: **"WHY-RSMA-4M"**

> **W**hy MLOps · **H**ow it differs from software · **Y**ou'll deploy many models ·  **R**eliable · **S**calable · **M**aintainable · **A**daptable · **4 M**yths

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Why This Course Exists | [§1](#1-why-this-course-exists) |
| 2 | What is MLOps? | [§2](#2-what-is-mlops) |
| 3 | Components of an ML System | [§3](#3-components-of-an-ml-system) |
| 4 | ML Systems Design (RSMA) | [§4](#4-ml-systems-design-rsma) |
| 5 | ML Research vs ML Production | [§5](#5-ml-research-vs-ml-production) |
| 6 | ML Systems vs Traditional Software | [§6](#6-ml-systems-vs-traditional-software) |
| 7 | Engineering Challenges with Large ML Models | [§7](#7-engineering-challenges-with-large-ml-models) |
| 8 | Four ML Production Myths | [§8](#8-four-ml-production-myths) |
| 9 | ML Engineering is More Engineering Than ML | [§9](#9-ml-engineering-is-more-engineering-than-ml) |
| 10 | Cheat Sheet | [§10](#10-cheat-sheet--exam-hacks) |

---

## 1. Why This Course Exists

### 👶 Easy Story
Think about a **toy car**. Building the toy car is fun — cut wood, paint it, stick wheels on. That's **making an ML algorithm**.

Now imagine a **real car** sold to a million people. Engineers spend 95% of their time on **seat belts, airbags, brakes, factory robots, paint lines, replacement parts, warranty, crash testing** — not on the *idea* of a car. The "idea" is the easy part!

**ML is the same.** The algorithm (the idea) is the smallest, most fun piece. Everything around it — data, plumbing, deployment, monitoring, retraining — is where the real work is. This course is about that **everything around it**.

### The core insight
> ❑ ML algorithms is the **less problematic part**
> ❑ The **hard part** is how to make algorithms work with other parts to solve real-world problems

### What "other parts"?
```
┌────────────────────────────────────────────────────┐
│ 🖥️  INTERFACE:       APIs, Dashboards              │
│ 📊  DATA:            Datasets, data pipelines       │
│ 🧠  ALGORITHMS:      The ML model itself            │
│ 🏗️  INFRASTRUCTURE:  CI/CD, version control, cloud  │
│ 💻  HARDWARE:        GPUs, edge devices, servers    │
└────────────────────────────────────────────────────┘
```

An ML **algorithm** is just one of five boxes. An ML **system** is all five working together.

> 🧠 **Need a refresher on the "Algorithms" box?** The companion repo [TS-01 / ML](https://github.com/rpaut03l/TS-01/tree/main/ML) has full theory + worked math + runnable practice code for every major algorithm (Regression, Decision Trees, Random Forest, K-NN, LDA, SVM, Neural Networks, Deep Learning, Parameter Estimation, Bayesian Decision Theory, and more). If a name in this lecture is unfamiliar, start there.

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

## 2. What is MLOps?

### 👶 Easy Story
**Dev** = the person who builds the thing (the chef who invents a recipe).
**Ops** = the person who runs the thing every day (the kitchen that serves 200 meals).
**MLOps** = a culture where the chef and the kitchen **talk every day**, use the **same tools**, and fix problems **together**.

### Formal definition (AWS)
> An ML **culture and practice** that unifies ML application **development (Dev)** with ML system **deployment and operations (Ops)**.

### Why the fuss?
Without MLOps:
- A data scientist trains a beautiful model in a Jupyter notebook.
- She hands it to an engineer over Slack: "please deploy this."
- The engineer doesn't know what Python libraries she used, what data she trained on, what version of the model is "the one."
- It takes 3 months to deploy. By then the data has drifted and the model is stale.

With MLOps:
- Data, code, models, and environments are **all versioned** together.
- Deployment is **automated** from the same repo.
- The model is **monitored** in production — alerts fire if accuracy drops.
- Retraining is a **scheduled job**, not a heroic effort.

### MLOps in one sentence
> **MLOps = DevOps for machine learning — but harder, because the "code" includes the data and the model.**

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

## 3. Components of an ML System

```
┌────────────────────────────────────────────────────┐
│                   ML SYSTEM                        │
│                                                    │
│   [Interface]  ← users, APIs, dashboards          │
│       ▲                                            │
│       │                                            │
│   [Algorithms] ← your model, training code        │
│       ▲                                            │
│       │                                            │
│   [Data]       ← collection, cleaning, pipelines  │
│       ▲                                            │
│       │                                            │
│  [Infrastructure] ← CI/CD, version control, K8s   │
│       ▲                                            │
│       │                                            │
│   [Hardware]   ← GPUs, TPUs, edge, cloud          │
└────────────────────────────────────────────────────┘
```

Each box can fail. Each box has its own tools, its own people, its own weekly headaches. MLOps is the discipline of **making all the boxes behave as one reliable system**.

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

## 4. ML Systems Design (RSMA)

> **Definition:** The process of defining the **interface, algorithms, data, infrastructure, and hardware** for an ML system to satisfy specified requirements.

The four requirements an ML system must meet:

### 👶 Easy Story with the cookie robot
- **R — Reliable:** the robot doesn't break when a customer asks a weird question. (Doesn't crash.)
- **S — Scalable:** works fine for 10 customers, and also for 10 million customers. (Doesn't slow down.)
- **M — Maintainable:** next year, a new intern can read the code and fix a bug. (Doesn't rot.)
- **A — Adaptable:** when new cookies come out, the robot learns about them without being rebuilt from scratch. (Doesn't get stale.)

### Checklist form
| | Requirement | Key question |
|---|---|---|
| ✅ | **Reliable** | Does the system produce correct outputs, even under failures, noisy inputs, or adversarial users? |
| ✅ | **Scalable** | Does performance degrade gracefully as data, models, and traffic grow? |
| ✅ | **Maintainable** | Can a different engineer (or future-you) understand, debug, and evolve the system? |
| ✅ | **Adaptable** | Can the system incorporate new data distributions, new labels, new business goals without a rewrite? |

### The questions this course will answer
- You have trained a model — **now what**?
- What are the different **components** of an ML system?
- How to do **data engineering**?
- How to **evaluate** your models — offline AND online?
- **Online prediction vs batch prediction** — what's the difference?
- How to **serve** a model on the cloud? On the edge?
- How to **continuously monitor** and deploy changes to ML systems?

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

## 5. ML Research vs ML Production

### 👶 Easy Story
**Research** = you're a chef on a cooking show. You have 30 minutes, a perfect kitchen, no customers, one dish, and the judges only care about the taste.

**Production** = you're running a real restaurant. Customers are hungry *now*. Kids spill things. The oven breaks. Allergies matter. Cost matters. Speed matters. Taste is only *one* of many things.

### Side-by-side comparison

```
┌──────────────────┬───────────────────────┬────────────────────────┐
│ DIMENSION        │ RESEARCH              │ PRODUCTION              │
├──────────────────┼───────────────────────┼────────────────────────┤
│ Objective        │ Model performance     │ Different stakeholders │
│                  │ (e.g. accuracy)       │ want different things   │
│ Computational    │ Fast training, high   │ Fast inference,         │
│ priority         │ throughput            │ low latency             │
│ Data             │ Static, clean,        │ Constantly shifting,    │
│                  │ benchmarked           │ dirty, incomplete       │
│ Fairness         │ Often an afterthought │ Must be considered      │
│                  │                       │ (legal, ethical)        │
│ Interpretability │ Nice-to-have          │ Often required          │
│                  │                       │ (audit, debugging)      │
└──────────────────┴───────────────────────┴────────────────────────┘
```

### Stakeholder objectives (why this is hard)
When the cookie robot goes into production, **different people want different things**:
- **The data scientist** — "model accuracy 95%"
- **The product manager** — "customers should click more"
- **The sales team** — "we need an explanation to give to customers"
- **The ML platform team** — "don't make our servers catch fire"
- **The manager** — "stay under budget"
- **The legal team** — "don't discriminate; comply with GDPR"

Building one model that makes all of them happy is *hard*. The research paper optimized only one of these (accuracy). Production has to balance all six.

### The fairness example (from the slides)
> The Berkeley study found that both face-to-face and online lenders **rejected 1.3 million creditworthy Black and Latino applicants** between 2008 and 2015. The researchers said those applicants *"would have been accepted had the applicant not been in these minority groups."*
>
> When they used the same income and credit scores but **deleted the race identifiers**, the applications were accepted.

This is why **fairness is not optional in production**. The dataset had racial bias, the model learned it, and real people were denied mortgages.

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

## 6. ML Systems vs Traditional Software

### 👶 Easy Story
**Traditional software** = a recipe card. You write the steps, put it in the kitchen, and every cake turns out the same. The recipe never changes on its own.

**ML system** = a chef who learns from experience. Today's cake comes out great. But tomorrow the flour is a different brand, the chef gets tired, and the cake is different. The chef IS the recipe, and the recipe changes when the chef's experience changes.

### What's different?

| | Traditional Software | ML System |
|---|---|---|
| **What you write** | Code (logic) | Code (logic) + **data** + **model weights** |
| **Determinism** | Same input → same output, forever | Same input → output **may drift** as model is retrained |
| **Testing** | Unit tests, integration tests | + data tests, model tests, distribution tests |
| **Debugging** | Read the stack trace | "It's the data. Or the model. Or the pipeline. Or all three." |
| **Versioning** | Version the code | Version the code **and** the data **and** the model **and** the environment |
| **Root-cause analysis** | Linear — trace through code | Non-linear — bug could be in any of the above |

### Separation of concerns
> In traditional software, **code** and **data** are cleanly separated.
>
> In ML, **code, data, and model** are tangled together. Change any one, and the behaviour of the whole thing changes. That's the core reason MLOps is harder than DevOps.

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

## 7. Engineering Challenges with Large ML Models

As models get bigger (BERT → GPT → LLaMA → GPT-4), new problems appear that didn't exist for a decision tree:

| Challenge | Example |
|---|---|
| **Memory** | A 70B-parameter LLM needs 140 GB just to load in FP16 — more than one GPU has |
| **Training cost** | Training a big model can cost **millions of dollars** in GPU time |
| **Inference latency** | A single forward pass can take **seconds**, not milliseconds |
| **Distributed training** | Model must be sharded across many GPUs — complex bugs, hard to debug |
| **Storage** | Model checkpoints are hundreds of gigabytes each; you can't just email one |
| **Carbon footprint** | Large-model training emits the same CO₂ as several transatlantic flights |
| **Data scale** | Training data is in the terabytes or petabytes — needs distributed storage |

The cookie robot of 2018 fit in a Jupyter notebook. The cookie robot of 2026 needs a dedicated team just to serve requests.

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

## 8. Four ML Production Myths

### Myth #1: "Deploying is hard"
> **Reality:** Deploying is **easy**. Deploying **reliably** is hard.

- Writing a Flask app that serves your model → 10 lines of code, done in an hour.
- Writing a service that handles 10,000 requests per second, with 99.99% uptime, automatic rollback, logging, monitoring, security, A/B testing, graceful degradation → a **team for a year**.

The first `docker run` is easy. The second million requests are hard.

---

### Myth #2: "You only deploy one or two ML models at a time"
> **Reality:** Big companies deploy **dozens or thousands** of models.

Examples from the slides:
- **Booking.com** — 150+ models in production at once
- **Uber** — **thousands** of models (pricing, ETA, fraud, matching, search…)

Each model has its own data, its own pipeline, its own bugs, its own monitoring. The tooling you build must scale to many models, not just one.

---

### Myth #3: "You won't need to update your models much"
> **Reality:** The pace of software delivery is accelerating. ML isn't exempt.

From the *State of DevOps 2021* report:
```
Elite performers deploy   973× more frequently
Elite performers deploy   6570× faster lead time
DevOps standards:
  - Etsy   deploys 50 times/day
  - Netflix  deploys 1000s of times/day
  - AWS    deploys every 11.7 seconds
```

If the company around your ML model deploys 1000 times/day, your ML model can't be the slow, scary thing that gets deployed once a quarter. **Continuous delivery must include the ML pipeline.**

---

### Myth #4: "ML will magically transform your business overnight"
> **Reality:** Magically — **possible**. Overnight — **no way**.

ML adoption is a multi-year journey. It takes time to:
1. Collect clean labeled data.
2. Build pipelines that don't break.
3. Train engineers and analysts to use the tools.
4. Earn stakeholder trust in model decisions.
5. Integrate with existing business processes.
6. Learn from failures and iterate.

Any vendor promising "ML in 30 days" is selling a **demo**, not a system.

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

## 9. ML Engineering is More Engineering Than ML

### 👶 Easy Story
Imagine you want to be a **chef**. You picture yourself inventing new recipes, tasting sauces, arranging plates. Then you get hired at a real restaurant and discover you spend 6 hours a day **washing vegetables, ordering stock, cleaning counters, fighting with the dishwasher**, and only 1 hour actually cooking.

**ML engineering is the same.**

### Where the time actually goes
From the slides:
> MLEs (Machine Learning Engineers) might spend **most of their time**:
> - Wrangling data
> - Understanding data
> - Setting up infrastructure
> - Deploying models
>
> **Instead of** training ML models.

### Typical ML engineer's week
```
  15%  Data cleaning, joining, deduping
  15%  Pipeline glue code (Airflow, DBT, etc.)
  15%  Debugging production bugs (wrong labels, missing features)
  10%  Meetings, code review
  10%  Monitoring and firefighting
  10%  Infrastructure (K8s, CI/CD, deployments)
   5%  Feature engineering
   5%  Model training and evaluation  ← the "ML" part
   5%  Writing tests
   5%  Documentation
   5%  Reading papers / upskilling
```

**Only ~5-10% of the job is the "interesting ML" part.** That's why this course is mostly about the other 90%.

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

## 10. Cheat Sheet & Exam Hacks

```
╔══════════════════════════════════════════════════════════════╗
║  MLOps LEC 1 ONE-LINERS                                      ║
╠══════════════════════════════════════════════════════════════╣
║  MLOps = culture unifying ML Dev + Ops                       ║
║  ML system = Interface + Data + Algorithms + Infra + HW      ║
║  ML Design requirements: Reliable · Scalable · Maintainable  ║
║                          · Adaptable  (RSMA)                ║
║  Research optimizes one metric, Production balances many     ║
║  Traditional SW: code separate from data                     ║
║  ML system: code + data + model tangled together             ║
║  Myth 1: Deploying is easy; deploying RELIABLY is hard       ║
║  Myth 2: Many companies deploy 100s-1000s of models          ║
║  Myth 3: Elite teams deploy thousands of times/day           ║
║  Myth 4: ML is magic, but not overnight                      ║
║  ML engineering ≈ 5-10% ML + 90% engineering                 ║
╚══════════════════════════════════════════════════════════════╝
```

### ⚡ Exam Red Flags

1. **"Define MLOps in one sentence."** — A culture and set of practices that unifies ML application **development** with ML system **deployment and operations**.

2. **"Name the 5 components of an ML system."** — Interface, Data, Algorithms, Infrastructure, Hardware.

3. **"What are the 4 requirements of ML Systems Design?"** — **R**eliable, **S**calable, **M**aintainable, **A**daptable. (RSMA.)

4. **"Give 3 differences between ML research and ML production."** — (a) Single objective vs many stakeholder objectives. (b) Fast training vs fast inference. (c) Static clean data vs shifting dirty data. (Fairness and interpretability are also valid answers.)

5. **"Why is ML systems different from traditional software?"** — Code + data + model weights are tangled; outputs can change as data changes; testing and versioning must cover all three.

6. **"Bust ML Production Myth #1."** — Deployment is easy; **reliable** deployment (uptime, monitoring, rollback, security) is what's hard.

7. **"Why do ML engineers spend most of their time not training models?"** — Because data wrangling, pipelines, infrastructure, debugging, and deployment are the bulk of the work.

8. **"What is the fairness example about?"** — Lenders rejected 1.3M creditworthy Black and Latino applicants 2008–2015; when the race identifiers were removed from the same applications, they were accepted — proving the model encoded racial bias.

[↑ Back to Top](#-mlops-lec-1--introduction-theory)

---

> **Next:** [💻 CODE →](mlops_lec01_intro_code.md) · [🎯 PRACTICE →](mlops_lec01_intro_practice.md)
>
> *MLOps · Lec 1 · github.com/rpaut03l/TS-02*
