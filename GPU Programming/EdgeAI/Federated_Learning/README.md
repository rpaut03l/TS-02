# 🤝 EdgeAI · Federated Learning & On-Device Training

### *Learn from everyone's data · Without anyone's data ever leaving their device*

> **Nav:** [← EdgeAI](../README.md) | [← Model Compression](../Model_Compression/README.md) | [← Deployment Frameworks](../Deployment_Frameworks/README.md) | [← TinyML](../TinyML/README.md) | **Federated Learning** | [Edge MLOps →](../Edge_MLOps/README.md) | [Security & Privacy →](../Security_Privacy/README.md)

---

## 👶 30-second story

100 children in 100 different classrooms. You want them all to get
smarter at the **same** subject — but you are not allowed to collect
their notebooks.

- **Cloud way:** every child sends their notebook to a central
  teacher. Teacher learns from everything. **Notebooks leave the
  classroom.**
- **Federated way:** every child does *some learning in their own
  notebook*. They each send only **"a small summary of what I
  learned today"**. The central teacher averages these summaries
  into a smarter textbook and sends it back. **Nobody's notebook
  ever leaves the classroom.**

That's **Federated Learning** (FL) in one picture.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [federated_learning_theory.md](federated_learning_theory.md) | Full theory — **FedAvg**, **FedSGD**, **FedProx**, cross-device vs cross-silo, the FL system architecture (clients, server, selection, aggregation), heterogeneity (non-IID data, stragglers, dropouts), communication efficiency (quantized updates, top-k), **Differential Privacy** + **Secure Aggregation**, when FL is the right answer |
| [federated_learning_code.md](federated_learning_code.md) | Runnable code — hand-rolled **FedAvg simulator in 60 lines** (clients as tensors, server as mean), same thing with **Flower** framework, intro to **TensorFlow Federated**, on-device fine-tuning with **LoRA adapters**, DP-SGD with Opacus |
| [federated_learning_practice.md](federated_learning_practice.md) | **Colab notebook** — simulate 20 clients each holding a **non-IID** shard of MNIST. Train FedAvg for 20 rounds. Plot global accuracy vs round. Compare with a single centralised run |

---

## 🎯 After reading this you should be able to…

- Draw the **FedAvg round** end-to-end from memory
- Tell **cross-device** from **cross-silo** FL with one real-world
  example each
- Explain why **non-IID** data makes FL hard (and what FedProx does
  about it)
- Implement a **toy FedAvg simulator** in under 100 lines of pure
  PyTorch
- Use the **Flower** framework for a realistic client/server setup
- Combine FL with **Differential Privacy** and **Secure Aggregation**
  to guarantee no single client's data can be extracted from the
  updates
- Spot when FL is **not** the right answer (centralised is often
  simpler)

---

## ⚡ The one-line intuition

> **Federated Learning moves the model to the data, instead of the
> data to the model.**

Everything else — aggregation math, privacy add-ons, selection
strategies — is a variation on that single idea.

---

> *GPU Programming · EdgeAI · Federated Learning · github.com/rpaut03l/TS-02*
