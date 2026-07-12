# DLOPS_EXAM_04 — Notebook Deep Dive (every class notebook, in detail)

[Hub](DLOPS_EXAM_00_Hub.md) | [Theory](DLOPS_EXAM_01_Theory.md) | [Numericals](DLOPS_EXAM_02_Numericals.md) | [Practice](DLOPS_EXAM_03_Practice.md) | [QA Bank](DLOPS_EXAM_05_QA_Bank.md)

## table of contents
- [nb1 — intro to pytorch](#nb1--intro-to-pytorch)
- [nb2 — basics of pytorch for dl](#nb2--basics-of-pytorch-for-dl)
- [nb3 — cnn classification and feature extraction](#nb3--cnn-classification-and-feature-extraction)
- [nb4 and nb5 — datasets, dataloaders, tinyvgg, loss curves](#nb4-and-nb5--datasets-dataloaders-tinyvgg-loss-curves)
- [nb6 — tensorboard experiment tracking](#nb6--tensorboard-experiment-tracking)
- [nb7 and nb8 — wandb sweeps](#nb7-and-nb8--wandb-sweeps)
- [nb9 — wandb artifacts pipeline versioning](#nb9--wandb-artifacts-pipeline-versioning)
- [nb10 — distributed training](#nb10--distributed-training)
- [nb11 — torchscript](#nb11--torchscript)
- [nb12-13 — onnx](#nb12-13--onnx)

---

## nb1 — intro to pytorch

Story: PyTorch's whole workflow in one notebook — tensor -> autograd -> model -> loss -> optimizer -> save.

Key content:
- Tensor creation (`tensor, zeros, ones, rand, randn, eye, arange`), attributes (`shape, dtype, device`).
- NumPy bridge: `torch.from_numpy(a)` and `t.numpy()` **share memory on CPU** — mutation propagates both ways.
- Operations: `torch.matmul` / `@` vs `torch.mul` / `*` — matmul is matrix product (shape rule (a,b)@(b,c)=(a,c)); mul is elementwise (shapes broadcast).
- Concatenation: `torch.cat([t1,t2], dim=0)` stacks along existing dim.
- Autograd demo: `requires_grad=True`, `loss.backward()`, gradients in `.grad`; `binary_cross_entropy_with_logits` as functional loss.
- Device modern API: `torch.accelerator.is_available()` / `current_accelerator()` (newer alternative to cuda.is_available shown in class — either accepted).
- First `nn.Sequential(Flatten, Linear, ReLU, Linear, Softmax)` on FashionMNIST + `optim.SGD` loop.
- `torch.save` / `torch.load` for tensors and state_dicts; `torch.no_grad()` for inference.

[back to top](#table-of-contents)

## nb2 — basics of pytorch for dl

Story: "From Building Blocks to Training Well" — the full classifier plus the *math* of optimizers and the overfitting toolkit.

Key content:
1. **Hand-written loop -> real API**: first manual gradient descent, then optimizer objects.
2. **Complete FashionMNIST classifier**: datasets -> DataLoader -> `nn.Module` model -> CrossEntropyLoss -> loop -> save/load.
3. **Activation zoo compared on plots**: relu, leaky_relu (`F.leaky_relu`), sigmoid, tanh, gelu (`F.gelu`) — shapes, ranges, when each saturates.
4. **Optimizer math (exam favorite)**:
   - SGD: `w = w - lr * g`
   - SGD + Momentum: `v = beta*v + g ; w = w - lr*v` (velocity remembers direction, beta≈0.9)
   - Adam: keeps two moving averages — m (mean of grads, beta1=0.9) and v (mean of squared grads, beta2=0.999):
     ```
     m = b1*m + (1-b1)*g
     v = b2*v + (1-b2)*g^2
     m_hat = m/(1-b1^t) ; v_hat = v/(1-b2^t)     # bias correction
     w = w - lr * m_hat / (sqrt(v_hat) + eps)
     ```
   - Intuition: SGD = walking, Momentum = rolling ball, Adam = rolling ball with self-adjusting shoe size per parameter.
5. **Learning rate effects**: too high -> loss explodes/oscillates; too low -> crawls; the "just right" curve.
6. **Overfitting + regularization**: Dropout (`nn.Dropout(p)` — randomly zeroes activations at train time, off at eval), weight decay (L2, `optim.Adam(..., weight_decay=1e-4)`), early stopping, more data/augmentation.

**Mnemonic — "SMA ladder: SGD walks, Momentum rolls, Adam adapts."**

[back to top](#table-of-contents)

## nb3 — cnn classification and feature extraction

Story: CIFAR-10, LeNet-style CNN, then the hybrid deep-features + RandomForest trick.

Flow:
1. CIFAR-10 via `torchvision.datasets`, transform = `ToTensor + Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))` -> pixel range [-1,1].
2. Network (see Numericals section 4 for full shape table): conv1(3,6,5) -> pool -> conv2(6,16,5) -> pool -> fc 400->120->84->10. No dropout version.
3. Train with `CrossEntropyLoss + SGD(lr=0.001, momentum=0.9)`, classic 2-epoch loop printing running loss every 2000 mini-batches.
4. Predictions: `torch.max(outputs, 1)` returns (values, indices) — indices are class predictions. Per-class accuracy computed.
5. **Feature extraction**: forward only up to fc2 -> 84-dim embedding per image.
6. **RandomForestClassifier** trained on embeddings; `rf.score()` on test embeddings; `feature_importances_` plotted to show which of the 84 features matter.
7. Persistence: `torch.save(net.state_dict(), path)` for the CNN; `joblib.dump(rf, path)` for the forest.

Why hybrid works (likely theory Q): CNN learns a representation where classes are almost linearly separable; a classic ML model on top is cheap, interpretable (feature importance), and trains in seconds.

[back to top](#table-of-contents)

## nb4 and nb5 — datasets, dataloaders, tinyvgg, loss curves

Story: pizza/steak/sushi-style image folders + FashionMNIST subset; from "become one with the data" to a full trained TinyVGG with augmentation comparison.

Part 1 — data pipeline:
- `os.walk` directory inspection ("become one with the data"), PIL and matplotlib visualization.
- Transform pipeline: `Resize -> RandomHorizontalFlip -> ToTensor` (train), plain for test.
- **Option 1: `ImageFolder(root, transform)`** — class-per-folder convention; `.classes`, `.class_to_idx`.
- **Option 2: custom `Dataset` replicating ImageFolder** — helper `find_classes(dir)` returning `(classes, class_to_idx)` using `os.scandir`, raising error if empty; then the ILG trio (`__init__` collects `pathlib.Path(...).glob("*/*.jpg")`, `__len__`, `__getitem__` opens image, applies transform, returns `(tensor, label)`).
- Random-image display helper; DataLoaders with `batch_size, shuffle, num_workers (os.cpu_count())`.
- Augmentation: `TrivialAugmentWide(num_magnitude_bins=31)` — one random augmentation at random strength per image; the modern "just works" policy.

Part 2 — TinyVGG:
```
class TinyVGG(nn.Module):
  block1: Conv(3,10,3) ReLU Conv(10,10,3) ReLU MaxPool(2)
  block2: Conv(10,10,3) ReLU Conv(10,10,3) ReLU MaxPool(2)
  classifier: Flatten -> Linear(10*13*13, num_classes)   # for 64x64 input
```
- Sanity test with a single image forward pass; `torchinfo.summary(model, input_size=(1,3,64,64))` for shape/param audit.
- `train_step(model, loader, loss_fn, opt)` and `test_step(...)` functions returning (loss, acc); `train()` combines both per epoch and returns a results dict `{"train_loss":[...], "train_acc":[...], "test_loss":[...], "test_acc":[...]}`.

Part 3 — **loss-curve diagnosis (huge exam favorite)**:
```
UNDERFITTING            IDEAL                   OVERFITTING
train loss high         both curves fall        train loss falls
test  loss high         together, small gap     test loss rises/flat
 \______                 \                        \        ___ test
        \____             \___                     \      /
  both flat high              \___ together         \____/  gap grows
                                                     train
```
- Fix overfitting: more data, augmentation, dropout, weight decay, early stopping, smaller model, transfer learning.
- Fix underfitting: bigger model, train longer, higher lr, fewer regularizers, better features.
- Model 0 (no aug) vs Model 1 (TrivialAugmentWide) results compared with dataframe + curves.
- Custom-image prediction: load with `torchvision.io.read_image`, convert to float /255, resize, unsqueeze(0) to add batch dim, predict.

**Mnemonic — gap logic: "Gap Grows = Generalization Gone" (overfit); "Both Bad = Bigger Brain needed" (underfit).**

[back to top](#table-of-contents)

## nb6 — tensorboard experiment tracking

Story: same TinyVGG world, but now every run writes to a log directory and you compare experiments like a scientist.

Key content:
- `create_writer(experiment_name, model_name, extra)` helper builds timestamped log dir: `runs/YYYY-MM-DD/experiment_name/model_name/extra` — so runs never overwrite.
- `train()` upgraded with a `writer` parameter: inside epoch loop
  ```
  writer.add_scalars("Loss", {"train_loss": train_loss, "test_loss": test_loss}, epoch)
  writer.add_scalars("Accuracy", {"train_acc": train_acc, "test_acc": test_acc}, epoch)
  writer.add_graph(model, input_to_model=torch.randn(32,3,64,64).to(device))
  writer.close()
  ```
- Also shown: `add_scalar`, `add_pr_curve(classname, labels, probs)` per class, `add_hparams(hparam_dict, metric_dict)` for the hyperparameter table view.
- **Experiment grid (section 7-9)**: vary (a) dataset size 10% vs 20%, (b) epochs 5 vs 10, (c) model variant — nested loops create 8 experiments, each with its own writer; `save_model()` utility saves each; best model picked from results and reloaded for prediction on a custom image.
- Launch: `%load_ext tensorboard` + `%tensorboard --logdir runs` in Colab, or CLI `tensorboard --logdir=runs`.

Design lesson (theory Q): change **one variable at a time**; start small (data subset, few epochs) and scale up — that's the experiment discipline DLOps preaches.

[back to top](#table-of-contents)

## nb7 and nb8 — wandb sweeps

Story: stop hand-tuning; describe the search space, let agents hunt.

Sweep anatomy (three steps, both notebooks):
1. **Config** (dict or YAML): `method` (grid/random/bayes), `metric` ({name, goal}), `parameters` — each param either `{"values":[...]}`, `{"value": x}` (fixed), `{"min":a,"max":b}` (range; can add `"distribution":"log_uniform_values"` for lr).
2. **`sweep_id = wandb.sweep(config, project=...)`** — registers the sweep on the server.
3. **`wandb.agent(sweep_id, function=train, count=N)`** — the agent pulls hyperparameter combos from the server, calls `train()` once per combo. `train()` must call `wandb.init()`, read `wandb.config.*`, and `wandb.log()` the metric named in config.

Extras covered:
- `wandb.define_metric("val_loss", summary="min")` — makes the run summary keep the best value, not the last.
- **Hyperband early termination** (nb7 sec 8): `"early_terminate": {"type": "hyperband", "min_iter": 3}` — kills clearly-bad runs early, saving compute.
- `wandb.Api()` to query best run programmatically; retrain a final model with best settings.
- Visuals to name in theory answers: **Parallel Coordinates Plot** (lines per run across hyperparam axes -> metric) and **Hyperparameter Importance Plot** (which knob correlates most with the metric).
- grid = exhaustive (only works with discrete `values`); random = independent samples (surprisingly strong baseline); bayes = builds a probabilistic model of metric vs hyperparams and picks the most promising next point.

[back to top](#table-of-contents)

## nb9 — wandb artifacts pipeline versioning

Story: git-for-data-and-models. Every dataset and model becomes a versioned, hash-deduplicated artifact with lineage.

Pipeline demonstrated (MNIST):
```
load_and_log()      -> Artifact "mnist-raw" (type=dataset), one .pt file per split
                       via artifact.new_file(name) inside `with` block; run.log_artifact(art)
preprocess_and_log()-> run.use_artifact("mnist-raw:latest").download()
                       -> Artifact "mnist-preprocessed" (type=dataset) with metadata=steps
train_and_log()     -> uses preprocessed data; saves model; Artifact "trained-model"
                       (type=model) via artifact.add_file("model.pth")
evaluate_and_log()  -> use_artifact("trained-model:latest"), test, wandb.log results
```
Key API distinctions (asked as one-liners):
- `wandb.init(project, job_type=...)` — job_type labels the pipeline stage ("load-data", "train", "eval").
- `artifact.new_file(name)` — create+write a fresh file directly into the artifact; `artifact.add_file(path)` — attach an existing file; `add_dir` for folders.
- `run.log_artifact(art)` — producer side (upload, auto-versions v0, v1... only when content hash changes).
- `run.use_artifact("name:alias")` + `.download()` — consumer side; also records lineage edge.
- Aliases: `latest`, `v0`, custom (`best`). Metadata dict travels with the artifact.
- **Graph view**: runs and artifacts alternate as nodes — full DAG of your pipeline; "exploded" view shows every version.

[back to top](#table-of-contents)

## nb10 — distributed training

Three mini-notebooks:
- **18a data parallel**: `model = nn.DataParallel(model)` when `torch.cuda.device_count() > 1`; a dummy RandomDataset shows input batch 30 splitting into 15+15 on 2 GPUs ("In Model" prints per-GPU shapes, "Outside" prints gathered shape).
- **18b internals**: manual `replicate(module, device_ids)` -> `scatter(input, device_ids)` -> `parallel_apply(replicas, inputs)` -> `gather(outputs, output_device)` — DataParallel is exactly these four.
- **18c model parallel**: ToyModel with `net1.to('cuda:0')`, `net2.to('cuda:1')`, forward moves activations `x.to('cuda:1')` between stages; loss/labels on the output device. Also the pipelining idea: split each batch into micro-batches so both GPUs work simultaneously instead of idling (naive model parallel keeps only one GPU busy at a time).

Contrast line to memorize: **DataParallel = same model everywhere, data split. ModelParallel = model split, data flows through. DP is single-process multi-thread (GIL-bound); DDP (DistributedDataParallel) is the multi-process production upgrade.**

[back to top](#table-of-contents)

## nb11 — torchscript

Story: freeze the model so C++/mobile can run it Python-free.

Content:
- Eager `MyCell(nn.Module)` with `torch.tanh(self.linear(x) + h)`.
- `torch.jit.trace(model, (x, h))` -> TracedModule; inspect `.graph` (low-level IR) and `.code` (readable Python-like).
- Trace limitation demo: a module with `if x.sum() > 0:` — trace bakes in the branch taken during tracing (with a TracerWarning); `torch.jit.script(model)` preserves the real control flow.
- Mixing: a scripted submodule inside a traced module and vice versa is allowed.
- Save/load: `traced.save("model.pt")`, `torch.jit.load("model.pt")`; the .pt archive contains code + parameters, loadable from **C++ (libtorch)** with `torch::jit::load` — the notebook's closing point.

[back to top](#table-of-contents)

## nb12-13 — onnx

Story: the universal interchange format.

- Export: `torch.onnx.export(model, dummy_input, "model.onnx", input_names, output_names, dynamic_axes)` — needs a dummy input because export traces the graph. `dynamic_axes={'input': {0: 'batch'}}` lets batch size vary at runtime.
- Validate: `onnx_model = onnx.load(path)` ; `onnx.checker.check_model(onnx_model)`.
- Run anywhere: `onnxruntime.InferenceSession(path)` ; `sess.run(None, {"input": np_array})` — note **NumPy in, NumPy out** (no torch tensors).
- Visualize: Netron viewer for the graph.
- Why ONNX (theory Q): decouple training framework from serving runtime; hardware-optimized runtimes (ONNX Runtime, TensorRT, OpenVINO, mobile) consume one format.

**Deployment decision tree:**
```
Need to serve the model?
 |- staying in Python + PyTorch -> just state_dict + load
 |- C++/mobile, PyTorch ecosystem -> TorchScript (trace if no branches, script if branches)
 |- non-PyTorch runtime / hardware accel / cross-framework -> ONNX
```

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Theory](DLOPS_EXAM_01_Theory.md) | [Numericals](DLOPS_EXAM_02_Numericals.md) | [Practice](DLOPS_EXAM_03_Practice.md) | [QA Bank](DLOPS_EXAM_05_QA_Bank.md)
