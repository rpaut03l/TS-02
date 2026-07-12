# DLOps Major Exam — Hub (12-Jul-2026)

> Course: DL-Ops (Prof. Bikash Santra, TA Prajwal Kajare) | Mode: Google Form + Meet | Code written in Form directly.

## Navigate
- [DLOPS_EXAM_01_Theory.md](DLOPS_EXAM_01_Theory.md) — every concept, notation, rule, function explained plainly
- [DLOPS_EXAM_02_Numericals.md](DLOPS_EXAM_02_Numericals.md) — shape math, conv output sizes, param counts, worked numbers
- [DLOPS_EXAM_03_Practice.md](DLOPS_EXAM_03_Practice.md) — copy-in-your-head code patterns, Google-Form writing tactics, traps
- [DLOPS_EXAM_04_Notebook_Deep_Dive.md](DLOPS_EXAM_04_Notebook_Deep_Dive.md) — every class notebook explained section by section
- [DLOPS_EXAM_05_QA_Bank.md](DLOPS_EXAM_05_QA_Bank.md) — 60 Q&A covering the entire course, drill-ready

## Module deep dives (one file per class module)
- [DLOPS_01_Intro_PyTorch.md](DLOPS_01_Intro_PyTorch.md) — tensors, autograd, first nn
- [DLOPS_02_Basics_PyTorch_DL.md](DLOPS_02_Basics_PyTorch_DL.md) — activations, losses, optimizer math, save/load
- [DLOPS_03_CNN_Feature_Extraction.md](DLOPS_03_CNN_Feature_Extraction.md) — CIFAR LeNet, features -> RandomForest, joblib
- [DLOPS_04_Datasets_DataLoaders.md](DLOPS_04_Datasets_DataLoaders.md) — built-in datasets, transforms, augmentation
- [DLOPS_05_Custom_Datasets_Training.md](DLOPS_05_Custom_Datasets_Training.md) — ImageFolder, custom Dataset, TinyVGG, loss curves
- [DLOPS_06_TensorBoard.md](DLOPS_06_TensorBoard.md) — SummaryWriter, scalars/graph/PR/hparams, experiment grid
- [DLOPS_07_WandB_Sweeps_Course.md](DLOPS_07_WandB_Sweeps_Course.md) — init/log/sweep/agent, define_metric, hyperband
- [DLOPS_08_WandB_Sweeps_Official.md](DLOPS_08_WandB_Sweeps_Official.md) — official sweep_config grammar, NLL pairing, plots
- [DLOPS_09_WandB_Artifacts.md](DLOPS_09_WandB_Artifacts.md) — data + model versioning pipeline, lineage
- [DLOPS_10_Distributed_Training.md](DLOPS_10_Distributed_Training.md) — DataParallel, RSPG primitives, model parallel, DDP
- [DLOPS_11_TorchScript.md](DLOPS_11_TorchScript.md) — trace vs script, jit save/load, C++
- [DLOPS_12_ONNX.md](DLOPS_12_ONNX.md) — export, checker, Netron, onnxruntime, verification

## Source repo for placement
This DLOps set belongs in **TS-02** (`github.com/rpaut03l/TS-02`), alongside
the existing `GPU Programming` and `MLOps` folders — create a `DLOps/` folder
there and drop all 18 files in flat (no subfolders; relative links assume
same-folder placement).

Nine of the twelve module files also carry a **cross-repo reference** line
near the top, linking out to the matching classical-ML topic folder in
**TS-01** (`github.com/rpaut03l/TS-01/tree/main/ML`) — e.g. module 3's
CNN+RandomForest links to TS-01's `Random-Forest` and `Deep-Learning`
folders. Modules 11 (TorchScript) and 12 (ONNX) are pure deployment
engineering and intentionally have no TS-01 link.

## Course map (what the notebooks covered)

```
1  Intro to PyTorch          --> tensors, autograd, first nn
2  Basics of PyTorch for DL  --> activations, losses, optimizers, save/load
3  CNN + Feature Extraction  --> CIFAR/LeNet-style, features -> RandomForest, joblib
4  Datasets & DataLoaders    --> built-in datasets, transforms, augmentation
5  Custom Datasets           --> ImageFolder, custom Dataset class, full train loop
6  TensorBoard               --> SummaryWriter, scalars, graphs, PR curves, hparams
7  W&B Sweeps (course ver.)  --> wandb.init/log/sweep/agent, define_metric
8  W&B Sweeps (official)     --> sweep_config: method/metric/parameters
9  W&B Artifacts             --> data + model versioning pipeline
10 Distributed Training      --> nn.DataParallel, scatter/gather, model parallel
11 TorchScript               --> trace vs script, jit.save/load
12-13 ONNX                   --> torch.onnx.export, onnxruntime InferenceSession
```

## Suggested reading order
1. Practice file first (code patterns you must write from memory in the Form).
2. Numericals (shape formulas — fastest marks).
3. Theory skim, mnemonics only, then sleep.
