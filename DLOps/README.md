# DLOps — Deep Learning Operations

> M.Tech AI (IIT Jodhpur) · TS-02 · Course: DL-Ops
> Prof. Bikash Santra · TA Prajwal Kajare
> Maintainer: Rohit Patel ([G25AIT2089](mailto:rohitpatel1675@gmail.com)) — CR, TS-02

A from-zero-to-mastery documentation set covering the full DLOps course arc:
PyTorch fundamentals → CNNs → data pipelines → experiment tracking (TensorBoard,
W&B Sweeps, W&B Artifacts) → distributed training → deployment (TorchScript, ONNX).

Every topic has **two layers**: the original class **notebook** (`notebooks/`)
and a **markdown deep-dive** (this folder) — line-by-line code explanations,
ASCII diagrams, worked numericals, mnemonics, cheatsheets, and exam-style
Q&A, written so a beginner can read it once and answer anything asked.

---

## Quick navigation

**Start here:** [DLOPS_EXAM_00_Hub.md](DLOPS_EXAM_00_Hub.md) — the master
index linking every file below, plus the exam-prep set (theory / numericals /
practice / notebook deep-dive / 60-question QA bank).

## Repo structure

```
TS-02/
└── DLOps/
    ├── README.md                          <- you are here
    ├── DLOPS_EXAM_00_Hub.md                master index + exam-prep nav
    ├── DLOPS_EXAM_01_Theory.md
    ├── DLOPS_EXAM_02_Numericals.md
    ├── DLOPS_EXAM_03_Practice.md
    ├── DLOPS_EXAM_04_Notebook_Deep_Dive.md
    ├── DLOPS_EXAM_05_QA_Bank.md
    │
    ├── DLOPS_01_Intro_PyTorch.md
    ├── DLOPS_02_Basics_PyTorch_DL.md
    ├── DLOPS_03_CNN_Feature_Extraction.md
    ├── DLOPS_04_Datasets_DataLoaders.md
    ├── DLOPS_05_Custom_Datasets_Training.md
    ├── DLOPS_06_TensorBoard.md
    ├── DLOPS_07_WandB_Sweeps_Course.md
    ├── DLOPS_08_WandB_Sweeps_Official.md
    ├── DLOPS_09_WandB_Artifacts.md
    ├── DLOPS_10_Distributed_Training.md
    ├── DLOPS_11_TorchScript.md
    ├── DLOPS_12_ONNX.md
    │
    └── notebooks/                          <- original class notebooks
        ├── 01_Intro_To_PyTorch/
        │   └── PyTorch_Tutorial.ipynb
        ├── 02_Basics_of_Pytorch_for_DL/
        │   └── Basics_of_Pytorch_for_DL.ipynb
        ├── 03_CNN_Feature_Extraction/
        │   └── 3_Classification_and_Feature_Extraction_using_CNN.ipynb
        ├── 04_Datasets_and_DataLoaders/
        │   └── 4_Datasets_and_DataLoaders.ipynb
        ├── 05_Custom_Datasets_Training/
        │   └── 5_DLops_Custom_Datasets_and_DataLoaders_Teaching.ipynb
        ├── 06_TensorBoard/
        │   └── 06_DLops_TensorBoard_Experiment_Tracking.ipynb
        ├── 07_WandB_Sweeps_Course/
        │   └── 07_DLops_Hyperparameter_Tuning_with_WandB_Sweeps.ipynb
        ├── 08_WandB_Sweeps_Official/
        │   └── 13c_Organizing_Hyperparameter_Sweeps_in_PyTorch_with_WandB.ipynb
        ├── 09_WandB_Artifacts/
        │   └── 14_Pipeline_Versioning_with_WandB_Artifacts.ipynb
        ├── 10_Distributed_Training/
        │   ├── 18a_data_parallel_tutorial.ipynb
        │   ├── 18b_parallelism_tutorial.ipynb
        │   └── 18c_model_parallel_tutorial.ipynb
        ├── 11_TorchScript/
        │   └── 15_Intro_to_TorchScript_tutorial.ipynb
        └── 12_ONNX/
            └── 16a_intro_onnx.ipynb
```

Each `.md` file's own top-of-file nav links Hub ↔ Prev ↔ Next, so you can
also just start at module 1 and read straight through to module 12.

---

## Module map — notebook ↔ markdown ↔ what it covers

| # | Topic | Notebook | Markdown | Covers |
|---|---|---|---|---|
| 01 | Intro to PyTorch | `01_Intro_To_PyTorch/PyTorch_Tutorial.ipynb` | [DLOPS_01](DLOPS_01_Intro_PyTorch.md) | tensors, autograd, first `nn.Sequential` |
| 02 | Basics for DL | `02_Basics_of_Pytorch_for_DL/Basics_of_Pytorch_for_DL.ipynb` | [DLOPS_02](DLOPS_02_Basics_PyTorch_DL.md) | activations, losses, SGD/Momentum/Adam math, dropout, save/load |
| 03 | CNN + Feature Extraction | `03_CNN_Feature_Extraction/3_Classification_and_Feature_Extraction_using_CNN.ipynb` | [DLOPS_03](DLOPS_03_CNN_Feature_Extraction.md) | CIFAR-10, LeNet-style CNN, CNN→RandomForest hybrid, joblib |
| 04 | Datasets & DataLoaders | `04_Datasets_and_DataLoaders/4_Datasets_and_DataLoaders.ipynb` | [DLOPS_04](DLOPS_04_Datasets_DataLoaders.md) | built-in datasets, transforms, TrivialAugmentWide |
| 05 | Custom Datasets | `05_Custom_Datasets_Training/5_DLops_Custom_Datasets_and_DataLoaders_Teaching.ipynb` | [DLOPS_05](DLOPS_05_Custom_Datasets_Training.md) | ImageFolder, custom `Dataset` class, TinyVGG, loss-curve diagnosis |
| 06 | TensorBoard | `06_TensorBoard/06_DLops_TensorBoard_Experiment_Tracking.ipynb` | [DLOPS_06](DLOPS_06_TensorBoard.md) | SummaryWriter, scalars/graph/PR-curve/hparams, experiment grid |
| 07 | W&B Sweeps (course) | `07_WandB_Sweeps_Course/07_DLops_Hyperparameter_Tuning_with_WandB_Sweeps.ipynb` | [DLOPS_07](DLOPS_07_WandB_Sweeps_Course.md) | init/log/sweep/agent, define_metric, hyperband |
| 08 | W&B Sweeps (official) | `08_WandB_Sweeps_Official/13c_Organizing_Hyperparameter_Sweeps_in_PyTorch_with_WandB.ipynb` | [DLOPS_08](DLOPS_08_WandB_Sweeps_Official.md) | sweep_config grammar, LogSoftmax+NLL pairing, importance plots |
| 09 | W&B Artifacts | `09_WandB_Artifacts/14_Pipeline_Versioning_with_WandB_Artifacts.ipynb` | [DLOPS_09](DLOPS_09_WandB_Artifacts.md) | data + model versioning pipeline, lineage graph |
| 10 | Distributed Training | `10_Distributed_Training/18a_*.ipynb`, `18b_*.ipynb`, `18c_*.ipynb` | [DLOPS_10](DLOPS_10_Distributed_Training.md) | `nn.DataParallel`, scatter/gather internals, model parallel, pipelining |
| 11 | TorchScript | `11_TorchScript/15_Intro_to_TorchScript_tutorial.ipynb` | [DLOPS_11](DLOPS_11_TorchScript.md) | trace vs script, `jit.save`/`jit.load`, C++ (libtorch) |
| 12 | ONNX | `12_ONNX/16a_intro_onnx.ipynb` | [DLOPS_12](DLOPS_12_ONNX.md) | `torch.onnx.export`, checker, Netron, `onnxruntime.InferenceSession` |

> Note: module 10 has three notebooks (a/b/c) because the class split
> DataParallel, its internals, and model parallel into three short notebooks —
> all three are explained together in one markdown file.

---

## How to run the notebooks

### Option A — Google Colab (recommended, zero setup)

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. `File → Upload notebook` → pick any `.ipynb` from `notebooks/<module>/`.
   *(Or, once this repo is pushed, `File → Open notebook → GitHub` and paste
   the repo URL — Colab will list every `.ipynb` in it directly.)*
3. `Runtime → Change runtime type → T4 GPU` for modules 3, 5, 6, 10 (anything
   CNN/training-heavy). CPU is fine for modules 1, 2, 11, 12.
4. `Runtime → Run all`. First cell in each notebook installs anything missing
   (`torchvision`, `wandb`, `tensorboard`, `onnx`, `onnxruntime`) via `!pip install`.
5. Modules 7, 8, 9 (W&B) will prompt `wandb.login()` — paste your API key from
   [wandb.ai/authorize](https://wandb.ai/authorize) once per session.

### Option B — Local (Mac/Linux, Apple Silicon friendly)

```bash
# 1. clone the repo
git clone https://github.com/rpaut03l/TS-02.git
cd TS-02/DLOps

# 2. create an isolated environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. install the shared dependency set
pip install torch torchvision torchaudio
pip install jupyterlab notebook
pip install tensorboard wandb onnx onnxruntime scikit-learn joblib torchinfo

# 4. launch Jupyter and open any notebook
jupyter lab notebooks/
```

Device check inside any notebook:

```python
import torch
device = "cuda" if torch.cuda.is_available() else (
         "mps" if torch.backends.mps.is_available() else "cpu")
print(device)     # "mps" on Apple Silicon, "cuda" with an NVIDIA GPU, else "cpu"
```

Module-specific extras:

| Module | Extra one-time setup |
|---|---|
| 06 TensorBoard | `tensorboard --logdir=runs` in a separate terminal, then open `http://localhost:6006` |
| 07-09 W&B | `wandb login` once in the terminal (or `wandb.login()` in-notebook) |
| 10 Distributed | needs ≥2 GPUs to see real parallelism; runs (unsplit) on 1 GPU/CPU too |
| 11 TorchScript | no extra installs; C++/libtorch section is read-only reference |
| 12 ONNX | `pip install onnx onnxruntime` (already in the block above) |

### Suggested reading order

```
notebook (run it) --> matching DLOPS_0N .md (read it) --> DLOPS_EXAM_05 QA Bank (drill it)
```

Run the notebook first so the code isn't abstract, then read the markdown for
the WHY behind every line, then self-test with the QA bank.

---

## How this fits the repo structure (placement steps)

This folder sits as a sibling to `GPU Programming/` and `MLOps/` inside TS-02:

```bash
cd ~/path/to/TS-02
mkdir -p DLOps/notebooks
# copy all DLOPS_*.md + this README.md into DLOps/
# copy the notebooks/ tree (per the structure above) into DLOps/notebooks/

git add DLOps/
git commit -m "DLOps: add README, module deep-dives, and source notebooks"
git push origin main
```

Rules this repo follows (TS-02 house style):
- All `.md` files live **flat** inside `DLOps/` — no subfolders — because
  their internal links are relative (`DLOPS_02_...md`, not a path).
- Notebooks live in **one level deeper**, `DLOps/notebooks/<module>/`, so they
  don't clutter the markdown listing and Colab's GitHub-open flow still finds
  them fine.
- Anchors and links are GitHub-safe: lowercase, hyphenated, ASCII-only.
- Nine of the twelve module `.md` files also cross-link to matching classical-ML
  topics in **TS-01** (`github.com/rpaut03l/TS-01/tree/main/ML`) — e.g. module 3
  (CNN + RandomForest) links to TS-01's `Random-Forest` and `Deep-Learning`
  folders. Modules 11 (TorchScript) and 12 (ONNX) are pure deployment
  engineering and intentionally have no TS-01 counterpart.

---

## Study system at a glance

```
DLOPS_EXAM_00_Hub.md
   |
   |-- DLOPS_EXAM_01_Theory.md         every concept, notation, rule
   |-- DLOPS_EXAM_02_Numericals.md     conv/param/shape formulas, worked
   |-- DLOPS_EXAM_03_Practice.md       write-from-memory code patterns
   |-- DLOPS_EXAM_04_Notebook_Deep_Dive.md   all 12 notebooks, section by section
   |-- DLOPS_EXAM_05_QA_Bank.md        60 Q&A, drill-ready
   |
   +-- DLOPS_01..12_*.md               one deep file per module (this README's table)
```

Each module file (01–12) independently contains: worked examples, ASCII
diagrams, mnemonics, a compact cheatsheet, and a 10-item exam-trap watch —
built to stand alone if you only need one topic, and to link together if you
read the whole set front to back.

---
[Back to Hub](DLOPS_EXAM_00_Hub.md)
