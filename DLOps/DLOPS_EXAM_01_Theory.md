# DLOPS_EXAM_01 — Theory (explained from zero)

[Hub](DLOPS_EXAM_00_Hub.md) | [Numericals](DLOPS_EXAM_02_Numericals.md) | [Practice](DLOPS_EXAM_03_Practice.md)

## table of contents
- [1. tensors — the one box that holds everything](#1-tensors--the-one-box-that-holds-everything)
- [2. autograd — the math that learns](#2-autograd--the-math-that-learns)
- [3. the training loop — 5 sacred steps](#3-the-training-loop--5-sacred-steps)
- [4. activations and losses](#4-activations-and-losses)
- [5. datasets, dataloaders, transforms](#5-datasets-dataloaders-transforms)
- [6. cnn + feature extraction + joblib](#6-cnn--feature-extraction--joblib)
- [7. experiment tracking — tensorboard](#7-experiment-tracking--tensorboard)
- [8. wandb — sweeps and artifacts](#8-wandb--sweeps-and-artifacts)
- [9. distributed training](#9-distributed-training)
- [10. deployment — torchscript and onnx](#10-deployment--torchscript-and-onnx)
- [11. master cheatsheet](#11-master-cheatsheet)
- [12. exam hacks and trap watch](#12-exam-hacks-and-trap-watch)

---

## 1. tensors — the one box that holds everything

Think of a tensor as a lunchbox that can be any size: a single number (0-d), a row of numbers (1-d, vector), a table (2-d, matrix), or a stack of tables (3-d and up, e.g. images).

Notation you must know:
- `shape` = the size of each dimension, e.g. `(64, 3, 32, 32)` = 64 images, 3 color channels, 32x32 pixels. **Order: N, C, H, W** (Number, Channels, Height, Width).
- `dtype` = kind of number (`torch.float32` default for weights, `torch.long` for class labels).
- `device` = where it lives: `cpu`, `cuda`, or `mps` (Mac).

Creation functions:
```
torch.tensor([1,2,3])     # from data
torch.zeros(2,3)          # all 0
torch.ones(2,3)           # all 1
torch.rand(2,3)           # uniform [0,1)
torch.randn(2,3)          # normal, mean 0
torch.arange(0,10,2)      # 0,2,4,6,8
torch.linspace(0,1,5)     # 5 evenly spaced points
torch.from_numpy(arr)     # numpy -> tensor (shares memory!)
```

Rule: `tensor.numpy()` and `torch.from_numpy()` SHARE memory on CPU — change one, the other changes.

**Mnemonic — "NCHW = Never Carry Heavy Watermelons"** for image tensor order.

[back to top](#table-of-contents)

## 2. autograd — the math that learns

Imagine every calculation leaves a breadcrumb trail. `loss.backward()` walks the trail backwards and writes, on every weight, "this is how much you contributed to the error" — that number is the **gradient** stored in `.grad`.

Rules:
1. Only tensors with `requires_grad=True` get gradients.
2. Gradients **accumulate** — that's why we call `optimizer.zero_grad()` every step (or old crumbs pile up).
3. `with torch.no_grad():` or `torch.inference_mode():` = "don't leave breadcrumbs" — used at evaluation, saves memory and time. `inference_mode` is the newer, faster one used in class.
4. `.detach()` cuts a tensor out of the trail.

```
w = torch.randn(3, requires_grad=True)
loss = (w**2).sum()
loss.backward()      # now w.grad exists
```

[back to top](#table-of-contents)

## 3. the training loop — 5 sacred steps

**Mnemonic — "ZFLBS: Zebras Find Lions Before Sunset"**
```
optimizer.zero_grad()        # Z - wipe old gradients
y_pred = model(X)            # F - forward pass
loss = loss_fn(y_pred, y)    # L - compute loss
loss.backward()              # B - backprop (compute grads)
optimizer.step()             # S - update weights
```
(Any order of Z is fine as long as it's before backward; class code does forward, loss, zero, backward, step — both accepted.)

Eval loop differences (three switches, all mandatory):
1. `model.eval()` — turns off dropout, freezes batchnorm stats.
2. `torch.inference_mode()` — no gradient tracking.
3. No `backward()`, no `step()`.
Back to training: `model.train()`.

```
        TRAIN                      EVAL
  model.train()              model.eval()
  grads ON                   inference_mode (grads OFF)
  dropout active             dropout off
  update weights             just measure
```

[back to top](#table-of-contents)

## 4. activations and losses

Activations = the "squish functions" that add non-linearity. Without them, 100 linear layers = 1 linear layer.

| function | formula | output range | use |
|---|---|---|---|
| ReLU | max(0,x) | [0, inf) | hidden layers default |
| LeakyReLU | x if x>0 else 0.01x | (-inf, inf) | fixes dead ReLU |
| Sigmoid | 1/(1+e^-x) | (0,1) | binary output prob |
| Tanh | (e^x-e^-x)/(e^x+e^-x) | (-1,1) | RNN-ish, centered |
| Softmax | e^xi / sum(e^xj) | sums to 1 | multi-class probs |
| GELU | x * Phi(x) | smooth ReLU | transformers |

Losses:
| loss | task | expects |
|---|---|---|
| `nn.MSELoss` | regression | float preds vs float targets |
| `nn.CrossEntropyLoss` | multi-class | **raw logits** + integer labels (it applies log-softmax inside!) |
| `F.nll_loss` | multi-class | log-probs (pair with `nn.LogSoftmax`) |
| `BCEWithLogitsLoss` | binary | raw logit + float 0/1 |

**Trap:** never put `Softmax` before `CrossEntropyLoss` — double softmax kills learning. **CrossEntropy = Softmax + NLL already inside.**

Optimizers seen in class: `SGD` (lr, momentum), `Adam` (adaptive, lr=1e-3 default), `RMSprop`. Mnemonic: **"SAR team rescues gradients" — SGD, Adam, RMSprop.**

[back to top](#table-of-contents)

## 5. datasets, dataloaders, transforms

- **Dataset** = a bookshelf. It knows two things: how many books (`__len__`) and how to hand you book number i (`__getitem__`).
- **DataLoader** = the librarian who brings books in batches, shuffled, using helpers (`num_workers`).

```
Dataset  --(indexing)-->  single (image, label)
DataLoader(dataset, batch_size=32, shuffle=True)
         --(iteration)--> batches of (32 images, 32 labels)
```

Custom Dataset skeleton (asked often):
```
class MyData(Dataset):
    def __init__(self, dir, transform=None): ...
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transform: img = self.transform(img)
        return img, label
```

Transforms (torchvision):
- `ToTensor()` — PIL [0,255] -> float tensor [0,1], HWC -> CHW. Always LAST before Normalize.
- `Normalize(mean, std)` — `(x-mean)/std` per channel.
- `Resize((H,W))`, `RandomHorizontalFlip(p=0.5)`, `TrivialAugmentWide()` (auto-augmentation used in class).
- `Compose([...])` chains them, order matters: Resize -> Augment -> ToTensor -> Normalize.

`ImageFolder(root, transform)` — folder names become class labels automatically. `.classes` gives the list.

**Mnemonic — "RAT-N": Resize, Augment, ToTensor, Normalize** — the transform order.

Rule: shuffle=True for train, shuffle=False for test.

[back to top](#table-of-contents)

## 6. cnn + feature extraction + joblib

CNN idea: instead of every pixel talking to every neuron (dense), a small window (kernel) slides across the image finding patterns — edges first, then shapes, then objects.

```
Input 3x32x32
  -> Conv2d(3,6,5)  -> ReLU -> MaxPool(2,2)   # 6x14x14
  -> Conv2d(6,16,5) -> ReLU -> MaxPool(2,2)   # 16x5x5
  -> Flatten (16*5*5=400)
  -> Linear(400,120) -> Linear(120,84) -> Linear(84,10)
```
(This LeNet-style net is exactly the class notebook 3.)

**Feature extraction hybrid (class special):** run images through the CNN, stop at the second-last layer, take that 84-dim vector as "features", then train a **RandomForestClassifier** (sklearn) on those features. Deep net = feature maker, classic ML = decision maker.

Saving:
- PyTorch model: `torch.save(model.state_dict(), 'm.pth')` then `model.load_state_dict(torch.load('m.pth'))`.
- sklearn model: `joblib.dump(rf, 'rf.joblib')`, `rf = joblib.load('rf.joblib')`.

Rule: prefer saving **state_dict** (just weights) over whole model (fragile, pickles class code).

[back to top](#table-of-contents)

## 7. experiment tracking — tensorboard

TensorBoard = a diary with graphs. You write entries with a `SummaryWriter`, then view at `tensorboard --logdir=runs`.

```
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/exp1')
writer.add_scalar('Loss/train', loss, epoch)         # one line chart
writer.add_scalars('Loss', {'train':a,'val':b}, ep)  # multiple lines
writer.add_graph(model, sample_input)                # model diagram
writer.add_pr_curve('pr', labels, probs, step)       # precision-recall
writer.add_hparams({'lr':0.01}, {'acc':0.9})         # hyperparam table
writer.close()                                       # ALWAYS close
```

**Mnemonic — "SGPH: Scalars, Graph, PR-curve, Hparams"** — the four add_* calls class used.

[back to top](#table-of-contents)

## 8. wandb — sweeps and artifacts

wandb = TensorBoard in the cloud + a robot that tries hyperparameters for you.

Core four:
```
wandb.login()
wandb.init(project='p', config={...})   # start a run
wandb.log({'loss': loss})               # record numbers
wandb.finish()                          # end run
```

**Sweeps (auto hyperparameter search):**
```
sweep_config = {
  'method': 'random',          # or 'grid', 'bayes'
  'metric': {'name':'val_loss', 'goal':'minimize'},
  'parameters': {
     'lr': {'min':1e-4, 'max':1e-1},        # range
     'batch_size': {'values':[16,32,64]},   # choices
     'optimizer': {'values':['adam','sgd']}
  }
}
sweep_id = wandb.sweep(sweep_config, project='p')
wandb.agent(sweep_id, function=train, count=10)
```
- grid = try everything; random = dice rolls; bayes = smart guessing that learns from previous runs.
- Inside `train()`, read values via `wandb.config.lr`.

**Artifacts (versioning data + models):**
```
art = wandb.Artifact('mnist', type='dataset')
art.add_file('data.pt')          # or add_dir
run.log_artifact(art)            # upload, becomes v0, v1...
art = run.use_artifact('mnist:latest')   # download side
path = art.download()
```
Artifacts build a lineage graph: raw data -> preprocessed -> model — full pipeline versioning.

**Mnemonic — "ILAF: Init, Log, Artifact, Finish"** and for sweep config **"MMP: Method, Metric, Parameters."**

[back to top](#table-of-contents)

## 9. distributed training

Two ways to use many GPUs:

1. **Data Parallel** (same model copied to every GPU, data split):
```
model = nn.DataParallel(model)   # that's it
model.to(device)
```
Under the hood 4 verbs: **replicate** the model, **scatter** the batch, **parallel_apply** forward on each, **gather** outputs back to GPU 0.
**Mnemonic — "RSPG: Real SREs Prefer GPUs."**

2. **Model Parallel** (model too big for one GPU — split layers across GPUs):
```
self.part1 = nn.Linear(...).to('cuda:0')
self.part2 = nn.Linear(...).to('cuda:1')
def forward(self,x):
    x = self.part1(x.to('cuda:0'))
    return self.part2(x.to('cuda:1'))   # move activations between GPUs
```

```
DATA PARALLEL                 MODEL PARALLEL
 GPU0 [full model] <- data/2   GPU0 [layers 1-3]
 GPU1 [full model] <- data/2        |  activations flow
 (batch split)                 GPU1 [layers 4-6]
 (model split)
```

Rule: DataParallel splits along **dim 0 (batch)**. Batch 30 on 2 GPUs = 15 each.

[back to top](#table-of-contents)

## 10. deployment — torchscript and onnx

**TorchScript** = freeze your Python model into a portable program that runs without Python (C++ servers, mobile).

Two roads:
- `torch.jit.trace(model, example_input)` — records one actual run. FAST but **misses if/else branches** (only records the path taken).
- `torch.jit.script(model)` — compiles the code itself. Keeps control flow (if, loops).

Rule: **model has if/else or loops depending on data -> use script. Plain feedforward -> trace is fine.**
Save/load: `traced.save('m.pt')`, `torch.jit.load('m.pt')`. Inspect: `.graph`, `.code`.

**ONNX** = universal model passport — export once, run anywhere (onnxruntime, TensorRT, mobile).
```
torch.onnx.export(model, dummy_input, 'model.onnx',
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}})
import onnx; onnx.checker.check_model(onnx.load('model.onnx'))
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx')
out = sess.run(None, {'input': x.numpy()})
```

**Mnemonic — "TS = Trace Simple, Script Smart."** ONNX steps: **"EC-RI: Export, Check, Runtime, Infer."**

[back to top](#table-of-contents)

## 11. master cheatsheet

```
SETUP        device = 'cuda' if torch.cuda.is_available() else 'cpu'
             torch.manual_seed(42); torch.cuda.manual_seed(42)
MODEL        nn.Sequential(nn.Flatten(), nn.Linear(784,128), nn.ReLU(), nn.Linear(128,10))
LOSS         nn.CrossEntropyLoss()  (logits + int labels)
OPTIM        torch.optim.Adam(model.parameters(), lr=1e-3)
LOOP         zero_grad -> forward -> loss -> backward -> step   (ZFLBS)
EVAL         model.eval() + torch.inference_mode()
DATA         DataLoader(ds, batch_size=32, shuffle=True)
TRANSFORM    Compose([Resize, Flip, ToTensor, Normalize])   (RAT-N)
SAVE         torch.save(model.state_dict(),'m.pth') / joblib.dump(rf,'rf.joblib')
TB           SummaryWriter -> add_scalar/graph/pr_curve/hparams -> close  (SGPH)
WANDB        init -> log -> finish; sweep(config)+agent  (ILAF, MMP)
DIST         nn.DataParallel(model)  (RSPG: replicate scatter parallel_apply gather)
DEPLOY       jit.trace (no branches) / jit.script (branches); onnx.export
```

[back to top](#table-of-contents)

## 12. exam hacks and trap watch

- **CrossEntropyLoss wants logits** — adding softmax before it is the classic MCQ trap.
- `optimizer.zero_grad()` missing -> gradients accumulate -> "why is my loss weird" question.
- `model.eval()` vs `torch.no_grad()` are DIFFERENT: eval changes layer behavior; no_grad stops gradient tracking. Full eval needs BOTH.
- `ToTensor()` already scales to [0,1] — Normalize afterwards uses that scale.
- DataParallel output gathers on device 0; loss computed there.
- trace vs script: any question mentioning "control flow / if statement" -> answer script.
- state_dict is an OrderedDict of param tensors, not the model.
- `shuffle=False` for test loader — reproducible eval.
- W&B sweep method question: bayes uses past results; random and grid don't.
- `num_workers` = parallel data loading processes; 0 = main process.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Numericals](DLOPS_EXAM_02_Numericals.md) | [Practice](DLOPS_EXAM_03_Practice.md)
