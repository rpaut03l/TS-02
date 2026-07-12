# DLOPS_10 — Distributed Training (nn.DataParallel, scatter/gather, model parallel)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Artifacts](DLOPS_09_WandB_Artifacts.md) | [Next: TorchScript](DLOPS_11_TorchScript.md)

> Story: one chef (GPU) cooking 30 dishes takes all night. Two strategies to
> use more chefs: **Data Parallel** — clone the full recipe book to every chef
> and split the ORDERS between them (15 dishes each); or **Model Parallel** —
> the recipe itself is too big for one kitchen, so chef 1 does prep, passes
> the tray to chef 2 who does the cooking. Three class mini-notebooks cover:
> the one-liner (18a), what's under its hood (18b), and the split-model
> pattern plus pipelining (18c).

> **Cross-repo reference (TS-01):** scaling considerations here sit on top of
> the fundamentals in TS-01 →
> [Foundations](https://github.com/rpaut03l/TS-01/tree/main/ML/Foundations)

## table of contents
- [1. the two parallelisms — the master diagram](#1-the-two-parallelisms--the-master-diagram)
- [2. notebook 18a — dataparallel, line by line](#2-notebook-18a--dataparallel-line-by-line)
- [3. reading the split — the in/outside print trick](#3-reading-the-split--the-inoutside-print-trick)
- [4. notebook 18b — the four primitives](#4-notebook-18b--the-four-primitives)
- [5. notebook 18c — model parallel, line by line](#5-notebook-18c--model-parallel-line-by-line)
- [6. pipelining — fixing the idle-gpu problem](#6-pipelining--fixing-the-idle-gpu-problem)
- [7. dataparallel vs distributeddataparallel](#7-dataparallel-vs-distributeddataparallel)
- [8. practical how-to and gotchas](#8-practical-how-to-and-gotchas)
- [9. mnemonics](#9-mnemonics)
- [10. cheatsheet](#10-cheatsheet)
- [11. exam hacks and trap watch](#11-exam-hacks-and-trap-watch)

---

## 1. the two parallelisms — the master diagram

```
DATA PARALLEL (speed)                MODEL PARALLEL (size)

      batch (30)                          batch (30, whole)
      /        \                               |
 GPU0[FULL     GPU1[FULL                 GPU0[layers 1-3]
 model copy]   model copy]                     |  activations .to('cuda:1')
   15 samples    15 samples              GPU1[layers 4-6]
      \        /                               |
   gather outputs on GPU0                   output (30)

 model fits on one GPU,               model TOO BIG for one GPU,
 want faster epochs                   split the layers themselves
```

Decision rule to recite: **fits-but-slow -> data parallel; doesn't-fit ->
model parallel; production multi-GPU -> DDP (section 7).**

[back to top](#table-of-contents)

## 2. notebook 18a — dataparallel, line by line

The complete toy setup:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# hyper-simple config
input_size, output_size = 5, 2
batch_size, data_size = 30, 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# a dummy dataset — random tensors, ILG contract from module 5
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len

rand_loader = DataLoader(RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

# a model that PRINTS its per-device batch size
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())
        return output

model = Model(input_size, output_size)

# THE key three lines:
if torch.cuda.device_count() > 1:                       # 1. count GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)                      # 2. wrap
model.to(device)                                        # 3. move to cuda:0
```

Line-by-line of the key three:
1. `torch.cuda.device_count()` — how many CUDA devices exist. The wrap is
   pointless (but harmless) on 1 GPU.
2. `nn.DataParallel(model)` — returns a WRAPPER module. Optional args:
   `device_ids=[0,1]` (which GPUs), `output_device=0` (where to gather).
3. `.to(device)` moves the base copy to cuda:0 — the "home" device from which
   replication happens each forward.

[back to top](#table-of-contents)

## 3. reading the split — the in/outside print trick

The run loop and its famous output:

```python
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```

On 2 GPUs, batch 30:

```
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
```

Read it like the exam wants:
- TWO "In Model" prints per step = forward ran on two replicas simultaneously,
  each seeing 15 (the batch split along **dim 0**).
- ONE "Outside" print with 30 = outputs were GATHERED back into a single
  tensor on the output device (cuda:0). Your training loop never notices.
- 100 samples / batch 30 -> batches of 30,30,30,10; the last splits 5+5.
- On 4 GPUs, 30 -> 8,8,8,6 (ceil split); on 1 GPU or CPU -> one print of 30.

Loss + backward happen on the gathered output on cuda:0 — grads flow back
into the replicas and are summed onto the base copy automatically.

[back to top](#table-of-contents)

## 4. notebook 18b — the four primitives

DataParallel unrolled — 18b performs each hidden step manually:

```python
import torch.nn.parallel as P

replicas = P.replicate(module, device_ids)       # 1. copy model to each GPU
inputs   = P.scatter(input, device_ids)          # 2. chunk batch along dim 0
replicas = replicas[:len(inputs)]                #    (trim if batch < #GPUs)
outputs  = P.parallel_apply(replicas, inputs)    # 3. forward all, in parallel
result   = P.gather(outputs, output_device)      # 4. concat results on target
```

What each primitive really does:
1. **replicate** — broadcasts parameters/buffers, builds N functional copies.
   Happens EVERY forward call (that's part of DP's overhead — remember it).
2. **scatter** — splits along dim 0 and ships each chunk to its GPU.
3. **parallel_apply** — launches each replica's forward in its own CUDA
   stream/thread — the actual parallelism.
4. **gather** — moves all outputs to output_device and cats along dim 0.

```
            replicate
   model ───────────────> [m0 on gpu0] [m1 on gpu1]
            scatter
   batch ───────────────>  [x0 15]      [x1 15]
            parallel_apply  [y0=m0(x0)] [y1=m1(x1)]
            gather
   [y0,y1] ─────────────>  y (30) on gpu0
```

The notebook also demos a practical use of manual placement: run
`nn.Embedding` on one GPU and an RNN on another inside one module — a taste
of model parallel before 18c formalizes it.

**Mnemonic — RSPG: "Real SREs Prefer GPUs" = Replicate, Scatter,
Parallel_apply, Gather.**

[back to top](#table-of-contents)

## 5. notebook 18c — model parallel, line by line

The pattern when the model itself must be split:

```python
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')   # stage 1 lives on 0
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')    # stage 2 lives on 1

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))    # ensure input on stage-1 GPU
        return self.net2(x.to('cuda:1'))            # SHIP activations to stage 2
```

Training loop — one subtlety:

```python
model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))                # input can start anywhere
labels = torch.randn(20, 5).to('cuda:1')            # LABELS on the OUTPUT device!
loss_fn(outputs, labels).backward()                 # autograd crosses GPUs fine
optimizer.step()
```

Line-by-line essentials:
- Layers are pinned to devices AT CONSTRUCTION (`.to('cuda:0')` per layer).
- forward() moves the ACTIVATIONS between stages — `x.to('cuda:1')` is the
  hop; that copy is the communication cost of model parallel.
- Labels must sit on the same device as the OUTPUT (cuda:1) or the loss op
  raises a device-mismatch error — the trap of this notebook.
- backward() just works across devices: autograd records the .to() hop and
  routes gradients back through it.
- One optimizer over model.parameters() covers params on BOTH GPUs.

[back to top](#table-of-contents)

## 6. pipelining — fixing the idle-gpu problem

Naive model parallel's shame, visualized:

```
time ->
GPU0:  [F1.........]                 [F3.........]
GPU1:               [F2.........]                 [F4.........]
        while GPU0 works, GPU1 IDLES, and vice versa — ~50% waste on 2 GPUs
```

The fix — split each batch into MICRO-batches and stream them:

```
time ->
GPU0:  [f1a][f1b][f1c][f1d]
GPU1:       [f2a][f2b][f2c][f2d]     <- stage 2 starts on micro-batch a
                                        while stage 1 works on b: OVERLAP
```

Sketch of the notebook's pipelined forward (structure > syntax):

```python
def forward(self, x):
    splits = iter(x.split(self.split_size, dim=0))   # micro-batches
    s_next = next(splits)
    s_prev = self.seq1(s_next).to('cuda:1')          # prime the pipe
    ret = []
    for s_next in splits:
        ret.append(self.seq2(s_prev))                # GPU1 works on prev...
        s_prev = self.seq1(s_next).to('cuda:1')      # ...while GPU0 does next
    ret.append(self.seq2(s_prev))                    # drain the pipe
    return torch.cat(ret)
```

- `split(split_size, dim=0)` makes the micro-batches; the loop keeps both
  stages busy on DIFFERENT micro-batches simultaneously.
- The notebook's timing comparison shows pipelined MP beating naive MP —
  the quantitative payoff.
- Trade-off to name: tiny micro-batches = more overlap but more per-launch
  overhead (and, in nets with BatchNorm, noisier statistics). There's a sweet
  spot; real systems (GPipe, PipeDream) build on exactly this idea.

[back to top](#table-of-contents)

## 7. dataparallel vs distributeddataparallel

The table every exam and interview loves:

| | nn.DataParallel (DP) | DistributedDataParallel (DDP) |
|---|---|---|
| processes | ONE process, multi-thread | one process PER GPU |
| python GIL | bottlenecked by it | bypassed (separate processes) |
| replication | re-replicates EVERY forward | model built once per process |
| gradient sync | gather to GPU0 (it becomes hot) | all-reduce, balanced |
| machines | single node only | multi-node capable |
| code effort | 1 line | init_process_group, launcher, DistributedSampler |
| verdict | quick experiments | production/serious training |

One-liner to memorize: **DP is a convenience, DDP is the standard; even on a
single machine DDP is faster because it avoids the GIL and per-step
replication.**

[back to top](#table-of-contents)

## 8. practical how-to and gotchas

Checking what you have:

```python
torch.cuda.is_available()      # any GPU?
torch.cuda.device_count()      # how many?
torch.cuda.get_device_name(0)  # what kind?
```

Colab: Runtime -> Change runtime type -> GPU (free tier gives ONE GPU —
DataParallel code still runs, just doesn't split; the "In Model" print shows
the full batch once. Say this if asked "what happens on 1 GPU?").

The state_dict gotcha (real-world classic, worth a mark anywhere):

```python
# DataParallel wraps your model as model.module — keys get 'module.' prefix!
torch.save(model.module.state_dict(), "clean.pth")   # save the INNER module
# or when loading a DP-saved file into a plain model:
state = {k.replace("module.", ""): v for k, v in state.items()}
```

Effective batch semantics: with DP, the DataLoader batch (30) is the GLOBAL
batch; each GPU sees 15. (With DDP each process loads its OWN batch via
DistributedSampler — global batch = per-gpu batch x world_size. Nice
contrast line.)

[back to top](#table-of-contents)

## 9. mnemonics

- **"Clone the cook vs split the recipe"** — data parallel vs model parallel.
- **RSPG — "Real SREs Prefer GPUs"** — Replicate, Scatter, Parallel_apply, Gather.
- **"Split on dim 0, gather on GPU 0."** — DP's two zeros.
- **"Labels live where outputs land"** — model-parallel loss device rule.
- **"Micro-batches keep both kitchens cooking"** — pipelining.
- **"DP = one process and a hot GPU0; DDP = one process per GPU, all equal."**
- **"module. prefix"** — the DP state_dict scar to strip.

[back to top](#table-of-contents)

## 10. cheatsheet

```
DETECT      torch.cuda.is_available() ; torch.cuda.device_count()
DP WRAP     if device_count()>1: model = nn.DataParallel(model)
            model.to('cuda:0')          # base copy on the output device
SPLIT MATH  batch B on G GPUs -> ~B/G each (30 on 2 -> 15+15; on 4 -> 8,8,8,6)
PRIMITIVES  replicate -> scatter(dim 0) -> parallel_apply -> gather(GPU0)
MODEL PAR   layer.to('cuda:k') at build; x.to('cuda:k') between stages in forward
LOSS DEVICE labels.to(output device) before loss_fn
PIPELINE    x.split(micro, dim=0); overlap seq1(next) with seq2(prev); cat results
DDP         1 proc/GPU, all-reduce grads, multi-node, needs DistributedSampler
SAVE        torch.save(model.module.state_dict(), ...)   # strip the wrapper
```

[back to top](#table-of-contents)

## 11. exam hacks and trap watch

1. "In Model" prints N times per step on N GPUs; "Outside" once, full batch —
   interpret-the-output question, guaranteed style.
2. DP splits along dim 0 ONLY — your batch dim must be first.
3. Gather target defaults to device_ids[0] -> GPU0 runs hotter (memory too);
   the classic DP imbalance point.
4. Model-parallel loss: labels on the OUTPUT stage's device, else RuntimeError.
5. DP re-replicates every forward — one reason DDP wins even single-node.
6. GIL: DP threads share one interpreter; DDP processes don't — the systems
   answer for "why is DDP faster?"
7. `model.module` to reach the real model inside a DP wrapper (for saving,
   attribute access).
8. Pipelining trades launch overhead for overlap — "why not micro-batch=1?"
9. batch 30 on 4 GPUs = 8,8,8,6 (ceil then remainder) — quick math check.
10. DataParallel on CPU/1-GPU runs fine, silently unsplit — "is it an error?" No.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Artifacts](DLOPS_09_WandB_Artifacts.md) | [Next: TorchScript](DLOPS_11_TorchScript.md)
