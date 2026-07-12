# DLOPS_01 — Intro to PyTorch (tensors, autograd, first nn)

[Hub](DLOPS_EXAM_00_Hub.md) | [Next: Basics for DL](DLOPS_02_Basics_PyTorch_DL.md)

> Story first: imagine PyTorch as a LEGO factory. **Tensors** are the bricks.
> **Autograd** is a helper who watches every brick you snap together and can tell
> you, afterwards, exactly which brick to nudge to make the final build better.
> **nn.Module** is the instruction booklet that groups bricks into reusable models.
> Master these three and every later module (CNNs, sweeps, deployment) is just
> arranging the same bricks in fancier shapes.

> **Cross-repo reference (TS-01):** the classical-ML groundwork behind autograd
> and "first nn" lives in TS-01 →
> [Neural-Networks](https://github.com/rpaut03l/TS-01/tree/main/ML/Neural-Networks) |
> [Parameter-Estimations-Guide](https://github.com/rpaut03l/TS-01/tree/main/ML/Parameter-Estimations-Guide)

## table of contents
- [1. setup and how to run](#1-setup-and-how-to-run)
- [2. tensors — creation](#2-tensors--creation)
- [3. tensor attributes and dtypes](#3-tensor-attributes-and-dtypes)
- [4. indexing, reshaping, joining](#4-indexing-reshaping-joining)
- [5. tensor math — elementwise vs matmul](#5-tensor-math--elementwise-vs-matmul)
- [6. numpy bridge — the shared memory trap](#6-numpy-bridge--the-shared-memory-trap)
- [7. devices — cpu, cuda, mps](#7-devices--cpu-cuda-mps)
- [8. autograd — full walkthrough](#8-autograd--full-walkthrough)
- [9. first neural network](#9-first-neural-network)
- [10. first training loop, line by line](#10-first-training-loop-line-by-line)
- [11. saving and loading](#11-saving-and-loading)
- [12. mnemonics](#12-mnemonics)
- [13. cheatsheet](#13-cheatsheet)
- [14. exam hacks and trap watch](#14-exam-hacks-and-trap-watch)

---

## 1. setup and how to run

Practical instructions (Colab or local):

```bash
# Colab: nothing to install, torch is preinstalled. Check version:
python -c "import torch; print(torch.__version__)"

# Local (Mac M1/M2 or Linux):
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision
```

Every notebook in this module starts with the same header — memorize it:

```python
import torch                      # the core library
import numpy as np                # for the numpy bridge demos
print(torch.__version__)          # sanity check
```

Line by line:
- `import torch` loads the tensor library, autograd engine, and nn namespace.
- Printing the version matters in DLOps: reproducibility starts with pinning versions.

[back to top](#table-of-contents)

## 2. tensors — creation

A tensor is an n-dimensional grid of numbers. Dimension count is called **rank**:

```
rank 0  scalar        7
rank 1  vector        [1, 2, 3]
rank 2  matrix        [[1,2],[3,4]]
rank 3  e.g. image    C x H x W
rank 4  batch of imgs N x C x H x W
```

All the creation functions, each with what it gives you:

```python
t = torch.tensor([[1, 2], [3, 4]])   # from Python data; dtype inferred (int64 here)
z = torch.zeros(2, 3)                # 2x3 of 0.0  (float32)
o = torch.ones(2, 3)                 # 2x3 of 1.0
e = torch.eye(3)                     # 3x3 identity matrix (1s on diagonal)
r = torch.rand(2, 3)                 # uniform random in [0, 1)
n = torch.randn(2, 3)                # normal (Gaussian), mean 0, std 1
ri = torch.randint(0, 10, (2, 3))    # random ints in [0, 10)
a = torch.arange(0, 10, 2)           # tensor([0, 2, 4, 6, 8]) — start, stop, step
l = torch.linspace(0, 1, 5)          # 5 evenly spaced: 0.00, 0.25, 0.50, 0.75, 1.00
```

"Like" variants copy shape+dtype of an existing tensor:

```python
torch.zeros_like(t)     # zeros, same shape/dtype as t
torch.ones_like(t)      # ones,  same shape/dtype as t
torch.rand_like(t, dtype=torch.float)  # override dtype when source is int
```

Explanation of each line:
- `torch.tensor(data)` COPIES the data. Nested lists must be rectangular.
- `rand` vs `randn`: rand is a fair dice in [0,1); randn is a bell curve centered at 0.
  Weights are usually initialized with randn-flavored schemes (small, centered).
- `arange` excludes the stop value (like Python range); `linspace` INCLUDES both ends.
- The `_like` family exists so you never hand-type shapes twice (fewer bugs).

**Mnemonic — "ZORRA-EL": Zeros, Ones, Rand, Randn, Arange, Eye, Linspace** — the seven creators.

[back to top](#table-of-contents)

## 3. tensor attributes and dtypes

Every tensor carries three ID cards. Print them constantly while debugging:

```python
x = torch.rand(3, 4)
print(x.shape)    # torch.Size([3, 4]) — size of each dimension
print(x.dtype)    # torch.float32     — the number type
print(x.device)   # cpu               — where it physically lives
```

dtype table (know these four):

| dtype | used for | notes |
|---|---|---|
| `torch.float32` | weights, activations | the default float |
| `torch.float16` | mixed precision | half memory, faster on GPU |
| `torch.int64` (`long`) | class labels | CrossEntropyLoss requires this |
| `torch.bool` | masks | from comparisons like `x > 0` |

Conversions:

```python
x.type(torch.float16)     # returns a converted copy
x.to(torch.int64)         # .to also converts dtype, not just device
```

Rule: **shape errors and dtype errors cause 90% of beginner exceptions.**
When anything breaks, the first debugging line is:

```python
print(x.shape, x.dtype, x.device)
```

[back to top](#table-of-contents)

## 4. indexing, reshaping, joining

Indexing works like NumPy — rows first, then columns:

```python
t = torch.arange(12).reshape(3, 4)
t[0]        # first row              -> shape (4,)
t[:, 0]     # first column           -> shape (3,)
t[1, 2]     # single element (scalar tensor)
t[..., -1]  # last column, any rank  ('...' = all leading dims)
t[t > 5]    # boolean mask selection -> 1-D tensor of matches
```

Reshaping family — same data, new view:

```python
t.reshape(4, 3)        # any compatible shape (may copy if needed)
t.view(4, 3)           # like reshape but REQUIRES contiguous memory
t.reshape(-1)          # -1 = "you compute this dim" -> flatten to (12,)
t.unsqueeze(0)         # add a dim of size 1 at position 0: (3,4)->(1,3,4)
t.squeeze()            # remove ALL size-1 dims
t.permute(1, 0)        # reorder dims (transpose for 2-D)
```

Why unsqueeze matters: models expect **batches**. One image (3,64,64) must
become (1,3,64,64) before `model(img)` — `img.unsqueeze(0)` does exactly that.

Joining:

```python
torch.cat([a, b], dim=0)    # glue along an EXISTING dim (rows if dim=0)
torch.stack([a, b], dim=0)  # create a NEW dim (two (3,4) -> one (2,3,4))
```

```
cat  dim=0:   [A]      stack dim=0:  [ [A] ]
              [B]                    [ [B] ]   <- new outer axis
```

**Mnemonic — "Cat joins the row, Stack builds a shelf."**

[back to top](#table-of-contents)

## 5. tensor math — elementwise vs matmul

Two multiplication worlds — the #1 conceptual MCQ:

```python
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[10., 20.], [30., 40.]])

a * b            # ELEMENTWISE: [[10, 40],[90,160]]  (same as torch.mul)
a @ b            # MATRIX PRODUCT (same as torch.matmul)
                 # [[1*10+2*30, 1*20+2*40],[3*10+4*30, 3*20+4*40]]
                 # = [[70, 100],[150, 220]]
```

Shape rules:
- Elementwise: shapes equal OR broadcastable (a (3,1) meets a (1,4) -> (3,4)).
- Matmul: **(a,b) @ (b,c) = (a,c)** — inner dims must match, they cancel out.

Broadcasting in one picture:

```
(3,1)   [x]      (1,4)  [a b c d]        result (3,4)
        [y]  op                      =   [xa xb xc xd]
        [z]                              [ya yb yc yd]
                                         [za zb zc zd]
```

Reductions and friends you saw in class:

```python
x.sum(), x.mean(), x.max(), x.min()
x.sum(dim=0)         # collapse rows -> per-column sums
x.argmax(dim=1)      # index of max along dim 1 (used for predictions!)
```

Rule: `dim=k` means "the k-th dimension DISAPPEARS in the result."

[back to top](#table-of-contents)

## 6. numpy bridge — the shared memory trap

```python
arr = np.ones(3)
t = torch.from_numpy(arr)   # numpy -> tensor,  SHARES memory (CPU)
back = t.numpy()            # tensor -> numpy,  SHARES memory (CPU)

arr += 1
print(t)     # tensor([2., 2., 2.])  <- t changed too!
```

Line by line:
- `from_numpy` doesn't copy; both objects point at the same bytes.
- Mutating either mutates both — great for speed, dangerous if you forget.
- Need independence? `torch.tensor(arr)` (copies) or `t.clone()`.
- GPU tensors can't `.numpy()` directly: `t.cpu().numpy()` first.
- Tensors with gradients need `t.detach().cpu().numpy()`.

**Mnemonic — "from_numpy = roommates (shared fridge); torch.tensor = new house (own fridge)."**

[back to top](#table-of-contents)

## 7. devices — cpu, cuda, mps

```python
# classic device-agnostic line (write this in EVERY exam answer):
device = "cuda" if torch.cuda.is_available() else "cpu"

# Mac Apple Silicon alternative:
device = "mps" if torch.backends.mps.is_available() else "cpu"

# newest API shown in class notebook 1:
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()

x = x.to(device)          # returns a NEW tensor on that device
model.to(device)          # moves module params IN-PLACE (module is special)
```

Rules:
1. All tensors in one operation must be on the SAME device, or RuntimeError.
2. `.to(device)` on a tensor returns a copy — reassign it (`x = x.to(device)`).
3. On a model, `.to` is in-place; reassignment optional but conventional.

```
CPU  [x] --.to('cuda')-->  GPU [x']    (copy travels over PCIe)
model.to(device)  moves ALL its weights at once
```

[back to top](#table-of-contents)

## 8. autograd — full walkthrough

The mental model: every op on a `requires_grad=True` tensor adds a node to an
invisible **computation graph**. `backward()` walks the graph in reverse,
applying the chain rule, depositing d(output)/d(leaf) into each leaf's `.grad`.

Complete worked example (do this by hand once — exams love it):

```python
w = torch.tensor(3.0, requires_grad=True)   # a leaf we want to tune
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)                       # data: no grad needed

y = w * x + b        # y = 3*2 + 1 = 7 ; graph:  w --*x--> (+b) --> y
loss = (y - 10)**2   # loss = (7-10)^2 = 9

loss.backward()      # chain rule time
print(w.grad)        # d(loss)/dw = 2*(y-10)*x = 2*(-3)*2 = -12
print(b.grad)        # d(loss)/db = 2*(y-10)*1 = -6
```

Every rule you must recite:

1. Only leaves created with `requires_grad=True` collect `.grad`.
2. **Gradients ACCUMULATE**: a second `backward()` adds on top. Hence
   `optimizer.zero_grad()` (or `w.grad.zero_()`) each iteration.
3. `backward()` needs a SCALAR (a single number). Losses are scalars by design.
4. Turn tracking off for inference:
   ```python
   with torch.no_grad():          # classic
       preds = model(x)
   with torch.inference_mode():   # newer, faster — preferred in class
       preds = model(x)
   ```
5. `y.detach()` returns a tensor cut out of the graph (no history).
6. The class also showed a functional loss with logits:
   ```python
   loss = torch.nn.functional.binary_cross_entropy_with_logits(z, target)
   ```
   "with_logits" = it applies sigmoid internally — feed RAW scores.

Graph picture:

```
   w --------\
              (mul) ---> (add) ---> y ---> (sub 10) ---> (square) ---> loss
   x --------/            ^
   b ---------------------/
   backward(): loss ==> gradients flow RIGHT-to-LEFT into w.grad, b.grad
```

**Mnemonic — "GAS ZeD": Gradients Accumulate, So Zero, Dear."**

[back to top](#table-of-contents)

## 9. first neural network

Class built this on FashionMNIST (28x28 grayscale, 10 clothing classes):

```python
from torch import nn

model = nn.Sequential(
    nn.Flatten(),                 # (N,1,28,28) -> (N,784) : unroll pixels
    nn.Linear(28*28, 512),        # 784 inputs -> 512 neurons (785*512 params)
    nn.ReLU(),                    # non-linearity: max(0,x)
    nn.Linear(512, 512),          # hidden -> hidden
    nn.ReLU(),
    nn.Linear(512, 10),           # -> 10 raw scores (LOGITS), one per class
).to(device)
```

Line by line:
- `nn.Sequential` chains layers; data flows top to bottom.
- `Flatten` keeps the batch dim, flattens the rest: exactly `torch.flatten(x,1)`.
- `Linear(in, out)` computes `x @ W.T + b`; params = (in+1)*out.
- ReLU between Linears is what makes it more than one big linear map.
- The LAST layer outputs logits — no Softmax here, because CrossEntropyLoss
  handles that internally (see module 2).

Turning logits into a prediction:

```python
logits = model(X)                          # (N, 10)
probs  = torch.softmax(logits, dim=1)      # rows sum to 1 (only for humans)
pred   = probs.argmax(dim=1)               # class index per sample
```

Peeking inside a model:

```python
for name, p in model.named_parameters():
    print(name, p.shape)     # e.g. 0.weight torch.Size([512, 784]) ...
```

[back to top](#table-of-contents)

## 10. first training loop, line by line

```python
loss_fn = nn.CrossEntropyLoss()                       # logits + int labels
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train_epoch(loader, model, loss_fn, optimizer):
    model.train()                                     # dropout/bn -> train mode
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)             # data to same device
        pred = model(X)                               # FORWARD
        loss = loss_fn(pred, y)                       # LOSS (scalar)
        optimizer.zero_grad()                         # ZERO old grads
        loss.backward()                               # BACKWARD (fill .grad)
        optimizer.step()                              # STEP: w -= lr * grad
        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}")         # .item() -> python float

def test_epoch(loader, model, loss_fn):
    model.eval()                                      # eval mode
    correct, total_loss = 0, 0
    with torch.no_grad():                             # no graph building
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    print(f"acc: {correct/len(loader.dataset):.3f}")
```

Why every line exists:
- `model.train()` / `model.eval()`: flips behavior of Dropout/BatchNorm.
- ZFLBS order inside the loop (zero can come before forward too — equivalent).
- `.item()` converts a 0-dim tensor to a plain Python number for printing —
  also DETACHES it, so your log list doesn't secretly hold the whole graph.
- `(pred.argmax(1) == y)` -> bool tensor; `.sum().item()` -> count of correct.

[back to top](#table-of-contents)

## 11. saving and loading

```python
# save ONLY the weights (recommended):
torch.save(model.state_dict(), "model.pth")

# load: rebuild the architecture first, then fill weights:
model = build_same_model()
model.load_state_dict(torch.load("model.pth"))
model.eval()                       # don't forget before inference!
```

- `state_dict()` = OrderedDict {param_name: tensor}. Portable, small, safe.
- Saving the whole model object (`torch.save(model, ...)`) pickles your class —
  breaks if code moves/renames. Exam answer: **prefer state_dict**.
- `torch.load(..., map_location="cpu")` loads GPU-saved weights onto CPU boxes.

[back to top](#table-of-contents)

## 12. mnemonics

- **ZORRA-EL** — the seven tensor creators (Zeros Ones Rand Randn Arange Eye Linspace).
- **"Cat joins the row, Stack builds a shelf"** — cat=existing dim, stack=new dim.
- **"from_numpy = roommates, torch.tensor = new house"** — shared vs copied memory.
- **GAS ZeD** — Gradients Accumulate So Zero, Dear.
- **ZFLBS** — Zero, Forward, Loss, Backward, Step.
- **SDD** — the three ID cards: Shape, Dtype, Device (print them when debugging).

[back to top](#table-of-contents)

## 13. cheatsheet

```
CREATE     tensor/zeros/ones/rand/randn/randint/arange/linspace/eye/_like
INSPECT    x.shape  x.dtype  x.device            (SDD)
RESHAPE    reshape/view/flatten(x,1)/unsqueeze(0)/squeeze/permute
JOIN       cat(dim existing)  stack(new dim)
MATH       * elementwise | @ matmul (a,b)@(b,c)=(a,c) | sum/mean/argmax(dim)
BRIDGE     torch.from_numpy(a) shares | torch.tensor(a) copies | t.cpu().numpy()
DEVICE     device = "cuda" if torch.cuda.is_available() else "cpu"; x=x.to(device)
AUTOGRAD   requires_grad=True -> loss.backward() -> w.grad ; zero_grad each step
NO-GRAD    with torch.inference_mode():   (eval-time)
MODEL      nn.Sequential(Flatten, Linear, ReLU, ..., Linear->logits)
PREDICT    model(X).argmax(dim=1)
SAVE/LOAD  torch.save(m.state_dict(),p) / m.load_state_dict(torch.load(p)); m.eval()
```

[back to top](#table-of-contents)

## 14. exam hacks and trap watch

1. `view` fails on non-contiguous tensors (e.g. after permute) — `reshape` is safe.
2. `backward()` twice without `retain_graph=True` -> RuntimeError (graph freed).
3. Forgetting `zero_grad` -> loss decreases weirdly then explodes: accumulate bug.
4. Integer tensors can't require grad — autograd is floats only.
5. `t.numpy()` on CUDA tensor -> error; chain `.detach().cpu().numpy()`.
6. `argmax(dim=1)` for row-wise predictions; `dim=0` is the column-wise trap answer.
7. Single image into a model: `img.unsqueeze(0)` first — models eat batches.
8. `torch.tensor(existing_tensor)` warns; use `clone().detach()` instead.
9. arange excludes stop; linspace includes it — off-by-one MCQ bait.
10. `.item()` only works on 1-element tensors — use `.tolist()` otherwise.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Next: Basics for DL](DLOPS_02_Basics_PyTorch_DL.md)
