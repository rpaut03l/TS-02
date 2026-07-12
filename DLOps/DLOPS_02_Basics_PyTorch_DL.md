# DLOPS_02 — Basics of PyTorch for DL (activations, losses, optimizers, save/load)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Intro](DLOPS_01_Intro_PyTorch.md) | [Next: CNN](DLOPS_03_CNN_Feature_Extraction.md)

> Story: module 1 taught you to snap bricks together. This module teaches you to
> train WELL. Think of training as teaching a dog tricks: the **loss** is how
> disappointed you are, the **activation** is how excitable each neuron is, the
> **optimizer** is your teaching strategy (strict repetition vs adaptive coaching),
> and **regularization** is making sure the dog learns the trick, not just YOUR
> living room. Class 2's title said it exactly: "From Building Blocks to Training Well."

> **Cross-repo reference (TS-01):** optimizer math and regularization here build
> directly on TS-01 →
> [Neural-Networks](https://github.com/rpaut03l/TS-01/tree/main/ML/Neural-Networks) |
> [Regression](https://github.com/rpaut03l/TS-01/tree/main/ML/Regression) |
> [Parameter-Estimations-Guide](https://github.com/rpaut03l/TS-01/tree/main/ML/Parameter-Estimations-Guide)

## table of contents
- [1. from hand-written loop to the real api](#1-from-hand-written-loop-to-the-real-api)
- [2. the complete fashionmnist classifier](#2-the-complete-fashionmnist-classifier)
- [3. activation functions — the full zoo](#3-activation-functions--the-full-zoo)
- [4. loss functions — which one when](#4-loss-functions--which-one-when)
- [5. optimizers — the math, worked numbers](#5-optimizers--the-math-worked-numbers)
- [6. learning rate — the most important knob](#6-learning-rate--the-most-important-knob)
- [7. overfitting and regularization toolkit](#7-overfitting-and-regularization-toolkit)
- [8. saving and loading — the full ritual](#8-saving-and-loading--the-full-ritual)
- [9. mnemonics](#9-mnemonics)
- [10. cheatsheet](#10-cheatsheet)
- [11. exam hacks and trap watch](#11-exam-hacks-and-trap-watch)

---

## 1. from hand-written loop to the real api

Class first trained y = wx + b by HAND to show what optimizers hide:

```python
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.01

for epoch in range(100):
    y_pred = w * X + b                 # forward, manual
    loss = ((y_pred - y)**2).mean()    # MSE, manual
    loss.backward()                    # autograd fills w.grad, b.grad
    with torch.no_grad():              # updates must NOT be tracked
        w -= lr * w.grad               # gradient descent step, manual
        b -= lr * b.grad
    w.grad.zero_(); b.grad.zero_()     # manual zeroing (trailing _ = in-place)
```

Then the same thing with the API — spot the 1:1 mapping:

```python
model = nn.Linear(1, 1)                              # holds w and b for you
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    loss = loss_fn(model(X), y)      # forward + loss
    optimizer.zero_grad()            # replaces the two .grad.zero_()
    loss.backward()
    optimizer.step()                 # replaces the two manual -= lines
```

The lesson: **optimizer.step() IS "w -= lr * grad" generalized** — every
optimizer just changes the formula inside step().

[back to top](#table-of-contents)

## 2. the complete fashionmnist classifier

The working base the rest of the course reuses:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 1) DATA — download once, transform to tensors
training_data = datasets.FashionMNIST(root="data", train=True,
                                      download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False,
                                  download=True, transform=ToTensor())

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# 2) MODEL — subclass style (vs Sequential in module 1)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()                       # NEVER skip this line
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 512),  nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        return self.stack(self.flatten(x))       # define the data path

model = NeuralNetwork().to(device)

# 3) LOSS + OPTIMIZER
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 4) LOOP — train/test functions per epoch (see module 1 sec 10)
for t in range(5):
    train_epoch(train_loader, model, loss_fn, optimizer)
    test_epoch(test_loader, model, loss_fn)
```

Why subclass instead of plain Sequential:
- You get a `forward()` where you can branch, reuse layers, print shapes.
- `super().__init__()` registers the module machinery — skipping it breaks
  parameter tracking (and loses you marks).
- `self.stack = nn.Sequential(...)` shows both styles compose.

Batch shapes flowing through (recite in the exam):

```
loader batch:  X (64, 1, 28, 28)   y (64,)  int64 labels 0..9
flatten:       (64, 784)
linear+relu:   (64, 512) -> (64, 512)
final linear:  (64, 10)   <- LOGITS
loss:          CrossEntropyLoss(logits (64,10), labels (64,)) -> scalar
```

[back to top](#table-of-contents)

## 3. activation functions — the full zoo

Why they exist (one-line exam answer): without non-linearity, any stack of
Linear layers collapses algebraically into ONE Linear layer — no curves,
no complex boundaries.

Each function, formula, plot shape, verdict:

```
ReLU        f(x) = max(0, x)
            ____/     kills negatives, cheap, default for hidden layers
            risk: "dying ReLU" — neuron stuck at 0 forever (zero gradient)

LeakyReLU   f(x) = x if x>0 else 0.01x
            ___/      tiny slope for negatives keeps gradient alive
                      (the fix for dying ReLU)

Sigmoid     f(x) = 1/(1+e^-x)      range (0,1)
             _--     use ONLY at output for binary probability;
            /        saturates at both ends -> vanishing gradients in deep nets

Tanh        f(x) = (e^x - e^-x)/(e^x + e^-x)   range (-1,1)
             _--     zero-centered sigmoid; classic in RNNs; still saturates
           _/

GELU        f(x) = x * Phi(x)   (Phi = Gaussian CDF)
            ___/     smooth ReLU; the transformer-era default

Softmax     f(x_i) = e^{x_i} / sum_j e^{x_j}     outputs sum to 1
                     turns a logits VECTOR into a probability distribution;
                     lives INSIDE CrossEntropyLoss — don't add it yourself
```

Code — module vs functional form (both appear in class):

```python
act = nn.ReLU(); y = act(x)          # module form: use inside Sequential
y = torch.relu(x)                    # functional: quick, inside forward()
y = F.leaky_relu(x, 0.01)            # functional with args
y = torch.sigmoid(x); y = torch.tanh(x); y = F.gelu(x)
```

Decision rule:

```
hidden layers   -> ReLU (or GELU if fancy)
binary output   -> Sigmoid (or better: no activation + BCEWithLogitsLoss)
multiclass out  -> nothing (logits) + CrossEntropyLoss
```

**Mnemonic — "RLSTGS: Real Learners Sometimes Try Great Snacks"** = ReLU,
LeakyReLU, Sigmoid, Tanh, GELU, Softmax.

[back to top](#table-of-contents)

## 4. loss functions — which one when

The loss is a single number scoring "how wrong" — the thing backward() differentiates.

| loss | task | inputs it expects | inside it |
|---|---|---|---|
| `nn.MSELoss` | regression | float preds, float targets | mean((p-t)^2) |
| `nn.L1Loss` | regression, robust | floats | mean(abs(p-t)) |
| `nn.CrossEntropyLoss` | multiclass | **raw logits (N,C)** + int labels (N,) | log-softmax + NLL |
| `nn.NLLLoss` / `F.nll_loss` | multiclass | LOG-probabilities | pair with nn.LogSoftmax |
| `nn.BCEWithLogitsLoss` | binary / multilabel | raw logit + float target 0/1 | sigmoid + BCE |

Worked CrossEntropy micro-example (do once by hand):

```python
logits = torch.tensor([[2.0, 0.5, 0.1]])   # one sample, 3 classes
label  = torch.tensor([0])                 # true class = 0
# softmax: e^2=7.39, e^0.5=1.65, e^0.1=1.11 ; sum=10.15
# p(class0) = 7.39/10.15 = 0.728
# CE loss  = -ln(0.728) = 0.317
print(nn.CrossEntropyLoss()(logits, label))   # tensor(0.3170...)
```

Three iron rules:
1. **CrossEntropyLoss eats LOGITS.** Adding Softmax first = classic bug/MCQ.
2. Labels for CE are `int64` class indices, NOT one-hot vectors.
3. "WithLogits" in a loss name always means "raw scores in, activation inside."

**Mnemonic — "CE = Chef's Special: softmax already cooked inside."**

[back to top](#table-of-contents)

## 5. optimizers — the math, worked numbers

All optimizers answer one question: given gradient g, how do I move w?

**1) SGD (Stochastic Gradient Descent)**

```
w = w - lr * g
```

Worked: w=1.0, g=4, lr=0.1 -> w = 1.0 - 0.4 = **0.6**.
"Stochastic" = the gradient comes from a mini-batch, not the whole dataset —
noisy but cheap, and the noise even helps escape shallow traps.

**2) SGD + Momentum**

```
v = beta * v + g          (beta ~ 0.9 ; v starts at 0)
w = w - lr * v
```

Worked, two steps with g=4 each, beta=0.9, lr=0.1:
- step1: v = 0.9*0 + 4 = 4      -> w -= 0.4
- step2: v = 0.9*4 + 4 = 7.6    -> w -= 0.76   <- accelerating!

Intuition: a rolling ball — consistent directions build speed, zig-zag
directions cancel out. Smooths ravines, blasts through small bumps.

**3) Adam (Adaptive Moment Estimation)**

```
m = b1*m + (1-b1)*g          # 1st moment: EMA of gradients   (b1=0.9)
v = b2*v + (1-b2)*g^2        # 2nd moment: EMA of squared g   (b2=0.999)
m_hat = m / (1 - b1^t)       # bias correction (t = step number,
v_hat = v / (1 - b2^t)       #   fixes the zero-start underestimate)
w = w - lr * m_hat / (sqrt(v_hat) + eps)     # eps ~ 1e-8
```

Worked, step t=1, g=2, lr=0.001:
- m = 0.1*2 = 0.2 ; v = 0.001*4 = 0.004
- m_hat = 0.2/0.1 = 2 ; v_hat = 0.004/0.001 = 4
- w -= 0.001 * 2/(2+1e-8) ≈ **0.001** — the step size self-normalizes!

Intuition: momentum (m) + a per-parameter automatic learning rate
(divide by sqrt(v_hat): parameters with big noisy gradients get smaller steps).

**RMSprop** = Adam without the m part (only the v scaling) — appears in the
class sweep configs as the third optimizer choice.

Code:

```python
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
torch.optim.Adam(model.parameters(), lr=1e-3)              # great default
torch.optim.RMSprop(model.parameters(), lr=1e-3)
torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # +L2
```

Comparison ladder:

```
SGD        walks      simple, needs lr tuning, generalizes well when tuned
+Momentum  rolls      faster valleys, smooths noise
RMSprop    adapts     per-param step from squared grads
Adam       rolls+adapts   the "just works" default, lr=1e-3
```

**Mnemonic — "SMA ladder: SGD walks, Momentum rolls, Adam adapts."**

[back to top](#table-of-contents)

## 6. learning rate — the most important knob

```
lr too HIGH:   loss  \/\/\/\  oscillates or explodes to NaN
lr right:      loss  \_____   smooth steady descent
lr too LOW:    loss  ------_  crawls; may look "stuck"
```

- Typical starting points: Adam 1e-3, SGD 1e-2 (with momentum 0.9).
- Sweep it on a LOG scale (1e-4 .. 1e-1) — that's why W&B configs use
  `log_uniform_values` for lr (module 7-8).
- Symptom table for the exam: NaN loss -> lr too high (or exploding grads);
  flat loss from step 0 -> lr too low OR dead ReLUs OR forgot zero_grad.

[back to top](#table-of-contents)

## 7. overfitting and regularization toolkit

Overfitting = the model memorizes the training set's quirks instead of the
underlying pattern. The tell: **train loss falls, validation loss rises.**

```
loss
  |\
  | \____________  train
  |   \    ______
  |    \__/        validation  <- turning point = start of overfitting
  +------------------ epochs      (early stopping would stop HERE)
```

The toolkit, with code, in order of "try first":

```python
# 1. MORE DATA / AUGMENTATION (module 4-5) — best fix, attacks the cause

# 2. DROPOUT — randomly zero p fraction of activations at TRAIN time
self.stack = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(512, 10),
)
# train mode: each forward pass a random 30% of activations become 0
#   (survivors scaled by 1/(1-p) so expected sum is unchanged)
# eval mode (model.eval()): dropout OFF — full network used
# effect: no neuron can rely on a specific partner -> redundant, robust features

# 3. WEIGHT DECAY (L2) — punish large weights inside the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# adds lambda*w to each gradient -> weights shrink toward 0 -> smoother function

# 4. EARLY STOPPING — track val loss, stop after `patience` bad epochs
best, patience, bad = float("inf"), 3, 0
for epoch in range(100):
    val = validate()
    if val < best: best, bad = val, 0; torch.save(model.state_dict(), "best.pth")
    else: bad += 1
    if bad >= patience: break     # restore best.pth afterwards

# 5. SMALLER MODEL — fewer params = less memorization capacity
```

Underfitting is the mirror image — both losses high. Fix by doing the
OPPOSITE: bigger model, longer training, higher lr, LESS regularization.

**Mnemonic — "DEWS-M cools an overheated (overfit) model: Dropout, Early
stopping, Weight decay, Smaller net, More data."**

[back to top](#table-of-contents)

## 8. saving and loading — the full ritual

```python
# --- SAVE (after/best-during training) ---
torch.save(model.state_dict(), "fmnist_model.pth")

# --- LOAD (fresh session / deployment) ---
model = NeuralNetwork()                                   # 1. same class
model.load_state_dict(torch.load("fmnist_model.pth",     # 2. weights in
                                 map_location=device))
model.to(device)                                          # 3. right device
model.eval()                                              # 4. eval mode!

# --- verify with one prediction ---
classes = ["T-shirt","Trouser","Pullover","Dress","Coat",
           "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
x, y = test_data[0]
with torch.no_grad():
    pred = model(x.unsqueeze(0).to(device))               # add batch dim!
    print(f"Predicted: {classes[pred.argmax(1).item()]}, Actual: {classes[y]}")
```

The four-step ritual is graded as a unit: **class -> load_state_dict ->
to(device) -> eval()**. Forgetting eval() means dropout still firing at
inference — predictions become randomly degraded (favorite trap question).

Checkpointing (resume training later) saves more than weights:

```python
torch.save({"epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "loss": loss}, "ckpt.pth")
```

[back to top](#table-of-contents)

## 9. mnemonics

- **RLSTGS** — activation zoo: ReLU, LeakyReLU, Sigmoid, Tanh, GELU, Softmax.
- **"CE = Chef's Special: softmax already cooked inside."**
- **SMA ladder** — SGD walks, Momentum rolls, Adam adapts.
- **DEWS-M** — overfit coolers: Dropout, Early stop, Weight decay, Smaller, More data.
- **"Load ritual: C-L-D-E"** — Class, Load, Device, Eval.
- **ZFLBS** — still the loop: Zero Forward Loss Backward Step.

[back to top](#table-of-contents)

## 10. cheatsheet

```
MODEL CLASS   class Net(nn.Module): super().__init__() ; define layers ; forward()
ACTIVATIONS   hidden=ReLU/GELU | binary out=BCEWithLogits | multi=logits+CE
LOSSES        MSE (regress) | CrossEntropy (multi, LOGITS+int labels)
              | BCEWithLogits (binary) | NLL (log-probs)
SGD           w -= lr*g
MOMENTUM      v = 0.9v + g ; w -= lr*v
ADAM          m,v EMAs -> bias-correct -> w -= lr*m_hat/(sqrt(v_hat)+eps)
LR SIGNS      NaN=too high | flat=too low/dead/no-zero_grad
DROPOUT       nn.Dropout(p) — train only, off at eval, scaled by 1/(1-p)
WEIGHT DECAY  optimizer(..., weight_decay=1e-4)  == L2 penalty
EARLY STOP    patience over val loss; keep best checkpoint
SAVE          torch.save(m.state_dict(), p)
LOAD          Class -> load_state_dict(torch.load(p)) -> to(device) -> eval()
```

[back to top](#table-of-contents)

## 11. exam hacks and trap watch

1. Softmax + CrossEntropyLoss together = double softmax = wrong (the #1 trap).
2. CE labels are int64 indices; float or one-hot -> RuntimeError.
3. Dropout at inference (forgot model.eval()) -> noisy, degraded predictions.
4. `super().__init__()` missing -> "cannot assign module before __init__" error.
5. Momentum beta and Adam beta1 are BOTH ~0.9 — different symbols, same value.
6. Adam bias correction exists because m,v start at 0 (early steps underestimate).
7. weight_decay is L2 regularization by another name — synonyms MCQ.
8. Early stopping monitors VALIDATION loss, never training loss.
9. In-place ops end with underscore: `zero_()`, `add_()` — style question.
10. Manual update must be inside `torch.no_grad()` or you corrupt the graph.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Intro](DLOPS_01_Intro_PyTorch.md) | [Next: CNN](DLOPS_03_CNN_Feature_Extraction.md)
