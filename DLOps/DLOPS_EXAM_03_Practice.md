# DLOPS_EXAM_03 — Practice (write-from-memory code + Google Form tactics)

[Hub](DLOPS_EXAM_00_Hub.md) | [Theory](DLOPS_EXAM_01_Theory.md) | [Numericals](DLOPS_EXAM_02_Numericals.md)

## table of contents
- [1. writing code in a google form — tactics](#1-writing-code-in-a-google-form--tactics)
- [2. pattern A — full training script](#2-pattern-a--full-training-script)
- [3. pattern B — custom dataset class](#3-pattern-b--custom-dataset-class)
- [4. pattern C — cnn class definition](#4-pattern-c--cnn-class-definition)
- [5. pattern D — tensorboard logging](#5-pattern-d--tensorboard-logging)
- [6. pattern E — wandb sweep](#6-pattern-e--wandb-sweep)
- [7. pattern F — save load deploy](#7-pattern-f--save-load-deploy)
- [8. pattern G — feature extraction + randomforest](#8-pattern-g--feature-extraction--randomforest)
- [9. python-in-form cheatsheet](#9-python-in-form-cheatsheet)
- [10. last-hour drill](#10-last-hour-drill)

---

## 1. writing code in a google form — tactics

You type into a plain text box: no autocomplete, no execution, no syntax check. So:

1. **Indent with 2 or 4 spaces consistently** — Google Form keeps spaces; never mix tabs.
2. **Write imports first, always** — graders scan for them: `import torch`, `import torch.nn as nn`, `from torch.utils.data import Dataset, DataLoader`.
3. **Skeleton first, fill later**: write the class/def lines and the 5-step loop before details, so even unfinished answers show structure.
4. **Comment your intent** — `# forward pass`, `# backprop` — cheap marks, proves understanding.
5. Draft on paper/notepad? No — form only, per rules. Type slowly, re-read once.
6. If unsure of an exact argument name, pick the most common (`lr=`, `batch_size=`) — partial credit beats blank.

[back to top](#table-of-contents)

## 2. pattern A — full training script

The one pattern that answers 50% of code questions. Memorize verbatim.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128), nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)              # forward
        loss = loss_fn(y_pred, y)      # loss
        optimizer.zero_grad()          # zero
        loss.backward()                # backward
        optimizer.step()               # step

    model.eval()
    correct = 0
    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
    print(f"epoch {epoch}: acc {correct/len(test_loader.dataset):.4f}")
```

**Mnemonic for the inner loop — ZFLBS ("Zebras Find Lions Before Sunset")** — even if you write F,L,Z,B,S order like class, both are correct.

[back to top](#table-of-contents)

## 3. pattern B — custom dataset class

```python
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = list(Path(root).glob("*/*.jpg"))
        self.transform = transform
        self.classes = sorted({p.parent.name for p in self.paths})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.class_to_idx[self.paths[idx].parent.name]
        if self.transform:
            img = self.transform(img)
        return img, label
```

**Mnemonic — "ILG: Init, Len, Getitem"** — the three mandatory methods. Forget one = broken Dataset.

Transforms + loaders:
```python
from torchvision import transforms
train_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(0.5),   # or TrivialAugmentWide()
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
test_loader  = DataLoader(ts, batch_size=32, shuffle=False)
```

[back to top](#table-of-contents)

## 4. pattern C — cnn class definition

```python
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)          # keep batch dim
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)               # raw logits, NO softmax
```
Trap: `super().__init__()` line is graded — don't skip. `flatten(x, 1)` starts flattening at dim 1 (spares batch).

[back to top](#table-of-contents)

## 5. pattern D — tensorboard logging

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/exp1")
for epoch in range(epochs):
    ...
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalars("Acc", {"train": tr_acc, "val": va_acc}, epoch)
writer.add_graph(model, torch.randn(1, 3, 32, 32).to(device))
writer.add_hparams({"lr": 1e-3, "bs": 32}, {"final_acc": acc})
writer.close()
# view: tensorboard --logdir=runs
```

[back to top](#table-of-contents)

## 6. pattern E — wandb sweep

```python
import wandb
wandb.login()

sweep_config = {
    "method": "random",                       # grid | random | bayes
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr":         {"min": 1e-4, "max": 1e-1},
        "batch_size": {"values": [16, 32, 64]},
        "epochs":     {"value": 5},
        "optimizer":  {"values": ["adam", "sgd"]},
    },
}

def train():
    with wandb.init() as run:
        cfg = wandb.config
        # build loaders with cfg.batch_size, optimizer with cfg.lr ...
        for epoch in range(cfg.epochs):
            ...
            wandb.log({"val_loss": val_loss, "epoch": epoch})

sweep_id = wandb.sweep(sweep_config, project="dlops-major")
wandb.agent(sweep_id, function=train, count=10)
```

Artifacts mini-pattern:
```python
run = wandb.init(project="p", job_type="train")
art = wandb.Artifact("model", type="model")
art.add_file("model.pth")
run.log_artifact(art)
# consume:
art = run.use_artifact("model:latest"); path = art.download()
```

**Mnemonics: MMP (Method, Metric, Parameters) for sweep config; ILAF (Init, Log, Artifact, Finish).**

[back to top](#table-of-contents)

## 7. pattern F — save load deploy

```python
# weights
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth")); model.eval()

# TorchScript
traced = torch.jit.trace(model, torch.randn(1, 3, 32, 32))   # no branches
scripted = torch.jit.script(model)                           # keeps if/loops
traced.save("model_ts.pt")
m = torch.jit.load("model_ts.pt")

# ONNX
torch.onnx.export(model, torch.randn(1, 3, 32, 32), "model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}})
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx")
out = sess.run(None, {"input": x.numpy()})
```

[back to top](#table-of-contents)

## 8. pattern G — feature extraction + randomforest

Class notebook 3's signature move — likely question.

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

def extract_features(loader):
    feats, labels = [], []
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            x = model.pool(F.relu(model.conv1(X.to(device))))
            x = model.pool(F.relu(model.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(model.fc1(x))
            x = F.relu(model.fc2(x))       # 84-dim features
            feats.append(x.cpu()); labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

Xtr, ytr = extract_features(train_loader)
rf = RandomForestClassifier(n_estimators=100).fit(Xtr, ytr)
joblib.dump(rf, "rf.joblib")
rf = joblib.load("rf.joblib")
print(rf.score(*extract_features(test_loader)))
```

[back to top](#table-of-contents)

## 9. python-in-form cheatsheet

```
IMPORT BLOCK (paste mentally at top of every answer)
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from torch.utils.data import Dataset, DataLoader
  from torchvision import datasets, transforms

DEVICE LINE   device = "cuda" if torch.cuda.is_available() else "cpu"
SEED          torch.manual_seed(42)
MOVE          model.to(device); X, y = X.to(device), y.to(device)
ACCURACY      (preds.argmax(1) == y).sum().item() / len(y)
SHAPES        x.shape, x.dtype, x.device
GRAD OFF      with torch.inference_mode():
F-STRING      print(f"loss: {loss.item():.4f}")
LIST COMP     [f(x) for x in xs]
DICT          {c: i for i, c in enumerate(classes)}
```

Mnemonic stack:
- **ZFLBS** — training loop order
- **ILG** — Dataset methods
- **RAT-N** — transform order
- **SGPH** — TensorBoard adds
- **ILAF / MMP** — wandb run / sweep config
- **RSPG** — DataParallel internals
- **TS = Trace Simple, Script Smart** — deployment choice
- **I Kicked 2 Penguins, So 1 cried** — conv output formula

[back to top](#table-of-contents)

## 10. last-hour drill

30-minute pair drill:
1. (10 min) Each writes Pattern A from blank memory; swap; the other marks the 5 loop steps + eval switches.
2. (5 min) Quiz each other the 8 mnemonics — expand each fully.
3. (5 min) Two shape numericals each: conv formula + param count (make up K, P, S).
4. (5 min) One writes Dataset class skeleton, the other writes sweep_config from memory.
5. (5 min) Rapid-fire traps: CE wants logits? trace vs script? eval switches? state_dict vs model? Normalize(0.5,0.5) range?

Then close the laptops and sleep — recall beats re-reading.

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Theory](DLOPS_EXAM_01_Theory.md) | [Numericals](DLOPS_EXAM_02_Numericals.md)
