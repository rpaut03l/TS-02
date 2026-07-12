# DLOPS_03 — CNN + Feature Extraction (CIFAR/LeNet, features -> RandomForest, joblib)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Basics](DLOPS_02_Basics_PyTorch_DL.md) | [Next: Datasets](DLOPS_04_Datasets_DataLoaders.md)

> Story: a dense network looks at an image the way you'd read a book with all
> pages shredded and mixed — every pixel connects to everything, position lost.
> A CNN instead slides a small magnifying glass (the **kernel**) across the
> image: same glass everywhere, so a cat's ear is a cat's ear in any corner.
> Then this module's signature twist: chop off the CNN's head, keep the body as
> a **feature factory**, and let a RandomForest do the final deciding.

> **Cross-repo reference (TS-01):** the RandomForest half of this module and its
> feature-space thinking connect straight to TS-01 →
> [Random-Forest](https://github.com/rpaut03l/TS-01/tree/main/ML/Random-Forest) |
> [Deep-Learning](https://github.com/rpaut03l/TS-01/tree/main/ML/Deep-Learning) |
> [Feature-Selection-DimRed](https://github.com/rpaut03l/TS-01/tree/main/ML/Feature-Selection-DimRed)

## table of contents
- [1. why convolutions beat dense layers on images](#1-why-convolutions-beat-dense-layers-on-images)
- [2. conv2d anatomy — every argument](#2-conv2d-anatomy--every-argument)
- [3. pooling](#3-pooling)
- [4. the class network, shape by shape](#4-the-class-network-shape-by-shape)
- [5. data: cifar-10 pipeline](#5-data-cifar-10-pipeline)
- [6. training and evaluating, line by line](#6-training-and-evaluating-line-by-line)
- [7. feature extraction — chopping the head](#7-feature-extraction--chopping-the-head)
- [8. randomforest on deep features](#8-randomforest-on-deep-features)
- [9. persistence: torch.save vs joblib](#9-persistence-torchsave-vs-joblib)
- [10. mnemonics](#10-mnemonics)
- [11. cheatsheet](#11-cheatsheet)
- [12. exam hacks and trap watch](#12-exam-hacks-and-trap-watch)

---

## 1. why convolutions beat dense layers on images

Count the parameters and be horrified:

```
Dense on a 32x32x3 image -> first hidden layer of 512:
  (32*32*3 + 1) * 512 = 3073 * 512 ≈ 1.57 MILLION params, layer one alone.

Conv2d(3, 6, kernel=5):
  (5*5*3 + 1) * 6 = 456 params. Total.
```

Three principles (name all three for full marks):
1. **Local connectivity** — each output pixel looks only at a KxK window.
2. **Weight sharing** — the SAME kernel slides everywhere (456 params reused
   across all positions) -> translation-tolerant detectors.
3. **Hierarchy** — early layers find edges, middle layers corners/textures,
   deep layers object parts. Stack + pool = growing receptive field.

Convolution in ASCII (3x3 kernel sliding on a 5x5 input, stride 1):

```
input 5x5          kernel 3x3        output 3x3
. . . . .           k k k            o . .
. # # # .    *      k k k     =      . . .
. # # # .           k k k            . . .
. # # # .        (dot product of     each o = sum(window * kernel) + bias
. . . . .         window & kernel)
```

[back to top](#table-of-contents)

## 2. conv2d anatomy — every argument

```python
nn.Conv2d(in_channels=3,    # channels coming IN (RGB image = 3)
          out_channels=6,   # number of FILTERS = channels going OUT
          kernel_size=5,    # 5x5 window (can be tuple (5,3))
          stride=1,         # slide step; 2 = skip every other position
          padding=0)        # zeros added around the border
```

- Each of the 6 filters is a (3,5,5) weight block + 1 bias -> 76 params;
  6 filters -> 456. General: **(K*K*Cin + 1) * Cout**.
- Output spatial size — THE formula:

```
O = floor((I - K + 2P) / S) + 1
```

Worked table:

| I | K | P | S | O |
|---|---|---|---|---|
| 32 | 5 | 0 | 1 | 28 |
| 28 | 3 | 1 | 1 | 28  ("same" trick: K=3,P=1 keeps size) |
| 14 | 5 | 0 | 1 | 10 |
| 224 | 7 | 3 | 2 | 112 (floor of 112.5) |

- Padding=“keep the borders in the conversation”: without it, edge pixels get
  seen fewer times and the map shrinks every layer.
- Stride>1 = built-in downsampling (used instead of pooling in modern nets).

**Mnemonic — "I Kicked 2 Penguins, So 1 cried" = (I - K + 2P)/S + 1.**

[back to top](#table-of-contents)

## 3. pooling

```python
self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
```

- Takes each 2x2 window, keeps the MAX ("strongest detector response wins").
- Halves H and W, channels untouched, **zero parameters**.
- Purpose: shrink compute, add small translation invariance, grow the
  receptive field of later layers.
- AvgPool2d = mean instead of max (softer summary; used at net ends sometimes).

```
input 4x4                     maxpool 2x2 -> output 2x2
 1  3 | 2  0                    3 | 2
 0  2 | 1  1        ->          -----
 -----+-----                    7 | 9
 7  5 | 4  9
 1  0 | 3  2
```

[back to top](#table-of-contents)

## 4. the class network, shape by shape

The exact notebook-3 model:

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)      # 3->6 channels, 5x5
        self.pool  = nn.MaxPool2d(2, 2)      # reused for both blocks
        self.conv2 = nn.Conv2d(6, 16, 5)     # 6->16 channels
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)              # keep batch dim (start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))              # <- 84-d "feature" layer
        return self.fc3(x)                   # logits, NO softmax

net = Net().to(device)
```

Full shape + parameter audit (write this table in any "trace the network" Q):

```
stage            output shape     out formula          params
input            3 x 32 x 32      -                    -
conv1(3,6,5)     6 x 28 x 28      (32-5)/1+1 = 28      (75+1)*6   = 456
relu             6 x 28 x 28      -                    0
pool(2,2)        6 x 14 x 14      28/2                 0
conv2(6,16,5)    16 x 10 x 10     (14-5)+1 = 10        (150+1)*16 = 2416
relu             16 x 10 x 10     -                    0
pool(2,2)        16 x 5 x 5       10/2                 0
flatten          400              16*5*5               0
fc1              120              -                    401*120    = 48120
fc2 (features!)  84               -                    121*84     = 10164
fc3              10               -                    85*10      = 850
                                            TOTAL params = 62,006
```

Note the design rhythm: **channels grow (3->6->16) while spatial shrinks
(32->14->5)** — trading "where" for "what."

[back to top](#table-of-contents)

## 5. data: cifar-10 pipeline

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),                              # [0,255] -> [0,1], HWC->CHW
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1] per channel
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
```

- CIFAR-10: 60k color images 32x32, 10 classes, 50k train / 10k test.
- Normalize math check: x in [0,1] -> (x-0.5)/0.5 in **[-1,1]** — recite it.
- Visual sanity check in the notebook: unnormalize (`img/2 + 0.5`) then
  `plt.imshow(np.transpose(npimg, (1,2,0)))` (CHW back to HWC for matplotlib).

[back to top](#table-of-contents)

## 6. training and evaluating, line by line

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):                       # class used 2 quick epochs
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:                 # print every 2000 mini-batches
            print(f"[{epoch+1}, {i+1}] loss: {running_loss/2000:.3f}")
            running_loss = 0.0
```

Evaluation with overall AND per-class accuracy:

```python
correct = total = 0
class_correct = [0]*10; class_total = [0]*10
net.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)      # (values, INDICES)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for lab, pred in zip(labels, predicted):
            class_total[lab] += 1
            class_correct[lab] += int(lab == pred)
print(f"overall: {100*correct/total:.1f}%")
for i in range(10):
    print(f"{classes[i]:>6}: {100*class_correct[i]/max(class_total[i],1):.1f}%")
```

- `torch.max(outputs, 1)` returns a TUPLE; the `_` discards max values, we
  keep indices = predicted classes. (Equivalent: `outputs.argmax(1)`.)
- Per-class accuracy exposes imbalance: the net might be great at ships,
  terrible at cats — an overall number hides that. Classic DLOps lesson.

[back to top](#table-of-contents)

## 7. feature extraction — chopping the head

The idea in one picture:

```
image -> [conv1-pool-conv2-pool-flatten-fc1-fc2] -> 84-d vector -> [fc3] -> 10 logits
          \_________ FEATURE EXTRACTOR _________/                  \HEAD/
                keep this (learned representation)             replace this
```

Implementation — run the forward pass but STOP at fc2:

```python
def extract_features(loader, model):
    feats, labels = [], []
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            x = model.pool(F.relu(model.conv1(X.to(device))))
            x = model.pool(F.relu(model.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(model.fc1(x))
            x = F.relu(model.fc2(x))           # 84-dim embeddings
            feats.append(x.cpu())              # back to CPU for sklearn
            labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

X_train, y_train = extract_features(trainloader, net)
X_test,  y_test  = extract_features(testloader, net)
print(X_train.shape)     # (50000, 84)
```

Why each detail:
- `inference_mode` — no gradients wanted, only representations.
- `.cpu()` before collecting — sklearn lives in NumPy land (CPU only).
- `torch.cat` stitches per-batch chunks into one big matrix.
- Why fc2 and not flatten(400)? Deeper layers are more abstract/compact;
  84 dims already encode "class-ish" information almost linearly separable.

[back to top](#table-of-contents)

## 8. randomforest on deep features

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)                     # trains in seconds on 84-d input
print("RF accuracy:", rf.score(X_test, y_test))

# which of the 84 learned features matter most?
import matplotlib.pyplot as plt
importances = rf.feature_importances_        # sums to 1.0 across features
plt.bar(range(84), importances)
plt.xlabel("feature index"); plt.ylabel("importance"); plt.show()
```

- RandomForest = many decision trees, each on a bootstrap sample + random
  feature subsets; prediction = majority vote. Robust, no scaling needed.
- `feature_importances_` = how much each embedding dimension reduced impurity
  across the forest — a peek into WHICH learned features drive decisions
  (interpretability the raw CNN lacks).

Why the hybrid at all (theory answer, 3 points):
1. CNN does what classic ML can't: learn features from raw pixels.
2. Forest does what the linear head can't: non-linear decision rules,
   feature importance, zero-epoch retraining if data shifts slightly.
3. Ops angle: retraining a forest on frozen embeddings is seconds and cheap —
   a practical pattern when the backbone stays fixed.

[back to top](#table-of-contents)

## 9. persistence: torch.save vs joblib

Two model families, two serializers:

```python
# PyTorch side
torch.save(net.state_dict(), "cifar_net.pth")
net2 = Net(); net2.load_state_dict(torch.load("cifar_net.pth")); net2.eval()

# sklearn side
import joblib
joblib.dump(rf, "rf_on_cnn_features.joblib")
rf2 = joblib.load("rf_on_cnn_features.joblib")
print(rf2.score(X_test, y_test))              # identical accuracy — proof
```

| | torch.save/state_dict | joblib.dump |
|---|---|---|
| what | tensors (weights) | whole sklearn object (numpy-optimized pickle) |
| load needs | the model CLASS defined | just joblib.load |
| for | nn.Module models | RandomForest, scalers, pipelines |

Deploying the pair = ship BOTH files + the feature-extraction code between them.

[back to top](#table-of-contents)

## 10. mnemonics

- **"I Kicked 2 Penguins, So 1 cried"** — O = (I-K+2P)/S + 1.
- **"LSH: Local, Shared, Hierarchical"** — the three CNN principles.
- **"Channels up, pixels down"** — the design rhythm 3->6->16 vs 32->14->5.
- **"MaxPool = strongest kid answers for the group"** — window max, 0 params.
- **"Chop the head, keep the body"** — feature extraction = forward to fc2.
- **"torch for torches (nn), joblib for gardens (forests)"** — who saves what.

[back to top](#table-of-contents)

## 11. cheatsheet

```
CONV OUT      O = floor((I-K+2P)/S)+1
CONV PARAMS   (K*K*Cin + 1) * Cout          pool params = 0
NETWORK       conv(3,6,5)->pool->conv(6,16,5)->pool->400->120->84->10 = 62,006
CIFAR NORM    ToTensor + Normalize(0.5s) -> [-1,1]
TRAIN         CE loss + SGD(lr=0.001, momentum=0.9), ZFLBS loop
PREDICT       _, pred = torch.max(outputs, 1)   or  outputs.argmax(1)
FEATURES      forward to fc2 -> (N, 84) -> .cpu().numpy()
FOREST        RandomForestClassifier(100).fit(Xtr,ytr); .score; .feature_importances_
SAVE          torch.save(net.state_dict(),*.pth) | joblib.dump(rf,*.joblib)
```

[back to top](#table-of-contents)

## 12. exam hacks and trap watch

1. Forgetting floor() in the conv formula on non-even divisions (224,K7,P3,S2 -> 112).
2. `fc1 = Linear(16*5*5, ...)` — the 400 comes from the LAST pool output; if the
   exam changes input size, recompute the flatten dim before writing fc1.
3. `torch.flatten(x, 1)` not `x.flatten()` — must preserve batch dim.
4. `torch.max(out,1)` returns (values, indices) — using values as predictions = trap.
5. Pooling has no parameters and no learning — "how many params in MaxPool?" -> 0.
6. Sklearn can't eat CUDA tensors — `.cpu().numpy()` first, or RuntimeError.
7. Feature extraction must run under eval + inference_mode, or dropout/bn noise
   contaminates embeddings.
8. `feature_importances_` sums to 1.0 — a "what do the values sum to?" one-liner.
9. joblib.load of a forest needs sklearn installed with a compatible version —
   version pinning is the DLOps moral.
10. Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) has THREE values because RGB = 3 channels.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Basics](DLOPS_02_Basics_PyTorch_DL.md) | [Next: Datasets](DLOPS_04_Datasets_DataLoaders.md)
