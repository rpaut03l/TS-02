# DLOPS_04 — Datasets & DataLoaders (built-in datasets, transforms, augmentation)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: CNN](DLOPS_03_CNN_Feature_Extraction.md) | [Next: Custom Datasets](DLOPS_05_Custom_Datasets_Training.md)

> Story: a Dataset is a bookshelf — it only knows two things: how many books it
> holds and how to hand you book number i. A DataLoader is the librarian — she
> shuffles the catalog, brings you books in stacks of 32, and even hires helpers
> (num_workers) to fetch stacks in advance so you never wait. Transforms are the
> book-prep desk: every book gets resized, maybe flipped for variety, and stamped
> into the standard tensor format before reaching you.

> **Cross-repo reference (TS-01):** the "become one with the data" and
> feature-prep mindset here echoes TS-01 →
> [Foundations](https://github.com/rpaut03l/TS-01/tree/main/ML/Foundations) |
> [Feature-Selection-DimRed](https://github.com/rpaut03l/TS-01/tree/main/ML/Feature-Selection-DimRed)

## table of contents
- [1. the two-object contract](#1-the-two-object-contract)
- [2. built-in datasets — getting data](#2-built-in-datasets--getting-data)
- [3. become one with the data](#3-become-one-with-the-data)
- [4. visualizing — pillow and matplotlib](#4-visualizing--pillow-and-matplotlib)
- [5. transforms — every one explained](#5-transforms--every-one-explained)
- [6. dataloader — every argument explained](#6-dataloader--every-argument-explained)
- [7. data augmentation — why and how](#7-data-augmentation--why-and-how)
- [8. how to run all of this](#8-how-to-run-all-of-this)
- [9. mnemonics](#9-mnemonics)
- [10. cheatsheet](#10-cheatsheet)
- [11. exam hacks and trap watch](#11-exam-hacks-and-trap-watch)

---

## 1. the two-object contract

```
Dataset (the bookshelf)                DataLoader (the librarian)
- __len__()      how many samples      - batches samples together
- __getitem__(i) one (data, label)     - shuffles order each epoch
                                       - parallel-loads with workers
                                       - iterable: for X, y in loader:
```

```python
from torch.utils.data import Dataset, DataLoader

sample_img, sample_label = dataset[0]     # Dataset: INDEXING, one at a time
for X_batch, y_batch in loader:           # DataLoader: ITERATION, batches
    ...
```

Rule to recite: **Dataset stores and serves ONE sample; DataLoader turns a
Dataset into shuffled, batched, parallel mini-batches.** All of PyTorch
training rests on this split — swap datasets freely, the loop never changes.

[back to top](#table-of-contents)

## 2. built-in datasets — getting data

torchvision ships classic datasets with auto-download:

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.FashionMNIST(
    root="data",           # where to store/download files
    train=True,            # True = training split, False = test split
    download=True,         # fetch from the internet if not present
    transform=ToTensor(),  # applied to the IMAGE on every access
    # target_transform=... # (optional) applied to the LABEL
)
test_data = datasets.FashionMNIST(root="data", train=False,
                                  download=True, transform=ToTensor())
print(len(train_data), len(test_data))     # 60000 10000
img, label = train_data[0]
print(img.shape, label)                    # torch.Size([1, 28, 28]) 9
```

Line by line:
- `root` — data lands in `data/FashionMNIST/raw`. Re-runs skip the download.
- `train=` picks the split; the SAME class serves both.
- `transform` runs lazily at ACCESS time (each `dataset[i]`), not at download —
  that's why augmentation can differ every epoch.
- `.classes` and `.class_to_idx` exist on these datasets too.

Class-taught principle — **"Start Small and Upgrade if Necessary"**: prototype
on a subset (the notebook even downloads a FashionMNIST SUBSET) so iteration
takes seconds; scale to the full data only once the pipeline works. This is a
core DLOps discipline, not laziness.

```python
from torch.utils.data import Subset
small = Subset(train_data, range(0, 6000))   # first 10% for fast experiments
```

[back to top](#table-of-contents)

## 3. become one with the data

Before modeling, WALK the data — the notebook does this with os.walk:

```python
import os
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"{len(dirnames)} dirs, {len(filenames)} images in '{dirpath}'")

walk_through_dir("data/pizza_steak_sushi")
# 3 dirs, 0 images in 'data/pizza_steak_sushi/train'
# 0 dirs, 78 images in 'data/pizza_steak_sushi/train/pizza'  ...
```

What you're checking (say these in a theory answer):
1. Class balance — 78 pizzas vs 12 sushis would bias the model.
2. Corrupt/odd files — a stray .txt in an image folder crashes loaders.
3. Image sizes/formats — decides your Resize target.
4. The folder convention — `root/split/class_name/*.jpg` — which is exactly
   what ImageFolder (module 5) expects.

[back to top](#table-of-contents)

## 4. visualizing — pillow and matplotlib

Two ways the notebook shows images:

```python
# (a) Pillow — open a random file directly
import random
from PIL import Image
from pathlib import Path

image_paths = list(Path("data/pizza_steak_sushi").glob("*/*/*.jpg"))
img_path = random.choice(image_paths)
img = Image.open(img_path)
print(img_path.parent.stem, img.height, img.width)   # class name from folder!
img            # displays in a notebook cell

# (b) matplotlib — after ToTensor, shape is CHW; matplotlib wants HWC
import matplotlib.pyplot as plt
import numpy as np
img_tensor, label = train_data[0]                    # (1, 28, 28)
plt.imshow(img_tensor.squeeze(), cmap="gray")        # grayscale: drop C
plt.title(train_data.classes[label]); plt.axis("off")

# color images: permute channels to the end
plt.imshow(color_tensor.permute(1, 2, 0))            # CHW -> HWC
```

The permute line is a guaranteed exam snippet: **PyTorch stores CHW,
matplotlib/PIL think HWC — `permute(1, 2, 0)` translates.**

[back to top](#table-of-contents)

## 5. transforms — every one explained

Transforms are functions applied to each sample at access time, chained with
Compose (order matters!):

```python
from torchvision import transforms

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),              # 1. every image -> 64x64
    transforms.RandomHorizontalFlip(p=0.5),   # 2. 50% chance mirror (train only)
    transforms.ToTensor(),                    # 3. PIL -> float tensor [0,1], CHW
    transforms.Normalize((0.5,), (0.5,)),     # 4. (x-mean)/std -> [-1,1]
])
```

Each transform in depth:

- **Resize((H, W))** — interpolates to a fixed size. Networks with Linear
  heads NEED fixed input sizes (the flatten dim is hard-coded).
  `Resize(64)` (single int) resizes the SHORTER side to 64, keeping aspect.
- **RandomHorizontalFlip(p)** — mirrors left-right with probability p.
  Safe for food/animals; NOT for digits/text (a flipped "3" isn't a "3").
- **ToTensor()** — the workhorse. Three jobs at once:
  1. PIL/ndarray -> torch.FloatTensor
  2. scales [0,255] uint8 -> [0.0, 1.0]
  3. reorders HWC -> CHW.
  Always placed AFTER PIL-based transforms, BEFORE Normalize.
- **Normalize(mean, std)** — per channel: `x = (x - mean)/std`.
  With (0.5, 0.5): 0 -> -1, 0.5 -> 0, 1 -> +1. Centered data trains faster
  (gradients better conditioned). Tuples must match channel count.
- **ToPILImage()** — the reverse of ToTensor (used for visualization).

```
PIL image (HWC, 0-255)
   | Resize          still PIL
   | RandomFlip      still PIL, maybe mirrored
   | ToTensor        float CHW in [0,1]
   | Normalize       float CHW in [-1,1]
   v
model-ready tensor
```

**Mnemonic — "RAT-N: Resize, Augment, ToTensor, Normalize"** — the canonical order.
PIL-transforms before ToTensor; tensor-transforms after.

[back to top](#table-of-contents)

## 6. dataloader — every argument explained

```python
from torch.utils.data import DataLoader
import os

train_loader = DataLoader(
    dataset=train_data,
    batch_size=32,             # samples per step; power of 2 by convention
    shuffle=True,              # reshuffle indices EVERY epoch (train only)
    num_workers=os.cpu_count(),# parallel loader subprocesses
    pin_memory=True,           # (GPU) page-locked RAM -> faster host->GPU copies
    drop_last=False,           # keep the final smaller batch
)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
```

Argument meanings and the WHY:
- **batch_size** — bigger = smoother gradients + better GPU utilization but
  more memory; smaller = noisier (regularizing) but slower per epoch.
  Batches/epoch = ceil(N/B): 60000/32 -> 1875.
- **shuffle=True (train)** — kills ordering correlations (all sneakers in a
  row would make gradients lurch class by class).
  **shuffle=False (test)** — evaluation must be reproducible.
- **num_workers** — each worker is a separate process pre-fetching batches;
  0 = load in the main process (slow but simplest to debug).
- **drop_last=True** — some setups (e.g. BatchNorm with batch 1) hate ragged
  last batches; dropping keeps every batch identical in size.

Peeking at one batch (the notebook's "quick look at the iterable"):

```python
imgs, labels = next(iter(train_loader))
print(imgs.shape, labels.shape)     # torch.Size([32, 1, 28, 28]) torch.Size([32])
```

`iter()` makes an iterator, `next()` pulls one batch — the standard debugging
one-liner before any training run.

[back to top](#table-of-contents)

## 7. data augmentation — why and how

Augmentation = manufacturing free variety. Each epoch the model sees slightly
different versions of the same images, so it must learn what a pizza IS,
not which exact pixels were in training file 0042.jpg.

```python
train_transform_aug = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),  # the star
    transforms.ToTensor(),
])
test_transform = transforms.Compose([          # NO augmentation on test!
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
```

**TrivialAugmentWide** — the modern "no-tuning" policy used in class:
for EACH image, pick ONE augmentation at random (rotate, shear, color jitter,
posterize, ...) at a RANDOM strength (0..31 bins). No schedule to tune, and it
matches or beats hand-crafted pipelines. `num_magnitude_bins` caps intensity.

Other common augments worth naming:

```python
transforms.RandomHorizontalFlip(0.5)
transforms.RandomRotation(degrees=15)
transforms.ColorJitter(brightness=0.2, contrast=0.2)
transforms.RandomCrop(64, padding=4)
```

Rules:
1. Augment TRAIN only. Augmenting test = evaluating on data you invented.
2. Augmentations must be label-preserving (flip a pizza = pizza; flip a "6" = "9"!).
3. Augmentation happens on the fly in `__getitem__` — epoch 1 and epoch 2 see
   different pixel versions; the dataset on disk never changes.
4. It's a regularizer: expect train accuracy to DROP slightly and test
   accuracy to RISE — that gap closing is the whole point (module 5 shows
   Model 0 vs Model 1 comparing exactly this).

```
without aug:  model memorizes exact pizzas  -> train 99%, test 74%  (gap!)
with aug:     every epoch new-ish pizzas    -> train 92%, test 81%  (gap shrinks)
```

[back to top](#table-of-contents)

## 8. how to run all of this

Practical, end-to-end minimal script (save as `data_pipeline.py`):

```python
import os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

train_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
])
test_tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

train_data = datasets.FashionMNIST("data", train=True,  download=True, transform=train_tf)
test_data  = datasets.FashionMNIST("data", train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_data, 32, shuffle=True,  num_workers=os.cpu_count())
test_loader  = DataLoader(test_data,  32, shuffle=False, num_workers=os.cpu_count())

X, y = next(iter(train_loader))
print(f"batch X: {X.shape} {X.dtype} | y: {y.shape} {y.dtype}")
# batch X: torch.Size([32, 1, 64, 64]) torch.float32 | y: torch.Size([32]) torch.int64
```

Run: `python data_pipeline.py`. If num_workers errors on Windows/Mac scripts,
wrap the body in `if __name__ == "__main__":` (multiprocessing quirk) or set
num_workers=0 — a real-world gotcha worth writing in a practical answer.

[back to top](#table-of-contents)

## 9. mnemonics

- **"Bookshelf and Librarian"** — Dataset serves one; DataLoader batches many.
- **ILG** — the Dataset trio: __Init__, __Len__, __Getitem__ (formalized in module 5).
- **RAT-N** — transform order: Resize, Augment, ToTensor, Normalize.
- **"ToTensor does 3 jobs: Type, /255, CHW"** — float, scale, channel-first.
- **"Shuffle the classroom, not the exam"** — shuffle train, never test.
- **"Augment = free clones with tiny disguises"** — label-preserving variety.

[back to top](#table-of-contents)

## 10. cheatsheet

```
BUILTIN     datasets.FashionMNIST(root, train, download, transform)
SUBSET      Subset(ds, range(k))            start small, upgrade later
INSPECT     len(ds); img,lab = ds[0]; ds.classes; ds.class_to_idx
WALK        os.walk(dir) -> count dirs/files per folder
SHOW        plt.imshow(t.squeeze(),cmap='gray') | color: t.permute(1,2,0)
TRANSFORMS  Compose([Resize, Flip/TrivialAugmentWide, ToTensor, Normalize])
TOTENSOR    PIL uint8 HWC [0,255] -> float32 CHW [0,1]
NORMALIZE   (x-mean)/std ; (0.5,0.5) => [-1,1]
LOADER      DataLoader(ds, batch_size=32, shuffle=True/False,
                       num_workers=os.cpu_count(), pin_memory=True)
PEEK        X,y = next(iter(loader)); print(X.shape)
BATCHES     ceil(N/B) per epoch
AUGMENT     train only; label-preserving; on-the-fly each epoch
```

[back to top](#table-of-contents)

## 11. exam hacks and trap watch

1. Transform runs at ACCESS time — "when is the transform applied?" -> in
   __getitem__, per access, per epoch (not at download).
2. matplotlib expects HWC — forgetting `permute(1,2,0)` shows garbage colors.
3. Normalize BEFORE ToTensor -> TypeError (Normalize needs tensors). Order = RAT-N.
4. Augmenting the test set = invalid evaluation — instant marks lost.
5. Flip augmentation on digit/text datasets breaks labels — mention it.
6. `Resize(64)` vs `Resize((64,64))` — int keeps aspect ratio, tuple forces square.
7. shuffle=True re-shuffles EVERY epoch, not once.
8. num_workers>0 inside a plain script needs the __main__ guard (Win/Mac spawn).
9. `next(iter(loader))` grabs one batch — the sanity-check idiom to memorize.
10. Labels dtype from loaders is int64 — exactly what CrossEntropyLoss wants;
    casting them to float is the self-inflicted trap.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: CNN](DLOPS_03_CNN_Feature_Extraction.md) | [Next: Custom Datasets](DLOPS_05_Custom_Datasets_Training.md)
