# DLOPS_05 — Custom Datasets & Full Training (ImageFolder, custom Dataset class, TinyVGG, loss curves)

[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Datasets](DLOPS_04_Datasets_DataLoaders.md) | [Next: TensorBoard](DLOPS_06_TensorBoard.md)

> Story: module 4 used ready-made bookshelves. Now you BUILD the bookshelf:
> first the easy way (ImageFolder reads your folder names as labels), then from
> raw wood (a custom Dataset class), then you train TinyVGG end-to-end twice —
> once plain, once with augmentation — and learn to READ loss curves like a
> doctor reads an X-ray. This is the longest class notebook (106 cells) and the
> single most likely source of "write the full pipeline" questions.

> **Cross-repo reference (TS-01):** TinyVGG and the overfit/underfit reading in
> this module build on TS-01 →
> [Deep-Learning](https://github.com/rpaut03l/TS-01/tree/main/ML/Deep-Learning) |
> [Neural-Networks](https://github.com/rpaut03l/TS-01/tree/main/ML/Neural-Networks)

## table of contents
- [1. option 1 — imagefolder](#1-option-1--imagefolder)
- [2. option 2 — custom dataset class, line by line](#2-option-2--custom-dataset-class-line-by-line)
- [3. dataloaders for both options](#3-dataloaders-for-both-options)
- [4. tinyvgg — the model](#4-tinyvgg--the-model)
- [5. torchinfo — auditing shapes and params](#5-torchinfo--auditing-shapes-and-params)
- [6. train_step, test_step, train — the reusable trio](#6-train_step-test_step-train--the-reusable-trio)
- [7. loss curves — reading the x-ray](#7-loss-curves--reading-the-x-ray)
- [8. model 0 vs model 1 — the augmentation experiment](#8-model-0-vs-model-1--the-augmentation-experiment)
- [9. predicting on a custom image](#9-predicting-on-a-custom-image)
- [10. mnemonics](#10-mnemonics)
- [11. cheatsheet](#11-cheatsheet)
- [12. exam hacks and trap watch](#12-exam-hacks-and-trap-watch)

---

## 1. option 1 — imagefolder

If your data follows the golden folder convention, one line builds the Dataset:

```
data/pizza_steak_sushi/
    train/
        pizza/  0001.jpg 0002.jpg ...
        steak/  ...
        sushi/  ...
    test/
        pizza/ ...  steak/ ...  sushi/ ...
```

```python
from torchvision import datasets

train_data = datasets.ImageFolder(root="data/pizza_steak_sushi/train",
                                  transform=data_transform,      # for images
                                  target_transform=None)         # for labels
test_data  = datasets.ImageFolder(root="data/pizza_steak_sushi/test",
                                  transform=test_transform)

print(train_data.classes)        # ['pizza', 'steak', 'sushi']  (alphabetical!)
print(train_data.class_to_idx)   # {'pizza': 0, 'steak': 1, 'sushi': 2}
img, label = train_data[0]       # transformed tensor, int label
```

What ImageFolder does under the hood (this becomes YOUR job in option 2):
1. Scan root for subfolders -> sorted folder names become `classes`.
2. Map name -> index in `class_to_idx`.
3. Collect every image path with its class index.
4. On `__getitem__`: open image, apply transform, return (tensor, index).

[back to top](#table-of-contents)

## 2. option 2 — custom dataset class, line by line

Why bother? Because real data is never that tidy — labels in CSVs, images in
buckets, multi-modal inputs. Writing a Dataset teaches you the contract.

Step 1 — helper to discover classes (the notebook's find_classes):

```python
import os
from typing import Tuple, List, Dict

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory)
                     if entry.is_dir())                     # sorted = stable ids
    if not classes:
        raise FileNotFoundError(f"No classes found in {directory}.")
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    return classes, class_to_idx
```

- `os.scandir` is a fast directory iterator; `is_dir()` skips stray files.
- `sorted(...)` guarantees pizza=0 on every machine — reproducibility again.
- Raising on empty catches a wrong path IMMEDIATELY instead of "0 samples" later.

Step 2 — the Dataset class (memorize this whole block):

```python
import pathlib
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderCustom(Dataset):                       # 1. subclass Dataset

    def __init__(self, targ_dir: str, transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))  # 2. all files
        self.transform = transform                                  # 3. store tf
        self.classes, self.class_to_idx = find_classes(targ_dir)    # 4. labels map

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.paths[index])            # helper: path -> PIL

    def __len__(self) -> int:
        return len(self.paths)                          # 5. dataset size

    def __getitem__(self, index: int):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name      # 6. label FROM folder
        class_idx = self.class_to_idx[class_name]       # 7. name -> int
        if self.transform:
            return self.transform(img), class_idx       # 8. transformed
        return img, class_idx                           # 9. raw fallback
```

Every numbered line, explained:
1. Subclassing `Dataset` is what makes DataLoader accept it.
2. `glob("*/*.jpg")` = "any class folder / any jpg" — collected ONCE at init
   (cheap: just paths, no pixels loaded yet).
3. Transform stored, applied lazily per access — enables per-epoch augmentation.
4. Reuse the helper so custom and ImageFolder agree on the label mapping.
5. `__len__` lets `len(ds)` and the loader's batch math work.
6. The label is literally the parent folder's name — the convention encoded.
7. Convert to int because losses want class indices.
8-9. Transform if given; the `if` makes the class usable for raw visualization too.

**Mnemonic — ILG: __Init__ (gather paths), __Len__ (count), __Getitem__ (serve one).**
Forget any of the three and the loader breaks.

```
loader asks for i --> __getitem__(i)
                        |-- open paths[i]  (disk -> PIL)
                        |-- label = parent folder name -> idx
                        |-- transform (resize/augment/tensor)
                        --> (tensor CHW, int)
```

[back to top](#table-of-contents)

## 3. dataloaders for both options

```python
import os
from torch.utils.data import DataLoader

BATCH = 32
train_loader = DataLoader(train_data_custom, batch_size=BATCH, shuffle=True,
                          num_workers=os.cpu_count())
test_loader  = DataLoader(test_data_custom, batch_size=BATCH, shuffle=False,
                          num_workers=os.cpu_count())
```

Identical to module 4 — that's the beauty of the contract: DataLoader cannot
tell ImageFolder and your custom class apart. If asked "what must a class
implement to be DataLoader-compatible?" -> `__len__` and `__getitem__`.

[back to top](#table-of-contents)

## 4. tinyvgg — the model

The course's workhorse CNN (a miniature of the VGG family: conv-conv-pool blocks):

```python
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 16 * 16, output_shape),   # for 64x64 input
        )
    def forward(self, x):
        return self.classifier(self.block_2(self.block_1(x)))

model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=3).to(device)
```

Shape trace for 64x64 input (padding=1, k=3 keeps size; each pool halves):

```
input        3 x 64 x 64
block_1      10 x 64 x 64 (convs, same-size) -> pool -> 10 x 32 x 32
block_2      10 x 32 x 32 -> pool             -> 10 x 16 x 16
flatten      10*16*16 = 2560
classifier   Linear(2560, 3) -> 3 logits (pizza/steak/sushi)
```

Sanity test with a single image BEFORE training (the notebook's habit —
adopt it in the exam, it earns method marks):

```python
img, label = next(iter(train_loader))
print(model_0(img.to(device))[0])   # 3 raw logits -> pipeline plumbing works
```

[back to top](#table-of-contents)

## 5. torchinfo — auditing shapes and params

```python
# pip install torchinfo
from torchinfo import summary
summary(model_0, input_size=(1, 3, 64, 64))
```

Output shows, per layer: output shape, #params, memory — the printed version
of the shape-trace table. Uses: catch a wrong flatten dim instantly, report
total params, estimate model size. In answers, quoting "verified with
torchinfo.summary" signals real workflow knowledge.

[back to top](#table-of-contents)

## 6. train_step, test_step, train — the reusable trio

The notebook refactors the loop into three functions — the exact trio that
modules 6-9 keep reusing. Learn it ONCE here.

```python
def train_step(model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)                          # logits (B, C)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    return train_loss / len(dataloader), train_acc / len(dataloader)

def test_step(model, dataloader, loss_fn):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            labels = logits.argmax(dim=1)
            test_acc += (labels == y).sum().item() / len(labels)
    return test_loss / len(dataloader), test_acc / len(dataloader)

from tqdm.auto import tqdm

def train(model, train_dataloader, test_dataloader, optimizer,
          loss_fn=nn.CrossEntropyLoss(), epochs=5):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        tr_loss, tr_acc = train_step(model, train_dataloader, loss_fn, optimizer)
        te_loss, te_acc = test_step(model, test_dataloader, loss_fn)
        print(f"Epoch {epoch+1} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| test_loss {te_loss:.4f} acc {te_acc:.4f}")
        for k, v in zip(results, (tr_loss, tr_acc, te_loss, te_acc)):
            results[k].append(v)
    return results
```

Design notes worth writing:
- Losses/accs are averaged over BATCHES (`/len(dataloader)`).
- The results dict of LISTS is what gets plotted as loss curves — and in
  module 6 it's what gets logged to TensorBoard. Same trio, growing powers.
- Softmax before argmax is redundant for the argmax itself (monotonic) — a
  fine one-line observation if asked.

[back to top](#table-of-contents)

## 7. loss curves — reading the x-ray

Plot from the results dict:

```python
import matplotlib.pyplot as plt

def plot_loss_curves(results):
    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="train_loss")
    plt.plot(epochs, results["test_loss"], label="test_loss")
    plt.title("Loss"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="train_acc")
    plt.plot(epochs, results["test_acc"], label="test_acc")
    plt.title("Accuracy"); plt.legend()
```

The three diagnoses (the notebook's section 9 — extremely likely exam figure):

```
UNDERFITTING              IDEAL                    OVERFITTING
loss                      loss                     loss
 |------- train            |\                       |\
 |------- test             | \\  both fall          | \______ train
 | both stuck HIGH         |  \\_ together,         |  \   ___----- test
 |                         |    \\_ tiny gap        |   \_/   gap GROWS
 +-------- epochs          +-------- epochs         +-------- epochs
model too weak /           just right               model memorizing
trained too little
```

Treatments (recite both lists):
- Overfitting -> more data, augmentation, dropout, weight decay, early
  stopping, smaller model, transfer learning.
- Underfitting -> bigger model (more hidden_units/layers), train longer,
  raise lr, reduce regularization, better features/transfer learning.
- The balance: you WANT to flirt with slight overfitting — it proves the
  model has enough capacity — then regularize back.

**Mnemonic — "Gap Grows = Generalization Gone" (overfit); "Both Bad = Bigger
Brain" (underfit).**

[back to top](#table-of-contents)

## 8. model 0 vs model 1 — the augmentation experiment

The notebook's controlled experiment — one variable changed:

```python
# Model 0: plain transforms (Resize + ToTensor)
# Model 1: Resize + TrivialAugmentWide(31) + ToTensor
train_tf_aug = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
])
# fresh datasets/loaders with the new transform, SAME TinyVGG class, same
# epochs, same optimizer -> results_0 vs results_1
```

Comparison the class did:

```python
import pandas as pd
df0, df1 = pd.DataFrame(results_0), pd.DataFrame(results_1)
# plot test_loss of both on one figure; inspect final rows
```

Expected reading (small dataset, few epochs — nuance matters):
- Model 0 shows the classic overfit gap sooner.
- Model 1's train accuracy is LOWER (augmented data is harder) but its
  test curve is closer to train — the gap shrinks. On tiny data/epochs the
  absolute test accuracy may not beat Model 0 yet — the honest takeaway the
  notebook makes: augmentation is a REGULARIZER, judged by the gap and by
  longer-run generalization, not by 5-epoch accuracy alone.

Experiment discipline on display (name it in theory answers): change ONE
variable, keep seeds/epochs/model identical, compare curves not vibes.

[back to top](#table-of-contents)

## 9. predicting on a custom image

The end-to-end payoff — predict on a photo the model never saw:

```python
import torchvision

img_path = "data/04-pizza-dad.jpeg"
img = torchvision.io.read_image(img_path)        # uint8 tensor (C,H,W), 0-255!
img = img.type(torch.float32) / 255.0            # match training scale [0,1]
transform = transforms.Resize((64, 64))
img = transform(img)                             # match training SIZE

model_1.eval()
with torch.inference_mode():
    pred = model_1(img.unsqueeze(0).to(device))  # add batch dim -> (1,3,64,64)
probs = torch.softmax(pred, dim=1)
label = class_names[probs.argmax(dim=1).item()]
print(label, probs.max().item())
```

The three matching rules (the notebook hammers them — so do exams):
1. **Same dtype/scale** — read_image gives uint8 0-255; training saw float 0-1.
2. **Same size** — resize to the training resolution (64x64).
3. **Same device + batch dim** — `.unsqueeze(0).to(device)`.
Any mismatch = wrong-but-silent predictions or a shape error.

[back to top](#table-of-contents)

## 10. mnemonics

- **ILG** — __Init__, __Len__, __Getitem__: the custom Dataset trio.
- **"Folder name IS the label"** — the ImageFolder convention.
- **"Same Scale, Same Size, Same Device (+batch)"** — the 3S custom-image rule.
- **"Gap Grows = Generalization Gone / Both Bad = Bigger Brain"** — curves.
- **"Trio powers up later: step, step, train"** — the same three functions gain
  a writer (module 6) and wandb.log (module 7) without changing shape.
- **"Channels wide, pixels slide"** — TinyVGG: same channels, halving maps.

[back to top](#table-of-contents)

## 11. cheatsheet

```
IMAGEFOLDER   datasets.ImageFolder(root, transform); .classes; .class_to_idx
FIND CLASSES  sorted(scandir dirs) -> (classes, {name: idx}); raise if empty
CUSTOM DS     __init__: glob('*/*.jpg')+transform+classes
              __len__: len(paths)
              __getitem__: PIL open -> label=parent.name -> transform -> (t, idx)
TINYVGG       [Conv3x3 p1, ReLU]x2 + Pool | x2 blocks | Flatten | Linear(h*16*16, C)
AUDIT         torchinfo.summary(model, input_size=(1,3,64,64))
TRIO          train_step (train mode, ZFLBS, avg loss/acc)
              test_step  (eval + inference_mode)
              train      (loop epochs -> results dict of 4 lists)
CURVES        plot train/test loss+acc; diagnose under/ideal/over
AUG EXP       Model0 plain vs Model1 TrivialAugmentWide — gap comparison
CUSTOM IMG    read_image -> /255 float -> Resize -> unsqueeze(0) -> eval+inference
```

[back to top](#table-of-contents)

## 12. exam hacks and trap watch

1. `glob("*/*.jpg")` misses .png/.jpeg — pattern must match the data (mention it).
2. Unsorted class list = different label ids per machine -> silent evaluation chaos.
3. `paths[index].parent.name` — the label comes from the FOLDER, not the filename.
4. Flatten dim for TinyVGG at 64x64 = hidden*16*16 (two pools: 64->32->16).
   Input 224? -> 224->112->56 -> hidden*56*56. Recompute, don't recite.
5. read_image returns uint8 0-255 — forgetting /255 makes the model see
   "impossibly bright" inputs and predict garbage (favorite practical trap).
6. Accuracy inside train_step divides by len(y_pred) per batch then by
   len(dataloader) — averaging of averages; fine for equal batches.
7. tqdm wraps range(epochs) — cosmetic; don't let it distract in reading code.
8. TrivialAugmentWide takes num_magnitude_bins (max 31) — the only knob.
9. Test transform NEVER contains augmentation — check before comparing models.
10. The "single image forward pass" sanity test before training = free method
    marks; write it.

[back to top](#table-of-contents)

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Prev: Datasets](DLOPS_04_Datasets_DataLoaders.md) | [Next: TensorBoard](DLOPS_06_TensorBoard.md)
