# DLOPS_EXAM_02 — Numericals (shape math with real numbers)

[Hub](DLOPS_EXAM_00_Hub.md) | [Theory](DLOPS_EXAM_01_Theory.md) | [Practice](DLOPS_EXAM_03_Practice.md)

## table of contents
- [1. conv output size formula](#1-conv-output-size-formula)
- [2. pooling output size](#2-pooling-output-size)
- [3. parameter counting](#3-parameter-counting)
- [4. lenet walkthrough end to end](#4-lenet-walkthrough-end-to-end)
- [5. linear layer and flatten math](#5-linear-layer-and-flatten-math)
- [6. normalize math](#6-normalize-math)
- [7. dataloader and batch math](#7-dataloader-and-batch-math)
- [8. dataparallel split math](#8-dataparallel-split-math)
- [9. cheatsheet of all formulas](#9-cheatsheet-of-all-formulas)

---

## 1. conv output size formula

The only formula you truly need:

```
O = floor( (I - K + 2P) / S ) + 1
I = input size, K = kernel, P = padding, S = stride
```

**Mnemonic — "I Kicked 2 Penguins, So 1 cried": (I - K + 2P)/S + 1.**

Worked examples:
- 32x32 input, K=5, P=0, S=1 -> (32-5+0)/1 + 1 = **28**
- 28x28, K=3, P=1, S=1 -> (28-3+2)/1 + 1 = **28** (padding=1 with k=3 keeps size — "same" trick)
- 224, K=7, P=3, S=2 -> (224-7+6)/2 + 1 = 111.5 -> floor -> **112**

Rules:
- Output channels = number of filters, nothing to compute.
- Same formula for H and W separately.
- floor() always — drop the fraction.

[back to top](#table-of-contents)

## 2. pooling output size

Same formula, usually P=0 and S=K:
```
MaxPool2d(2,2) on 28x28 -> (28-2)/2 + 1 = 14   (halves the size)
MaxPool2d(2,2) on 10x10 -> 5
```
Rule of thumb: pool(2,2) = divide H,W by 2. Channels unchanged. Pooling has **0 parameters**.

[back to top](#table-of-contents)

## 3. parameter counting

```
Conv2d:  params = (K*K*Cin + 1) * Cout        (+1 = bias per filter)
Linear:  params = (in_features + 1) * out_features
```

Examples:
- Conv2d(3,6,5): (5*5*3 + 1)*6 = 76*6 = **456**
- Conv2d(6,16,5): (5*5*6 + 1)*16 = 151*16 = **2416**
- Linear(400,120): 401*120 = **48120**
- Linear(120,84): 121*84 = **10164**
- Linear(84,10): 85*10 = **850**

LeNet total = 456+2416+48120+10164+850 = **62006 params.**

[back to top](#table-of-contents)

## 4. lenet walkthrough end to end

Class notebook 3 network on CIFAR-10 (3x32x32):

```
input          3 x 32 x 32
conv1(3,6,5)   6 x 28 x 28    (32-5)/1+1=28
pool(2,2)      6 x 14 x 14
conv2(6,16,5)  16 x 10 x 10   (14-5)+1=10
pool(2,2)      16 x 5 x 5
flatten        400            16*5*5
fc1            120
fc2            84   <-- feature extraction taps HERE for RandomForest
fc3            10             (10 CIFAR classes)
```

If exam changes numbers, just re-run the formula stage by stage — this table pattern is the answer format.

[back to top](#table-of-contents)

## 5. linear layer and flatten math

- Flatten of (N, C, H, W) -> (N, C*H*W). Batch dim survives!
  - (64, 16, 5, 5) -> (64, 400)
- Linear(400,120) on (64,400) -> (64,120). Rule: last dim in must equal in_features.
- Matmul shape rule: (a,b) @ (b,c) = (a,c).

[back to top](#table-of-contents)

## 6. normalize math

`Normalize(mean=0.5, std=0.5)` after ToTensor (range [0,1]):
```
out = (x - 0.5)/0.5  -> range [-1, 1]
x=0 -> -1 ; x=1 -> +1 ; x=0.5 -> 0
```
Classic MCQ: "after ToTensor + Normalize(0.5,0.5) the pixel range is?" -> **[-1,1]**.

[back to top](#table-of-contents)

## 7. dataloader and batch math

```
batches per epoch = ceil(N / batch_size)      if drop_last=False
                  = floor(N / batch_size)     if drop_last=True
```
- 50000 images, batch 32 -> 50000/32 = 1562.5 -> **1563** batches (last one has 16).
- Total steps for E epochs = E * batches_per_epoch.

[back to top](#table-of-contents)

## 8. dataparallel split math

Batch B on G GPUs -> each GPU gets ceil(B/G) (roughly B/G):
- batch 30, 2 GPUs -> 15 each. Output re-gathered to shape (30, ...) on cuda:0.
- batch 30, 4 GPUs -> 8,8,8,6.

[back to top](#table-of-contents)

## 9. cheatsheet of all formulas

```
CONV OUT     O = floor((I - K + 2P)/S) + 1
POOL OUT     same formula; pool(2,2) => size/2, 0 params
CONV PARAMS  (K*K*Cin + 1) * Cout
FC PARAMS    (in + 1) * out
FLATTEN      (N,C,H,W) -> (N, C*H*W)
MATMUL       (a,b)@(b,c) = (a,c)
NORMALIZE    (x - mean)/std ; ToTensor+Norm(0.5,0.5) => [-1,1]
BATCHES      ceil(N/B)
DP SPLIT     batch/num_gpus per device, gathered on device 0
```

**Exam hack:** write the stage-by-stage shape table (section 4 style) even for partial marks — examiners give steps marks for the formula line + substitution + result.

---
[Hub](DLOPS_EXAM_00_Hub.md) | [Theory](DLOPS_EXAM_01_Theory.md) | [Practice](DLOPS_EXAM_03_Practice.md)
