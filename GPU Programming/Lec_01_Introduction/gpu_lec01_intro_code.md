# 💻 GPU Programming Lec 1 — Introduction: CODE

### *Runnable Python + shell — see the GPU, use it, measure it*

> **Nav:** [← Lec 1 README](README.md) | [📖 THEORY](gpu_lec01_intro_theory.md) | **CODE** | [🎯 PRACTICE →](gpu_lec01_intro_practice.md)

---

## 🏗️ Setup — Google Colab (easiest)

1. Open [colab.research.google.com](https://colab.research.google.com).
2. Menu: **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**.
3. All the code below runs on the free tier — no install, no credit card.

On your own laptop, you need NVIDIA drivers + CUDA toolkit + `cupy` / `numba` — much more setup, so start with Colab.

---

## Ex 1. Check that you actually have a GPU

### 👶 What this does
Before you write any GPU code, **prove** you have a GPU. `nvidia-smi` is the standard NVIDIA management tool — it tells you the model, driver version, free memory, and utilization.

```python
# In a Colab/Kaggle cell:
!nvidia-smi
```

Typical Colab output (free tier):
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
+-------------------------------+----------------------+----------------------+
| GPU  Name                     | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf  Pwr:Usage   |         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4                 | 00000000:00:04.0 Off |                    0 |
| N/A   42C    P8    9W /  70W  |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 👶 What the numbers mean
- **Tesla T4** — a real datacenter GPU with ~2,560 CUDA cores and 16 GB of GDDR6 memory.
- **70W max power** — low-power; your gaming RTX 4090 draws 450W.
- **15360 MiB** — 15 GB free. That's how much data you can fit on the GPU.
- **0% GPU-Util** — nothing is using the GPU yet. We'll fix that.

---

## Ex 2. Check PyTorch / TensorFlow see the GPU

```python
# PyTorch
import torch
print("PyTorch sees GPU?:", torch.cuda.is_available())
print("Device name:      ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("CUDA version:     ", torch.version.cuda)
```

```python
# If you use TensorFlow instead
import tensorflow as tf
print("TF GPUs:", tf.config.list_physical_devices("GPU"))
```

If these return empty / False, your runtime type is CPU-only — go back to Runtime → Change runtime type and pick GPU.

---

## Ex 3. NumPy baseline — CPU vector add

### 👶 What this does
We'll add two arrays of 10 million numbers on the CPU using NumPy. This is our **baseline** — whatever speedup we see from the GPU will be measured against this number.

```python
import numpy as np
import time

N = 10_000_000     # 10 million elements

a_cpu = np.random.rand(N).astype(np.float32)
b_cpu = np.random.rand(N).astype(np.float32)

t0 = time.perf_counter()
c_cpu = a_cpu + b_cpu
t1 = time.perf_counter()

print(f"NumPy (CPU) vector add: {(t1 - t0)*1000:.3f} ms")
print(f"First 5 results: {c_cpu[:5]}")
```

**Typical Colab CPU time:** ~20-40 ms for 10 M float adds.

---

## Ex 4. CuPy — NumPy on the GPU

### 👶 What this does
**CuPy** is NumPy with the word `np` replaced by `cp`. Same API. But the arrays live on the GPU and operations run on thousands of cores in parallel.

```python
import cupy as cp
import time

N = 10_000_000
a_gpu = cp.random.rand(N, dtype=cp.float32)
b_gpu = cp.random.rand(N, dtype=cp.float32)

# Warmup — the first GPU op includes kernel compilation and allocation
for _ in range(5):
    _ = a_gpu + b_gpu
cp.cuda.Stream.null.synchronize()    # wait for the GPU to finish

# Measure
t0 = time.perf_counter()
c_gpu = a_gpu + b_gpu
cp.cuda.Stream.null.synchronize()    # MUST wait; GPU calls are async
t1 = time.perf_counter()

print(f"CuPy (GPU) vector add: {(t1 - t0)*1000:.3f} ms")
print(f"First 5 results (back on CPU for printing): {cp.asnumpy(c_gpu[:5])}")
```

**Typical Colab T4 time:** ~0.5 ms — that's **~50× faster than CPU** for this operation.

### 👶 Important detail — `synchronize()`
GPU operations are **asynchronous**. When you write `c_gpu = a_gpu + b_gpu`, the CPU instantly returns — the GPU might not be done yet. If you time without `synchronize()`, you'll measure how long it took to **submit** the work, not how long the work actually took. Always sync before measuring.

---

## Ex 5. Side-by-side comparison

### 👶 What this does
Let's put CPU NumPy and GPU CuPy head to head on the *same* problem, properly warmed up and synchronized.

```python
import numpy as np, cupy as cp, time

def bench_numpy(n, trials=10):
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    for _ in range(3): _ = a + b     # warmup
    t0 = time.perf_counter()
    for _ in range(trials):
        _ = a + b
    t1 = time.perf_counter()
    return (t1 - t0) / trials * 1000

def bench_cupy(n, trials=10):
    a = cp.random.rand(n, dtype=cp.float32)
    b = cp.random.rand(n, dtype=cp.float32)
    for _ in range(3): _ = a + b
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(trials):
        _ = a + b
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / trials * 1000

for N in [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
    cpu_ms = bench_numpy(N)
    gpu_ms = bench_cupy(N)
    speedup = cpu_ms / gpu_ms
    print(f"N={N:>12,}   CPU {cpu_ms:>8.3f} ms   GPU {gpu_ms:>8.3f} ms   speedup {speedup:>7.1f}×")
```

**Expected pattern:**
```
N=      10,000    CPU    0.020 ms   GPU    0.050 ms   speedup    0.4×
N=     100,000    CPU    0.180 ms   GPU    0.060 ms   speedup    3.0×
N=   1,000,000    CPU    2.1   ms   GPU    0.15  ms   speedup   14×
N=  10,000,000    CPU   22    ms   GPU    0.5   ms   speedup   44×
N= 100,000,000    CPU  220    ms   GPU    4.2   ms   speedup   52×
```

### 👶 Observations
- **For tiny arrays the CPU wins!** Kernel launch overhead + memory transfer kills the GPU advantage when N is small. There's a **break-even point** around 100K elements where the GPU starts to pay off.
- **The bigger the problem, the bigger the speedup.** This is exactly the "colour 1000 sheep" story from the theory.
- **The speedup saturates** — eventually you hit the GPU's memory bandwidth limit.

---

## Ex 6. Amdahl's Law — live

### 👶 What this does
Let's prove Amdahl's Law with real code. We build a program that has a **serial** part (runs on the CPU) and a **parallel** part (runs on the GPU). We'll vary how much is parallel, measure the total time, and plot the speedup ceiling.

```python
import numpy as np, cupy as cp, time, matplotlib.pyplot as plt

def mixed_program(P_parallel, parallel_work=10_000_000, serial_work=100_000):
    """
    P_parallel = fraction of work that is GPU-accelerated
    parallel_work = # of float adds we would do if everything were parallel
    serial_work = # of float adds forced to run on CPU
    """
    # Serial part on CPU
    x = np.random.rand(serial_work).astype(np.float32)
    t0 = time.perf_counter()
    _ = x + x
    t_serial = time.perf_counter() - t0

    # Parallel part — run fraction P on GPU
    work_gpu = int(parallel_work * P_parallel)
    work_cpu_leftover = parallel_work - work_gpu

    # GPU portion
    y = cp.random.rand(work_gpu, dtype=cp.float32) if work_gpu > 0 else cp.zeros(1, cp.float32)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    _ = y + y
    cp.cuda.Stream.null.synchronize()
    t_par = time.perf_counter() - t0

    # CPU leftover
    z = np.random.rand(work_cpu_leftover).astype(np.float32) if work_cpu_leftover > 0 else np.zeros(1, np.float32)
    t0 = time.perf_counter()
    _ = z + z
    t_cpu_leftover = time.perf_counter() - t0

    return t_serial + t_par + t_cpu_leftover

baseline = mixed_program(P_parallel=0.0)    # everything on CPU
print(f"Baseline (all CPU): {baseline*1000:.3f} ms")

results = []
for P in [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]:
    t = mixed_program(P_parallel=P)
    speedup = baseline / t
    results.append((P, t*1000, speedup))
    print(f"P={P:>4}  → total {t*1000:>7.3f} ms   speedup {speedup:>5.2f}×")
```

### 👶 What you should see
As `P` goes up, the speedup goes up — but slowly at first, then rapidly near P = 0.95+. That's exactly Amdahl's Law: a small serial chunk is a big deal.

---

## Ex 7. First taste of `numba.cuda` — writing your own kernel

### 👶 What this does
CuPy is great when your operation maps to a NumPy function. But sometimes you need to write a **custom** GPU function ("kernel"). **Numba** lets you write CUDA kernels in pure Python — it compiles them to GPU machine code at runtime.

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(a, b, c):
    i = cuda.grid(1)                  # which thread am I?
    if i < a.size:
        c[i] = a[i] + b[i]

N = 10_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.zeros_like(a)

# Move data to device
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(a)

# Launch config — 256 threads per block, enough blocks to cover N
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
cuda.synchronize()

# Copy result back
c = d_c.copy_to_host()
print(f"First 5: {c[:5]}")
print(f"Correct? {np.allclose(c, a + b)}")
```

### 👶 What just happened
1. **`@cuda.jit`** — "compile this Python function as a CUDA kernel."
2. **`cuda.grid(1)`** — asks "what's my unique thread index?" This is where the parallelism happens. 10 million threads run simultaneously; each grabs one index `i` and handles element `a[i] + b[i]`.
3. **`threads_per_block = 256`** — we launch the kernel with 256 threads per block. The GPU schedules these in groups of 32 (called **warps** — see Lec 2).
4. **`blocks_per_grid = ceil(N / 256)`** — enough blocks to cover all 10 M elements. Each block has 256 threads, so we need N/256 blocks.
5. **`cuda.to_device` / `copy_to_host`** — explicit data transfer across PCIe. This is the slowest step — keep it to a minimum.

### 👶 Why write kernels when CuPy exists?
- When CuPy doesn't have the operation you need.
- When you want to **fuse** several operations into one kernel (avoid memory bounces).
- To learn what's *actually* happening under the hood.

---

## Ex 8. Memory transfer is the tax — measure it

### 👶 What this does
The theory said "data transfer over PCIe is the slowest step." Let's measure it with our own eyes.

```python
import numpy as np, cupy as cp, time

N = 100_000_000
a_cpu = np.random.rand(N).astype(np.float32)

# Measure CPU → GPU transfer
t0 = time.perf_counter()
a_gpu = cp.asarray(a_cpu)
cp.cuda.Stream.null.synchronize()
t1 = time.perf_counter()
h2d_ms = (t1 - t0) * 1000

# Measure computation on GPU
t0 = time.perf_counter()
b_gpu = a_gpu * 2 + 1
cp.cuda.Stream.null.synchronize()
t1 = time.perf_counter()
compute_ms = (t1 - t0) * 1000

# Measure GPU → CPU transfer
t0 = time.perf_counter()
b_cpu = cp.asnumpy(b_gpu)
t1 = time.perf_counter()
d2h_ms = (t1 - t0) * 1000

print(f"Host → Device transfer: {h2d_ms:.2f} ms   ({N*4 / 1e9 / (h2d_ms/1000):.2f} GB/s)")
print(f"Compute on GPU:         {compute_ms:.2f} ms")
print(f"Device → Host transfer: {d2h_ms:.2f} ms   ({N*4 / 1e9 / (d2h_ms/1000):.2f} GB/s)")
print()
print(f"Compute is {h2d_ms/compute_ms:.1f}× FASTER than the transfer alone.")
```

### 👶 The headline number
On a Colab T4 you'll see something like:
```
Host → Device:  200 ms   (2 GB/s over PCIe 3)
Compute:          5 ms
Device → Host:  200 ms
```
**Transferring data takes 40× longer than the computation itself.** This is why in real GPU code you:
- Move data to the GPU **once**.
- Do **many** operations on it.
- Move the final result back **once**.

Bouncing back and forth for every tiny op will **annihilate** your speedup.

---

## 🧭 What we learned

1. **`nvidia-smi`** is your first check. (Ex 1–2)
2. **CuPy = NumPy on the GPU.** Same API, huge speedup for large N. (Ex 3–5)
3. **Break-even point.** Small arrays are faster on CPU. (Ex 5)
4. **Amdahl's Law is real.** Speedup is capped by the serial fraction. (Ex 6)
5. **Custom kernels via numba** when CuPy isn't enough. (Ex 7)
6. **PCIe is slow.** Keep data on the GPU as long as possible. (Ex 8)

---

> **Next:** [🎯 PRACTICE →](gpu_lec01_intro_practice.md) · [← 📖 THEORY](gpu_lec01_intro_theory.md)
>
> *GPU Programming · Lec 1 · github.com/rpaut03l/TS-02*
