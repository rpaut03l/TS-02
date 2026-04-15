# 🎯 GPU Programming Lec 1 — Introduction: PRACTICE

### *Google Colab notebook · free T4 GPU · paste cells one at a time*

> **Nav:** [← Lec 1 README](README.md) | [📖 THEORY](gpu_lec01_intro_theory.md) | [💻 CODE](gpu_lec01_intro_code.md) | **PRACTICE**

---

## 🏗️ How to use this file

1. Open [colab.research.google.com](https://colab.research.google.com) → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. Verify the GPU is attached with Cell 1.
4. Run the rest of the cells top-to-bottom.
5. Try the **Challenge Exercises** at the end.

No local setup needed. Colab's free tier has everything.

---

## Cell 1 — Confirm GPU + environment

### 👶 What this does
Prove you have a real GPU and that Python can see it. If this fails, change the runtime type before going further.

```python
!nvidia-smi
```

```python
import torch, numpy, sys
print("Python:", sys.version.split()[0])
print("NumPy :", numpy.__version__)
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("GPU?  :", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—")
```

Expected: **Tesla T4**, **CUDA 12.x**, `GPU? True`.

---

## Cell 2 — Install CuPy (if not already available)

```python
# Colab usually has cupy pre-installed; if not, this installs it for CUDA 12
!pip install -q cupy-cuda12x
import cupy as cp
print("CuPy :", cp.__version__)
print("CUDA compute capability:", cp.cuda.Device(0).compute_capability)
```

---

## Cell 3 — The moment of truth: CuPy beats NumPy

### 👶 What this does
Run a big vector operation on CPU (NumPy) and GPU (CuPy). Measure both. See the speedup with your own eyes.

```python
import numpy as np, cupy as cp, time

N = 50_000_000

# CPU version
a_cpu = np.random.rand(N).astype(np.float32)
b_cpu = np.random.rand(N).astype(np.float32)

# Warmup
for _ in range(3): _ = a_cpu + b_cpu

t0 = time.perf_counter()
for _ in range(20):
    c_cpu = a_cpu + b_cpu
cpu_time = (time.perf_counter() - t0) / 20 * 1000

# GPU version
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_cpu)

for _ in range(3): _ = a_gpu + b_gpu
cp.cuda.Stream.null.synchronize()

t0 = time.perf_counter()
for _ in range(20):
    c_gpu = a_gpu + b_gpu
cp.cuda.Stream.null.synchronize()
gpu_time = (time.perf_counter() - t0) / 20 * 1000

print(f"CPU (NumPy):  {cpu_time:>7.2f} ms")
print(f"GPU (CuPy) :  {gpu_time:>7.2f} ms")
print(f"Speedup    :  {cpu_time / gpu_time:>7.1f}×")

# Verify correctness
print(f"\nResults match? {np.allclose(c_cpu, cp.asnumpy(c_gpu))}")
```

### 👶 What you should see
Roughly a **30-50× speedup** for the T4 on 50 M float adds. If the speedup is less, check that the runtime is set to GPU (not CPU).

---

## Cell 4 — The break-even point

### 👶 What this does
For **tiny** problems the CPU is actually faster (GPU has launch overhead). Find the size at which GPU starts winning.

```python
import numpy as np, cupy as cp, time
import matplotlib.pyplot as plt

def bench(fn, trials=50):
    for _ in range(3): fn()
    if hasattr(fn, "__self__") or True:  # sync for cupy
        try: cp.cuda.Stream.null.synchronize()
        except: pass
    t0 = time.perf_counter()
    for _ in range(trials):
        fn()
    try: cp.cuda.Stream.null.synchronize()
    except: pass
    return (time.perf_counter() - t0) / trials * 1000

sizes = [10**k for k in range(2, 9)]   # 100 to 100M
cpu_times, gpu_times = [], []

for N in sizes:
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    ga = cp.asarray(a); gb = cp.asarray(b)
    cpu_times.append(bench(lambda: a + b))
    gpu_times.append(bench(lambda: ga + gb))

print(f"{'N':>12} {'CPU (ms)':>12} {'GPU (ms)':>12} {'speedup':>10}")
for N, c, g in zip(sizes, cpu_times, gpu_times):
    print(f"{N:>12,} {c:>12.4f} {g:>12.4f} {c/g:>10.2f}x")

plt.figure(figsize=(7, 4))
plt.loglog(sizes, cpu_times, "o-", label="NumPy (CPU)")
plt.loglog(sizes, gpu_times, "s-", label="CuPy (GPU)")
plt.xlabel("N (elements)"); plt.ylabel("time (ms)")
plt.title("CPU vs GPU — vector add"); plt.grid(which="both", alpha=0.3); plt.legend()
plt.show()
```

### 👶 What the chart tells you
- The CPU line is a **straight line** on log-log axes — the time is proportional to N (serial processing of one thing at a time).
- The GPU line is **flat at first** (launch overhead dominates) then goes up — but much more slowly.
- The **lines cross** somewhere around 100K-1M elements on a T4. That's the **break-even point** — below it, just use the CPU.

---

## Cell 5 — Matrix multiply — where GPUs really shine

### 👶 What this does
Matrix multiply is the core operation of every neural network. We'll multiply two 4096×4096 matrices on CPU and GPU and see how the GPU absolutely crushes it.

```python
import numpy as np, cupy as cp, time

N = 4096

A_cpu = np.random.rand(N, N).astype(np.float32)
B_cpu = np.random.rand(N, N).astype(np.float32)

# CPU
t0 = time.perf_counter()
C_cpu = A_cpu @ B_cpu
cpu_time = time.perf_counter() - t0
print(f"CPU matmul {N}x{N}: {cpu_time*1000:.1f} ms")

# GPU
A_gpu = cp.asarray(A_cpu); B_gpu = cp.asarray(B_cpu)
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
C_gpu = A_gpu @ B_gpu
cp.cuda.Stream.null.synchronize()
gpu_time = time.perf_counter() - t0
print(f"GPU matmul {N}x{N}: {gpu_time*1000:.1f} ms")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")

# How many FLOPs was that?
flops = 2 * N * N * N
print(f"\nTheoretical work: {flops/1e9:.1f} GFLOPs")
print(f"CPU throughput: {flops / cpu_time / 1e9:.1f} GFLOPS")
print(f"GPU throughput: {flops / gpu_time / 1e12:.2f} TFLOPS")
```

### 👶 What the numbers mean
A T4 will typically deliver ~**5-8 TFLOPS** on float32 matmul — that's **trillions** of floating-point operations per second. A CPU (the Colab host) gets maybe 100-300 GFLOPS — **20-50× less**.

---

## Cell 6 — Amdahl's Law — see the ceiling

### 👶 What this does
Build a program with a tiny serial part and a big parallel part. Vary the serial fraction and measure the speedup. The plot will look exactly like the Amdahl's Law curve from theory.

```python
import numpy as np, cupy as cp, time, matplotlib.pyplot as plt

def mixed(serial_ms_target, gpu_work_n=20_000_000):
    """Serial CPU busy-wait + parallel GPU add."""
    # Serial chunk on CPU — fake it with a small-ish numpy op scaled to target
    serial_data = np.random.rand(int(serial_ms_target * 200_000)).astype(np.float32)
    t0 = time.perf_counter()
    _ = serial_data * 2.0 + 1.0
    t_s = time.perf_counter() - t0

    # Parallel chunk on GPU
    a = cp.random.rand(gpu_work_n, dtype=cp.float32)
    b = cp.random.rand(gpu_work_n, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    _ = a + b
    cp.cuda.Stream.null.synchronize()
    t_p = time.perf_counter() - t0
    return t_s + t_p, t_s, t_p

# Baseline total
total, s, p = mixed(serial_ms_target=5)
print(f"Baseline: total {total*1000:.3f} ms | serial {s*1000:.3f} ms | parallel {p*1000:.3f} ms")

# What's the parallel fraction?
P_parallel = p / total
print(f"Parallel fraction P ≈ {P_parallel:.3f}")

# Theoretical Amdahl max speedup (pretend serial CPU is N=1 and GPU is "N workers")
def amdahl(P, N): return 1 / ((1 - P) + P / N)

N_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 10000]
speedups = [amdahl(P_parallel, n) for n in N_vals]

plt.figure(figsize=(7, 4))
plt.plot(N_vals, speedups, "o-")
plt.axhline(1/(1-P_parallel), ls="--", c="r", label=f"Max speedup = {1/(1-P_parallel):.1f}×")
plt.xscale("log"); plt.xlabel("# workers N"); plt.ylabel("speedup vs serial")
plt.title(f"Amdahl's Law, P_parallel = {P_parallel:.3f}"); plt.legend()
plt.grid(which="both", alpha=0.3); plt.show()
```

### 👶 Reading the chart
The curve climbs fast at first then **plateaus** at `1 / (1 − P)`. No matter how many workers you add, you can't go past the red dashed line. That's Amdahl's Law.

---

## Cell 7 — First custom kernel with `numba.cuda`

```python
from numba import cuda
import numpy as np

@cuda.jit
def saxpy(a, x, y, out):
    """out[i] = a * x[i] + y[i]  (SAXPY — the classic benchmark)"""
    i = cuda.grid(1)
    if i < x.size:
        out[i] = a * x[i] + y[i]

N = 10_000_000
x = np.random.rand(N).astype(np.float32)
y = np.random.rand(N).astype(np.float32)
out = np.zeros_like(x)

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array_like(x)

threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

saxpy[blocks_per_grid, threads_per_block](2.5, d_x, d_y, d_out)
cuda.synchronize()

out = d_out.copy_to_host()
print(f"First 5: {out[:5]}")
print(f"Correct? {np.allclose(out, 2.5 * x + y)}")
```

### 👶 What you did
You wrote CUDA in **pure Python**. No C++, no `nvcc`, no build system. Numba compiled your function to GPU machine code at runtime. This is the "training wheels" version of CUDA — Lec 2 shows the full C/C++ version.

---

## Cell 8 — Memory transfer cost (the PCIe tax)

```python
import numpy as np, cupy as cp, time

sizes_mb = [1, 10, 100, 500, 1000]
print(f"{'Size (MB)':>12} {'H2D (ms)':>12} {'D2H (ms)':>12} {'H2D (GB/s)':>14} {'D2H (GB/s)':>14}")

for mb in sizes_mb:
    n = (mb * 1024 * 1024) // 4   # float32 = 4 bytes
    host = np.random.rand(n).astype(np.float32)

    # Warmup
    _ = cp.asarray(host); cp.cuda.Stream.null.synchronize()

    # H2D
    t0 = time.perf_counter()
    dev = cp.asarray(host)
    cp.cuda.Stream.null.synchronize()
    h2d_ms = (time.perf_counter() - t0) * 1000

    # D2H
    t0 = time.perf_counter()
    host_back = cp.asnumpy(dev)
    d2h_ms = (time.perf_counter() - t0) * 1000

    h2d_gbs = mb / 1024 / (h2d_ms / 1000)
    d2h_gbs = mb / 1024 / (d2h_ms / 1000)
    print(f"{mb:>12} {h2d_ms:>12.2f} {d2h_ms:>12.2f} {h2d_gbs:>14.2f} {d2h_gbs:>14.2f}")
```

### 👶 What you're looking at
- **H2D / D2H** = Host-to-Device / Device-to-Host transfer time
- Colab's PCIe gives you ~6-12 GB/s. Inside the GPU, HBM is ~300 GB/s. So **GPU memory is ~30-50× faster than the PCIe bus**.
- If your workload does a round-trip for every tiny computation, you waste most of your time on transfers — exactly what the theory section warned about.

---

## 🏋️ Challenge Exercises

### Challenge 1 — Find the break-even point exactly
Using Cell 4's benchmark, locate the smallest N for which `gpu_time < cpu_time`. Share the number in your notebook. (Typical answer on T4: between 50K and 500K elements.)

### Challenge 2 — Bigger matmul, bigger speedup
Rerun Cell 5 with `N = 512`, `1024`, `2048`, `4096`, `8192`. Plot CPU time and GPU time vs N. Which grows more steeply? (Answer: CPU scales ~O(N³), GPU also scales O(N³) but with a much smaller constant factor.)

### Challenge 3 — Try SAXPY with different block sizes
In Cell 7, rerun the saxpy kernel with `threads_per_block` in `[32, 64, 128, 256, 512, 1024]`. Time each. Which is fastest on the T4? Why? (Hint: warp size is 32; block sizes should be a multiple of 32. Also check occupancy.)

### Challenge 4 — Amdahl stress test
Pick a target speedup of 50×. Using `max speedup = 1 / (1 − P)`, solve for the required P. (Answer: P = 0.98.) Then compute what P needs to be for 100× and 1000×. (0.99 and 0.999.) Each additional 9 in the parallel fraction adds a 10× to the ceiling.

### Challenge 5 — Round-trip penalty
Modify Cell 3 so that you **move data to the GPU, do 1 op, copy back to host, and repeat 20 times** instead of keeping data on the GPU. Measure the total time. Compare. You should see the GPU version becoming *slower than* the CPU because of PCIe overhead. Write a one-line comment explaining what happened.

### Challenge 6 — Discover `torch.cuda`
Repeat Cell 3 using `torch.tensor` on `device='cuda'` instead of CuPy. Same speedup? (Usually yes.) This is how PyTorch uses the GPU under the hood.

---

## 📝 Wrap-up — reflect

Before you close the notebook, answer these in a markdown cell:

1. What's the **smallest problem size** for which the GPU beats the CPU on vector add?
2. What was the speedup you got on 4096×4096 matrix multiply? **Why is matmul a better case for the GPU than vector add?** (Hint: think operations per loaded byte — **arithmetic intensity**.)
3. Using the Amdahl curve, what's the **maximum speedup** for your `P_parallel` from Cell 6? How many workers do you need to hit 90% of that maximum?

---

> **Next:** [← Lec 1 THEORY](gpu_lec01_intro_theory.md) · [Lec 2 →](../Lec_02_CPU_CUDA_Basics/README.md)
>
> *GPU Programming · Lec 1 · github.com/rpaut03l/TS-02*
