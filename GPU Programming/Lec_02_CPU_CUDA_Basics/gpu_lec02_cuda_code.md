# 💻 GPU Programming Lec 2 — CPU · CUDA · GPU Architecture: CODE

### *Real CUDA C (nvcc) + Numba equivalents — runnable on Google Colab*

> **Nav:** [← Lec 2 README](README.md) | [📖 THEORY](gpu_lec02_cuda_theory.md) | **CODE** | [🎯 PRACTICE →](gpu_lec02_cuda_practice.md)

---

## 🏗️ Setup on Google Colab

1. **Runtime → Change runtime type → T4 GPU → Save**
2. `nvcc` (NVIDIA's CUDA compiler) is pre-installed on Colab — you can compile and run CUDA C directly.
3. Every snippet below has an equivalent **Python numba version** at the end if you prefer to stay in Python.

Check `nvcc` is available:
```python
!nvcc --version
```

---

## Ex 1. Hello World from the GPU

### 👶 What this does
Every CUDA tutorial starts with this: launch a kernel that prints its thread index. You'll see 16 threads all shouting "hello" — proof that your GPU is running parallel code.

Write the CUDA C source to a file using Colab's `%%writefile` magic:
```python
%%writefile hello.cu
#include <cstdio>

__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d (block %d, warp lane %d)\n",
           tid, blockIdx.x, threadIdx.x % 32);
}

int main() {
    // Launch 2 blocks × 8 threads = 16 threads total
    hello_kernel<<<2, 8>>>();
    cudaDeviceSynchronize();   // wait for the kernel to finish printing
    return 0;
}
```

Compile and run:
```python
!nvcc hello.cu -o hello
!./hello
```

**Expected output (order may vary — threads print concurrently):**
```
Hello from thread 0 (block 0, warp lane 0)
Hello from thread 1 (block 0, warp lane 1)
Hello from thread 2 (block 0, warp lane 2)
...
Hello from thread 15 (block 1, warp lane 7)
```

### 👶 What the code means
- `__global__` — "this function is a CUDA kernel. Launched from CPU, runs on GPU."
- `blockIdx.x * blockDim.x + threadIdx.x` — the **global thread index** formula. You'll see this everywhere.
- `<<<2, 8>>>` — launch with **2 blocks of 8 threads**. Total = 16 threads.
- `cudaDeviceSynchronize()` — wait until the GPU finishes, otherwise your `printf`s might still be in-flight when `main` exits.

---

## Ex 2. Vector Add in CUDA C — the canonical example

### 👶 What this does
Add two arrays of 1 million floats on the GPU. This is vector add in its full CUDA C form — host memory allocation, explicit `cudaMemcpy`, kernel launch, copy back, free.

```python
%%writefile vecadd.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vec_add(const float *a, const float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {               // guard against the "tail" threads
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20;     // 1,048,576
    size_t bytes = N * sizeof(float);

    // 1. Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f * i;
        h_b[i] = 2.0f * i;
    }

    // 2. Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 3. Copy host → device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 4. Launch kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;   // ceil division
    vec_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);

    // 5. Copy device → host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 6. Verify one element
    printf("h_c[100] = %f (expected %f)\n", h_c[100], 3.0f * 100);

    // 7. Free
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

```python
!nvcc vecadd.cu -o vecadd && ./vecadd
```

**Expected output:**
```
h_c[100] = 300.000000 (expected 300.000000)
```

### 👶 The 7-step structure
Look at the comments in `main`. This matches **exactly** the 7-step workload flow from the theory:
1. Allocate host → 2. Allocate device → 3. H2D → 4. Launch → 5. D2H → 6. Use → 7. Free.

Every CUDA program has this structure. Memorize it.

---

## Ex 3. Thread indexing cheat-sheet

### 👶 What this does
CUDA gives you three indices per thread: `threadIdx`, `blockIdx`, `blockDim`. Combining them lets you compute 1D, 2D, or 3D global indices.

```python
%%writefile indexing.cu
#include <cstdio>

__global__ void show_indices() {
    // 1D version
    int tid_1d = blockIdx.x * blockDim.x + threadIdx.x;

    // 2D version (if you launched with a 2D grid & block)
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_2d = tid_y * (gridDim.x * blockDim.x) + tid_x;

    printf("Block (%d,%d)  Thread (%d,%d)  1D=%d  2D=%d\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y,
           tid_1d, global_2d);
}

int main() {
    dim3 block(4, 2);    // 4 threads in x, 2 in y → 8 threads per block
    dim3 grid(2, 2);     // 2 blocks in x, 2 in y → 4 blocks
    show_indices<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```python
!nvcc indexing.cu -o indexing && ./indexing
```

### 👶 The important built-in variables
Inside every kernel, CUDA gives you:
- `threadIdx.{x,y,z}` — your index within your block
- `blockIdx.{x,y,z}` — your block's index within the grid
- `blockDim.{x,y,z}` — size of a block (how many threads it has)
- `gridDim.{x,y,z}` — size of the grid (how many blocks it has)

Mastering these 4 variables is mastering CUDA indexing.

---

## Ex 4. Warp divergence demo

### 👶 What this does
Write a kernel with an `if` that splits the warp in half. Time it. Then write the same kernel where the branch is taken by all-or-none threads. See the difference.

```python
%%writefile divergence.cu
#include <cstdio>
#include <cuda_runtime.h>

// Bad: half the threads take branch A, half take branch B
__global__ void diverged(float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = 0.0f;
    if (i % 2 == 0) {                // even threads — warp diverges
        for (int k = 0; k < 1000; ++k) x += sinf((float)k);
    } else {
        for (int k = 0; k < 1000; ++k) x += cosf((float)k);
    }
    out[i] = x;
}

// Good: whole warps agree on the branch (warp-aligned branching)
__global__ void coherent(float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int warp_id = (i / 32);         // which warp am I in?
    float x = 0.0f;
    if (warp_id % 2 == 0) {         // whole warps take the same branch
        for (int k = 0; k < 1000; ++k) x += sinf((float)k);
    } else {
        for (int k = 0; k < 1000; ++k) x += cosf((float)k);
    }
    out[i] = x;
}

int main() {
    const int N = 1 << 20;
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    diverged<<<blocks, threads>>>(d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    diverged<<<blocks, threads>>>(d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_diverged;
    cudaEventElapsedTime(&ms_diverged, start, stop);

    cudaEventRecord(start);
    coherent<<<blocks, threads>>>(d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_coherent;
    cudaEventElapsedTime(&ms_coherent, start, stop);

    printf("Diverged: %.3f ms\n", ms_diverged);
    printf("Coherent: %.3f ms\n", ms_coherent);
    printf("Divergence slowdown: %.2fx\n", ms_diverged / ms_coherent);

    cudaFree(d_out);
    return 0;
}
```

```python
!nvcc divergence.cu -o divergence && ./divergence
```

**Expected:** `coherent` is ~1.7-2× faster than `diverged`. The divergent kernel runs both branches sequentially for every warp.

---

## Ex 5. Shared memory — a simple reduction

### 👶 What this does
Compute the sum of an array using **shared memory** inside each block. This is the classic "parallel reduction" pattern — every CUDA programmer writes it at some point.

```python
%%writefile reduce.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void block_sum(const float *in, float *block_results, int N) {
    __shared__ float tile[256];     // 256 floats = 1 KB of shared memory

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Each thread loads one element into shared memory
    tile[tid] = (gid < N) ? in[gid] : 0.0f;
    __syncthreads();                // wait for all threads to finish loading

    // 2. Tree-reduction: each iteration halves the active threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            tile[tid] += tile[tid + stride];
        }
        __syncthreads();            // wait after every step
    }

    // 3. Thread 0 writes the block's result
    if (tid == 0) {
        block_results[blockIdx.x] = tile[0];
    }
}

int main() {
    const int N = 1 << 20;       // 1M elements
    const int TPB = 256;
    const int BLOCKS = (N + TPB - 1) / TPB;

    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;   // expected sum = N

    float *d_in, *d_block_results;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_block_results, BLOCKS * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    block_sum<<<BLOCKS, TPB>>>(d_in, d_block_results, N);

    // Copy block results back and sum on host (cheap — only BLOCKS values)
    float *h_block_results = (float*)malloc(BLOCKS * sizeof(float));
    cudaMemcpy(h_block_results, d_block_results, BLOCKS * sizeof(float),
               cudaMemcpyDeviceToHost);
    float total = 0.0f;
    for (int i = 0; i < BLOCKS; ++i) total += h_block_results[i];

    printf("Sum = %.0f (expected %d)\n", total, N);

    free(h_in); free(h_block_results);
    cudaFree(d_in); cudaFree(d_block_results);
    return 0;
}
```

```python
!nvcc reduce.cu -o reduce && ./reduce
```

### 👶 What just happened
- **`__shared__ float tile[256];`** — declares a 256-float buffer that's visible to all threads in this block. Only 1 KB of shared memory.
- **`__syncthreads()`** — all threads in the block wait until everyone reaches this line. Needed after writing to shared memory and before reading it.
- **Tree reduction** — at step 1, thread 0 sums elements 0 and 128; thread 1 sums 1 and 129; etc. At step 2, thread 0 sums 0 and 64. After log₂(256) = 8 steps, thread 0 holds the total for the whole block.

This is the GPU way to think about parallelism: **halve the work each step using all active threads**.

---

## Ex 6. Measuring coalesced vs strided access

### 👶 What this does
Write two kernels that read 100 M floats. The first uses stride-1 (coalesced — fast). The second uses stride-32 (non-coalesced — slow). Measure both.

```python
%%writefile coalesce.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void stride1(const float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * 2.0f;     // thread i reads in[i]
}

__global__ void stride32(const float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = (i * 32) % N;                  // stride 32 — non-coalesced
    if (i < N) out[i] = in[j] * 2.0f;
}

int main() {
    const int N = 1 << 25;   // 32 M
    size_t bytes = N * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);

    int TPB = 256;
    int BLOCKS = (N + TPB - 1) / TPB;

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    // warmup
    stride1<<<BLOCKS, TPB>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    for (int k = 0; k < 10; ++k) stride1<<<BLOCKS, TPB>>>(d_in, d_out, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms1; cudaEventElapsedTime(&ms1, s, e);

    cudaEventRecord(s);
    for (int k = 0; k < 10; ++k) stride32<<<BLOCKS, TPB>>>(d_in, d_out, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms32; cudaEventElapsedTime(&ms32, s, e);

    printf("stride-1  (coalesced):  %.3f ms\n", ms1 / 10);
    printf("stride-32 (scattered):  %.3f ms\n", ms32 / 10);
    printf("Slowdown from bad pattern: %.1fx\n", ms32 / ms1);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

```python
!nvcc coalesce.cu -o coalesce && ./coalesce
```

**Expected:** the scattered version is 3-10× slower. **This is the cost of ignoring the memory access pattern.**

---

## Ex 7. Same 7 examples in Python (Numba) — no nvcc needed

### 👶 What this does
If CUDA C feels overwhelming, here are the key ideas in Python. Same kernels, same launch syntax, compiled by Numba at runtime.

```python
from numba import cuda
import numpy as np

# --- Vector add ---
@cuda.jit
def vec_add_py(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

N = 1_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.zeros_like(a)

d_a = cuda.to_device(a); d_b = cuda.to_device(b); d_c = cuda.device_array_like(a)
vec_add_py[(N + 255) // 256, 256](d_a, d_b, d_c)
cuda.synchronize()
c = d_c.copy_to_host()
print("Vector add:", np.allclose(c, a + b))
```

```python
# --- Block reduction with shared memory ---
@cuda.jit
def block_sum_py(in_arr, block_results):
    tile = cuda.shared.array(shape=256, dtype=np.float32)

    tid = cuda.threadIdx.x
    gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    tile[tid] = in_arr[gid] if gid < in_arr.size else 0.0
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            tile[tid] += tile[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        block_results[cuda.blockIdx.x] = tile[0]

N = 1 << 20
TPB = 256
BLOCKS = (N + TPB - 1) // TPB

a = np.ones(N, dtype=np.float32)
d_a = cuda.to_device(a)
d_block_results = cuda.device_array(BLOCKS, dtype=np.float32)

block_sum_py[BLOCKS, TPB](d_a, d_block_results)
total = d_block_results.copy_to_host().sum()
print(f"Reduction sum: {total:.0f} (expected {N})")
```

---

## 🧭 What we learned

1. **CUDA C launch syntax** — `kernel<<<blocks, threads>>>(args)` with `nvcc`. (Ex 1-2)
2. **The 7-step host → device → launch → device → host flow.** (Ex 2)
3. **Thread indexing** — `blockIdx.x * blockDim.x + threadIdx.x` in 1D, dim3 for 2D/3D. (Ex 3)
4. **Warp divergence** is real and expensive. Align branches to warp boundaries. (Ex 4)
5. **Shared memory + `__syncthreads`** for block-local reductions. (Ex 5)
6. **Coalesced access** — stride-1 wins, scattered loses 3-10×. (Ex 6)
7. **Python equivalents via Numba** when you'd rather stay in Python. (Ex 7)

---

> **Next:** [🎯 PRACTICE →](gpu_lec02_cuda_practice.md) · [← 📖 THEORY](gpu_lec02_cuda_theory.md)
>
> *GPU Programming · Lec 2 · github.com/rpaut03l/TS-02*
