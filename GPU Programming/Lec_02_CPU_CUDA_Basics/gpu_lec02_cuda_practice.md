# 🎯 GPU Programming Lec 2 — CPU · CUDA · GPU Architecture: PRACTICE

### *Google Colab notebook · write, compile, and run real CUDA on a free T4 GPU*

> **Nav:** [← Lec 2 README](README.md) | [📖 THEORY](gpu_lec02_cuda_theory.md) | [💻 CODE](gpu_lec02_cuda_code.md) | **PRACTICE**

---

## 🏗️ How to use this file

1. Open [colab.research.google.com](https://colab.research.google.com) → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU**.
3. Paste each cell below into a fresh cell and run it in order.
4. You'll compile **real CUDA C** with `nvcc` and run it on the T4 — just like a workstation with a GPU.

---

## Cell 1 — Verify nvcc, driver, and GPU

```python
!nvcc --version
!nvidia-smi
```

If both commands succeed, you have everything needed to compile and run CUDA C.

---

## Cell 2 — Hello world CUDA kernel (compile + run)

### 👶 What this does
Write a 10-line CUDA C file, compile with `nvcc`, and run. Every thread prints its ID. If this works, everything works.

```python
%%writefile hello.cu
#include <cstdio>

__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", tid);
}

int main() {
    hello_kernel<<<1, 8>>>();       // 1 block, 8 threads
    cudaDeviceSynchronize();
    return 0;
}
```

```python
!nvcc hello.cu -o hello && ./hello
```

Expected: 8 lines, one per thread. Order may vary — they run concurrently.

---

## Cell 3 — Vector add with correctness check

```python
%%writefile vecadd.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vec_add(const float *a, const float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;     // 1M
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { h_a[i] = i; h_b[i] = 2*i; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int TPB = 256, BLK = (N + TPB - 1) / TPB;
    vec_add<<<BLK, TPB>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Correctness check
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) errors++;
    }
    printf("Errors: %d / %d\n", errors, N);
    printf("h_c[100] = %.0f (expected 300)\n", h_c[100]);

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

```python
!nvcc vecadd.cu -o vecadd && ./vecadd
```

Expected: `Errors: 0 / 1048576` and `h_c[100] = 300`.

---

## Cell 4 — Time different block sizes

### 👶 What this does
Block size matters! Too small wastes warp slots; too large hurts occupancy. Let's time the vector add with different block sizes.

```python
%%writefile bench_blocksize.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vec_add(const float *a, const float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 25;   // 32 M
    size_t bytes = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    printf("%10s %12s\n", "blocksize", "time (ms)");
    for (int bs : block_sizes) {
        int blocks = (N + bs - 1) / bs;
        // warmup
        vec_add<<<blocks, bs>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        cudaEventRecord(s);
        for (int k = 0; k < 20; ++k)
            vec_add<<<blocks, bs>>>(d_a, d_b, d_c, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        printf("%10d %12.3f\n", bs, ms / 20);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

```python
!nvcc bench_blocksize.cu -o bench_blocksize && ./bench_blocksize
```

### 👶 What to look for
Typically 128, 256, or 512 win. 32 is too small (only one warp, poor occupancy). 1024 is usually suboptimal on the T4 too (limits occupancy). Block sizes that are multiples of the warp size (32) always win over weird sizes like 100 or 300.

---

## Cell 5 — Warp divergence penalty

```python
%%writefile divergence.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void diverged(float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = 0.0f;
    if (i & 1) {
        for (int k = 0; k < 500; ++k) x += sinf((float)k);
    } else {
        for (int k = 0; k < 500; ++k) x += cosf((float)k);
    }
    out[i] = x;
}

__global__ void coherent(float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int warp = i / 32;
    float x = 0.0f;
    if (warp & 1) {
        for (int k = 0; k < 500; ++k) x += sinf((float)k);
    } else {
        for (int k = 0; k < 500; ++k) x += cosf((float)k);
    }
    out[i] = x;
}

int main() {
    const int N = 1 << 20;
    float *d_out; cudaMalloc(&d_out, N * sizeof(float));
    int TPB = 256, BLK = (N + TPB - 1) / TPB;

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    // warmup
    diverged<<<BLK, TPB>>>(d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    for (int k = 0; k < 10; ++k) diverged<<<BLK, TPB>>>(d_out, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_d; cudaEventElapsedTime(&ms_d, s, e);

    cudaEventRecord(s);
    for (int k = 0; k < 10; ++k) coherent<<<BLK, TPB>>>(d_out, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_c; cudaEventElapsedTime(&ms_c, s, e);

    printf("Diverged: %.3f ms\n", ms_d / 10);
    printf("Coherent: %.3f ms\n", ms_c / 10);
    printf("Slowdown: %.2fx\n", ms_d / ms_c);
    cudaFree(d_out);
}
```

```python
!nvcc divergence.cu -o divergence && ./divergence
```

Expected: divergent version is ~1.5-2× slower.

---

## Cell 6 — Memory coalescing measurement

```python
%%writefile coalesce.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void stride1(const float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * 2.0f;
}

__global__ void stride32(const float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = (i * 32) % N;
    if (i < N) out[i] = in[j] * 2.0f;
}

int main() {
    const int N = 1 << 25;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    int TPB = 256, BLK = (N + TPB - 1) / TPB;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

    stride1<<<BLK, TPB>>>(d_in, d_out, N); cudaDeviceSynchronize();

    cudaEventRecord(s);
    for (int k = 0; k < 20; ++k) stride1<<<BLK, TPB>>>(d_in, d_out, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms1; cudaEventElapsedTime(&ms1, s, e);

    cudaEventRecord(s);
    for (int k = 0; k < 20; ++k) stride32<<<BLK, TPB>>>(d_in, d_out, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms32; cudaEventElapsedTime(&ms32, s, e);

    printf("stride-1  (coalesced):  %.3f ms\n", ms1/20);
    printf("stride-32 (scattered):  %.3f ms\n", ms32/20);
    printf("slowdown: %.2fx\n", ms32/ms1);
    cudaFree(d_in); cudaFree(d_out);
}
```

```python
!nvcc coalesce.cu -o coalesce && ./coalesce
```

### 👶 Lesson
Same kernel body, same compute cost, **only the access pattern** differs — and non-coalesced is dramatically slower. Always design kernels so thread `i` accesses element `i`, not element `perm[i]`.

---

## Cell 7 — Shared memory reduction

```python
%%writefile reduce.cu
#include <cstdio>
#include <cuda_runtime.h>

#define TPB 256

__global__ void block_sum(const float *in, float *block_results, int N) {
    __shared__ float tile[TPB];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    tile[tid] = (gid < N) ? in[gid] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) tile[tid] += tile[tid + stride];
        __syncthreads();
    }
    if (tid == 0) block_results[blockIdx.x] = tile[0];
}

int main() {
    const int N = 1 << 20;
    const int BLK = (N + TPB - 1) / TPB;
    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_br;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_br, BLK * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    block_sum<<<BLK, TPB>>>(d_in, d_br, N);

    float *h_br = (float*)malloc(BLK * sizeof(float));
    cudaMemcpy(h_br, d_br, BLK * sizeof(float), cudaMemcpyDeviceToHost);
    float total = 0.0f;
    for (int i = 0; i < BLK; ++i) total += h_br[i];

    printf("Sum = %.0f (expected %d)\n", total, N);
    free(h_in); free(h_br);
    cudaFree(d_in); cudaFree(d_br);
}
```

```python
!nvcc reduce.cu -o reduce && ./reduce
```

---

## Cell 8 — Tiled matrix multiply (the classic)

### 👶 What this does
Matrix multiply using **shared memory tiling** — each block loads an 16×16 tile of A and B into fast shared memory, multiplies locally, moves to the next tile. This is the canonical "write fast CUDA code" exercise.

```python
%%writefile matmul_tiled.cu
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matmul_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    int n_tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < n_tiles; ++t) {
        // Load a tile of A and a tile of B into shared memory
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        // Multiply the two tiles and accumulate
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes), *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N*N; ++i) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    // warmup
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    for (int k = 0; k < 10; ++k) matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    ms /= 10;

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    // Expected: each element = N * 1 * 2 = 2 * N
    printf("C[0] = %.0f (expected %d)\n", h_C[0], 2 * N);
    double flops = 2.0 * N * N * N;
    printf("Time: %.3f ms, throughput: %.2f GFLOPS\n", ms, flops / (ms / 1000.0) / 1e9);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

```python
!nvcc matmul_tiled.cu -o matmul_tiled && ./matmul_tiled
```

### 👶 Expected on T4
Around 500-1500 GFLOPS. A hand-tuned cuBLAS would reach 5-6 TFLOPS — we have a 5-10× gap, which is normal for a textbook implementation. Optimizing further (bigger tiles, double-buffering, vectorized loads) is the subject of entire courses.

---

## Cell 9 — Trace the 7-step lifecycle live

```python
%%writefile lifecycle.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>

__global__ void add_one(float *x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] += 1.0f;
}

int main() {
    using clk = std::chrono::high_resolution_clock;
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    auto t0 = clk::now();

    // 1. CPU program starts.
    float *h = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h[i] = 0.0f;
    auto t_alloc_host = clk::now();

    // 2. Device allocation
    float *d; cudaMalloc(&d, bytes);
    auto t_alloc_dev = clk::now();

    // 3. H→D transfer
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    auto t_h2d = clk::now();

    // 4/5. Launch + execute
    add_one<<<(N + 255) / 256, 256>>>(d, N);
    cudaDeviceSynchronize();
    auto t_kernel = clk::now();

    // 6. D→H transfer
    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
    auto t_d2h = clk::now();

    // 7. Free and done
    cudaFree(d); free(h);
    auto t_done = clk::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    printf("Host alloc     : %7.3f ms\n", ms(t0, t_alloc_host));
    printf("Device alloc   : %7.3f ms\n", ms(t_alloc_host, t_alloc_dev));
    printf("H -> D copy    : %7.3f ms\n", ms(t_alloc_dev, t_h2d));
    printf("Kernel compute : %7.3f ms\n", ms(t_h2d, t_kernel));
    printf("D -> H copy    : %7.3f ms\n", ms(t_kernel, t_d2h));
    printf("Free           : %7.3f ms\n", ms(t_d2h, t_done));
    printf("TOTAL          : %7.3f ms\n", ms(t0, t_done));
}
```

```python
!nvcc lifecycle.cu -o lifecycle && ./lifecycle
```

### 👶 Reading the results
Most of your time will be in the **H→D** and **D→H** copies. The kernel compute itself is tiny. **This is the #1 GPU optimization lesson**: don't ping-pong data across PCIe.

---

## 🏋️ Challenge Exercises

### Challenge 1 — 2D kernel
Rewrite `vec_add` to process a 2D matrix: add two `N×N` matrices. Use `dim3 block(16, 16)` and `dim3 grid(N/16, N/16)`. Verify with `np.allclose`.

### Challenge 2 — Warp divergence auditor
Write a kernel with various `if` patterns and see how much they slow down. Which patterns are free, and which are expensive?
- `if (i == 0)` — diverges (only 1 thread active).
- `if (i < 100)` — diverges once.
- `if ((i / 32) % 2 == 0)` — warp-aligned, no divergence.
- `if (i % 2 == 0)` — worst case, half-diverges every warp.

### Challenge 3 — Compare shared vs global reduction
Take Cell 7's reduction and write a **naive** version that only uses global memory (no `__shared__`). Time both. Shared should win by 5-10×.

### Challenge 4 — Tile-size sweep for matmul
In Cell 8, try `TILE = 8, 16, 32`. How does performance change? Why? (Hint: too small wastes per-block launch cost; too large hurts occupancy and may exceed shared memory.)

### Challenge 5 — Async memcpy
Replace one of the `cudaMemcpy` calls with `cudaMemcpyAsync` on a stream. See how `cudaDeviceSynchronize` is now required to guarantee the copy finished. Bonus: overlap compute with a second copy using two streams.

### Challenge 6 — Compute the max CPI for the tiled matmul
From the TFLOPs number, compute the "effective arithmetic intensity" (FLOPs per byte of global memory traffic). Compare to the T4's roofline: HBM bandwidth ~320 GB/s → how many FLOPs/byte before you hit the memory wall?

---

## 📝 Wrap-up

Before closing the notebook, answer:

1. Which of the 7 lifecycle steps in Cell 9 was the **slowest**? By how much?
2. In Cell 4, which block size was fastest? Was it a power of 2?
3. In Cell 8, what was your GFLOPS? How does that compare to the T4's peak ~8 TFLOPS? What does the gap tell you about your tile-sized matmul's efficiency?

---

> **Next:** [← Lec 2 THEORY](gpu_lec02_cuda_theory.md) · [← Lec 1](../Lec_01_Introduction/README.md) · [← GPU Programming](../README.md)
>
> *GPU Programming · Lec 2 · github.com/rpaut03l/TS-02*
