# Results

## Machine

| | |
|---|---|
| cpu | Intel(R) Xeon(R) Processor @ 2.10GHz |
| logical_cpus | 4 |
| physical_cores | 4 |
| simd | sse4_2 avx avx2 fma avx512f |
| caches | L1d cache 192 KiB (4 instances), L2 cache 8 MiB (4 instances), L3 cache 260 MiB (1 instance) |
| virtualised | True |
| os | Ubuntu 24.04.5 LTS (kernel 6.18.44-fc-v233) |
| python | 3.13.18 |
| numpy | 2.5.3 |
| gcc | gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| mojo | Mojo 1.2.0 (2016d0cd) |
| c_flags | -O3 -march=native |

Warm-up runs: 1. Timed repeats are adaptive: at least 5, otherwise enough to fill ~1 s, capped at 200 (see the `repeats` column of results.csv). Times are medians of the timed repeats; input generation is excluded.

## Problem sizes

| Kernel | n | Meaning |
|---|---:|---|
| matmul_naive | 384 | n x n float64 matrices |
| matmul_tiled | 384 | n x n float64 matrices |
| mandelbrot | 640 | n x n pixels, 200 max iterations |
| nbody | 1,024 | n bodies, 5 steps, O(n^2) forces |
| reduction | 10,000,000 | n int64 values |
| prefix_sum | 10,000,000 | n int64 values |
| hash_table | 2,000,000 | n inserts + n lookups (50% hits) |

## Median time

| Kernel | Python | NumPy | C (1 thread) | Mojo (1 thread) | Mojo (4 threads) |
|---|---:|---:|---:|---:|---:|
| matmul_naive | 2.60 s | n/a | 69.1 ms | 70.8 ms | n/a |
| matmul_tiled | 2.51 s | 1.9 ms | 10.4 ms | 6.3 ms | 2.4 ms |
| mandelbrot | 2.08 s | 350.7 ms | 88.4 ms | 19.0 ms | 5.0 ms |
| nbody | 1.06 s | 94.4 ms | 10.4 ms | 9.4 ms | 2.6 ms |
| reduction | 75.2 ms | 3.0 ms | 3.4 ms | 3.8 ms | 851 µs |
| prefix_sum | 593.3 ms | 37.6 ms | 18.3 ms | 16.9 ms | 3.8 ms |
| hash_table | 1.04 s | n/a | 111.3 ms | 109.7 ms | n/a |

## Speed-up over pure Python (higher is better)

| Kernel | Python | NumPy | C (1 thread) | Mojo (1 thread) | Mojo (4 threads) |
|---|---:|---:|---:|---:|---:|
| matmul_naive | 1.0× | n/a | 37.7× | 36.8× | n/a |
| matmul_tiled | 1.0× | 1,335× | 241× | 400× | 1,049× |
| mandelbrot | 1.0× | 5.9× | 23.5× | 110× | 413× |
| nbody | 1.0× | 11.2× | 102× | 112× | 409× |
| reduction | 1.0× | 24.7× | 22.2× | 20.0× | 88.3× |
| prefix_sum | 1.0× | 15.8× | 32.4× | 35.0× | 156× |
| hash_table | 1.0× | n/a | 9.4× | 9.5× | n/a |

n/a = no sensible implementation in that language (see README).

## Start-up and compile costs

| What | Median wall time |
|---|---:|
| Python interpreter start (no-op kernel) | 17.2 ms |
| Python + import numpy (no-op kernel) | 98.2 ms |
| C binary start (no-op kernel) | 1.3 ms |
| Mojo binary start (no-op kernel) | 8.8 ms |
| gcc -O3 compile of kernels.c | 253.0 ms |
| mojo build of kernels.mojo | 1.89 s |
| mojo run kernels.mojo noop (JIT compile + run) | 1.89 s |
