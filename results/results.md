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
| matmul_naive | 2.61 s | n/a | 71.2 ms | 73.0 ms | n/a |
| matmul_tiled | 2.48 s | 1.9 ms | 8.9 ms | 6.5 ms | 2.0 ms |
| mandelbrot | 2.03 s | 365.2 ms | 79.9 ms | 18.8 ms | 4.9 ms |
| nbody | 1.03 s | 94.7 ms | 10.5 ms | 9.5 ms | 2.5 ms |
| reduction | 72.8 ms | 3.0 ms | 3.3 ms | 3.1 ms | 929 µs |
| prefix_sum | 580.6 ms | 35.7 ms | 16.9 ms | 16.4 ms | 2.7 ms |
| hash_table | 946.4 ms | n/a | 116.9 ms | 110.3 ms | n/a |

## Speed-up over pure Python (higher is better)

| Kernel | Python | NumPy | C (1 thread) | Mojo (1 thread) | Mojo (4 threads) |
|---|---:|---:|---:|---:|---:|
| matmul_naive | 1.0× | n/a | 36.7× | 35.8× | n/a |
| matmul_tiled | 1.0× | 1,327× | 280× | 383× | 1,242× |
| mandelbrot | 1.0× | 5.5× | 25.3× | 108× | 409× |
| nbody | 1.0× | 10.9× | 98.1× | 109× | 407× |
| reduction | 1.0× | 23.9× | 22.4× | 23.8× | 78.4× |
| prefix_sum | 1.0× | 16.3× | 34.4× | 35.4× | 215× |
| hash_table | 1.0× | n/a | 8.1× | 8.6× | n/a |

n/a = no sensible implementation in that language (see README).

## Start-up and compile costs

| What | Median wall time |
|---|---:|
| Python interpreter start (no-op kernel) | 16.8 ms |
| Python + import numpy (no-op kernel) | 94.6 ms |
| C binary start (no-op kernel) | 1.4 ms |
| Mojo binary start (no-op kernel) | 9.8 ms |
| gcc -O3 compile of kernels.c | 250.7 ms |
| mojo build of kernels.mojo | 1.87 s |
| mojo run kernels.mojo noop (JIT compile + run) | 1.98 s |
