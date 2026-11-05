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
| word_count | 4,000,000 | n words (~22 MB of text), ~4,700 distinct |

## Median time

| Kernel | Python | NumPy | C (1 thread) | C OpenMP (4 threads) | Mojo (1 thread) | Mojo (4 threads) |
|---|---:|---:|---:|---:|---:|---:|
| matmul_naive | 2.48 s | n/a | 69.0 ms | n/a | 72.1 ms | n/a |
| matmul_tiled | 2.46 s | 1.9 ms | 9.2 ms | 3.6 ms | 6.3 ms | 1.7 ms |
| mandelbrot | 2.04 s | 359.0 ms | 82.5 ms | 21.6 ms | 19.0 ms | 4.8 ms |
| nbody | 1.00 s | 88.8 ms | 10.5 ms | 2.9 ms | 9.3 ms | 2.5 ms |
| reduction | 73.6 ms | 3.0 ms | 3.2 ms | 860 µs | 3.1 ms | 839 µs |
| prefix_sum | 580.3 ms | 36.2 ms | 16.9 ms | 2.7 ms | 14.7 ms | 3.2 ms |
| hash_table | 911.5 ms | n/a | 96.5 ms | n/a | 102.4 ms | n/a |
| word_count | 615.4 ms | 517.7 ms | 79.5 ms | n/a | 83.1 ms | n/a |

## Speed-up over pure Python (higher is better)

| Kernel | Python | NumPy | C (1 thread) | C OpenMP (4 threads) | Mojo (1 thread) | Mojo (4 threads) |
|---|---:|---:|---:|---:|---:|---:|
| matmul_naive | 1.0× | n/a | 35.9× | n/a | 34.3× | n/a |
| matmul_tiled | 1.0× | 1,325× | 268× | 677× | 392× | 1,439× |
| mandelbrot | 1.0× | 5.7× | 24.7× | 94.3× | 107× | 421× |
| nbody | 1.0× | 11.3× | 95.9× | 342× | 108× | 394× |
| reduction | 1.0× | 24.5× | 22.7× | 85.5× | 23.7× | 87.7× |
| prefix_sum | 1.0× | 16.0× | 34.3× | 211× | 39.5× | 182× |
| hash_table | 1.0× | n/a | 9.4× | n/a | 8.9× | n/a |
| word_count | 1.0× | 1.2× | 7.7× | n/a | 7.4× | n/a |

n/a = no sensible implementation in that language (see README).

## Start-up and compile costs

| What | Median wall time |
|---|---:|
| Python interpreter start (no-op kernel) | 19.4 ms |
| Python + import numpy (no-op kernel) | 92.8 ms |
| C binary start (no-op kernel) | 1.3 ms |
| Mojo binary start (no-op kernel) | 8.4 ms |
| gcc -O3 compile of kernels.c | 309.9 ms |
| mojo build of kernels.mojo | 1.88 s |
| mojo run kernels.mojo noop (JIT compile + run) | 1.88 s |
