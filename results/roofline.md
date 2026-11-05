# Roofline analysis

Measured roofs on this machine (best of 3 runs of `build/roofline_probe`):

| Roof | All threads | 1 thread |
|---|---:|---:|
| Peak FP64 FMA throughput | 327 GFLOP/s | 84 GFLOP/s |
| Memory bandwidth, read-only sweep | 60.0 GB/s | 14.9 GB/s |
| Memory bandwidth, STREAM triad | 51.0 GB/s | 12.8 GB/s |

Ridge point = peak / read bandwidth = **5.4 operations per byte**. Kernels to the left are memory-bound; kernels to the right can only be limited by compute.

| Kernel | n | Intensity (op/byte) | Bound | Implementation | Achieved | Roof | % of roof |
|---|---:|---:|---|---|---:|---:|---:|
| matmul_tiled | 1,024 | 85.3 | compute | C OpenMP | 38.9 Gop/s | 326.6 | 12 % |
| matmul_tiled | 1,024 | 85.3 | compute | Mojo parallelize | 63.0 Gop/s | 326.6 | 19 % |
| mandelbrot | 2,048 | ∞ | compute | C OpenMP | 8.4 Gop/s | 326.6 | 3 % |
| mandelbrot | 2,048 | ∞ | compute | Mojo parallelize | 37.4 Gop/s | 326.6 | 11 % |
| nbody | 4,096 | 640 | compute | C OpenMP | 38.9 Gop/s | 326.6 | 12 % |
| nbody | 4,096 | 640 | compute | Mojo parallelize | 43.5 Gop/s | 326.6 | 13 % |
| reduction | 100,000,000 | 0.125 | memory | C OpenMP | 5.6 Gop/s (45.0 GB/s) | 7.5 | 75 % |
| reduction | 100,000,000 | 0.125 | memory | Mojo parallelize | 6.4 Gop/s (51.3 GB/s) | 7.5 | 85 % |
| prefix_sum | 100,000,000 | 0.0625 | memory | C OpenMP | 3.2 Gop/s (51.6 GB/s) | 3.8 | 86 % |
| prefix_sum | 100,000,000 | 0.0625 | memory | Mojo parallelize | 3.3 Gop/s (52.5 GB/s) | 3.8 | 87 % |

Operation and byte counts:

- **matmul_tiled**: 2n³ flops; bytes = compulsory traffic for A, B and C (3·8n²), an upper bound on intensity. With 64-wide tiles and no reuse beyond a tile it would fall to ~5 flop/byte, still at or past the ridge.
- **mandelbrot**: 8 flops per useful iteration × total iterations (from the checksum); no array traffic at all (only a running count), so intensity is effectively infinite.
- **nbody**: ~20 flops per pair (sqrt and divide counted as 1 each) × 5 steps × n²; bytes ≈ 16 doubles per body per step (positions, masses, velocities read and written).
- **reduction**: 1 integer add per 8-byte element read.
- **prefix_sum**: Two-pass parallel scan: each element is read twice (8 + 8 bytes), written once (8) and pulled into cache before the write (write-allocate, 8): 2 adds per 32 bytes.
