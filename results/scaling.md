# Thread scaling

Median time (speed-up over the same implementation on 1 thread). All checksums agree across thread counts and languages.

| Kernel | Size | Implementation | 1 thr | 2 thr | 3 thr | 4 thr |
|---|---|---|---:|---:|---:|---:|
| matmul_tiled | 1,024 × 1,024 float64 | C OpenMP | 220.0 ms (1.00×) | 107.1 ms (2.05×) | 84.6 ms (2.60×) | 55.2 ms (3.98×) |
| matmul_tiled | 1,024 × 1,024 float64 | Mojo parallelize | 137.4 ms (1.00×) | 68.2 ms (2.01×) | 44.8 ms (3.07×) | 34.1 ms (4.03×) |
| mandelbrot | 2,048 × 2,048 pixels | C OpenMP | 830.2 ms (1.00×) | 428.9 ms (1.94×) | 281.6 ms (2.95×) | 214.1 ms (3.88×) |
| mandelbrot | 2,048 × 2,048 pixels | Mojo parallelize | 186.1 ms (1.00×) | 94.2 ms (1.98×) | 64.5 ms (2.88×) | 48.2 ms (3.86×) |
| nbody | 4,096 bodies, 5 steps | C OpenMP | 168.0 ms (1.00×) | 84.7 ms (1.98×) | 57.2 ms (2.94×) | 43.1 ms (3.90×) |
| nbody | 4,096 bodies, 5 steps | Mojo parallelize | 152.9 ms (1.00×) | 76.4 ms (2.00×) | 51.4 ms (2.98×) | 38.6 ms (3.96×) |
| reduction | 100 M int64 (800 MB) | C OpenMP | 71.4 ms (1.00×) | 37.0 ms (1.93×) | 24.3 ms (2.94×) | 17.8 ms (4.02×) |
| reduction | 100 M int64 (800 MB) | Mojo parallelize | 60.9 ms (1.00×) | 30.6 ms (1.99×) | 21.2 ms (2.87×) | 15.6 ms (3.91×) |
| prefix_sum | 100 M int64 (800 MB in + 800 MB out) | C OpenMP | 245.6 ms (1.00×) | 123.9 ms (1.98×) | 89.0 ms (2.76×) | 62.0 ms (3.96×) |
| prefix_sum | 100 M int64 (800 MB in + 800 MB out) | Mojo parallelize | 227.2 ms (1.00×) | 118.0 ms (1.93×) | 81.7 ms (2.78×) | 60.9 ms (3.73×) |
