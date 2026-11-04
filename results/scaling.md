# Thread scaling

Median time (speed-up over the same implementation on 1 thread). All checksums agree across thread counts and languages.

| Kernel | Size | Implementation | 1 thr | 2 thr | 3 thr | 4 thr |
|---|---|---|---:|---:|---:|---:|
| matmul_tiled | 1,024 × 1,024 float64 | C OpenMP | 200.0 ms (1.00×) | 104.7 ms (1.91×) | 73.6 ms (2.72×) | 51.8 ms (3.86×) |
| matmul_tiled | 1,024 × 1,024 float64 | Mojo parallelize | 138.6 ms (1.00×) | 69.9 ms (1.98×) | 46.6 ms (2.98×) | 34.6 ms (4.01×) |
| mandelbrot | 2,048 × 2,048 pixels | C OpenMP | 825.9 ms (1.00×) | 421.0 ms (1.96×) | 280.3 ms (2.95×) | 214.0 ms (3.86×) |
| mandelbrot | 2,048 × 2,048 pixels | Mojo parallelize | 189.8 ms (1.00×) | 95.1 ms (2.00×) | 63.6 ms (2.99×) | 48.8 ms (3.89×) |
| nbody | 4,096 bodies, 5 steps | C OpenMP | 169.9 ms (1.00×) | 85.3 ms (1.99×) | 60.4 ms (2.81×) | 42.5 ms (4.00×) |
| nbody | 4,096 bodies, 5 steps | Mojo parallelize | 153.3 ms (1.00×) | 76.4 ms (2.01×) | 52.2 ms (2.94×) | 38.8 ms (3.95×) |
| reduction | 100 M int64 (800 MB) | C OpenMP | 69.1 ms (1.00×) | 34.6 ms (2.00×) | 23.4 ms (2.95×) | 17.3 ms (4.00×) |
| reduction | 100 M int64 (800 MB) | Mojo parallelize | 57.1 ms (1.00×) | 30.4 ms (1.88×) | 20.5 ms (2.79×) | 15.7 ms (3.63×) |
| prefix_sum | 100 M int64 (800 MB in + 800 MB out) | C OpenMP | 235.8 ms (1.00×) | 127.5 ms (1.85×) | 85.0 ms (2.77×) | 63.4 ms (3.72×) |
| prefix_sum | 100 M int64 (800 MB in + 800 MB out) | Mojo parallelize | 231.0 ms (1.00×) | 127.4 ms (1.81×) | 81.3 ms (2.84×) | 75.0 ms (3.08×) |
