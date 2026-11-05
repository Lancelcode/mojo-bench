# mojo-bench

How fast is Mojo in practice? This repository compares **Mojo** with **pure Python**, **NumPy** and **C** on seven classic compute kernels, on the same machine, with the same inputs, checking that every implementation produces the same answer before any time is reported.

It is meant to be honest rather than flattering: where Mojo wins, the analysis says why; where it does not, that is reported too. Every number below was measured by `make bench` and is in [`results/results.csv`](results/results.csv).

![Speed-up over pure Python](results/speedup.png)

## What is compared, and why

| Kernel | What it stresses | Size used |
|---|---|---|
| `matmul_naive` | Textbook i-j-k triple loop; strided memory access, no blocking | 384 × 384 float64 |
| `matmul_tiled` | Cache blocking + SIMD over the inner loop | 384 × 384 float64 |
| `mandelbrot` | Pure compute with a data-dependent early exit (branchy, hard to auto-vectorise) | 640 × 640 pixels, 200 max iterations |
| `nbody` | O(n²) floating-point maths with `sqrt` and division | 1,024 bodies, 5 steps |
| `reduction` | Streaming sum; memory bandwidth | 10 M int64 |
| `prefix_sum` | Inclusive scan; a serial dependency chain plus a write stream | 10 M int64 |
| `hash_table` | Random memory access, branches; 2 M inserts then 2 M lookups (50 % hits) | 2 M keys |

The implementations:

- **Python**: lists, loops and `dict`, written the way a competent Python programmer would without extension modules. This is the baseline (1×) and the source of truth for correctness.
- **NumPy**: the idiomatic "make Python fast" route, only where a natural array formulation exists. A naive triple loop and an open-addressing hash table have no sensible NumPy form, so they are shown as n/a. For `matmul_tiled`, NumPy's `a @ b` calls OpenBLAS: an expertly tuned GEMM, so treat it as the ceiling rather than a like-for-like peer.
- **C**: plain C compiled with `gcc -O3 -march=native` (GCC's default dialect), no intrinsics. Any SIMD comes from GCC's auto-vectoriser, which is what most C code actually gets.
- **Mojo (1 thread)**: idiomatic Mojo 1.2 using explicit SIMD types and `vectorize` where it helps.
- **Mojo (all cores)**: the same code split into one chunk per core with `parallelize`, for the five kernels that parallelise naturally.

The fair, like-for-like comparison is **C (1 thread) against Mojo (1 thread)**. The parallel Mojo column shows what multi-threading adds on top. A multi-threaded C (OpenMP) comparison is part of the scaling study.

## Results

Machine: Intel Xeon (Sapphire Rapids, AVX-512) cloud VM with 4 vCPUs, Ubuntu 24.04, gcc 13.3.0 (`-O3 -march=native`), Mojo 1.2.0, Python 3.13.18, NumPy 2.5.3. Full details are in [`results/results.md`](results/results.md) and [`results/machine.json`](results/machine.json).

### Median time

| Kernel | Python | NumPy | C (1 thread) | Mojo (1 thread) | Mojo (4 threads) |
|---|---:|---:|---:|---:|---:|
| matmul_naive | 2.61 s | n/a | 71.2 ms | 73.0 ms | n/a |
| matmul_tiled | 2.48 s | 1.9 ms | 8.9 ms | 6.5 ms | 2.0 ms |
| mandelbrot | 2.03 s | 365.2 ms | 79.9 ms | 18.8 ms | 4.9 ms |
| nbody | 1.03 s | 94.7 ms | 10.5 ms | 9.5 ms | 2.5 ms |
| reduction | 72.8 ms | 3.0 ms | 3.3 ms | 3.1 ms | 929 µs |
| prefix_sum | 580.6 ms | 35.7 ms | 16.9 ms | 16.4 ms | 2.7 ms |
| hash_table | 946.4 ms | n/a | 116.9 ms | 110.3 ms | n/a |

### Speed-up over pure Python (higher is better)

| Kernel | NumPy | C (1 thread) | Mojo (1 thread) | Mojo (4 threads) |
|---|---:|---:|---:|---:|
| matmul_naive | n/a | 36.7× | 35.8× | n/a |
| matmul_tiled | 1,327× | 280× | 383× | 1,242× |
| mandelbrot | 5.5× | 25.3× | 108× | 409× |
| nbody | 10.9× | 98.1× | 109× | 407× |
| reduction | 23.9× | 22.4× | 23.8× | 78.4× |
| prefix_sum | 16.3× | 34.4× | 35.4× | 215× |
| hash_table | n/a | 8.1× | 8.6× | n/a |

### Start-up and compile costs

| What | Median wall time |
|---|---:|
| Python interpreter start (no-op) | 16.8 ms |
| Python + `import numpy` (no-op) | 94.6 ms |
| C binary start (no-op) | 1.4 ms |
| Mojo binary start (no-op) | 9.8 ms |
| `gcc -O3` compile of `kernels.c` | 250.7 ms |
| `mojo build` of `kernels.mojo` | 1.87 s |
| `mojo run kernels.mojo` (JIT compile + run, no-op) | 1.98 s |

## Analysis: where Mojo wins, where it doesn't, and why

**Same algorithm, same machine code: no difference.** On `matmul_naive`, C and Mojo are within 3 % of each other (71.2 vs 73.0 ms, about 1.6 GFLOP/s each). Both are scalar loops walking down a column of B with a stride of 384 × 8 bytes, and neither compiler may reorder the floating-point sum. Mojo is not magic: written the same way as C, it compiles to much the same code and runs at the same speed. The ~36× over Python is simply the cost of the interpreter.

**Explicit SIMD is where single-threaded Mojo pulls ahead.** The clearest case is `mandelbrot`: Mojo is **4.3× faster than C** on one core (18.8 vs 79.9 ms). The escape-time loop has a data-dependent exit, which GCC's auto-vectoriser will not touch, so the C code processes one pixel at a time. The Mojo version uses `SIMD[DType.float64, 8]` vectors, so 8 pixels per AVX-512 register, with a mask that freezes escaped lanes and a loop that ends when all 8 have escaped. This is the case Mojo is designed for: SIMD as a normal, portable type in the language rather than compiler-specific intrinsics. A C programmer *could* write the same thing with AVX-512 intrinsics. The point is that idiomatic C doesn't, and idiomatic Mojo does.

**Sometimes the difference is a compiler default, not the language.** On `matmul_tiled`, Mojo is 1.37× faster than C (6.5 vs 8.9 ms; 17.5 vs 12.8 GFLOP/s) with the same blocked i-k-j algorithm. The cause is measurable: GCC detects this CPU as Sapphire Rapids but defaults to `-mprefer-vector-width=256`, so it uses half-width AVX-512 vectors. Rebuilding the C with `-mprefer-vector-width=512` brought it to **6.6 ms**, level with Mojo. The same flag made the C n-body *slower* (10.6 → 13.3 ms), which is why GCC is cautious by default. Mojo's `simd_width_of[DType.float64]()` picks the full 8-lane width. To reproduce:

```bash
gcc -O3 -march=native -mprefer-vector-width=512 -o build/kernels_c512 kernels/c/kernels.c -lm
```

```bash
./build/kernels_c512 matmul_tiled 384 2 30
```

A second compiler detail mattered here too. An earlier version of this benchmark compiled the C with `-std=c11`, which, as a side effect, stops GCC fusing `a*b + c` into a single FMA instruction (in its strict ISO C modes GCC treats floating-point contraction as off). The C binary then contained no FMA instructions at all, against 55 in the Mojo binary, and the tiled C matmul was 10.4 ms instead of 8.9 ms. The C here now uses GCC's default dialect, as a plain `gcc -O3 -march=native` would. Each of these flags moved a result by 15–25 %, so benchmark comparisons must state them.

**When the compiler already vectorises, it's a tie.** On `nbody`, C and Mojo differ by about 10 % (10.5 vs 9.5 ms; roughly 500–550 M pair interactions per second). GCC auto-vectorises the inner loop of the C code, including `sqrt` and division, so writing it with explicit SIMD in Mojo gains little. The same applies to `reduction` (3.3 vs 3.1 ms) and `prefix_sum` (16.9 vs 16.4 ms): the differences are within run-to-run noise.

**Memory-bound kernels: everyone hits the same wall.** A single core sums the 80 MB `reduction` array at about 25–26 GB/s in C, Mojo *and* NumPy. At that point the limit is how fast one core can pull data in, not the language. NumPy is level with compiled code here because `a.sum()` is a single call into a C loop with no per-element Python overhead. The 80 MB array does fit in the 260 MB of L3 that the VM reports, so after warm-up this is probably cache bandwidth rather than DRAM bandwidth. The thread-scaling study uses 800 MB arrays to measure DRAM.

**Pointer chasing: Python's dict is already C.** `hash_table` shows the smallest gap: C and Mojo do an insert or lookup in about 28–29 ns, Python in about 240 ns, so roughly 8–9×. Python's `dict` is a highly optimised C hash table, so what you pay for in Python is interpreter dispatch and boxed integers, not the hashing. The compiled versions are limited by cache misses on a 64 MB table, which no language feature removes.

**Parallelism is where the big numbers come from, and it isn't free.** With `parallelize` across 4 cores, Mojo reaches 3.7× (n-body) and 3.8× (Mandelbrot) over its own single-threaded code. Tiled matmul (3.2×) and reduction (3.3×) scale less well: the runs take only 1–2 ms, so fixed costs such as waking threads become visible, and the reduction is limited by memory bandwidth. The scan's 6.1× figure is not trustworthy: back-to-back 4-thread runs of the same scan gave medians of 3.0 ms and 6.8 ms, so at this size it is dominated by something other than the code (the scaling study measures it properly at a larger size). The best Mojo matmul, at 57 GFLOP/s on 4 cores, roughly equals single-threaded OpenBLAS at 61 GFLOP/s. A decade of hand-tuned micro-kernels (register blocking, packing, prefetching) still gets four times as much out of each core as a straightforward tiled loop.

**Start-up and compile cost are real.** A compiled Mojo binary takes about 10 ms to start, against 1.4 ms for C: it initialises its runtime and thread pool. Compilation is the bigger cost: `mojo build` takes about 1.9 s for this file against 0.25 s for `gcc -O3`. `mojo run` pays that compile on *every* invocation, so for a script that does under a second of work, Python (17 ms to start) or a pre-built binary is the better choice. All kernel timings above exclude start-up and compilation.

### Summary

- Mojo is not a faster C. Written like C, it runs like C.
- Its advantage is that SIMD and parallelism are first-class and portable. `mandelbrot` (4.3× over C on one core) shows that clearly, and `parallelize` turns 4 cores into a 3.7–3.8× speed-up for compute-heavy kernels with a few lines of code.
- Compared with Python, the speed-ups are dramatic (8× to 1,000×+), but most of that gain is also available from C. For array-shaped problems, NumPy gets a good part of the way there with no new language.

## Benchmarking caveats

- **One machine.** This is a 4-vCPU cloud VM (`hypervisor` flag set), shared with other tenants. Results will differ on your hardware, especially for memory-bound kernels.
- **Noise.** Repeats are adaptive: at least 5, otherwise as many as fit in about 1 s, capped at 200. Across two complete runs, medians agreed within about 3 % for most rows and within 7 % for every single-threaded row except NumPy n-body (13 %). Three of the very short 4-thread runs (tiled matmul, reduction and scan, all under 4 ms) varied by 10–20 % between runs, while their minimum times stayed within 8 %. The 4-thread scan is worse still (see the analysis). The single-threaded C tiled matmul also moved by up to ±10 % with code changes elsewhere in the file that did not touch its loop, which looks like code-layout and alignment sensitivity. Treat differences under about 10 % as a tie. Single-run figures quoted in the analysis (such as the `-mprefer-vector-width=512` experiment) were measured separately with 30 repeats.
- **Medians, warm caches.** Each implementation has one warm-up run, and times are medians of the timed repeats. Input generation is excluded, as is process start-up (measured separately above). Min, mean and standard deviation are in the CSV.
- **Sizes are set by pure Python.** Problem sizes were chosen so that pure Python takes 1–3 s per run. That makes several compiled runs short (milliseconds), which favours low-overhead code and limits parallel speed-up. Bigger problems would show the compiled languages scaling better.
- **Idiomatic, not heroic.** No implementation uses intrinsics, hand-written assembly or libraries other than NumPy. A determined C programmer can match every Mojo result here. The question this repo asks is what you get from reasonable code in each language.
- **Correctness first.** The runner exits with an error if any checksum disagrees with pure Python. Inputs come from integer formulas, so matmul, Mandelbrot, reduction, scan and hash-table checksums must match *exactly*. N-body sums floating-point forces in a different order in each implementation, so it allows a relative difference of 1e-9; the largest observed was about 3e-14.
- **Floating-point model.** No `-ffast-math` anywhere. Both GCC (in its default dialect) and Mojo fuse `a*b + c` into FMA instructions, which round differently from Python. At the benchmark size every Mandelbrot iteration count still matches Python exactly. At 2,048 × 2,048, the FMA builds differ from Python by 12 iterations out of 225.6 M (a handful of boundary pixels), so Mandelbrot allows a relative difference of 1e-6.
- **Toolchain versions matter.** Mojo is evolving quickly. In Mojo 1.x the standard library is imported as `std.*`, `UnsafePointer` gave way to `Pointer`, and `parallelize` moved to `max.algorithm` in the `max-core` package. The code here targets Mojo 1.2.0 and max-core 26.7.0 as pinned in `requirements.txt`.

## How to reproduce

Requirements: Linux x86-64 (the Mojo wheels need glibc ≥ 2.34), Python 3.10+ and gcc.

```bash
python3 -m pip install -r requirements.txt
```

```bash
make bench
```

`make bench` builds the C and Mojo code into `build/`, runs every kernel in every implementation, checks the answers, and rewrites `results/results.csv`, `results/results.md`, `results/machine.json`, `results/startup.csv` and `results/speedup.png`. It takes about two minutes on the machine above.

Other targets:

```bash
make quick
```

builds everything and runs a tiny-size correctness pass, which is what CI runs on every push and pull request.

```bash
python3 bench/run.py --kernels nbody mandelbrot --impls c mojo mojo_par
```

runs a subset.

Every implementation is also a standalone program with the same interface, e.g. `./build/kernels_mojo nbody 1024 1 10 4` (kernel, size, warm-up runs, timed repeats, threads).

## Repository layout

```
kernels/python/pure.py        pure-Python reference implementations
kernels/python/numpy_impl.py  NumPy implementations
kernels/python/harness.py     shared timing harness for the two above
kernels/c/kernels.c           C implementations
kernels/mojo/kernels.mojo     Mojo implementations (serial SIMD + parallel paths)
bench/run.py                  runner: correctness checks, timing, CSV/Markdown
bench/plot.py                 speed-up chart
results/                      the measured results shown above
```
