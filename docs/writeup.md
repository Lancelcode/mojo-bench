# Is Mojo really faster than Python? I measured it against Python, NumPy and C

Mojo is often introduced with a headline number: tens of thousands of times faster than Python. As a software engineering student moving towards high-performance computing, I wanted to know what that means for ordinary code, written the way I would write it, on a machine I can describe. So I built [mojo-bench](https://github.com/Lancelcode/mojo-bench): eight classic kernels implemented in pure Python, NumPy, C and Mojo, all checked for identical answers before anything is timed.

This post covers what I compared, how, what I found, and what I'd tell anyone weighing Mojo for a project.

## What I compared

The kernels are deliberately varied, because "fast" depends on what the code spends its time doing:

- **Matrix multiplication**, naive (the textbook triple loop) and tiled (cache blocking plus SIMD)
- **Mandelbrot set**: pure arithmetic with a data-dependent early exit
- **N-body step**: O(n²) floating point with square roots and divisions
- **Reduction** and **prefix sum** over 10 million integers: streaming memory access
- **Hash table** inserts and lookups: random memory access
- **Word-frequency count** over 22 MB of generated text: string processing

Each has a pure-Python version (the 1× baseline), a NumPy version where one makes sense, C compiled with `gcc -O3 -march=native`, and Mojo 1.2. The five kernels that parallelise naturally also have multi-threaded versions: OpenMP in C and `parallelize` in Mojo.

The rule I set myself was **idiomatic, not heroic**: no intrinsics, no assembly, nothing a reasonable programmer in that language wouldn't write. A determined C expert can match every result here; the question is what you get from normal code.

## Method

All inputs come from integer formulas, so every language works on exactly the same data, and most checksums can be compared *exactly* rather than within a tolerance. The runner refuses to report a time if any implementation disagrees with Python. Each measurement has a warm-up run and then enough repeats to fill about a second (at least five), and I report the median.

The test machine is a 4-vCPU cloud VM on an Intel Sapphire Rapids Xeon with AVX-512. Being a shared VM, it is noisy. I ran everything twice to quantify that, and treat differences under about 10–15 % as a tie. Everything is reproducible with `make bench`, and CI re-checks correctness on every push.

## Results

| Kernel | NumPy | C (1 thread) | Mojo (1 thread) | C OpenMP (4) | Mojo (4) |
|---|---:|---:|---:|---:|---:|
| Matmul, naive | n/a | 36× | 34× | n/a | n/a |
| Matmul, tiled | 1,325× | 268× | 392× | 677× | 1,439× |
| Mandelbrot | 5.7× | 25× | 107× | 94× | 421× |
| N-body | 11× | 96× | 108× | 342× | 394× |
| Reduction | 25× | 23× | 24× | 86× | 88× |
| Prefix sum | 16× | 34× | 40× | 211× | 182× |
| Hash table | n/a | 9.4× | 8.9× | n/a | n/a |
| Word count | 1.2× | 7.7× | 7.4× | n/a | n/a |

*Speed-up over pure Python; higher is better.*

### Finding 1: written like C, Mojo runs like C

On the naive matrix multiply, C took 69 ms and Mojo 72 ms. Same algorithm, same scalar loop, same speed. The same goes for the hash table (97 vs 102 ms) and the word count (80 vs 83 ms). Mojo compiles to native code through LLVM, so it's no surprise that it lands where a compiled language lands. The 35× over Python is the cost of the interpreter, not a property of Mojo.

### Finding 2: SIMD is where Mojo earns its keep

On Mandelbrot, Mojo was **4.3× faster than C** on one core (19 vs 83 ms). The inner loop of a Mandelbrot pixel exits as soon as the point escapes, and GCC won't auto-vectorise a loop with a data-dependent exit, so the C code does one pixel at a time. In Mojo, SIMD vectors are an ordinary type: I wrote the loop over `SIMD[DType.float64, 8]`, with a mask that freezes pixels once they escape, and got 8 pixels per AVX-512 instruction without leaving the language. You can do the same in C with intrinsics, but that is precisely what idiomatic C doesn't do.

### Finding 3: some "language" differences are compiler flags

Mojo's tiled matmul was 1.46× faster than C's. Before calling that a Mojo win, I checked the generated code. GCC recognises Sapphire Rapids but by default prefers 256-bit vectors, half of AVX-512. With `-mprefer-vector-width=512`, the C dropped from about 9 ms to 6.6 ms, level with Mojo. (The same flag made the C n-body *slower*, which is presumably why it isn't the default.)

I also found a mistake in my own first version. I had compiled the C with `-std=c11`, which quietly stops GCC from fusing multiply-add into FMA instructions. The C binary had zero FMAs; the Mojo one had 55. Fixing it made the C matmul about 15 % faster. Small flags move results by 15–25 %, which is a good reason to distrust any benchmark that doesn't state them.

### Finding 4: memory doesn't care what language you use

The reduction ran at about 25 GB/s on one core in C, Mojo *and* NumPy. Once a kernel is limited by how fast data arrives, the language stops mattering, and NumPy is just as good because `a.sum()` is a single call into compiled code.

### Finding 5: Python is already C in places

Python's `dict`, `str.split` and `collections.Counter` are implemented in C. That is why the hash table and word count show only 7–9× gaps: idiomatic Python there is mostly native code already. Surprisingly, my vectorised NumPy word count was only 1.2× faster than plain Python. Finding word boundaries with array operations and sorting 4 million keys costs about as much as it saves.

## Scaling and the roofline

For the parallel kernels I ran 1 to 4 threads at larger sizes, with 800 MB arrays for the streaming kernels so that they actually hit main memory. Both OpenMP and Mojo's `parallelize` scaled almost perfectly, 3.7–4.0× on 4 threads, with no meaningful difference between the two. Mojo's parallel API is as effective as OpenMP and takes about as much code.

To understand *why* each kernel performs as it does, I measured the machine's limits with a small probe: 327 GFLOP/s of peak FP64 throughput and 51–60 GB/s of memory bandwidth on 4 cores. That puts the ridge of the roofline at about 5.4 operations per byte. The reduction and prefix sum sit far to the left of it. At 4 threads they reach 75–100 % of the measured bandwidth, so they are memory-bound and no language can rescue them. Matmul, n-body and Mandelbrot are compute-bound, yet reach only 3–19 % of peak. The gap is down to things like register blocking (which OpenBLAS does and my tiled loop doesn't), divide and square-root throughput in n-body, and scalar code in the C Mandelbrot. On a 4-core slice, scaling never really flattens for compute-bound kernels. The streaming kernels arrive at the bandwidth roof exactly at 4 threads, so more cores would not help them.

## What I learned

1. **Mojo isn't a faster C; it's a more convenient one for performance work.** SIMD and parallelism are first-class, portable parts of the language. When a kernel needs them and the C compiler won't provide them on its own, Mojo wins clearly. When it doesn't, it's a tie.
2. **Measure before explaining.** Twice, a gap I was ready to attribute to the language turned out to be a compiler default. Looking at the instruction mix and trying one flag took minutes and changed the story.
3. **Correctness checks catch real bugs.** Exact checksums flagged an FMA rounding difference in Mandelbrot at a larger size (12 iterations out of 225 million) and a weak checksum of my own that summed to zero.
4. **Start-up and compile time matter for small jobs.** A compiled Mojo binary takes about 8–10 ms to start against about 1 ms for C, and `mojo build` takes about 1.9 s against 0.3 s for GCC. For quick scripts, Python is still the pragmatic choice.
5. **The ecosystem is young.** Between Mojo versions the standard library moved under `std.*`, pointer types changed, and `parallelize` moved into a separate `max-core` package. Expect to read source code, not just documentation.

## Would I use Mojo for my final-year project?

For numerical kernels where SIMD and multi-core parallelism are the point, yes. It gave me C-level performance, with vectorisation I could express directly instead of hoping the compiler found it. For anything where the hot path is already NumPy, a C library or a Python built-in, the gains are smaller than the marketing suggests, and the measurements here show why.

The code, raw results and charts are all on GitHub. If you run `make bench` on different hardware, I'd love to see how your numbers compare.
