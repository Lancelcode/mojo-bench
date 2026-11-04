# Mojo implementations of the mojo-bench kernels (Mojo 1.2).
#
# Usage (same protocol as every other implementation):
#   kernels_mojo <kernel> <n> <warmup> <repeats> [threads]
# prints
#   RESULT <kernel> <checksum> <t1,t2,...>
#
# threads = 0 (or omitted) runs the single-threaded code path, which uses
# explicit SIMD where it helps. threads = T > 0 runs the parallel code path,
# splitting the work into exactly T chunks handed to `parallelize`, so T is
# the number of threads doing work at once (as long as T <= cores).

from std.sys import argv
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from std.algorithm import vectorize
from std.memory.alloc import unsafe_alloc
from std.math import sqrt, iota
from max.algorithm import parallelize

comptime F64 = DType.float64
comptime I64 = DType.int64
comptime W = simd_width_of[F64]()  # 8 lanes of Float64 with AVX-512
comptime FPtr = Pointer[Float64, MutUntrackedOrigin]
comptime IPtr = Pointer[Int64, MutUntrackedOrigin]
comptime UPtr = Pointer[UInt64, MutUntrackedOrigin]


def falloc(n: Int) -> FPtr:
    return unsafe_alloc[Float64](n, alignment=64)


def ialloc(n: Int) -> IPtr:
    return unsafe_alloc[Int64](n, alignment=64)


def chunk(n: Int, parts: Int, idx: Int) -> Tuple[Int, Int]:
    """Half-open range [lo, hi) of the idx-th of `parts` near-equal chunks."""
    var lo = n * idx // parts
    var hi = n * (idx + 1) // parts
    return (lo, hi)


# ===------------------------------------------------------------- matmul


def mat_inputs(n: Int, a: FPtr, b: FPtr):
    for i in range(n):
        for j in range(n):
            a.unsafe_store(i * n + j, Float64((7 * i + 3 * j) % 11 - 5))
            b.unsafe_store(i * n + j, Float64((5 * i + 13 * j) % 7 - 3))


def matmul_naive(n: Int, a: FPtr, b: FPtr, c: FPtr):
    """Textbook i-j-k triple loop, scalar: the compiled baseline."""
    for i in range(n):
        for j in range(n):
            var acc: Float64 = 0.0
            for k in range(n):
                acc += a.unsafe_load(i * n + k) * b.unsafe_load(k * n + j)
            c.unsafe_store(i * n + j, acc)


comptime TILE = 64


def matmul_rows(n: Int, a: FPtr, b: FPtr, c: FPtr, row_lo: Int, row_hi: Int):
    """Blocked i-k-j matmul over rows [row_lo, row_hi), SIMD over j."""
    for i in range(row_lo, row_hi):
        for j in range(n):
            c.unsafe_store(i * n + j, 0.0)
    for ii in range(row_lo, row_hi, TILE):
        var ie = min(ii + TILE, row_hi)
        for kk in range(0, n, TILE):
            var ke = min(kk + TILE, n)
            for jj in range(0, n, TILE):
                var je = min(jj + TILE, n)
                for i in range(ii, ie):
                    for k in range(kk, ke):
                        var aik = a.unsafe_load(i * n + k)
                        var crow = c.unsafe_offset(i * n + jj)
                        var brow = b.unsafe_offset(k * n + jj)

                        def axpy[w: Int](j: Int) {var crow, var brow, var aik}:
                            crow.unsafe_store(
                                j,
                                crow.unsafe_load[width=w](j)
                                + SIMD[F64, w](aik) * brow.unsafe_load[width=w](j),
                            )

                        vectorize[W](je - jj, axpy)


def matmul_tiled(n: Int, a: FPtr, b: FPtr, c: FPtr, threads: Int):
    if threads <= 0:
        matmul_rows(n, a, b, c, 0, n)
        return

    def work(t: Int) {var n, var a, var b, var c, var threads}:
        var r = chunk(n, threads, t)
        matmul_rows(n, a, b, c, r[0], r[1])

    parallelize(work, threads)


def mat_checksum(n: Int, c: FPtr) -> String:
    var acc: Float64 = 0.0
    for i in range(n):
        for j in range(n):
            acc += c.unsafe_load(i * n + j) * Float64((i + 2 * j) % 7)
    return String(Int(acc))


# ===------------------------------------------------------------- mandelbrot

comptime MAX_ITER = 200


def mandel_rows(n: Int, row_lo: Int, row_hi: Int) -> Int:
    """Escape-time counts for rows [row_lo, row_hi), W pixels per SIMD vector."""
    var total = 0
    var nf = Float64(n)
    for i in range(row_lo, row_hi):
        var cy = SIMD[F64, W](-1.25 + 2.5 * Float64(i) / nf)
        for j in range(0, n, W):
            var cx = -2.0 + 2.5 * (iota[F64, W]() + Float64(j)) / nf
            var x = SIMD[F64, W](0)
            var y = SIMD[F64, W](0)
            var k = SIMD[I64, W](0)
            for _ in range(MAX_ITER):
                var active = (x * x + y * y).le(4.0)
                if not active.reduce_or():
                    break
                k += active.select(SIMD[I64, W](1), SIMD[I64, W](0))
                var xt = x * x - y * y + cx
                y = active.select(2.0 * x * y + cy, y)
                x = active.select(xt, x)
            # Only count lanes that are real pixels (n need not divide by W).
            for lane in range(min(W, n - j)):
                total += Int(k[lane])
    return total


def mandelbrot(n: Int, threads: Int) -> Int:
    if threads <= 0:
        return mandel_rows(n, 0, n)
    var partial = ialloc(threads)

    def work(t: Int) {var n, var threads, var partial}:
        # Interleave rows so every thread gets a fair share of the expensive
        # rows near the set, rather than one thread owning the middle band.
        var acc = 0
        for i in range(t, n, threads):
            acc += mandel_rows(n, i, i + 1)
        partial.unsafe_store(t, Int64(acc))

    parallelize(work, threads)
    var total = 0
    for t in range(threads):
        total += Int(partial.unsafe_load(t))
    partial.unsafe_free()
    return total


# ===------------------------------------------------------------- n-body

comptime STEPS = 5
comptime DT = 0.01
comptime EPS2 = 0.01


def nbody_accel(
    n: Int, x: FPtr, y: FPtr, z: FPtr, m: FPtr,
    vx: FPtr, vy: FPtr, vz: FPtr, lo: Int, hi: Int,
):
    """Update velocities of bodies [lo, hi): SIMD over the inner j loop."""
    for i in range(lo, hi):
        var xi = x.unsafe_load(i)
        var yi = y.unsafe_load(i)
        var zi = z.unsafe_load(i)
        var ax = SIMD[F64, W](0)
        var ay = SIMD[F64, W](0)
        var az = SIMD[F64, W](0)
        var j = 0
        while j + W <= n:
            var dx = x.unsafe_load[width=W](j) - xi
            var dy = y.unsafe_load[width=W](j) - yi
            var dz = z.unsafe_load[width=W](j) - zi
            var d2 = dx * dx + dy * dy + dz * dz + EPS2
            var s = m.unsafe_load[width=W](j) / (d2 * sqrt(d2))
            ax += dx * s
            ay += dy * s
            az += dz * s
            j += W
        var sx = ax.reduce_add()
        var sy = ay.reduce_add()
        var sz = az.reduce_add()
        while j < n:  # scalar tail
            var dx = x.unsafe_load(j) - xi
            var dy = y.unsafe_load(j) - yi
            var dz = z.unsafe_load(j) - zi
            var d2 = dx * dx + dy * dy + dz * dz + EPS2
            var s = m.unsafe_load(j) / (d2 * sqrt(d2))
            sx += dx * s
            sy += dy * s
            sz += dz * s
            j += 1
        vx.unsafe_store(i, vx.unsafe_load(i) + sx * DT)
        vy.unsafe_store(i, vy.unsafe_load(i) + sy * DT)
        vz.unsafe_store(i, vz.unsafe_load(i) + sz * DT)


def nbody(
    n: Int, x0: FPtr, y0: FPtr, z0: FPtr, m: FPtr,
    x: FPtr, y: FPtr, z: FPtr, vx: FPtr, vy: FPtr, vz: FPtr, threads: Int,
):
    for i in range(n):
        x.unsafe_store(i, x0.unsafe_load(i))
        y.unsafe_store(i, y0.unsafe_load(i))
        z.unsafe_store(i, z0.unsafe_load(i))
        vx.unsafe_store(i, 0.0)
        vy.unsafe_store(i, 0.0)
        vz.unsafe_store(i, 0.0)
    for _ in range(STEPS):
        if threads <= 0:
            nbody_accel(n, x, y, z, m, vx, vy, vz, 0, n)
        else:

            def work(t: Int) {
                var n, var threads, var x, var y, var z, var m,
                var vx, var vy, var vz
            }:
                var r = chunk(n, threads, t)
                nbody_accel(n, x, y, z, m, vx, vy, vz, r[0], r[1])

            parallelize(work, threads)
        for i in range(n):
            x.unsafe_store(i, x.unsafe_load(i) + vx.unsafe_load(i) * DT)
            y.unsafe_store(i, y.unsafe_load(i) + vy.unsafe_load(i) * DT)
            z.unsafe_store(i, z.unsafe_load(i) + vz.unsafe_load(i) * DT)


# ===------------------------------------------------------------- reduction / scan


def int_inputs(n: Int, a: IPtr):
    for i in range(n):
        a.unsafe_store(i, Int64((i * 2654435761) % 1000003))


def sum_range(a: IPtr, lo: Int, hi: Int) -> Int64:
    comptime WI = simd_width_of[I64]()
    var acc = SIMD[I64, WI](0)
    var i = lo
    while i + WI <= hi:
        acc += a.unsafe_load[width=WI](i)
        i += WI
    var total = acc.reduce_add()
    while i < hi:
        total += a.unsafe_load(i)
        i += 1
    return total


def reduction(n: Int, a: IPtr, threads: Int) -> Int64:
    if threads <= 0:
        return sum_range(a, 0, n)
    var partial = ialloc(threads)

    def work(t: Int) {var n, var a, var threads, var partial}:
        var r = chunk(n, threads, t)
        partial.unsafe_store(t, sum_range(a, r[0], r[1]))

    parallelize(work, threads)
    var total: Int64 = 0
    for t in range(threads):
        total += partial.unsafe_load(t)
    partial.unsafe_free()
    return total


def scan_range(a: IPtr, p: IPtr, lo: Int, hi: Int, offset: Int64):
    var acc = offset
    for i in range(lo, hi):
        acc += a.unsafe_load(i)
        p.unsafe_store(i, acc)


def prefix_sum(n: Int, a: IPtr, p: IPtr, threads: Int):
    if threads <= 0:
        scan_range(a, p, 0, n, 0)
        return
    # Two-pass parallel scan: (1) sum each chunk, (2) serial exclusive scan of
    # the T chunk sums, (3) scan each chunk starting from its offset.
    var sums = ialloc(threads)

    def pass1(t: Int) {var n, var a, var threads, var sums}:
        var r = chunk(n, threads, t)
        sums.unsafe_store(t, sum_range(a, r[0], r[1]))

    parallelize(pass1, threads)
    var run: Int64 = 0
    for t in range(threads):
        var s = sums.unsafe_load(t)
        sums.unsafe_store(t, run)
        run += s

    def pass2(t: Int) {var n, var a, var p, var threads, var sums}:
        var r = chunk(n, threads, t)
        scan_range(a, p, r[0], r[1], sums.unsafe_load(t))

    parallelize(pass2, threads)
    sums.unsafe_free()


def scan_checksum(n: Int, p: IPtr) -> Int64:
    var acc: Int64 = 0
    for i in range(n):
        acc += p.unsafe_load(i) % 1000003
    return acc


# ===------------------------------------------------------------- hash table

comptime FIB = UInt64(0x9E3779B97F4A7C15)


def hash_table(
    n: Int, keys: UPtr, queries: UPtr, tk: UPtr, tv: IPtr, cap: Int, shift: UInt64
) -> Tuple[Int64, Int64]:
    """Open addressing, linear probing, Fibonacci hashing; key 0 = empty."""
    var mask = cap - 1
    for h in range(cap):
        tk.unsafe_store(h, 0)
    for i in range(n):
        var k = keys.unsafe_load(i)
        var h = Int((k * FIB) >> shift)
        while True:
            var slot = tk.unsafe_load(h)
            if slot == 0 or slot == k:
                break
            h = (h + 1) & mask
        tk.unsafe_store(h, k)
        tv.unsafe_store(h, Int64(i))
    var found: Int64 = 0
    var vsum: Int64 = 0
    for i in range(n):
        var q = queries.unsafe_load(i)
        var h = Int((q * FIB) >> shift)
        while True:
            var slot = tk.unsafe_load(h)
            if slot == 0:
                break
            if slot == q:
                found += 1
                vsum += tv.unsafe_load(h)
                break
            h = (h + 1) & mask
    return (found, vsum)


# ===------------------------------------------------------------- driver


def now() -> Float64:
    return Float64(perf_counter_ns()) / 1e9


def report(kernel: String, checksum: String, times: List[Float64]):
    var s = String("RESULT ", kernel, " ", checksum, " ")
    for r in range(len(times)):
        if r > 0:
            s += ","
        s += String(times[r])
    print(s)


def main() raises:
    var args = argv()
    var kernel = String(args[1])
    if kernel == "noop":
        print("RESULT noop 0 0")
        return
    var n = Int(String(args[2]))
    var warmup = Int(String(args[3]))
    var repeats = Int(String(args[4]))
    var threads = Int(String(args[5])) if len(args) > 5 else 0
    var times = List[Float64]()
    var total_runs = warmup + repeats

    if kernel == "matmul_naive" or kernel == "matmul_tiled":
        var a = falloc(n * n)
        var b = falloc(n * n)
        var c = falloc(n * n)
        mat_inputs(n, a, b)
        for r in range(total_runs):
            var t0 = now()
            if kernel == "matmul_naive":
                matmul_naive(n, a, b, c)
            else:
                matmul_tiled(n, a, b, c, threads)
            if r >= warmup:
                times.append(now() - t0)
        report(kernel, mat_checksum(n, c), times)
        a.unsafe_free()
        b.unsafe_free()
        c.unsafe_free()
    elif kernel == "mandelbrot":
        var total = 0
        for r in range(total_runs):
            var t0 = now()
            total = mandelbrot(n, threads)
            if r >= warmup:
                times.append(now() - t0)
        report(kernel, String(total), times)
    elif kernel == "nbody":
        var x0 = falloc(n)
        var y0 = falloc(n)
        var z0 = falloc(n)
        var m = falloc(n)
        var x = falloc(n)
        var y = falloc(n)
        var z = falloc(n)
        var vx = falloc(n)
        var vy = falloc(n)
        var vz = falloc(n)
        for i in range(n):
            x0.unsafe_store(i, Float64((i * 37) % 101) / 101 - 0.5)
            y0.unsafe_store(i, Float64((i * 53) % 103) / 103 - 0.5)
            z0.unsafe_store(i, Float64((i * 71) % 107) / 107 - 0.5)
            m.unsafe_store(i, 1.0 + Float64(i % 7) / 7)
        for r in range(total_runs):
            var t0 = now()
            nbody(n, x0, y0, z0, m, x, y, z, vx, vy, vz, threads)
            if r >= warmup:
                times.append(now() - t0)
        var sx: Float64 = 0
        var sy: Float64 = 0
        var sz: Float64 = 0
        var sv: Float64 = 0
        for i in range(n):
            sx += x.unsafe_load(i)
            sy += y.unsafe_load(i)
            sz += z.unsafe_load(i)
            sv += vx.unsafe_load(i) + vy.unsafe_load(i) + vz.unsafe_load(i)
        report(kernel, String(sx + 2 * sy + 3 * sz + sv), times)
        for p in [x0, y0, z0, m, x, y, z, vx, vy, vz]:
            p.unsafe_free()
    elif kernel == "reduction" or kernel == "prefix_sum":
        var a = ialloc(n)
        var p = ialloc(n)
        int_inputs(n, a)
        var s: Int64 = 0
        for r in range(total_runs):
            var t0 = now()
            if kernel == "reduction":
                s = reduction(n, a, threads)
            else:
                prefix_sum(n, a, p, threads)
            if r >= warmup:
                times.append(now() - t0)
        if kernel == "prefix_sum":
            s = scan_checksum(n, p)
        report(kernel, String(s), times)
        a.unsafe_free()
        p.unsafe_free()
    elif kernel == "hash_table":
        var keys = unsafe_alloc[UInt64](n)
        var queries = unsafe_alloc[UInt64](n)
        var seq = unsafe_alloc[UInt64](2 * n)
        var xs: UInt64 = 42
        for i in range(2 * n):
            xs = xs * 48271 % 2147483647
            seq.unsafe_store(i, xs)
        for i in range(n):
            keys.unsafe_store(i, seq.unsafe_load(i))
            queries.unsafe_store(
                i, seq.unsafe_load(i) if i % 2 == 0 else seq.unsafe_load(n + i)
            )
        var cap = 1
        var bits = 0
        while cap < 2 * n:
            cap <<= 1
            bits += 1
        var tk = unsafe_alloc[UInt64](cap)
        var tv = ialloc(cap)
        var res = (Int64(0), Int64(0))
        for r in range(total_runs):
            var t0 = now()
            res = hash_table(n, keys, queries, tk, tv, cap, UInt64(64 - bits))
            if r >= warmup:
                times.append(now() - t0)
        report(kernel, String(res[1] * 7 + res[0]), times)
        for p in [keys, queries, seq, tk]:
            p.unsafe_free()
        tv.unsafe_free()
    else:
        print("UNSUPPORTED", kernel)
