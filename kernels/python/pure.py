"""Pure-Python reference implementations (lists, loops, dict - no NumPy).

These are deliberately written the way a competent Python programmer would
write them without reaching for extension modules. They are the baseline that
every speed-up in the results is measured against, and the source of truth
for correctness checks.
"""

import math
from itertools import accumulate

from harness import run

# ---------------------------------------------------------------- matmul


def mat_inputs(n):
    a = [[float((7 * i + 3 * j) % 11 - 5) for j in range(n)] for i in range(n)]
    b = [[float((5 * i + 13 * j) % 7 - 3) for j in range(n)] for i in range(n)]
    return n, a, b


def matmul_naive(state):
    n, a, b = state
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ai, ci = a[i], c[i]
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += ai[k] * b[k][j]
            ci[j] = s
    return c


def matmul_tiled(state, bs=32):
    """Blocked i-k-j loop order: unit-stride access to rows of B and C."""
    n, a, b = state
    c = [[0.0] * n for _ in range(n)]
    for ii in range(0, n, bs):
        for kk in range(0, n, bs):
            for jj in range(0, n, bs):
                jend = min(jj + bs, n)
                for i in range(ii, min(ii + bs, n)):
                    ai, ci = a[i], c[i]
                    for k in range(kk, min(kk + bs, n)):
                        aik, bk = ai[k], b[k]
                        for j in range(jj, jend):
                            ci[j] += aik * bk[j]
    return c


def mat_checksum(c):
    """Weighted sum plus sum of squares: exact in float64 and sensitive to any wrong element."""
    n = len(c)
    return int(sum(v * ((3 * i + 5 * j) % 13 + 1) + v * v
               for i in range(n) for j, v in enumerate(c[i])))


# ---------------------------------------------------------------- mandelbrot

MAX_ITER = 200


def mandelbrot(n):
    total = 0
    for i in range(n):
        y0 = -1.25 + 2.5 * i / n
        for j in range(n):
            x0 = -2.0 + 2.5 * j / n
            x = y = 0.0
            k = 0
            while k < MAX_ITER and x * x + y * y <= 4.0:
                x, y = x * x - y * y + x0, 2.0 * x * y + y0
                k += 1
            total += k
    return total


# ---------------------------------------------------------------- n-body

STEPS, DT, EPS2 = 5, 0.01, 0.01


def nbody_inputs(n):
    x = [((i * 37) % 101) / 101 - 0.5 for i in range(n)]
    y = [((i * 53) % 103) / 103 - 0.5 for i in range(n)]
    z = [((i * 71) % 107) / 107 - 0.5 for i in range(n)]
    m = [1.0 + (i % 7) / 7 for i in range(n)]
    return n, x, y, z, m


def nbody(state):
    n, x0, y0, z0, m = state
    x, y, z = list(x0), list(y0), list(z0)  # each repeat starts from the same state
    vx, vy, vz = [0.0] * n, [0.0] * n, [0.0] * n
    for _ in range(STEPS):
        for i in range(n):
            xi, yi, zi = x[i], y[i], z[i]
            ax = ay = az = 0.0
            for j in range(n):
                dx, dy, dz = x[j] - xi, y[j] - yi, z[j] - zi
                d2 = dx * dx + dy * dy + dz * dz + EPS2
                s = m[j] / (d2 * math.sqrt(d2))
                ax += dx * s
                ay += dy * s
                az += dz * s
            vx[i] += ax * DT
            vy[i] += ay * DT
            vz[i] += az * DT
        for i in range(n):
            x[i] += vx[i] * DT
            y[i] += vy[i] * DT
            z[i] += vz[i] * DT
    return x, y, z, vx, vy, vz


def nbody_checksum(out):
    x, y, z, vx, vy, vz = out
    return sum(x) + 2 * sum(y) + 3 * sum(z) + sum(vx) + sum(vy) + sum(vz)


# ---------------------------------------------------------------- reduction / scan


def int_array(n):
    return [(i * 2654435761) % 1000003 for i in range(n)]


def scan_checksum(p):
    return sum(v % 1000003 for v in p)


# ---------------------------------------------------------------- hash table

PM_A, PM_M = 48271, 2147483647  # Park-Miller "minimal standard" generator


def hash_inputs(n):
    x, seq = 42, []
    for _ in range(2 * n):
        x = x * PM_A % PM_M
        seq.append(x)
    keys, absent = seq[:n], seq[n:]
    queries = [keys[i] if i % 2 == 0 else absent[i] for i in range(n)]
    return keys, queries


def hash_table(state):
    keys, queries = state
    table = {}
    for i, k in enumerate(keys):
        table[k] = i
    found = vsum = 0
    for q in queries:
        v = table.get(q)
        if v is not None:
            found += 1
            vsum += v
    return found, vsum


KERNELS = {
    "matmul_naive": (mat_inputs, matmul_naive, mat_checksum),
    "matmul_tiled": (mat_inputs, matmul_tiled, mat_checksum),
    "mandelbrot": (lambda n: n, mandelbrot, lambda t: t),
    "nbody": (nbody_inputs, nbody, nbody_checksum),
    "reduction": (int_array, sum, lambda s: s),
    "prefix_sum": (int_array, lambda a: list(accumulate(a)), scan_checksum),
    "hash_table": (hash_inputs, hash_table, lambda r: r[1] * 7 + r[0]),
}

if __name__ == "__main__":
    run(KERNELS)
