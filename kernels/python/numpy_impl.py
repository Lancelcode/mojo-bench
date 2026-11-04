"""NumPy implementations: the idiomatic "make Python fast" answer.

Only kernels with a natural array formulation are implemented. A naive
triple-loop matmul and an open-addressing hash table have no sensible NumPy
equivalent, so those report UNSUPPORTED and appear as "n/a" in the results.
`matmul_tiled` uses `a @ b`, which calls the bundled BLAS (OpenBLAS): an
expertly blocked, vectorised GEMM, i.e. the ceiling rather than a peer.
"""

import numpy as np

from harness import run

# ---------------------------------------------------------------- matmul


def mat_inputs(n):
    i, j = np.indices((n, n))
    a = ((7 * i + 3 * j) % 11 - 5).astype(np.float64)
    b = ((5 * i + 13 * j) % 7 - 3).astype(np.float64)
    return a, b


def mat_checksum(c):
    i, j = np.indices(c.shape)
    return int((c * ((3 * i + 5 * j) % 13 + 1) + c * c).sum())


# ---------------------------------------------------------------- mandelbrot

MAX_ITER = 200


def mandelbrot(n):
    """Vectorised escape-time loop that drops escaped points each iteration."""
    j = np.arange(n, dtype=np.float64)
    cx = np.tile(-2.0 + 2.5 * j / n, n)
    cy = np.repeat(-1.25 + 2.5 * j / n, n)
    idx = np.arange(n * n)
    x = np.zeros(n * n)
    y = np.zeros(n * n)
    counts = np.zeros(n * n, dtype=np.int64)
    for _ in range(MAX_ITER):
        active = x * x + y * y <= 4.0
        idx, x, y, cx, cy = idx[active], x[active], y[active], cx[active], cy[active]
        if idx.size == 0:
            break
        counts[idx] += 1
        x, y = x * x - y * y + cx, 2.0 * x * y + cy
    return int(counts.sum())


# ---------------------------------------------------------------- n-body

STEPS, DT, EPS2 = 5, 0.01, 0.01


def nbody_inputs(n):
    i = np.arange(n)
    x = (i * 37) % 101 / 101 - 0.5
    y = (i * 53) % 103 / 103 - 0.5
    z = (i * 71) % 107 / 107 - 0.5
    m = 1.0 + (i % 7) / 7
    return x, y, z, m


def nbody(state):
    x, y, z, m = (a.copy() for a in state)
    vx, vy, vz = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for _ in range(STEPS):
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        dz = z[None, :] - z[:, None]
        d2 = dx * dx + dy * dy + dz * dz + EPS2
        s = m[None, :] / (d2 * np.sqrt(d2))
        vx += (dx * s).sum(axis=1) * DT
        vy += (dy * s).sum(axis=1) * DT
        vz += (dz * s).sum(axis=1) * DT
        x += vx * DT
        y += vy * DT
        z += vz * DT
    return x, y, z, vx, vy, vz


def nbody_checksum(out):
    x, y, z, vx, vy, vz = out
    return float(x.sum() + 2 * y.sum() + 3 * z.sum() + vx.sum() + vy.sum() + vz.sum())


# ---------------------------------------------------------------- reduction / scan


def int_array(n):
    return np.arange(n, dtype=np.int64) * 2654435761 % 1000003


# ---------------------------------------------------------------- word count


def text_bytes(n):
    from pure import make_text  # same generator as the reference, so identical input

    return np.frombuffer(make_text(n).encode("ascii"), dtype=np.uint8)


def word_count(b):
    """Vectorised: find word boundaries, build an exact base-257 key per word
    (exact for words up to 7 letters, a 64-bit hash beyond), then np.unique."""
    w = (b != 32) & (b != 10)
    starts = np.flatnonzero(w & ~np.concatenate(([False], w[:-1])))
    ends = np.flatnonzero(w & ~np.concatenate((w[1:], [False]))) + 1
    lens = ends - starts
    key = np.zeros(starts.size, dtype=np.uint64)
    for j in range(int(lens.max())):
        m = lens > j
        key[m] = key[m] * np.uint64(257) + b[starts[m] + j].astype(np.uint64)
    _, counts = np.unique(key, return_counts=True)
    return counts


def counts_checksum(c):
    c = c.astype(np.int64)
    return f"{int(c.sum())}/{c.size}/{int(c.max())}/{int((c * c).sum())}"


KERNELS = {
    "matmul_tiled": (mat_inputs, lambda s: s[0] @ s[1], mat_checksum),
    "mandelbrot": (lambda n: n, mandelbrot, lambda t: t),
    "nbody": (nbody_inputs, nbody, nbody_checksum),
    "reduction": (int_array, lambda a: int(a.sum()), lambda s: s),
    "prefix_sum": (int_array, np.cumsum, lambda p: int((p % 1000003).sum())),
    "word_count": (text_bytes, word_count, counts_checksum),
}

if __name__ == "__main__":
    run(KERNELS)
