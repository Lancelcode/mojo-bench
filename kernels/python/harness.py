"""Shared timing harness for the Python and NumPy implementations.

Every implementation in this repository (Python, NumPy, C, Mojo) speaks the
same tiny protocol so the runner can treat them identically:

    <impl> <kernel> <n> <warmup> <repeats> [threads]

and prints exactly one line on stdout:

    RESULT <kernel> <checksum> <t1,t2,...>

where the times are seconds for each timed repeat. Input generation happens
before timing starts; the checksum is computed after timing finishes.
"""

import sys
import time


def run(kernels):
    """`kernels` maps a name to (setup(n) -> state, run(state) -> out, checksum(out))."""
    kernel = sys.argv[1]
    if kernel == "noop":  # used to measure process start-up cost
        print("RESULT noop 0 0")
        return
    n, warmup, repeats = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    if kernel not in kernels:
        print(f"UNSUPPORTED {kernel}")
        sys.exit(3)
    setup, body, checksum = kernels[kernel]
    state = setup(n)
    out = None
    for _ in range(warmup):
        out = body(state)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = body(state)
        times.append(time.perf_counter() - t0)
    print(f"RESULT {kernel} {checksum(out)!r} {','.join(f'{t:.9f}' for t in times)}")
