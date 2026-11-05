#!/usr/bin/env python3
"""Benchmark runner for mojo-bench.

Runs every kernel in every implementation, checks that all implementations
agree with the pure-Python reference, and writes:

    <out>/results.csv   one row per (kernel, implementation)
    <out>/results.md    Markdown tables: machine, sizes, times, speed-ups
    <out>/machine.json  machine and toolchain details
    <out>/startup.csv   process start-up and compile costs

Usage:
    python3 bench/run.py                 # full run (what `make bench` does)
    python3 bench/run.py --profile quick # tiny sizes, correctness only (CI)
"""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
PY = sys.executable
NCPU = os.cpu_count() or 1

# Problem sizes. "full" sizes are chosen so that pure Python takes roughly
# 0.5-3 s per run; "quick" is for the CI correctness pass.
# Repeats are adaptive: a pilot run estimates the cost, then each
# implementation gets enough repeats to fill `budget_s` seconds (clamped to
# [repeats, max_repeats]). Fast kernels therefore get many samples, which
# matters on a noisy shared machine; slow pure-Python runs get the minimum.
PROFILES = {
    "full": {
        "warmup": 1,
        "repeats": 5,
        "max_repeats": 200,
        "budget_s": 1.0,
        "sizes": {
            "matmul_naive": 384,
            "matmul_tiled": 384,
            "mandelbrot": 640,
            "nbody": 1024,
            "reduction": 10_000_000,
            "prefix_sum": 10_000_000,
            "hash_table": 2_000_000,
        },
    },
    "quick": {
        "warmup": 0,
        "repeats": 1,
        "max_repeats": 1,
        "budget_s": 0.0,
        "sizes": {
            "matmul_naive": 37,
            "matmul_tiled": 101,
            "mandelbrot": 67,
            "nbody": 67,
            "reduction": 100_003,
            "prefix_sum": 100_003,
            "hash_table": 10_007,
        },
    },
}

SIZE_NOTES = {
    "matmul_naive": "n x n float64 matrices",
    "matmul_tiled": "n x n float64 matrices",
    "mandelbrot": "n x n pixels, 200 max iterations",
    "nbody": "n bodies, 5 steps, O(n^2) forces",
    "reduction": "n int64 values",
    "prefix_sum": "n int64 values",
    "hash_table": "n inserts + n lookups (50% hits)",
}

# Relative tolerance per kernel. Integer-valued checksums must match exactly,
# with one exception: compiled code may fuse x*x - y*y + c into an FMA, which
# rounds differently from Python and can change the escape iteration of a
# handful of boundary pixels in Mandelbrot (12 of 225.6 M iterations at
# 2048 x 2048; none at the benchmark size). n-body sums floating-point forces
# in different orders (SIMD lanes), so a tiny relative difference is expected.
TOLERANCE = {"mandelbrot": 1e-6, "nbody": 1e-9}

# Kernels with a parallel code path. The naive matmul is the deliberately
# simple scalar baseline, and the hash table is a serial dependency chain.
PARALLEL_KERNELS = {"matmul_tiled", "mandelbrot", "nbody", "reduction", "prefix_sum"}
PARALLEL_IMPLS = {"mojo_par"}

SERIAL_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# name -> (label, argv prefix, threads argument or None, extra env)
IMPLS = {
    "python": ("Python", [PY, str(ROOT / "kernels/python/pure.py")], None, SERIAL_ENV),
    "numpy": ("NumPy", [PY, str(ROOT / "kernels/python/numpy_impl.py")], None, SERIAL_ENV),
    "c": ("C (1 thread)", [str(BUILD / "kernels_c")], None, SERIAL_ENV),
    "mojo": ("Mojo (1 thread)", [str(BUILD / "kernels_mojo")], 0, {}),
    "mojo_par": (f"Mojo ({NCPU} threads)", [str(BUILD / "kernels_mojo")], NCPU, {}),
}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str)).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def machine_info():
    lscpu = sh(["lscpu"])

    def field(name):
        m = re.search(rf"^{name}:\s*(.+)$", lscpu, re.M)
        return m.group(1).strip() if m else ""

    flags = field("Flags").split()
    os_name = platform.platform()
    if Path("/etc/os-release").exists():
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)', Path("/etc/os-release").read_text(), re.M)
        if m:
            os_name = f"{m.group(1)} (kernel {platform.release()})"
    try:
        import numpy

        np_ver = numpy.__version__
    except ImportError:
        np_ver = "not installed"
    cores = field("Core\\(s\\) per socket")
    sockets = field("Socket\\(s\\)")
    return {
        "cpu": field("Model name") or platform.processor(),
        "logical_cpus": NCPU,
        "physical_cores": int(cores) * int(sockets) if cores.isdigit() and sockets.isdigit() else None,
        "simd": " ".join(f for f in ("sse4_2", "avx", "avx2", "fma", "avx512f") if f in flags),
        "caches": ", ".join(f"{k} {field(k)}" for k in ("L1d cache", "L2 cache", "L3 cache") if field(k)),
        "virtualised": "hypervisor" in flags,
        "os": os_name,
        "python": platform.python_version(),
        "numpy": np_ver,
        "gcc": sh(["gcc", "--version"]).splitlines()[0] if shutil.which("gcc") else "not found",
        "mojo": sh(["mojo", "--version"]) or "not found",
        "c_flags": "-O3 -march=native",
    }


def run_impl(impl, kernel, n, warmup, repeats, threads=None):
    """Run one implementation; returns (checksum, [times]) or None if unsupported."""
    _, argv, default_threads, env = IMPLS[impl]
    if impl in PARALLEL_IMPLS and kernel not in PARALLEL_KERNELS:
        return None
    t = default_threads if threads is None else threads
    cmd = argv + [kernel, str(n), str(warmup), str(repeats)] + ([] if t is None else [str(t)])
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env}, timeout=1800)
    for line in proc.stdout.splitlines():
        if line.startswith("UNSUPPORTED"):
            return None
        if line.startswith("RESULT"):
            _, _, checksum, times = line.split()
            return checksum, [float(x) for x in times.split(",")]
    raise RuntimeError(f"{impl}/{kernel} failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")


def agrees(kernel, value, reference):
    if value == reference:
        return True, 0.0
    try:
        a, b = float(value), float(reference)
    except ValueError:
        return False, float("inf")
    err = abs(a - b) / max(abs(b), 1e-300)
    return err <= TOLERANCE.get(kernel, 0.0), err


def measure_startup(repeats=5):
    """Wall-clock cost of starting each runtime and of compiling the code."""
    rows = []

    def wall(cmd, runs):
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            subprocess.run(cmd, capture_output=True, check=True)
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    rows.append(("Python interpreter start (no-op kernel)", wall([PY, str(ROOT / "kernels/python/pure.py"), "noop"], repeats)))
    rows.append(("Python + import numpy (no-op kernel)", wall([PY, str(ROOT / "kernels/python/numpy_impl.py"), "noop"], repeats)))
    rows.append(("C binary start (no-op kernel)", wall([str(BUILD / "kernels_c"), "noop"], repeats)))
    rows.append(("Mojo binary start (no-op kernel)", wall([str(BUILD / "kernels_mojo"), "noop"], repeats)))
    with tempfile.TemporaryDirectory() as tmp:
        rows.append(("gcc -O3 compile of kernels.c", wall(
            ["gcc", "-O3", "-march=native", "-o", f"{tmp}/k", str(ROOT / "kernels/c/kernels.c"), "-lm"], 3)))
        rows.append(("mojo build of kernels.mojo", wall(
            ["mojo", "build", "-O3", str(ROOT / "kernels/mojo/kernels.mojo"), "-o", f"{tmp}/k"], 3)))
    rows.append(("mojo run kernels.mojo noop (JIT compile + run)", wall(
        ["mojo", "run", str(ROOT / "kernels/mojo/kernels.mojo"), "noop"], 3)))
    return rows


def fmt_time(s):
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.1f} ms"
    return f"{s * 1e6:.0f} µs"


def fmt_speedup(x):
    return f"{x:,.0f}×" if x >= 100 else f"{x:.1f}×"


def write_markdown(path, machine, profile, rows, startup, impls):
    by = {(r["kernel"], r["impl"]): r for r in rows}
    kernels = list(dict.fromkeys(r["kernel"] for r in rows))
    out = ["# Results", ""]
    out += ["## Machine", "", "| | |", "|---|---|"]
    for k, v in machine.items():
        out.append(f"| {k} | {v} |")
    out += ["", f"Warm-up runs: {profile['warmup']}. Timed repeats are adaptive: at least {profile['repeats']}, "
            f"otherwise enough to fill ~{profile['budget_s']:g} s, capped at {profile['max_repeats']} "
            "(see the `repeats` column of results.csv). Times are medians of the timed repeats; "
            "input generation is excluded.", ""]
    out += ["## Problem sizes", "", "| Kernel | n | Meaning |", "|---|---:|---|"]
    for k in kernels:
        out.append(f"| {k} | {profile['sizes'][k]:,} | {SIZE_NOTES[k]} |")
    head = "| Kernel | " + " | ".join(IMPLS[i][0] for i in impls) + " |"
    sep = "|---|" + "---:|" * len(impls)
    out += ["", "## Median time", "", head, sep]
    for k in kernels:
        cells = [fmt_time(by[(k, i)]["median_s"]) if (k, i) in by else "n/a" for i in impls]
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out += ["", "## Speed-up over pure Python (higher is better)", "", head, sep]
    for k in kernels:
        cells = []
        for i in impls:
            r = by.get((k, i))
            cells.append(fmt_speedup(r["speedup_vs_python"]) if r and r["speedup_vs_python"] else "n/a")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out += ["", "n/a = no sensible implementation in that language (see README).", ""]
    if startup:
        out += ["## Start-up and compile costs", "", "| What | Median wall time |", "|---|---:|"]
        out += [f"| {name} | {fmt_time(t)} |" for name, t in startup]
        out.append("")
    Path(path).write_text("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=PROFILES, default="full")
    ap.add_argument("--kernels", nargs="*", help="subset of kernels to run")
    ap.add_argument("--impls", nargs="*", default=list(IMPLS), choices=list(IMPLS))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--no-startup", action="store_true", help="skip start-up/compile timings")
    args = ap.parse_args()

    profile = PROFILES[args.profile]
    kernels = args.kernels or list(profile["sizes"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    machine = machine_info()
    print(json.dumps(machine, indent=2))

    rows, failures = [], []
    for kernel in kernels:
        n = profile["sizes"][kernel]
        results = {}
        repeats_used = {}
        for impl in args.impls:
            repeats = profile["repeats"]
            if profile["budget_s"] > 0:
                pilot = run_impl(impl, kernel, n, 0, 1)
                if pilot is None:
                    print(f"  {kernel:13s} {impl:9s} n/a")
                    continue
                est = max(pilot[1][0], 1e-6)
                repeats = min(max(repeats, int(profile["budget_s"] / est)), profile["max_repeats"])
            res = run_impl(impl, kernel, n, profile["warmup"], repeats)
            if res is None:
                print(f"  {kernel:13s} {impl:9s} n/a")
                continue
            results[impl] = res
            repeats_used[impl] = repeats
        reference = results["python"][0] if "python" in results else next(iter(results.values()))[0]
        py_median = statistics.median(results["python"][1]) if "python" in results else None
        for impl, (checksum, times) in results.items():
            ok, err = agrees(kernel, checksum, reference)
            if not ok:
                failures.append(f"{kernel}/{impl}: checksum {checksum} != reference {reference}")
            med = statistics.median(times)
            rows.append({
                "kernel": kernel, "impl": impl, "n": n,
                "threads": IMPLS[impl][2] if IMPLS[impl][2] else 1,
                "warmup": profile["warmup"], "repeats": repeats_used[impl],
                "median_s": med, "min_s": min(times), "mean_s": statistics.fmean(times),
                "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
                "speedup_vs_python": py_median / med if py_median and med > 0 else None,
                "checksum": checksum, "rel_err_vs_python": err, "verified": ok,
            })
            print(f"  {kernel:13s} {impl:9s} {fmt_time(med):>10s}  checksum {checksum} {'OK' if ok else 'MISMATCH'}")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "machine.json").write_text(json.dumps(machine, indent=2) + "\n")

    startup = []
    if not args.no_startup:
        startup = measure_startup()
        with open(out / "startup.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["what", "median_s"])
            w.writerows(startup)
    write_markdown(out / "results.md", machine, profile, rows, startup, args.impls)
    print(f"wrote {out}/results.csv and {out}/results.md")

    if failures:
        print("CORRECTNESS FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("all implementations agree with the pure-Python reference")


if __name__ == "__main__":
    main()
