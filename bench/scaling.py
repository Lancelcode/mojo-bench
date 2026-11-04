#!/usr/bin/env python3
"""Thread-scaling study: C (OpenMP) vs Mojo (parallelize), 1..all cores.

Each parallel kernel is run with 1, 2, ... N threads through its parallel
code path. Speed-up is relative to the same implementation at 1 thread, so it
measures how well the code scales, not which language is faster (absolute
times are in the CSV and the table).

Sizes are larger than in the main benchmark because pure Python is not
involved: the two streaming kernels use arrays far bigger than the L3 cache so
that they hit DRAM bandwidth.

Writes <out>/scaling.csv, <out>/scaling.md and <out>/scaling.png.
Usage: python3 bench/scaling.py [--out results] [--max-threads N]
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run import NCPU, ROOT, run_impl  # noqa: E402

SIZES = {
    "matmul_tiled": (1024, "1,024 × 1,024 float64"),
    "mandelbrot": (2048, "2,048 × 2,048 pixels"),
    "nbody": (4096, "4,096 bodies, 5 steps"),
    "reduction": (100_000_000, "100 M int64 (800 MB)"),
    "prefix_sum": (100_000_000, "100 M int64 (800 MB in + 800 MB out)"),
}
IMPLS = [("c_omp", "C OpenMP", "#6250d6", "o"), ("mojo_par", "Mojo parallelize", "#eda100", "s")]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BUDGET_S, MIN_REPEATS, MAX_REPEATS = 1.0, 5, 50


def measure(impl, kernel, n, threads):
    pilot = run_impl(impl, kernel, n, 0, 1, threads=threads)
    repeats = min(max(MIN_REPEATS, int(BUDGET_S / max(pilot[1][0], 1e-6))), MAX_REPEATS)
    checksum, times = run_impl(impl, kernel, n, 1, repeats, threads=threads)
    return checksum, statistics.median(times), min(times), repeats


def plot(rows, out, max_threads):
    kernels = list(SIZES)
    fig, axes = plt.subplots(1, len(kernels), figsize=(14, 3.8), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    threads = list(range(1, max_threads + 1))
    for ax, kernel in zip(axes, kernels):
        ax.set_facecolor(SURFACE)
        ax.plot(threads, threads, color=INK2, lw=1, ls="--", label="ideal")
        for impl, name, colour, marker in IMPLS:
            pts = sorted((r["threads"], r["speedup"]) for r in rows if r["kernel"] == kernel and r["impl"] == impl)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, lw=2, marker=marker, ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
            ax.annotate(f"{pts[-1][1]:.1f}×", (pts[-1][0], pts[-1][1]), xytext=(6, -3 if impl == "c_omp" else 6),
                        textcoords="offset points", fontsize=8, color=INK)
        ax.set_title(kernel, fontsize=10, color=INK, loc="left")
        ax.set_xticks(threads)
        ax.set_xlim(0.7, max_threads + 0.9)
        ax.set_ylim(0, max_threads + 0.5)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlabel("threads", fontsize=9, color=INK2)
    axes[0].set_ylabel("speed-up over 1 thread", fontsize=9, color=INK2)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Thread scaling: speed-up over the same implementation on 1 thread", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "scaling.png", facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--max-threads", type=int, default=NCPU)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for kernel, (n, _) in SIZES.items():
        reference = None
        for impl, *_ in IMPLS:
            base = None
            for t in range(1, args.max_threads + 1):
                checksum, med, best, reps = measure(impl, kernel, n, t)
                reference = reference or checksum
                if abs(float(checksum) - float(reference)) > 1e-9 * abs(float(reference)):
                    raise SystemExit(f"checksum mismatch {kernel}/{impl}/{t}: {checksum} vs {reference}")
                base = base or med
                rows.append({"kernel": kernel, "impl": impl, "n": n, "threads": t, "median_s": med,
                             "min_s": best, "repeats": reps, "speedup": base / med,
                             "efficiency": base / med / t, "checksum": checksum})
                print(f"{kernel:13s} {impl:9s} t={t}  {med * 1e3:9.2f} ms  speed-up {base / med:4.2f}")

    with open(out / "scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = {(r["kernel"], r["impl"], r["threads"]): r for r in rows}
    md = ["# Thread scaling", "",
          "Median time (speed-up over the same implementation on 1 thread). "
          "All checksums agree across thread counts and languages.", ""]
    head = "| Kernel | Size | Implementation | " + " | ".join(f"{t} thr" for t in range(1, args.max_threads + 1)) + " |"
    md += [head, "|---|---|---|" + "---:|" * args.max_threads]
    for kernel, (n, desc) in SIZES.items():
        for impl, name, *_ in IMPLS:
            cells = []
            for t in range(1, args.max_threads + 1):
                r = by[(kernel, impl, t)]
                cells.append(f"{r['median_s'] * 1e3:.1f} ms ({r['speedup']:.2f}×)")
            md.append(f"| {kernel} | {desc} | {name} | " + " | ".join(cells) + " |")
    (out / "scaling.md").write_text("\n".join(md) + "\n")
    plot(rows, out, args.max_threads)
    print(f"wrote {out}/scaling.csv, scaling.md, scaling.png")


if __name__ == "__main__":
    main()
