#!/usr/bin/env python3
"""Roofline-style analysis of the parallel kernels.

1. Runs build/roofline_probe to measure this machine's roofs: peak FMA
   throughput and memory bandwidth (read-only sweep and STREAM triad), on
   1 thread and on all threads.
2. Estimates each kernel's arithmetic intensity (operations per byte of DRAM
   traffic) from the algorithm as implemented.
3. Takes the measured all-thread times from results/scaling.csv and reports
   achieved performance, the roof that applies, and whether each kernel is
   memory- or compute-bound here.

Writes results/roofline.md, results/roofline.json and results/roofline.png.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
IMPLS = [("c_omp", "C OpenMP", "#6250d6", "o"), ("mojo_par", "Mojo parallelize", "#eda100", "s")]
INF_AI = 3000.0  # where "no DRAM traffic" kernels are drawn


def kernel_model(kernel, n, checksum):
    """(operations, DRAM bytes, note) for one run of the kernel as implemented."""
    if kernel == "matmul_tiled":
        return 2 * n**3, 3 * 8 * n**2, (
            "2n³ flops; bytes = compulsory traffic for A, B and C (3·8n²), an upper bound on intensity. "
            "With 64-wide tiles and no reuse beyond a tile it would fall to ~5 flop/byte, still at or past the ridge.")
    if kernel == "mandelbrot":
        return 8 * int(checksum), 0, (
            "8 flops per useful iteration × total iterations (from the checksum); "
            "no array traffic at all (only a running count), so intensity is effectively infinite.")
    if kernel == "nbody":
        return 20 * 5 * n * n, 5 * 16 * 8 * n, (
            "~20 flops per pair (sqrt and divide counted as 1 each) × 5 steps × n²; "
            "bytes ≈ 16 doubles per body per step (positions, masses, velocities read and written).")
    if kernel == "reduction":
        return n, 8 * n, "1 integer add per 8-byte element read."
    if kernel == "prefix_sum":
        return 2 * n, 32 * n, (
            "Two-pass parallel scan: each element is read twice (8 + 8 bytes), written once (8) "
            "and pulled into cache before the write (write-allocate, 8): 2 adds per 32 bytes.")
    raise KeyError(kernel)


def probe(runs=3):
    """Best of several probe runs: a roof is the most the machine can deliver,
    and on a shared VM individual runs can come in 15-20 % low."""
    exe = ROOT / "build" / "roofline_probe"
    if not exe.exists():
        raise SystemExit("build/roofline_probe missing: run `make build` first")
    roofs = {}
    for _ in range(runs):
        for line in subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines():
            if line.startswith("PROBE"):
                _, name, threads, value = line.split()
                key = f"{name}_{threads}t"
                roofs[key] = max(roofs.get(key, 0.0), float(value))
    return roofs


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results")
    roofs = probe()
    nthreads = max(int(k.split("_")[-1][:-1]) for k in roofs)
    peak, bw_read, bw_triad = (roofs[f"{k}_{nthreads}t"] for k in ("peak_gflops", "read_gbs", "triad_gbs"))
    ridge = peak / bw_read

    rows = list(csv.DictReader(open(out / "scaling.csv")))
    points = []
    for r in rows:
        if int(r["threads"]) != nthreads:
            continue
        n = int(r["n"])
        ops, nbytes, note = kernel_model(r["kernel"], n, r["checksum"])
        t = float(r["median_s"])
        ai = ops / nbytes if nbytes else float("inf")
        attainable = min(peak, ai * bw_read)
        points.append({
            "kernel": r["kernel"], "impl": r["impl"], "n": n, "ai": ai,
            "achieved_gops": ops / t / 1e9, "achieved_gbs": nbytes / t / 1e9,
            "attainable_gops": attainable, "fraction_of_roof": ops / t / 1e9 / attainable,
            "bound": "memory" if ai < ridge else "compute", "note": note,
        })

    (out / "roofline.json").write_text(json.dumps({"roofs": roofs, "ridge_flop_per_byte": ridge,
                                                   "points": points}, indent=2, default=str) + "\n")

    # ---- Markdown
    md = ["# Roofline analysis", "",
          f"Measured roofs on this machine (best of 3 runs of `build/roofline_probe`):", "",
          "| Roof | All threads | 1 thread |", "|---|---:|---:|",
          f"| Peak FP64 FMA throughput | {peak:.0f} GFLOP/s | {roofs['peak_gflops_1t']:.0f} GFLOP/s |",
          f"| Memory bandwidth, read-only sweep | {bw_read:.1f} GB/s | {roofs['read_gbs_1t']:.1f} GB/s |",
          f"| Memory bandwidth, STREAM triad | {bw_triad:.1f} GB/s | {roofs['triad_gbs_1t']:.1f} GB/s |",
          "",
          f"Ridge point = peak / read bandwidth = **{ridge:.1f} operations per byte**. Kernels to the left are "
          "memory-bound; kernels to the right can only be limited by compute.", "",
          f"| Kernel | n | Intensity (op/byte) | Bound | Implementation | Achieved | Roof | % of roof |",
          "|---|---:|---:|---|---|---:|---:|---:|"]
    for p in points:
        ai = "∞" if p["ai"] == float("inf") else f"{p['ai']:.3g}"
        name = dict((i, n) for i, n, *_ in IMPLS)[p["impl"]]
        extra = f" ({p['achieved_gbs']:.1f} GB/s)" if p["bound"] == "memory" else ""
        md.append(f"| {p['kernel']} | {p['n']:,} | {ai} | {p['bound']} | {name} | "
                  f"{p['achieved_gops']:.1f} Gop/s{extra} | {p['attainable_gops']:.1f} | {100 * p['fraction_of_roof']:.0f} % |")
    md += ["", "Operation and byte counts:", ""]
    for kernel in dict.fromkeys(p["kernel"] for p in points):
        md.append(f"- **{kernel}**: " + next(p["note"] for p in points if p["kernel"] == kernel))
    md.append("")
    (out / "roofline.md").write_text("\n".join(md))

    # ---- Chart
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [0.01 * 1.15**i for i in range(110)]
    ax.plot(xs, [min(peak, x * bw_read) for x in xs], color=INK, lw=2, label=f"roof: read bandwidth {bw_read:.0f} GB/s")
    ax.plot(xs, [min(peak, x * bw_triad) for x in xs], color=INK2, lw=1, ls="--",
            label=f"roof: triad bandwidth {bw_triad:.0f} GB/s")
    ax.text(ridge * 1.3, peak * 1.12, f"FP64 FMA peak {peak:.0f} GFLOP/s", fontsize=8, color=INK2)
    ax.axvline(ridge, color=GRID, lw=1)
    for impl, name, colour, marker in IMPLS:
        pts = [p for p in points if p["impl"] == impl]
        ax.scatter([min(p["ai"], INF_AI) for p in pts], [p["achieved_gops"] for p in pts], s=70, color=colour,
                   marker=marker, edgecolors=SURFACE, linewidths=1.5, label=name, zorder=3)
    # Hand-placed label offsets (points) so labels clear the marks and the roof.
    offsets = {"prefix_sum": (10, -4, "left"), "reduction": (10, -4, "left"), "matmul_tiled": (0, 10, "center"),
               "nbody": (0, 10, "center"), "mandelbrot": (0, -20, "center")}
    for kernel in dict.fromkeys(p["kernel"] for p in points):
        p = max((q for q in points if q["kernel"] == kernel), key=lambda q: q["achieved_gops"])
        x = min(p["ai"], INF_AI)
        label = kernel + ("\n(no DRAM traffic)" if p["ai"] == float("inf") else "")
        dx, dy, ha = offsets.get(kernel, (8, 8, "left"))
        ax.annotate(label, (x, p["achieved_gops"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, INF_AI * 2)
    ax.set_ylim(0.3, peak * 2.5)
    ax.set_xlabel("arithmetic intensity (operations per byte of DRAM traffic)", color=INK2)
    ax.set_ylabel("achieved performance (G operations/s)", color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6, which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_title(f"Roofline, {nthreads} threads: measured roofs, measured kernels", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "roofline.png", facecolor=SURFACE)
    print((out / "roofline.md").read_text())


if __name__ == "__main__":
    main()
