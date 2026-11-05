#!/usr/bin/env python3
"""Draw results/speedup.png: speed-up over pure Python per kernel (log scale).

Usage: python3 bench/plot.py [results_dir]
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Fixed categorical order: an implementation always gets the same colour.
SERIES = [
    ("numpy", "NumPy", "#2a78d6"),
    ("c", "C (1 thread)", "#eb6834"),
    ("mojo", "Mojo (1 thread)", "#1baf7a"),
    ("mojo_par", "Mojo (all cores)", "#eda100"),
]


def label(x):
    return f"{x:,.0f}×" if x >= 100 else f"{x:.1f}×"


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = list(csv.DictReader(open(out / "results.csv")))
    kernels = list(dict.fromkeys(r["kernel"] for r in rows))
    speed = {(r["kernel"], r["impl"]): float(r["speedup_vs_python"]) for r in rows if r["speedup_vs_python"]}

    bar_h, gap = 0.19, 0.02
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s, (impl, name, colour) in enumerate(SERIES):
        ys, xs = [], []
        for k, kernel in enumerate(kernels):
            if (kernel, impl) in speed:
                ys.append(k + (s - 1.5) * (bar_h + gap))
                xs.append(speed[(kernel, impl)])
        # Bars grow from the Python baseline (1×) to the measured speed-up.
        ax.barh(ys, [x - 1 for x in xs], height=bar_h, color=colour, label=name, left=1)
        for y, x in zip(ys, xs):
            ax.text(x * 1.08, y, label(x), va="center", ha="left", fontsize=7.5, color=INK2)

    ax.axvline(1, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1, max(speed.values()) * 4)
    ax.set_yticks(range(len(kernels)), kernels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Speed-up over pure Python (log scale, higher is better; Python = 1×)", color=INK2)
    ax.tick_params(axis="x", colors=INK2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("mojo-bench: median speed-up over pure Python", loc="left", fontsize=13, color=INK, pad=12)
    fig.text(0.01, 0.01, "Missing bar = no sensible implementation in that language. See results.md for times.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "speedup.png", facecolor=SURFACE)
    print(f"wrote {out / 'speedup.png'}")


if __name__ == "__main__":
    main()
