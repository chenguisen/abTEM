#!/usr/bin/env python3
"""Plot V1a merged figure — Stage A vs B vacuum unitarity comparison.

Reads docs/data/v1a_vacuum_{A,B}.npz and generates a single merged PDF.
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "lines.linewidth": 0.8, "lines.markersize": 3,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
WONG = {
    "blue": "#0072B2", "orange": "#E69F00", "red": "#D55E00",
    "green": "#009E73", "black": "#000000", "grey": "#555555",
}
FIG_W = 183 / 25.4
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NO_LABELS = "--no-labels" in sys.argv


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.38))
    gs = fig.add_gridspec(1, 2, wspace=0.45, left=0.09, right=0.97,
                          top=0.88, bottom=0.22)

    for col, stage in enumerate(["A", "B"]):
        base = os.path.join(DATA_DIR, f"v1a_vacuum_{stage}")
        data = np.load(base + ".npz")
        with open(base + ".json") as f:
            manifest = json.load(f)

        depths = data["depths"]
        I_ratio = data["I_ratio"]
        max_dev = float(data["max_deviation"])
        px_A = manifest["params"]["sampling_A_per_px"]
        n_sl = manifest["params"]["num_slices"]

        ax = fig.add_subplot(gs[0, col])
        ax.plot(depths, I_ratio, "o-", color=WONG["blue"], markersize=3, linewidth=0.8)
        ax.axhline(y=1.0, color=WONG["grey"], linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Depth (Å)")
        if col == 0:
            ax.set_ylabel("I / I$_0$")
        ylo = min(float(I_ratio.min()) - 0.0002, 0.9998)
        yhi = max(float(I_ratio.max()) + 0.0002, 1.0002)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.6f"))
        ax.text(0.98, 0.92, f"max |1 $-$ I/I$_0$| = {max_dev:.2e}",
                transform=ax.transAxes, ha="right", fontsize=6.5, color=WONG["black"])
        ax.text(0.02, 0.06, f"V=0, {px_A:.3f} Å/px, {n_sl} slices",
                transform=ax.transAxes, fontsize=6, color=WONG["black"])
        if not NO_LABELS:
            ax.text(-0.14, 1.05, "ab"[col], transform=ax.transAxes, fontsize=9,
                    fontweight="bold", va="bottom", ha="left")
        ax.set_title(f"Stage {stage}", fontsize=7, color=WONG["grey"], pad=2)

    out = os.path.join(DATA_DIR, "v1a_vacuum.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
