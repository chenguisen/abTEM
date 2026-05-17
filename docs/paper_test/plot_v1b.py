#!/usr/bin/env python3
"""Plot V1b merged figure — Stage A vs B SrTiO3 unitarity comparison.

Reads docs/data/v1b_unitarity_{A,B}.npz and generates a single merged PDF
with Stage A and B overlaid for direct comparison.
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

STAGE_CFG = {
    "A": {"color": WONG["blue"], "marker": "o", "label": "Stage A (0.20 Å/px, PASS)"},
    "B": {"color": WONG["red"],  "marker": "s", "label": "Stage B (0.05 Å/px, FAIL)"},
}


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.38))
    gs = fig.add_gridspec(1, 2, wspace=0.45, left=0.09, right=0.97,
                          top=0.88, bottom=0.22)

    # ── Panel (a): I/I₀ vs depth, both stages overlaid ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axhline(y=1.0, color=WONG["grey"], linestyle="--", linewidth=0.5, alpha=0.6)

    for stage in ["A", "B"]:
        base = os.path.join(DATA_DIR, f"v1b_unitarity_{stage}")
        data = np.load(base + ".npz")
        with open(base + ".json") as f:
            manifest = json.load(f)
        cfg = STAGE_CFG[stage]

        depths = data["depths"]
        I_ratio = data["I_ratio"]
        max_dev = float(data["max_deviation"])
        px_A = manifest["params"]["sampling_A_per_px"]

        ax_a.plot(depths, I_ratio, cfg["marker"] + "-", color=cfg["color"],
                  markersize=3, linewidth=0.8, label=cfg["label"])
        ax_a.text(0.98, 0.12 if stage == "B" else 0.22,
                  f"{stage}: max |1 $-$ I/I$_0$| = {max_dev:.2e}",
                  transform=ax_a.transAxes, ha="right", fontsize=6,
                  color=cfg["color"])

    ax_a.set_xlabel("Depth (Å)")
    ax_a.set_ylabel("I / I$_0$")
    ax_a.legend(loc="lower left", framealpha=0.9)
    if not NO_LABELS:
        ax_a.text(-0.14, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (b): |1 - I/I₀| vs depth, both stages overlaid ──
    ax_b = fig.add_subplot(gs[0, 1])

    for stage in ["A", "B"]:
        base = os.path.join(DATA_DIR, f"v1b_unitarity_{stage}")
        data = np.load(base + ".npz")
        cfg = STAGE_CFG[stage]
        depths = data["depths"]
        I_ratio = data["I_ratio"]
        deviation = np.abs(1.0 - np.array(I_ratio))
        ax_b.semilogy(depths, deviation, cfg["marker"] + "-", color=cfg["color"],
                      markersize=3, linewidth=0.8, label=cfg["label"])

    ax_b.axhline(y=1e-6, color=WONG["grey"], linestyle=":", linewidth=0.5, alpha=0.5)
    ax_b.set_xlabel("Depth (Å)")
    ax_b.set_ylabel("|1 $-$ I/I$_0$|")
    ax_b.legend(loc="lower right", framealpha=0.9)
    if not NO_LABELS:
        ax_b.text(-0.14, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    out = os.path.join(DATA_DIR, "v1b_unitarity.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
