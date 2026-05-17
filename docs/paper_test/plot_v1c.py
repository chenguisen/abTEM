#!/usr/bin/env python3
"""Plot V1c merged figure — Stage A vs B Fresnel BSC residual comparison.

Reads docs/data/v1c_bsc_residual_{A,B}.npz and generates a single merged PDF.
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

SC = {"A": {"color": WONG["blue"], "marker": "o", "label": "A (0.12 Å/px)"},
      "B": {"color": WONG["red"],  "marker": "s", "label": "B (0.05 Å/px)"}}


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.55))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8],
                          hspace=0.50, wspace=0.50,
                          left=0.08, right=0.97, top=0.90, bottom=0.18)

    # ── Panel (a): Forward I/I₀ vs depth ──
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.axhline(y=1.0, color=WONG["grey"], linestyle="--", linewidth=0.5, alpha=0.6)

    for stage in ["A", "B"]:
        base = os.path.join(DATA_DIR, f"v1c_bsc_residual_{stage}")
        data = np.load(base + ".npz")
        with open(base + ".json") as f:
            manifest = json.load(f)
        cfg = SC[stage]
        depths = data["depths"]
        fwd_ratios = data["fwd_ratios"]
        loss_pct = float(data["forward_loss_pct"])
        px_A = manifest["params"]["sampling_A_per_px"]

        ax_a.plot(depths, fwd_ratios, cfg["marker"] + "-", color=cfg["color"],
                  markersize=3, linewidth=0.8, label=f"{cfg['label']} (loss={loss_pct:.3f}%)")
        ax_a.text(0.02, 0.18 + 0.08 * (1 if stage == "B" else 0),
                  f"Stage {stage}: {px_A:.3f} Å/px",
                  transform=ax_a.transAxes, fontsize=6, color=cfg["color"])

    ax_a.set_xlabel("Depth (Å)")
    ax_a.set_ylabel("Forward I / I$_0$")
    ax_a.set_ylim(0.994, 1.0005)
    ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax_a.legend(loc="lower left", framealpha=0.9)
    if not NO_LABELS:
        ax_a.text(-0.14, 1.04, "a", transform=ax_a.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (b): BSC intensity vs depth ──
    ax_b = fig.add_subplot(gs[1, :2])

    for stage in ["A", "B"]:
        base = os.path.join(DATA_DIR, f"v1c_bsc_residual_{stage}")
        data = np.load(base + ".npz")
        cfg = SC[stage]
        depths = data["depths"]
        bsc_I = data["bsc_I"]

        ax_b.semilogy(depths, bsc_I, cfg["marker"] + "-", color=cfg["color"],
                      markersize=3, linewidth=0.8, label=cfg["label"])

    ax_b.set_xlabel("Depth (Å)  (0 = entrance surface)")
    ax_b.set_ylabel("Σ|BSC|²")
    ax_b.invert_xaxis()
    ax_b.legend(loc="upper left", framealpha=0.9)
    if not NO_LABELS:
        ax_b.text(-0.14, 1.04, "b", transform=ax_b.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (c): Energy budget bars ──
    ax_c = fig.add_subplot(gs[:, 2])
    bar_width = 0.35
    x_positions = {"A": [0.8, 1.8], "B": [1.2, 2.2]}
    for i, stage in enumerate(["A", "B"]):
        base = os.path.join(DATA_DIR, f"v1c_bsc_residual_{stage}")
        data = np.load(base + ".npz")
        cfg = SC[stage]
        I0 = float(data["I0"])
        I_fwd_exit = float(data["I_fwd_exit"])
        I_bsc_ent = float(data["I_bsc_entrance"])
        energy_bal = float(data["energy_balance"])
        xs = x_positions[stage]

        ax_c.bar([xs[0]], [I_fwd_exit / I0], width=bar_width,
                 color=cfg["color"], edgecolor="white", linewidth=0.5,
                 label=f"Stage {stage}")
        ax_c.text(xs[0], 1.004, f"{I_fwd_exit/I0:.4f}", ha="center",
                  fontsize=5.5, color=cfg["color"])
        ax_c.annotate(f"S{stage}: |T|²+|R|²−1 = {energy_bal:.2e}",
                      xy=(0.5, 0.95 - 0.07 * i), xycoords="axes fraction",
                      ha="center", fontsize=6, color=cfg["color"])

    ax_c.set_ylabel("I / I$_0$  (forward channel)")
    ax_c.set_ylim(0.994, 1.006)
    ax_c.set_xticks([1, 2])
    ax_c.set_xticklabels(["I$_0$", "I$_{fwd}$ (exit)"])
    ax_c.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax_c.axhline(y=1.0, color=WONG["grey"], linestyle=":", linewidth=0.5)
    if not NO_LABELS:
        ax_c.text(-0.28, 1.04, "c", transform=ax_c.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    out = os.path.join(DATA_DIR, "v1c_bsc_residual.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
