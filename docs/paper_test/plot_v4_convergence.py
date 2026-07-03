#!/usr/bin/env python3
"""Plot V4 figures from saved NPZ data.

Reads docs/data/v4_convergence_{stage}.npz and generates the PDF figure
showing ΔI/I₀ and NCC vs convergence threshold ε.

Usage:
  python plot_v4_convergence.py A
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
    "font.size": 7,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "lines.linewidth": 0.8,
    "lines.markersize": 3,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
WONG = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "red":    "#D55E00",
    "green":  "#009E73",
    "black":  "#000000",
    "grey":   "#555555",
}
FIG_W = 183 / 25.4
NO_LABELS = "--no-labels" in sys.argv

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def plot_v4(stage):
    base = os.path.join(DATA_DIR, f"v4_convergence_{stage}")
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)

    eps        = data["eps"]
    delta_I    = data["delta_I"]
    ncc_vals   = data["ncc"]
    eps_ref    = float(data["eps_ref"])
    I_ref      = float(data["I_ref"])

    params     = manifest["params"]
    px_A       = params["sampling_A_per_px"]

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1],
                          wspace=0.50, left=0.08, right=0.97, top=0.88, bottom=0.22)

    # ── Panel (a): ΔI/I₀ vs ε ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.loglog(eps, delta_I, "o-", color=WONG["blue"], markersize=4, linewidth=0.8)
    ax_a.axvline(x=1e-7, color=WONG["grey"], linestyle=":", linewidth=0.5, alpha=0.5)
    ax_a.axhline(y=2e-4, color=WONG["red"], linestyle="--", linewidth=0.5, alpha=0.5,
                 label="2×10⁻⁴ criterion")
    ax_a.legend(loc="lower left")
    ax_a.set_xlabel("Convergence threshold ε")
    ax_a.set_ylabel("|I/I₀ − I$_\\mathregular{ref}$| / I$_\\mathregular{ref}$")
    ax_a.invert_xaxis()
    if not NO_LABELS:
        ax_a.text(-0.14, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")
    ax_a.text(0.98, 0.06, f"I$_\\mathregular{{ref}}$ = {I_ref:.8f}",
              transform=ax_a.transAxes, ha="right", fontsize=6, color=WONG["black"])
    ax_a.text(0.02, 0.06, f"SriTiO$_3$, {px_A:.3f} Å/px",
              transform=ax_a.transAxes, fontsize=6, color=WONG["black"])

    # ── Panel (b): 1−NCC vs ε ──
    ax_b = fig.add_subplot(gs[0, 1])
    one_minus_ncc = np.array([max(1e-16, 1.0 - n) for n in ncc_vals])
    ax_b.loglog(eps, one_minus_ncc, "s-", color=WONG["orange"], markersize=4, linewidth=0.8)
    ax_b.axvline(x=1e-7, color=WONG["grey"], linestyle=":", linewidth=0.5, alpha=0.5)
    ax_b.set_xlabel("Convergence threshold ε")
    ax_b.set_ylabel("1 − NCC")
    ax_b.invert_xaxis()
    if not NO_LABELS:
        ax_b.text(-0.14, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")
    ax_b.text(0.02, 0.06, f"NCC vs ε={eps_ref:.0e}",
              transform=ax_b.transAxes, fontsize=6, color=WONG["black"])

    out = os.path.join(DATA_DIR, f"v4_convergence_{stage}.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    stages = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not stages:
        stages = ["A"]
    for s in stages:
        s_upper = s.upper()
        if s_upper not in ("A", "B"):
            continue
        plot_v4(s_upper)
