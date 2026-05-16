#!/usr/bin/env python3
"""Plot V1c figures from saved NPZ data — no simulation re-run needed.

Reads docs/data/v1c_bsc_residual_{stage}.npz and generates the corresponding
PDF figure.  Iterate on figure appearance without re-running the simulation.

Usage:
  python plot_v1c.py A        # plot Stage A only
  python plot_v1c.py B        # plot Stage B only
  python plot_v1c.py A B     # plot both
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Comms Phys figure compliance (§1.4) ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "lines.linewidth": 0.8,
    "lines.markersize": 4,
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
FIG_W = 183 / 25.4  # mm → inches (double column)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def plot_v1c(stage):
    """Read NPZ and generate PDF figure for the given stage."""
    base = os.path.join(DATA_DIR, f"v1c_bsc_residual_{stage}")
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)

    fwd_I       = data["fwd_I"]
    fwd_ratios  = data["fwd_ratios"]
    bsc_I       = data["bsc_I"]
    depths      = data["depths"]
    I0          = float(data["I0"])
    I_fwd_exit  = float(data["I_fwd_exit"])
    I_bsc_ent   = float(data["I_bsc_entrance"])
    energy_bal  = float(data["energy_balance"])
    loss_pct    = float(data["forward_loss_pct"])

    params    = manifest["params"]
    px_A      = params["sampling_A_per_px"]
    thick_A   = params["thickness_A"]
    n_ep      = params["num_exit_planes"]

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.55))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8],
                          hspace=0.50, wspace=0.50,
                          left=0.08, right=0.97, top=0.94, bottom=0.18)

    # ── Panel (a): Forward I/I0 vs depth ──
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.plot(depths, fwd_ratios, "o-", color=WONG["blue"],
              markersize=3, linewidth=0.8)
    ax_a.axhline(y=1.0, color=WONG["grey"], linestyle="--",
                 linewidth=0.5, alpha=0.6)
    ax_a.set_xlabel("Depth (Å)")
    ax_a.set_ylabel("Forward I / I$_0$")
    ax_a.set_ylim(min(fwd_ratios) - 0.0002, 1.0002)
    ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    # Annotations — use textcolor, placed outside data area
    ax_a.text(0.98, 0.18, f"total loss = {loss_pct:.3f}%", transform=ax_a.transAxes,
              ha="right", fontsize=6.5, color=WONG["black"])
    ax_a.text(0.02, 0.18, f"sampling = {px_A:.3f} Å/px",
              transform=ax_a.transAxes, fontsize=6, color=WONG["black"])
    # Panel label
    ax_a.text(-0.14, 1.04, "a", transform=ax_a.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Panel (b): BSC intensity vs depth ──
    ax_b = fig.add_subplot(gs[1, :2])
    ax_b.semilogy(depths, bsc_I, "s-", color=WONG["red"],
                  markersize=3, linewidth=0.8)
    ax_b.set_xlabel("Depth (Å)  (0 = entrance surface)")
    ax_b.set_ylabel("Σ|BSC|²")
    ax_b.invert_xaxis()
    ax_b.text(0.02, 0.92, f"entrance Σ|BSC|² = {bsc_I[0]:.2e}",
              transform=ax_b.transAxes, fontsize=6.5, color=WONG["black"])
    ax_b.text(0.02, 0.84, f"bottom = {bsc_I[-1]:.1e}",
              transform=ax_b.transAxes, fontsize=6.5, color=WONG["black"])
    ax_b.text(0.98, 0.92, f"{px_A:.3f} Å/px, {n_ep} exit planes",
              transform=ax_b.transAxes, ha="right", fontsize=6, color=WONG["black"])
    # Panel label
    ax_b.text(-0.14, 1.04, "b", transform=ax_b.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Panel (c): Energy budget bar ──
    ax_c = fig.add_subplot(gs[:, 2])
    ax_c.bar(["I$_0$", "I$_{fwd}$", "Σ|BSC|²\n× 10³"],
             [I0, I_fwd_exit, I_bsc_ent * 1e3],
             color=[WONG["grey"], WONG["blue"], WONG["red"]],
             width=0.55, edgecolor="white", linewidth=0.5)
    ax_c.set_ylabel("Integrated intensity")
    ax_c.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # Energy balance annotation — inside bar area
    ax_c.text(0.5, 0.92,
              f"|I$_{{fwd}}$ + I$_{{bsc}}$ − I$_0$| / I$_0$\n= {energy_bal:.2e}",
              transform=ax_c.transAxes, ha="center", fontsize=6.5,
              color=WONG["black"],
              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                        ec=WONG["grey"], alpha=0.8))
    ax_c.text(0.5, 0.04, f"{thick_A:.0f} Å", transform=ax_c.transAxes,
              ha="center", fontsize=6.5, color=WONG["black"])
    # Panel label
    ax_c.text(-0.28, 1.04, "c", transform=ax_c.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    out = os.path.join(DATA_DIR, f"v1c_bsc_residual_{stage}.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    stages = sys.argv[1:] if len(sys.argv) > 1 else ["A", "B"]
    for s in stages:
        s_upper = s.upper()
        if s_upper not in ("A", "B"):
            print(f"Unknown stage: {s}, skipping")
            continue
        plot_v1c(s_upper)
