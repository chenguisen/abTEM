#!/usr/bin/env python3
"""Plot V3 figures from saved NPZ data.

Reads docs/data/v3_slice_{stage}.npz and generates the PDF figure
showing pairwise NCC convergence and 1−NCC vs Δz with O(Δz²) reference.

Usage:
  python plot_v3_slice.py A
  python plot_v3_slice.py B
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
    "axes.titlesize": 9,
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


def plot_v3(stage):
    base = os.path.join(DATA_DIR, f"v3_slice_{stage}")
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)

    dz          = data["dz"]
    ncc_pw      = data["ncc_pairwise"]       # pairwise NCC
    ncc_vf      = data["ncc_vs_finest"]      # vs finest Δz
    phase_rms   = data["phase_rms"]
    dz_finest   = float(data["dz_finest"])

    params      = manifest["params"]
    px_A        = params["sampling_A_per_px"]
    thick_A     = params["thickness_A"]

    # ── Figure ──
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1],
                          wspace=0.50, left=0.10, right=0.97, top=0.88, bottom=0.22)

    # ── Panel (a): NCC (pairwise & vs finest) vs Δz ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(dz, ncc_pw, "o-", color=WONG["blue"], markersize=4, linewidth=0.8,
              label="consecutive Δz")
    ax_a.plot(dz, ncc_vf, "s--", color=WONG["orange"], markersize=3, linewidth=0.8,
              label="vs Δz={:.3f} Å".format(dz_finest))
    ax_a.axhline(y=1.0, color=WONG["grey"], linestyle=":", linewidth=0.5, alpha=0.5)
    ax_a.legend(loc="lower right")
    ax_a.set_xlabel("Coarser Δz (Å)")
    ax_a.set_ylabel("NCC")
    ax_a.invert_xaxis()
    if not NO_LABELS:
        ax_a.text(-0.14, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")
    ax_a.text(0.02, 0.06, f"SriTiO$_3$, {thick_A:.0f} Å, {px_A:.3f} Å/px",
              transform=ax_a.transAxes, fontsize=6, color=WONG["black"])

    # ── Panel (b): 1−NCC vs Δz (log-log, O(Δz²) reference) ──
    ax_b = fig.add_subplot(gs[0, 1])
    one_minus_ncc = np.array([max(1e-16, 1.0 - n) for n in ncc_vf])
    ax_b.loglog(dz, one_minus_ncc, "o-", color=WONG["blue"],
                markersize=4, linewidth=0.8, label="1 − NCC")
    # O(Δz²) reference: anchor at finest pair
    if len(dz) >= 2:
        ref = one_minus_ncc[-1] * (dz / dz[-1])**2
        ax_b.loglog(dz, ref, "--", color=WONG["grey"], linewidth=0.6,
                    label="O(Δz²)")
    ax_b.legend(loc="lower right")
    ax_b.set_xlabel("Coarser Δz (Å)")
    ax_b.set_ylabel("1 − NCC")
    ax_b.invert_xaxis()
    if not NO_LABELS:
        ax_b.text(-0.14, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")
    ax_b.text(0.02, 0.06, f"ref: Δz={dz_finest:.3f} Å",
              transform=ax_b.transAxes, fontsize=6, color=WONG["black"])
    ax_b.text(0.02, 0.94, f"min NCC$_{{pw}}$ = {min(ncc_pw):.6f}",
              transform=ax_b.transAxes, fontsize=6, color=WONG["black"])

    out = os.path.join(DATA_DIR, f"v3_slice_{stage}.pdf")
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
            print(f"Unknown stage: {s}, skipping")
            continue
        plot_v3(s_upper)
