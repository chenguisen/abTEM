#!/usr/bin/env python3
"""Plot P4 diagnostic-potential spatial correlation and IPR.

Reads docs/data/p4_correlation.npz and generates a single merged PDF
plus a PNG for HTML embedding.
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def main():
    base = os.path.join(DATA_DIR, "p4_correlation")
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)

    V_2d = data["V_2d"]
    grad_mag = data["grad_mag"]
    I_bsc = data["I_bsc"]
    bsc_residual = data["bsc_residual"]
    z_IPR = data["z_IPR"]
    IPR_bsc = data["IPR_bsc"]
    IPR_fwd = data["IPR_fwd"]
    px_A = float(data["px_A"])

    results = manifest["results"]
    params = manifest["params"]

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.72))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9],
                          hspace=0.55, wspace=0.52,
                          left=0.07, right=0.97, top=0.90, bottom=0.16)

    ny, nx = V_2d.shape
    extent = [0, nx * px_A, 0, ny * px_A]

    # ── Panel (a): V(R) — projected potential ──
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(V_2d, extent=extent, cmap="inferno", aspect="equal")
    ax_a.set_xlabel("x (Å)")
    ax_a.set_ylabel("y (Å)")
    cb_a = plt.colorbar(im_a, ax=ax_a, label="V (eV·Å)", shrink=0.82)
    ax_a.text(-0.22, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Panel (b): |∇V| — commutator driver ──
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(grad_mag, extent=extent, cmap="inferno", aspect="equal")
    ax_b.set_xlabel("x (Å)")
    ax_b.set_ylabel("y (Å)")
    cb_b = plt.colorbar(im_b, ax=ax_b, label="|∇V| (eV)", shrink=0.82)
    ax_b.text(-0.22, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Panel (c): V vs |∇V| scatter — commutator localisation proof ──
    ax_c = fig.add_subplot(gs[0, 2])
    step = 4
    v_sub = V_2d[::step, ::step].ravel()
    g_sub = grad_mag[::step, ::step].ravel()
    ax_c.loglog(v_sub[v_sub > 0], g_sub[v_sub > 0], ".", color=WONG["blue"],
                markersize=1, alpha=0.4)
    r_V_grad = results["pearson_r_V_grad"]
    ax_c.text(0.95, 0.05, f"r = {r_V_grad:.4f}", transform=ax_c.transAxes,
             ha="right", fontsize=7, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax_c.set_xlabel("V (eV·Å)")
    ax_c.set_ylabel("|∇V| (eV)")
    ax_c.text(-0.28, 1.05, "c", transform=ax_c.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Panel (d): BSC residual — null test ──
    ax_d = fig.add_subplot(gs[1, 0])
    im_d = ax_d.imshow(np.log10(bsc_residual + 1e-40), extent=extent,
                       cmap="inferno", aspect="equal")
    ax_d.set_xlabel("x (Å)")
    ax_d.set_ylabel("y (Å)")
    plt.colorbar(im_d, ax=ax_d, label=r"log$_{10}$ |ΔI|", shrink=0.82)
    ax_d.text(-0.22, 1.05, "d", transform=ax_d.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Panel (e): |∇V| vs BSC residual scatter ──
    ax_e = fig.add_subplot(gs[1, 1])
    b_sub = bsc_residual[::step, ::step].ravel()
    mask = b_sub > 0
    ax_e.loglog(g_sub[mask], b_sub[mask], ".", color=WONG["orange"],
                markersize=1, alpha=0.4)
    r_grad_bsc = results["pearson_r_grad_bsc"]
    ax_e.text(0.95, 0.05, f"r = {r_grad_bsc:.4f}", transform=ax_e.transAxes,
             ha="right", fontsize=7, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax_e.set_xlabel("|∇V| (eV)")
    ax_e.set_ylabel("BSC residual")
    ax_e.text(-0.28, 1.05, "e", transform=ax_e.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Panel (f): IPR vs depth ──
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.plot(z_IPR, IPR_bsc, "-", color=WONG["blue"],
              linewidth=0.8, label="CVDMS + BSC")
    ax_f.plot(z_IPR, IPR_fwd, "--", color=WONG["orange"],
              linewidth=0.8, label="CVDMS forward")
    ax_f.axhline(1.0, color=WONG["grey"], linestyle=":", linewidth=0.5, alpha=0.5)
    ax_f.set_xlabel("Depth (Å)")
    ax_f.set_ylabel("IPR × area")
    ax_f.legend(loc="upper right", framealpha=0.9, fontsize=6)
    ax_f.text(-0.28, 1.05, "f", transform=ax_f.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    out_pdf = os.path.join(DATA_DIR, "p4_correlation.pdf")
    fig.savefig(out_pdf, dpi=300)
    out_png = os.path.join(DATA_DIR, "p4_correlation.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
