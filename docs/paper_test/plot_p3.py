#!/usr/bin/env python3
"""Plot P3 HOLZ FOLZ ring Δz sweep — CVDMS vs Fourier with convergent probe.

Reads docs/data/p3_holz.npz and generates PDF + PNG.
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
NO_LABELS = "--no-labels" in sys.argv


def main():
    base = os.path.join(DATA_DIR, "p3_holz")
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)

    dz_sweep = manifest["params"]["dz_sweep"]
    g_folz = float(data["g_folz_analytic"])
    dq = float(data["dq_A_per_px"])
    params = manifest["params"]

    sweep = manifest["sweep"]

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.58))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.50,
                          left=0.10, right=0.97, top=0.90, bottom=0.18)

    dz_colors = {0.4: WONG["blue"], 0.8: WONG["red"], 1.0: WONG["green"]}

    # ── Panel (a): Diffraction pattern (CVDMS, Δz=0.8 Å) ──
    ax_a = fig.add_subplot(gs[0, 0])
    dp = data["dp_c_dz0.8"] if "dp_c_dz0.8" in data else data["dp_cvdms"]
    ny, nx = dp.shape
    extent = [-nx//2 * dq, nx//2 * dq, -ny//2 * dq, ny//2 * dq]
    im = ax_a.imshow(np.log10(dp + 1e-30), extent=extent,
                     cmap="inferno", aspect="equal")
    ax_a.set_xlim(-7, 7); ax_a.set_ylim(-7, 7)
    ax_a.set_xlabel(r"$q_x$ ($\mathrm{\AA}^{-1}$)")
    ax_a.set_ylabel(r"$q_y$ ($\mathrm{\AA}^{-1}$)")
    if not NO_LABELS:
        ax_a.text(-0.14, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (b): Radial profiles for all Δz ──
    ax_b = fig.add_subplot(gs[0, 1])
    for dz in dz_sweep:
        q = data[f"q_axis_dz{dz}"]
        rc = data[f"radial_c_dz{dz}"]
        rf = data[f"radial_f_dz{dz}"]
        ax_b.semilogy(q, rc, "-", color=dz_colors[dz], linewidth=0.7,
                      label=f"CVDMS Δz={dz:.1f}")
        ax_b.semilogy(q, rf, "--", color=dz_colors[dz], linewidth=0.7,
                      label=f"Fourier Δz={dz:.1f}")
    ax_b.axvline(g_folz, color=WONG["grey"], linestyle=":", linewidth=0.7,
                 alpha=0.6, label=f"Analytic {g_folz:.3f}")
    ax_b.set_xlim(0, 8)
    ax_b.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax_b.set_ylabel("Radial intensity")
    ax_b.legend(loc="upper right", framealpha=0.9, fontsize=5.5, ncol=2)
    if not NO_LABELS:
        ax_b.text(-0.14, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (c): FOLZ contrast vs Δz — key commutator sensitivity ──
    ax_c = fig.add_subplot(gs[1, 0])
    dz_vals = sorted(dz_sweep)
    contrast_c = [sweep[f"dz_{dz}"]["contrast_cvdms"] for dz in dz_vals]
    contrast_f = [sweep[f"dz_{dz}"]["contrast_fourier"] for dz in dz_vals]
    ratio_cf = [c/f for c, f in zip(contrast_c, contrast_f)]

    ax_c.bar(np.array(dz_vals) - 0.05, contrast_c, 0.08, color=WONG["blue"],
             label="CVDMS")
    ax_c.bar(np.array(dz_vals) + 0.05, contrast_f, 0.08, color=WONG["orange"],
             label="Fourier")
    ax_c.set_xlabel("Slice thickness Δz (Å)")
    ax_c.set_ylabel("FOLZ ring contrast (peak/bg)")
    ax_c.legend(loc="upper right", framealpha=0.9, fontsize=6)

    # Add ratio labels
    for i, (dz, r) in enumerate(zip(dz_vals, ratio_cf)):
        ax_c.text(dz, max(contrast_c[i], contrast_f[i]) * 1.05,
                  f"×{r:.1f}", ha="center", fontsize=6, color=WONG["red"],
                  fontweight="bold")
    if not NO_LABELS:
        ax_c.text(-0.14, 1.05, "c", transform=ax_c.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (d): Summary ──
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    lines = [
        f"Si [001], {params['energy_eV']/1000:.0f} keV",
        f"Convergent probe: {params['probe_semiangle_mrad']:.0f} mrad",
        f"$\\Delta x$ = {params['sampling_A_per_px']:.3f} Å/px",
        f"t = {params['thickness_A']:.0f} Å",
        "",
        f"$g_{{\\rm FOLZ}}$ (analytic) = {g_folz:.3f} Å$^{{-1}}$",
        "",
        "  Δz      g(CVDMS)   g(Fourier)   contrast ratio",
    ]
    for dz in dz_vals:
        r = sweep[f"dz_{dz}"]
        cr = r["contrast_cvdms"] / r["contrast_fourier"]
        lines.append(f"  {dz:.1f} Å    {r['g_folz_cvdms']:.3f}      {r['g_folz_fourier']:.3f}        ×{cr:.2f}")
    lines += [
        "",
        f"DP NCC (C−F): {sweep['dz_0.4']['dp_ncc']:.6f} – {sweep['dz_1.0']['dp_ncc']:.6f}",
    ]
    for i, line in enumerate(lines):
        ax_d.text(0.0, 0.95 - i * 0.07, line, transform=ax_d.transAxes,
                 fontsize=6, va="top", fontfamily="monospace")

    out_pdf = os.path.join(DATA_DIR, "p3_holz.pdf")
    out_png = os.path.join(DATA_DIR, "p3_holz.png")
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
