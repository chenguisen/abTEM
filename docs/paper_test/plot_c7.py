#!/usr/bin/env python3
"""Plot C7 phase diagram — (ρ,η) dimensionless stability map.

Reads docs/data/c7_phase.npz/.json and generates PDF + PNG.
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
    "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "lines.linewidth": 0.8, "lines.markersize": 5,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
WONG = {
    "blue": "#0072B2", "orange": "#E69F00", "red": "#D55E00",
    "green": "#009E73", "black": "#000000", "grey": "#555555",
}
FIG_W = 183 / 25.4
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

REGIME_COLORS = {"convergent": WONG["green"], "conditional": WONG["orange"],
                 "divergent": WONG["red"]}
REGIME_MARKERS = {"convergent": "o", "conditional": "s", "divergent": "X"}
MATERIAL_COLORS = {"SrTiO3": WONG["blue"], "Si": WONG["green"], "Au": WONG["red"]}


def main():
    base = os.path.join(DATA_DIR, "c7_phase")
    with open(base + ".json") as f:
        manifest = json.load(f)

    results = manifest["results"]
    # Exclude Δz > 1.0 Å (physically invalid — projection approximation fails)
    results = [r for r in results if r["dz_A"] <= 1.0]
    params = manifest["params"]

    # Group by material and voltage
    materials = sorted(set(r["material"] for r in results))
    voltages = sorted(set(r["voltage_keV"] for r in results))

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.72))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.48,
                          left=0.10, right=0.97, top=0.90, bottom=0.14)

    # ── (a) (ρ,η) scatter colored by regime ──
    ax_a = fig.add_subplot(gs[0, :])
    for regime, color in REGIME_COLORS.items():
        pts = [r for r in results if r["regime"] == regime]
        if not pts: continue
        # Plot per material with different markers
        for mat in materials:
            m_pts = [r for r in pts if r["material"] == mat]
            if not m_pts: continue
            rho = [r["rho"] for r in m_pts]
            eta = [r["eta"] for r in m_pts]
            marker = REGIME_MARKERS[regime]
            ax_a.scatter(rho, eta, c=color, marker=marker, s=30,
                        edgecolors="white", linewidth=0.3, zorder=3,
                        label=f"{mat} {regime}" if len(pts) > 0 else "")

    ax_a.set_xlabel("$\\rho = \\Delta z\\,/\\,\\ell_\\mathrm{mfp}$")
    ax_a.set_ylabel("$\\eta = r_F\\,/\\,w_\\mathrm{col}$")
    ax_a.set_xlim(0.005, 0.4)
    ax_a.set_ylim(0.002, 0.03)
    ax_a.text(-0.08, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # Add regime region labels
    ax_a.text(0.02, 0.023, "Convergent\n(Si, all E, all $\\Delta z$)",
              fontsize=5.5, color=WONG["green"], alpha=0.7)
    ax_a.text(0.15, 0.012, "Conditional (SrTiO$_3$)", fontsize=5.5,
              color=WONG["orange"], alpha=0.7)
    ax_a.text(0.6, 0.008, "Divergent (Au)", fontsize=5.5,
              color=WONG["red"], alpha=0.7)

    # ── (b) ρ vs Δz grouped by material/voltage ──
    ax_b = fig.add_subplot(gs[1, 0])
    for mat in materials:
        mat_pts = [r for r in results if r["material"] == mat]
        for V in voltages:
            v_pts = [r for r in mat_pts if r["voltage_keV"] == V]
            if not v_pts: continue
            dz_arr = np.array([r["dz_A"] for r in v_pts])
            rho_arr = np.array([r["rho"] for r in v_pts])
            regime_arr = [r["regime"] for r in v_pts]
            color = MATERIAL_COLORS[mat]
            linestyle = {"30.0": "-", "100.0": "--", "300.0": ":"}[f"{V:.1f}"]
            ax_b.plot(dz_arr, rho_arr, linestyle, color=color, linewidth=1.0,
                     marker="o", markersize=3, markerfacecolor="white",
                     alpha=0.8)
            # Mark regime
            for dz, rho, regime in zip(dz_arr, rho_arr, regime_arr):
                m = REGIME_MARKERS[regime]
                ax_b.plot(dz, rho, m, color=REGIME_COLORS[regime],
                         markersize=4, markeredgewidth=0.5, zorder=4)

    ax_b.set_xlabel("$\\Delta z$ (Å)")
    ax_b.set_ylabel("$\\rho = \\Delta z\\,/\\,\\ell_\\mathrm{mfp}$")
    ax_b.text(-0.18, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # Legend entries
    from matplotlib.lines import Line2D
    leg_mat = [Line2D([0],[0], color=MATERIAL_COLORS[m], linewidth=1.0,
                      label=m) for m in materials]
    leg_E = [Line2D([0],[0], color="black", linestyle=ls, linewidth=1.0,
                    label=f"{int(V)} keV")
             for V, ls in zip(voltages, ["-", "--", ":"])]
    ax_b.legend(handles=leg_mat + leg_E, loc="upper left", fontsize=5.5,
               framealpha=0.9, ncol=2)

    # ── (c) Estimated ρ_c boundary ──
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.axis("off")

    # Compute regime statistics
    conv_rhos = [r["rho"] for r in results if r["regime"] == "convergent"]
    div_rhos = [r["rho"] for r in results if r["regime"] == "divergent"]
    cond_rhos = [r["rho"] for r in results if r["regime"] == "conditional"]
    rho_boundary = (max(conv_rhos) + min(div_rhos)) / 2 if conv_rhos and div_rhos else float('nan')

    lines = [
        "CVDMS phase diagram at t = 200 Å",
        f"3 materials × 3 voltages × 4 Δz ≤ 1 Å",
        f"({36} valid points; Δz>1 Å excluded)",
        "",
        f"ρ_c ≈ {rho_boundary:.3f} (conv/div boundary)",
        "",
        "Regime counts:",
        f"  Convergent:   {len(conv_rhos)} (all Si)",
        f"  Conditional:  {len(cond_rhos)} (all SrTiO$_3$)",
        f"  Divergent:    {len(div_rhos)} (all Au)",
        "",
        "Key finding:",
        "Phase diagram collapses to",
        "material-Z axis. For t = 200 Å,",
        "regime is dominated by V_rms,",
        "not by Δz or E. η dynamic",
        "range too small (0.003-0.017)",
        "to cross phase boundaries.",
    ]
    for i, line in enumerate(lines):
        ax_c.text(0.0, 0.97 - i * 0.045, line, transform=ax_c.transAxes,
                 fontsize=5.5, va="top", fontfamily="monospace")

    out_pdf = os.path.join(DATA_DIR, "c7_phase.pdf")
    out_png = os.path.join(DATA_DIR, "c7_phase.png")
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
