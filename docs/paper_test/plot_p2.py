#!/usr/bin/env python3
"""Plot P2 channeling Pendellösung — CVDMS vs Fourier Δz sweep.

Reads docs/data/p2_channeling.npz and generates PDF + PNG.
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
    base = os.path.join(DATA_DIR, "p2_channeling")
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)

    dz_sweep = data["dz_sweep"]
    xi_ch = float(data["xi_ch_analytic"])
    V_g = float(data["V_g"])

    # Load all Δz results
    all_z, all_Ic, all_If = {}, {}, {}
    xi_c, xi_f = {}, {}
    for dz in dz_sweep:
        key = f"{dz}".replace(".", "_")  # match save format
        all_z[dz] = data[f"z_dz{dz}"]
        all_Ic[dz] = data[f"I_c_dz{dz}"]
        all_If[dz] = data[f"I_f_dz{dz}"]
        xi_c[dz] = float(data[f"xi_c_dz{dz}"])
        xi_f[dz] = float(data[f"xi_f_dz{dz}"])

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.70))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.52,
                          left=0.13, right=0.97, top=0.90, bottom=0.18)

    dz_colors = {0.2: WONG["blue"], 0.4: WONG["orange"],
                 0.8: WONG["red"], 1.0: WONG["green"]}

    # ── Panel (a): On-column |ψ|² vs depth — CVDMS for all Δz ──
    ax_a = fig.add_subplot(gs[0, 0])
    for dz in sorted(dz_sweep):
        z = all_z[dz]; Ic = all_Ic[dz]
        ax_a.plot(z, Ic * 1e6, "-", color=dz_colors[dz],
                  linewidth=0.7, label=f"CVDMS Δz={dz:.1f} Å")
    ax_a.set_xlabel("Depth (Å)")
    ax_a.set_ylabel(r"On-column $|\psi|^2$ ($\times 10^{-6}$)")
    ax_a.legend(loc="upper right", framealpha=0.9, fontsize=5.5)
    if not NO_LABELS:
        ax_a.text(-0.14, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (b): Detrended + ACF for the most divergent Δz (0.8 Å) ──
    ax_b = fig.add_subplot(gs[0, 1])
    dz_div = 0.8
    z_div = all_z[dz_div]
    Ic_div = all_Ic[dz_div]
    If_div = all_If[dz_div]

    # Detrend
    for I, label, color in [(Ic_div, "CVDMS", WONG["blue"]),
                              (If_div, "Fourier", WONG["orange"])]:
        trend = np.polyfit(z_div, I, 1)
        det = I - np.polyval(trend, z_div)
        det -= det.mean()
        dz_eff = z_div[1] - z_div[0]
        acf = np.correlate(det, det, mode="full")
        acf = acf[len(acf)//2:]
        acf_norm = acf / acf[0]
        lags = np.arange(len(acf)) * dz_eff
        ax_b.plot(lags, acf_norm, "-" if "CVDMS" in label else "--",
                  color=color, linewidth=0.8,
                  label=f"{label}  ξ={xi_c[dz_div] if 'CVDMS' in label else xi_f[dz_div]:.1f} Å")
    ax_b.axvline(xi_ch, color=WONG["grey"], linestyle="--",
                 linewidth=0.7, alpha=0.6, label=f"Bulk ξ={xi_ch:.0f} Å")
    ax_b.set_xlim(0, 160)
    ax_b.set_xlabel("Lag (Å)")
    ax_b.set_ylabel("Normalized ACF")
    ax_b.legend(loc="lower left", framealpha=0.9, fontsize=6)
    ax_b.set_title(f"Δz = {dz_div:.1f} Å  (Δξ = {abs(xi_c[dz_div]-xi_f[dz_div]):.1f} Å)", fontsize=7)
    if not NO_LABELS:
        ax_b.text(-0.14, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (c): ξ vs Δz — key commutator sensitivity plot ──
    ax_c = fig.add_subplot(gs[1, 0])
    dz_vals = sorted(dz_sweep)
    xi_c_arr = [xi_c[dz] for dz in dz_vals]
    xi_f_arr = [xi_f[dz] for dz in dz_vals]
    ax_c.plot(dz_vals, xi_c_arr, "o-", color=WONG["blue"],
              linewidth=1.0, markersize=4, label="CVDMS")
    ax_c.plot(dz_vals, xi_f_arr, "s--", color=WONG["orange"],
              linewidth=1.0, markersize=4, label="Fourier")
    ax_c.axhline(xi_ch, color=WONG["grey"], linestyle=":",
                 linewidth=0.8, label=f"Bulk ξ={xi_ch:.0f} Å")
    ax_c.fill_between(dz_vals, xi_c_arr, xi_f_arr, alpha=0.15, color=WONG["red"])
    ax_c.set_xlabel("Slice thickness Δz (Å)")
    ax_c.set_ylabel("Pendellösung period ξ (Å)", labelpad=2)
    ax_c.legend(loc="lower left", framealpha=0.9, fontsize=6)
    if not NO_LABELS:
        ax_c.text(-0.12, 1.06, "c", transform=ax_c.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (d): Parameter summary ──
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    params = manifest["params"]
    lines = [
        f"SrTiO$_3$ [001], {params['energy_eV']/1000:.0f} keV",
        f"Probe: 10 mrad on Sr column",
        f"$\\Delta x$ = {params['sampling_A_per_px']:.3f} Å/px",
        f"Thickness: {params['thickness_A']:.0f} Å",
        "",
        f"$V_g$ (bulk) = {V_g:.1f} eV·Å",
        f"$\\xi_g$ (bulk) = {xi_ch:.0f} Å",
        "",
        "  Δz      ξ(CVDMS)   ξ(Fourier)   Δξ",
    ]
    for dz in dz_vals:
        dxi = abs(xi_c[dz] - xi_f[dz])
        lines.append(f"  {dz:.1f} Å     {xi_c[dz]:.1f} Å      {xi_f[dz]:.1f} Å       {dxi:.1f} Å")
    for i, line in enumerate(lines):
        ax_d.text(0.0, 0.95 - i * 0.075, line, transform=ax_d.transAxes,
                 fontsize=6, va="top", fontfamily="monospace")

    out_pdf = os.path.join(DATA_DIR, "p2_channeling.pdf")
    out_png = os.path.join(DATA_DIR, "p2_channeling.png")
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
