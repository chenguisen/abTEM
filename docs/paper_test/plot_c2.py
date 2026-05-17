#!/usr/bin/env python3
"""Plot C2 thickness sweep — CVDMS vs Fourier agreement vs thickness.

Reads docs/data/c2_thickness.npz and generates PDF + PNG.
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
    "lines.linewidth": 0.8, "lines.markersize": 4,
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
    base = os.path.join(DATA_DIR, "c2_thickness")
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)

    t = data["thickness_A"]
    ncc_arr = data["ncc"]
    onem = data["one_minus_ncc"]
    phase_rms = data["phase_rms_mrad"]
    amp_err = data["amp_rms_error"]
    dp_ncc_arr = data["dp_ncc"]

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.65))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.35,
                          left=0.10, right=0.97, top=0.90, bottom=0.16)

    # ── Panel (a): 1 − NCC vs thickness (log-log) ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.loglog(t, onem, "o-", color=WONG["blue"], linewidth=1.0, markersize=5)
    # Fit power law
    log_t = np.log(t); log_e = np.log(onem)
    slope, intercept = np.polyfit(log_t, log_e, 1)
    ax_a.loglog(t, np.exp(intercept) * t**slope, "--", color=WONG["grey"],
                linewidth=0.7, alpha=0.6, label=f"∝ t$^{{{slope:.2f}}}$")
    ax_a.set_xlabel("Thickness (Å)")
    ax_a.set_ylabel("1 − NCC")
    ax_a.legend(loc="upper left", framealpha=0.9, fontsize=6)
    if not NO_LABELS:
        ax_a.text(-0.14, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (b): Phase RMS vs thickness ──
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.loglog(t, phase_rms, "s-", color=WONG["orange"], linewidth=1.0, markersize=5)
    log_p = np.log(phase_rms)
    slope_p, intercept_p = np.polyfit(log_t, log_p, 1)
    ax_b.loglog(t, np.exp(intercept_p) * t**slope_p, "--", color=WONG["grey"],
                linewidth=0.7, alpha=0.6, label=f"∝ t$^{{{slope_p:.2f}}}$")
    ax_b.set_xlabel("Thickness (Å)")
    ax_b.set_ylabel("Phase RMS error (mrad)")
    ax_b.legend(loc="upper left", framealpha=0.9, fontsize=6)
    if not NO_LABELS:
        ax_b.text(-0.14, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (c): NCC, DP NCC, and amplitude error vs t ──
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(t, ncc_arr, "o-", color=WONG["blue"], linewidth=1.0,
              label="NCC (wave function)")
    ax_c.plot(t, dp_ncc_arr, "s--", color=WONG["green"], linewidth=1.0,
              label="NCC (diffraction)")
    ax_c.set_xlabel("Thickness (Å)")
    ax_c.set_ylabel("NCC")
    ax_c.legend(loc="lower left", framealpha=0.9, fontsize=6)
    ax_c.set_ylim(0.85, 1.005)
    if not NO_LABELS:
        ax_c.text(-0.14, 1.05, "c", transform=ax_c.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    # ── Panel (d): Summary ──
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    params = manifest["params"]
    lines = [
        f"SrTiO$_3$ [001], {params['energy_eV']/1000:.0f} keV",
        f"$\\Delta z$ = {params['dz_A']} Å",
        f"$\\Delta x$ = {params['sampling_A_per_px']:.3f} Å/px",
        f"Probe: 10 mrad on Sr column",
        "",
        f"Error scaling: 1−NCC ∝ t$^{{{slope:.2f}}}$",
        f"Phase RMS ∝ t$^{{{slope_p:.2f}}}$",
        "",
        "  t (Å)      NCC      Phase RMS   DP NCC",
    ]
    for i in range(len(t)):
        lines.append(f"  {t[i]:>5.0f}    {ncc_arr[i]:.5f}    {phase_rms[i]:>6.0f} mrad  {dp_ncc_arr[i]:.5f}")
    for i, line in enumerate(lines):
        ax_d.text(0.0, 0.97 - i * 0.06, line, transform=ax_d.transAxes,
                 fontsize=5.5, va="top", fontfamily="monospace")

    out_pdf = os.path.join(DATA_DIR, "c2_thickness.pdf")
    out_png = os.path.join(DATA_DIR, "c2_thickness.png")
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
