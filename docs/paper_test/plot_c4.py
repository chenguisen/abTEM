#!/usr/bin/env python3
"""Plot C4 antialiasing comparison — bandwidth explosion negative control.

Reads docs/data/c4_antialias.npz and generates PDF + PNG.
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


def main():
    base = os.path.join(DATA_DIR, "c4_antialias")
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)

    psi_aa = data["psi_aa"]
    psi_ref = data["psi_ref"]
    amp_aa = data["amp_aa"]
    amp_ref = data["amp_ref"]
    phase_err = data["phase_err_aa"]
    r_freq = data["r_freq"]
    pwr_aa = data["pwr_aa"]
    pwr_ref = data["pwr_ref"]
    overflow = bool(manifest["results"]["aa_off_overflow"])

    res = manifest["results"]
    params = manifest["params"]

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.70))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.52,
                          left=0.08, right=0.98, top=0.92, bottom=0.14)

    # ── (a) Exit wave amplitude — reference ──
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(amp_ref, cmap="inferno", origin="lower",
                        extent=[0, amp_ref.shape[1]*params["sampling_A_per_px"],
                                0, amp_ref.shape[0]*params["sampling_A_per_px"]])
    ax_a.set_xlabel("x (Å)"); ax_a.set_ylabel("y (Å)")
    plt.colorbar(im_a, ax=ax_a, fraction=0.046)
    ax_a.set_title(f"Ref: $\\Delta z$=0.2 Å + AA", fontsize=7)
    ax_a.text(-0.18, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── (b) Exit wave amplitude — AA=ON ──
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(amp_aa, cmap="inferno", origin="lower",
                        extent=[0, amp_aa.shape[1]*params["sampling_A_per_px"],
                                0, amp_aa.shape[0]*params["sampling_A_per_px"]])
    ax_b.set_xlabel("x (Å)"); ax_b.set_ylabel("y (Å)")
    plt.colorbar(im_b, ax=ax_b, fraction=0.046)
    ax_b.set_title(f"AA=ON: $\\Delta z$=0.4 Å", fontsize=7)
    ax_b.text(-0.18, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── (c) Phase error AA=ON vs ref ──
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.imshow(phase_err * 1000, cmap="RdBu_r", origin="lower",
                        extent=[0, phase_err.shape[1]*params["sampling_A_per_px"],
                                0, phase_err.shape[0]*params["sampling_A_per_px"]],
                        vmin=-800, vmax=800)
    ax_c.set_xlabel("x (Å)"); ax_c.set_ylabel("y (Å)")
    plt.colorbar(im_c, ax=ax_c, fraction=0.046, label="mrad")
    ax_c.set_title(f"Phase err AA=ON (RMS={res['phase_rms_aa_mrad']:.0f} mrad)", fontsize=7)
    ax_c.text(-0.18, 1.05, "c", transform=ax_c.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── (d) Radial power spectra ──
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.semilogy(r_freq, pwr_ref + 1e-30, "-", color=WONG["grey"], linewidth=1.0,
                  label="Ref ($\\Delta z$=0.2, AA=ON)")
    ax_d.semilogy(r_freq, pwr_aa + 1e-30, "--", color=WONG["blue"], linewidth=1.0,
                  label="AA=ON ($\\Delta z$=0.4)")
    nyq = psi_aa.shape[0] // 2
    ax_d.axvline(2/3 * nyq, color=WONG["red"], linewidth=0.5, linestyle=":",
                 alpha=0.6)
    ax_d.text(2/3 * nyq * 1.02, ax_d.get_ylim()[1]*0.5, "$\\frac{2}{3}f_\\mathrm{Nyq}$",
              color=WONG["red"], fontsize=5.5)
    ax_d.set_xlabel("Spatial frequency (px)"); ax_d.set_ylabel("Power")
    ax_d.legend(loc="lower left", framealpha=0.9, fontsize=5.5)
    ax_d.text(-0.18, 1.05, "d", transform=ax_d.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── (e) Metrics bar chart ──
    ax_e = fig.add_subplot(gs[1, 1])
    metrics = ["NCC", "1$-$NCC", "Phase RMS\n(mrad)", "Amp err"]
    aa_vals = [res["ncc_aa_vs_ref"], 1 - res["ncc_aa_vs_ref"],
               res["phase_rms_aa_mrad"], res["amp_rms_aa"]]
    x = np.arange(len(metrics))
    bars = ax_e.bar(x, aa_vals, 0.4, color=WONG["blue"], label="AA=ON")
    ax_e.set_xticks(x); ax_e.set_xticklabels(metrics, fontsize=6)
    # Annotate values
    for bar, val in zip(bars, aa_vals):
        ypos = bar.get_height() + max(aa_vals) * 0.02
        ax_e.text(bar.get_x() + bar.get_width()/2, ypos,
                  f"{val:.4f}" if val < 10 else f"{val:.0f}",
                  ha="center", va="bottom", fontsize=5.5)
    ax_e.set_ylabel("Value")
    ax_e.text(-0.18, 1.05, "e", transform=ax_e.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── (f) Summary ──
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    lines = [
        f"SrTiO$_3$ [001], {params['energy_eV']/1000:.0f} keV",
        f"$\\Delta x$ = {params['sampling_A_per_px']:.3f} Å/px",
        f"t = {params['thickness_A']:.0f} Å",
        f"Probe: 10 mrad on Sr column",
        "",
        f"NCC vs ref: {res['ncc_aa_vs_ref']:.6f}",
        f"Phase RMS vs ref: {res['phase_rms_aa_mrad']:.0f} mrad",
        f"AA-off overflow: {overflow}",
        "",
        "Conclusion:",
        "Antialiasing is required",
        "at $\\Delta x$ = 0.05 Å/px to",
        "prevent bandwidth explosion.",
    ]
    for i, line in enumerate(lines):
        ax_f.text(0.0, 0.97 - i * 0.065, line, transform=ax_f.transAxes,
                 fontsize=5.5, va="top", fontfamily="monospace")

    out_pdf = os.path.join(DATA_DIR, "c4_antialias.pdf")
    out_png = os.path.join(DATA_DIR, "c4_antialias.png")
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
