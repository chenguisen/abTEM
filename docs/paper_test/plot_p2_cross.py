#!/usr/bin/env python3
"""Plot P2 cross-material channeling — Si, SrTiO3, Au Δz sweep comparison.

Reads docs/data/p2_{si,srtio3,au}.npz or p2_channeling.npz (for SrTiO3).
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

MATERIALS = [
    {"name": "Si", "label": "Si [001] 100 keV", "color": WONG["green"],
     "npz": "p2_si.npz", "json": "p2_si.json", "Z": 14},
    {"name": "SrTiO3", "label": "SrTiO$_3$ [001] 30 keV", "color": WONG["blue"],
     "npz": "p2_channeling.npz", "json": "p2_channeling.json", "Z": "mixed"},
    {"name": "Au", "label": "Au [001] 300 keV", "color": WONG["red"],
     "npz": "p2_au.npz", "json": "p2_au.json", "Z": 79},
]


def load_material(mat):
    base = os.path.join(DATA_DIR, mat["npz"].replace(".npz", ""))
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)
    dz_sweep = data["dz_sweep"]
    xi_c = [float(data[f"xi_c_dz{dz}"]) for dz in dz_sweep]
    xi_f = [float(data[f"xi_f_dz{dz}"]) for dz in dz_sweep]
    acf_c = [manifest["sweep"][f"dz_{dz}"]["acf_amp_cvdms"] for dz in dz_sweep]
    acf_f = [manifest["sweep"][f"dz_{dz}"]["acf_amp_fourier"] for dz in dz_sweep]
    return {"dz": list(dz_sweep), "xi_c": xi_c, "xi_f": xi_f,
            "acf_c": acf_c, "acf_f": acf_f,
            "xi_ch": float(data["xi_ch_analytic"]),
            "V_g": float(data["V_g"])}


def main():
    mats = {}
    for mat in MATERIALS:
        base = os.path.join(DATA_DIR, mat["npz"].replace(".npz", ""))
        if os.path.exists(base + ".npz"):
            mats[mat["name"]] = {**mat, **load_material(mat)}
            print(f"Loaded {mat['name']}")
        else:
            print(f"SKIP {mat['name']}: {base}.npz not found")

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.75))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.55,
                          left=0.08, right=0.97, top=0.90, bottom=0.16)

    # ── Row 1: ACF amplitude vs Δz (key commutator fingerprint) ──
    # ── Row 2: ξ vs Δz ──
    for col, (name, m) in enumerate(mats.items()):
        # Panel: ACF amplitude vs Δz
        ax_acf = fig.add_subplot(gs[0, col])
        dz = m["dz"]
        ax_acf.plot(dz, m["acf_c"], "o-", color=WONG["blue"],
                    linewidth=1.2, markersize=5, label="CVDMS")
        ax_acf.plot(dz, m["acf_f"], "s--", color=WONG["orange"],
                    linewidth=1.0, markersize=4, label="Fourier")
        ax_acf.fill_between(dz, m["acf_c"], m["acf_f"], alpha=0.12, color=WONG["red"])
        ax_acf.set_xlabel("Δz (Å)")
        ax_acf.set_ylabel("ACF amplitude")
        ax_acf.set_title(m["label"], fontsize=7, fontweight="bold")
        if col == 0:
            ax_acf.legend(loc="lower right", framealpha=0.9, fontsize=5.5)
        # Annotate ratio
        ratio = m["acf_c"][-1] / m["acf_f"][-1] if m["acf_f"][-1] > 0 else 1
        ax_acf.text(0.95, 0.08, f"ACF ratio: ×{ratio:.2f}",
                    transform=ax_acf.transAxes, ha="right", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Panel: ξ vs Δz
        ax_xi = fig.add_subplot(gs[1, col])
        ax_xi.plot(dz, m["xi_c"], "o-", color=WONG["blue"],
                   linewidth=1.2, markersize=5, label="CVDMS")
        ax_xi.plot(dz, m["xi_f"], "s--", color=WONG["orange"],
                   linewidth=1.0, markersize=4, label="Fourier")
        ax_xi.axhline(m["xi_ch"], color=WONG["grey"], linestyle=":",
                      linewidth=0.7, alpha=0.6, label=f"Bulk ξ={m['xi_ch']:.0f}")
        ax_xi.fill_between(dz, m["xi_c"], m["xi_f"], alpha=0.12, color=WONG["red"])
        ax_xi.set_xlabel("Δz (Å)")
        ax_xi.set_ylabel("ξ (Å)")
        if col == 0:
            ax_xi.legend(loc="upper right", framealpha=0.9, fontsize=5.5)
        # Annotate Δξ
        dxi = abs(m["xi_c"][-1] - m["xi_f"][-1])
        ax_xi.text(0.95, 0.92, f"Δξ={dxi:.1f} Å",
                   transform=ax_xi.transAxes, ha="right", fontsize=6,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    out_pdf = os.path.join(DATA_DIR, "p2_cross_material.pdf")
    out_png = os.path.join(DATA_DIR, "p2_cross_material.png")
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
