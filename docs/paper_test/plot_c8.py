#!/usr/bin/env python3
"""Plot C8 float32 precision floor — I/I0 diffusion vs (Δz, thickness).

Reads docs/data/c8_float32.npz/.json and generates PDF + PNG.
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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


def main():
    base = os.path.join(DATA_DIR, "c8_float32")
    data = np.load(base + ".npz")
    with open(base + ".json") as f:
        manifest = json.load(f)

    results = manifest["results"]
    params = manifest["params"]
    dz_uniq = sorted(set(r["dz_A"] for r in results))
    t_uniq = sorted(set(r["thickness_A"] for r in results))

    # Build 2D array
    I_grid = np.zeros((len(dz_uniq), len(t_uniq)))
    slices_grid = np.zeros((len(dz_uniq), len(t_uniq)), dtype=int)
    for r in results:
        i = dz_uniq.index(r["dz_A"])
        j = t_uniq.index(r["thickness_A"])
        I_grid[i, j] = r["I_ratio"]
        slices_grid[i, j] = r["n_slices"]

    # Per-slice error: 1 - I/I0^(1/n)
    eps_grid = 1 - I_grid ** (1 / np.maximum(slices_grid, 1))

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.72))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.35,
                          left=0.10, right=0.97, top=0.90, bottom=0.14)

    # ── (a) I/I₀ vs thickness for each Δz ──
    ax_a = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(dz_uniq)))
    for i, dz in enumerate(dz_uniq):
        pts = [r for r in results if r["dz_A"] == dz]
        t_arr = np.array([r["thickness_A"] for r in pts])
        I_arr = np.array([r["I_ratio"] for r in pts])
        ax_a.plot(t_arr, I_arr, "o-", color=colors[i], linewidth=1.0,
                 markersize=4, label=f"$\\Delta z$={dz:.1f} Å")
    ax_a.axhline(0.99, color=WONG["red"], linewidth=0.5, linestyle=":",
                 alpha=0.6)
    ax_a.text(t_uniq[-1]*0.6, 0.990, "1% loss", color=WONG["red"],
              fontsize=5.5, va="bottom")
    ax_a.set_xlabel("Thickness (Å)")
    ax_a.set_ylabel("$I/I_0$")
    ax_a.legend(loc="lower left", framealpha=0.9, fontsize=5.5, ncol=2)
    ax_a.text(-0.18, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── (b) I/I₀ vs n_slices (collapse check) ──
    ax_b = fig.add_subplot(gs[0, 1])
    for r in results:
        color = plt.cm.viridis(r["dz_A"] / max(dz_uniq))
        ax_b.plot(r["n_slices"], r["I_ratio"], "o", color=color,
                 markersize=4, markeredgewidth=0.3, markeredgecolor="white")
    # Fit: I/I0 ≈ 1 - α * n_slices
    n_all = np.array([r["n_slices"] for r in results])
    I_all = np.array([r["I_ratio"] for r in results])
    loss = 1 - I_all
    alpha = np.sum(n_all * loss) / np.sum(n_all**2) if n_all.sum() > 0 else 0
    n_fit = np.linspace(0, max(n_all), 100)
    ax_b.plot(n_fit, 1 - alpha * n_fit, "--", color=WONG["grey"],
             linewidth=0.7, alpha=0.6,
             label=f"$1-\\alpha n$ ($\\alpha$={alpha:.1e})")
    ax_b.set_xlabel("Number of slices")
    ax_b.set_ylabel("$I/I_0$")
    ax_b.legend(loc="lower left", framealpha=0.9, fontsize=5.5)
    ax_b.text(-0.18, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── (c) Per-slice error ε vs Δz ──
    ax_c = fig.add_subplot(gs[1, 0])
    eps_mean = eps_grid.mean(axis=1)
    eps_std = eps_grid.std(axis=1)
    ax_c.errorbar(dz_uniq, eps_mean, yerr=eps_std, fmt="s-", color=WONG["blue"],
                 linewidth=1.0, markersize=6, capsize=3)
    ax_c.set_xlabel("$\\Delta z$ (Å)")
    ax_c.set_ylabel("Per-slice diffusion $\\varepsilon$")
    ax_c.text(-0.18, 1.05, "c", transform=ax_c.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── (d) Summary ──
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    # Compute per-slice error for each run
    eps_all = np.array([1 - r["I_ratio"]**(1/max(r["n_slices"],1)) for r in results])
    lines = [
        f"SrTiO$_3$ [001], {params['energy_eV']/1000:.0f} keV",
        f"$\\Delta x$ = {params['sampling_A_per_px']} Å/px",
        f"$\\varepsilon$ = {float(eps_all.mean()):.1e}",
        "",
        "Float32 precision floor:",
        f"  $\\langle\\varepsilon\\rangle$ = {alpha:.1e} per slice",
        f"  I/I$_0$ > 0.99 → n < {1/alpha*0.01:.0f} slices",
        "",
        "Key: float32 error is diffusive,",
        "NOT amplificative. I/I$_0$ < 1",
        "in all cases. Finer $\\Delta z$",
        "(more slices) → more loss.",
        "",
        "This is opposite to commutator",
        "error which grows with $\\Delta z$.",
        "$\\therefore$ optimal $\\Delta z$",
        "balances commutator error",
        "(coarse) and float32 diffusion",
        "(fine).",
    ]
    for i, line in enumerate(lines):
        ax_d.text(0.0, 0.97 - i * 0.045, line, transform=ax_d.transAxes,
                 fontsize=5.5, va="top", fontfamily="monospace")

    out_pdf = os.path.join(DATA_DIR, "c8_float32.pdf")
    out_png = os.path.join(DATA_DIR, "c8_float32.png")
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
