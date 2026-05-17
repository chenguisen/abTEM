#!/usr/bin/env python3
"""Plot V2 figures from saved NPZ data.

Reads docs/data/v2{a,b,c}_{subtest}_{stage}.npz and generates the
corresponding PDF figure.

Usage:
  python plot_v2_analytical.py A        # plot Stage A only
  python plot_v2_analytical.py B        # plot Stage B (V2c only)
  python plot_v2_analytical.py A B      # plot both
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def plot_v2(stage):
    # Check which data files exist
    has_a = os.path.exists(os.path.join(DATA_DIR, f"v2a_probe_{stage}.npz"))
    has_b = os.path.exists(os.path.join(DATA_DIR, f"v2b_weak_phase_{stage}.npz"))
    has_c = os.path.exists(os.path.join(DATA_DIR, f"v2c_homogeneous_{stage}.npz"))

    # ── Load available data ──
    if has_a:
        data_a = np.load(os.path.join(DATA_DIR, f"v2a_probe_{stage}.npz"))
        with open(os.path.join(DATA_DIR, f"v2a_probe_{stage}.json")) as f:
            manifest_a = json.load(f)
        z_a = data_a["z"]
        ncc_a = data_a["ncc"]
        ph_rms_a = data_a["phase_rms"]
        amp_rms_a = data_a["amp_rms"]
        px_A = manifest_a["params"]["sampling_A_per_px"]
    if has_b:
        with open(os.path.join(DATA_DIR, f"v2b_weak_phase_{stage}.json")) as f:
            manifest_b = json.load(f)
        ph_b = manifest_b["results"]["phase_rms"]
        passed_b = manifest_b["results"]["passed"]
    if has_c:
        data_c = np.load(os.path.join(DATA_DIR, f"v2c_homogeneous_{stage}.npz"))
        with open(os.path.join(DATA_DIR, f"v2c_homogeneous_{stage}.json")) as f:
            manifest_c = json.load(f)
        z_c = data_c["z"]
        amp_rms_c = data_c["amp_rms"]
        max_amp_c = max(amp_rms_c)

    # For Stage B (V2a/b skipped), use a single-panel layout for V2c only
    if stage == "B" and not has_a:
        _plot_stage_b(manifest_c, z_c, amp_rms_c, max_amp_c)
        return

    n_panels = 2 if has_c else 1

    # ── Figure ──
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.35))
    gs = fig.add_gridspec(1, n_panels, width_ratios=[1] * n_panels,
                          wspace=0.50 if n_panels > 1 else 0,
                          left=0.08, right=0.97, top=0.88, bottom=0.22)

    # ── Panel (a): Probe Fresnel — NCC & phase RMS vs z ──
    if has_a:
        probe_semi = manifest_a["params"].get("probe_semiangle_mrad", "?")
        ax_a = fig.add_subplot(gs[0, 0])
        ax_a_r = ax_a.twinx()
        ax_a.plot(z_a, ncc_a, "o-", color=WONG["blue"], markersize=3, linewidth=0.8)
        ax_a_r.plot(z_a, ph_rms_a, "s--", color=WONG["orange"], markersize=3, linewidth=0.8)
        ax_a.axhline(y=1.0, color=WONG["grey"], linestyle=":", linewidth=0.5, alpha=0.5)
        ax_a.set_xlabel("Depth (Å)")
        ax_a.set_ylabel("NCC", color=WONG["blue"])
        ax_a_r.set_ylabel("Phase RMS (rad)", color=WONG["orange"])
        ax_a.tick_params(axis="y", labelcolor=WONG["blue"])
        ax_a_r.tick_params(axis="y", labelcolor=WONG["orange"])
        ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.6f"))
        ax_a.text(-0.14, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")
        ax_a.text(0.02, 0.06, f"Probe {probe_semi} mrad, V=0, {px_A:.3f} Å/px",
                  transform=ax_a.transAxes, fontsize=6, color=WONG["black"])
        ax_a.text(0.98, 0.92,
                  f"NCC_min = {min(ncc_a):.6f}\nphase_RMS_max = {max(ph_rms_a):.2e}",
                  transform=ax_a.transAxes, ha="right", fontsize=6,
                  color=WONG["black"], va="top")

    # ── Panel (b): Homogeneous V — amplitude RMS vs z ──
    if has_c:
        ax_idx = 1 if has_a else 0
        ax_b = fig.add_subplot(gs[0, ax_idx])
        ax_b.plot(z_c, amp_rms_c, "o-", color=WONG["blue"], markersize=3, linewidth=0.8)
        ax_b.set_xlabel("Depth (Å)")
        ax_b.set_ylabel("Amplitude RMS error")
        ax_b.text(-0.14, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")
        ax_b.text(0.02, 0.06, f"V$_0$=10 eV, {px_A:.3f} Å/px",
                  transform=ax_b.transAxes, fontsize=6, color=WONG["black"])
        ax_b.text(0.98, 0.92, f"amp_RMS_max = {max_amp_c:.2e}",
                  transform=ax_b.transAxes, ha="right", fontsize=6,
                  color=WONG["black"], va="top")

    # weak phase annotation at bottom
    if has_b:
        wp_label = (f"Weak phase (V2b): phase_RMS = {ph_b:.2e}  "
                    f"{'PASS' if passed_b else 'FAIL'}")
        fig.text(0.5, 0.03, wp_label, ha="center", fontsize=6.5, color=WONG["black"])

    out = os.path.join(DATA_DIR, f"v2_analytical_{stage}.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out}")


def _plot_stage_b(manifest_c, z_c, amp_rms_c, max_amp_c):
    """Single-panel figure for Stage B (V2c only)."""
    px_A = manifest_c["params"]["sampling_A_per_px"]

    fig = plt.figure(figsize=(FIG_W * 0.5, FIG_W * 0.35))
    gs = fig.add_gridspec(1, 1, left=0.18, right=0.95, top=0.88, bottom=0.22)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(z_c, amp_rms_c, "o-", color=WONG["blue"], markersize=3, linewidth=0.8)
    ax.set_xlabel("Depth (Å)")
    ax.set_ylabel("Amplitude RMS error")
    ax.text(0.02, 0.06, f"V$_0$=10 eV, {px_A:.3f} Å/px, 627×627",
            transform=ax.transAxes, fontsize=6, color=WONG["black"])
    ax.text(0.98, 0.92, f"amp_RMS_max = {max_amp_c:.2e}",
            transform=ax.transAxes, ha="right", fontsize=6,
            color=WONG["black"], va="top")

    out = os.path.join(DATA_DIR, f"v2_analytical_B.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    stages = sys.argv[1:] if len(sys.argv) > 1 else ["A"]
    for s in stages:
        s_upper = s.upper()
        if s_upper not in ("A", "B"):
            print(f"Unknown stage: {s}, skipping")
            continue
        plot_v2(s_upper)
