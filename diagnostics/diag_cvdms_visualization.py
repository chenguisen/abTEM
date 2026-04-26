"""
CVDMS divergence visualization and CBED correctness verification.

Generates publication-quality figures saved to docs/figures/:
  Fig 1: CBED comparison grid (log scale)
  Fig 2: CBED comparison grid (linear scale)
  Fig 3: CVDMS vs Fourier side-by-side comparison
  Fig 4: Taylor series convergence per order
  Fig 5: Critical frequency map
  Fig 6: Intensity conservation vs thickness
  Fig 7: Thick sample overflow boundary

Usage:
    python diag_cvdms_visualization.py
"""
import sys, os, warnings, json, pickle, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
import ase, abtem
from abtem.multislice import CVDMSMultislice, FourierMultislice
from abtem.core import config as abtem_config
abtem_config.config["fft"] = "numpy"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "figures")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cbed_viz_cache")
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ======================================================================
# Global matplotlib settings
# ======================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
})

# ======================================================================
# Helper functions
# ======================================================================

def make_potential_probe(sampling, energy, gpts=(128, 128), slice_thickness=1.0,
                         semiangle=9.4):
    """Build Si potential + probe at given sampling and energy."""
    bulk = ase.build.bulk("Si", cubic=True) * (4, 4, 4)
    pot = abtem.Potential(
        bulk, gpts=gpts, slice_thickness=slice_thickness, sampling=sampling,
    )
    probe = abtem.Probe(semiangle_cutoff=semiangle, energy=energy).match_grid(pot)
    return pot, probe


def cache_key(*args):
    h = hashlib.md5(str(args).encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{h}.json")


def run_cbed(pot, probe, algorithm, n_slices, cache=True):
    """Run multislice and return CBED intensity array."""
    ck = cache_key("cbed", n_slices, str(algorithm.__class__.__name__),
                   pot.sampling, probe.energy)
    if cache and os.path.exists(ck):
        with open(ck, "r") as f:
            return json.load(f)

    result = probe.multislice(pot[:n_slices], algorithm=algorithm)
    dp = result.diffraction_patterns()
    arr = np.asarray(dp.array)
    cbed = np.abs(arr).astype(np.float64)

    if cache:
        data = {"shape": list(cbed.shape), "max": float(np.max(cbed)),
                "has_inf": bool(np.any(np.isinf(cbed))),
                "has_nan": bool(np.any(np.isnan(cbed)))}
        with open(ck, "w") as f:
            json.dump(data, f)
    return {"array": cbed}


def log_scale(img, C=1.5e6):
    """Log-scale CBED intensity: log10(1 + C * img / max(img))."""
    flat = img.ravel()
    peak = np.max(flat[np.isfinite(flat)] + 0.0)
    if peak <= 0:
        return np.zeros_like(img)
    return np.log10(1.0 + C * img / peak)


def linear_scale(img):
    """Normalize to [0, 1]."""
    flat = img.ravel()
    peak = np.max(flat[np.isfinite(flat)] + 0.0)
    if peak <= 0:
        return np.zeros_like(img)
    return img / peak


def safe_array(arr):
    """Ensure numeric array with inf/nan replaced by 0."""
    a = np.asarray(arr, dtype=np.float64)
    a[np.isinf(a) | np.isnan(a)] = 0.0
    return a


# ======================================================================
# Figure generation
# ======================================================================

def fig_cbed_grid(output_prefix="fig_cbed"):
    """Fig 1 & 2: CBED pattern grids (log and linear) for multi-parameter combos.

    Tests voltage x sampling, at fixed moderate thickness (25 slices at 1A ~ 25nm).
    """
    energies = [30e3, 80e3, 200e3, 300e3]
    samplings = [0.05, 0.10, 0.20]
    n_slices = 25
    gpts = (128, 128)

    fig_log, axes_log = plt.subplots(len(energies), len(samplings),
                                      figsize=(3.5 * len(samplings), 3.5 * len(energies)))
    fig_lin, axes_lin = plt.subplots(len(energies), len(samplings),
                                      figsize=(3.5 * len(samplings), 3.5 * len(energies)))

    for i, energy in enumerate(energies):
        for j, sampling in enumerate(samplings):
            print(f"  CBED grid: {energy/1e3:.0f}keV, {sampling:.2f}A ...")
            pot, probe = make_potential_probe(sampling, energy, gpts=gpts,
                                               slice_thickness=1.0)
            result = probe.multislice(pot[:n_slices],
                                       algorithm=CVDMSMultislice(order=1, max_terms=50))
            dp = result.diffraction_patterns()
            cbed = safe_array(np.abs(np.asarray(dp.array)))

            # Log scale
            log_cbed = log_scale(cbed)
            im_log = axes_log[i][j].imshow(log_cbed, cmap="inferno", aspect="auto",
                                            origin="lower")
            axes_log[i][j].set_title(f"{energy/1e3:.0f} keV, {sampling:.2f} A",
                                      fontsize=13)
            if i == 0 and j == 0:
                axes_log[i][j].set_ylabel("y [px]", fontsize=12)
            plt.colorbar(im_log, ax=axes_log[i][j], fraction=0.046, pad=0.04)

            # Linear scale
            lin_cbed = linear_scale(cbed)
            im_lin = axes_lin[i][j].imshow(lin_cbed, cmap="inferno", aspect="auto",
                                            origin="lower")
            axes_lin[i][j].set_title(f"{energy/1e3:.0f} keV, {sampling:.2f} A",
                                      fontsize=13)
            plt.colorbar(im_lin, ax=axes_lin[i][j], fraction=0.046, pad=0.04)

    fig_log.suptitle("CVDMS CBED Patterns (log scale, C=1.5e6)", fontsize=18, y=1.01)
    fig_lin.suptitle("CVDMS CBED Patterns (linear scale, normalized)", fontsize=18, y=1.01)
    fig_log.tight_layout()
    fig_lin.tight_layout()
    fig_log.savefig(os.path.join(FIGS_DIR, f"{output_prefix}_log.png"), dpi=300)
    fig_lin.savefig(os.path.join(FIGS_DIR, f"{output_prefix}_linear.png"), dpi=300)
    plt.close("all")
    print(f"  Saved {output_prefix}_log.png and {output_prefix}_linear.png")


def fig_cbed_side_by_side(output_name="fig_cbed_side_by_side"):
    """Fig 3: CVDMS vs Fourier side-by-side for key parameter points.

    2x2 layout per point: CVDMS/Fourier columns, log/linear rows.
    """
    params = [
        (80e3, 0.10, 30, "80 keV, 0.10 A, 30 nm"),
        (200e3, 0.05, 30, "200 keV, 0.05 A, 30 nm"),
        (30e3, 0.10, 25, "30 keV, 0.10 A, 25 nm"),
    ]
    gpts = (128, 128)

    fig, axes_all = plt.subplots(len(params), 4, figsize=(14, 4 * len(params)))
    if len(params) == 1:
        axes_all = axes_all.reshape(1, -1)

    for p_idx, (energy, sampling, n_sl, label) in enumerate(params):
        print(f"  Side-by-side: {label} ...")
        pot, probe = make_potential_probe(sampling, energy, gpts=gpts,
                                           slice_thickness=1.0)

        for algo_name, Algo in [("CVDMS", CVDMSMultislice), ("Fourier", FourierMultislice)]:
            if algo_name == "CVDMS":
                algo = Algo(order=1, max_terms=50)
            else:
                algo = Algo(order=1)
            result = probe.multislice(pot[:n_sl], algorithm=algo)
            dp = result.diffraction_patterns()
            cbed = safe_array(np.abs(np.asarray(dp.array)))

            col = 0 if algo_name == "CVDMS" else 1
            axes_all[p_idx][col].imshow(log_scale(cbed), cmap="inferno",
                                         aspect="auto", origin="lower")
            axes_all[p_idx][col].set_title(f"{algo_name} (log)", fontsize=13)

            axes_all[p_idx][col + 2].imshow(linear_scale(cbed), cmap="inferno",
                                             aspect="auto", origin="lower")
            axes_all[p_idx][col + 2].set_title(f"{algo_name} (linear)", fontsize=13)

        axes_all[p_idx][0].set_ylabel(label, fontsize=14)

    fig.suptitle("CVDMS vs Fourier Multislice CBED Comparison", fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"{output_name}.png"), dpi=300)
    plt.close("all")
    print(f"  Saved {output_name}.png")


def fig_taylor_convergence(output_name="fig_taylor_convergence"):
    """Fig 4: Taylor series term amplitude vs order at different voltages."""
    from abtem.cvdms import _cvdms_forward_scattering
    from abtem.finite_difference import _laplace_operator_stencil
    from abtem.core.energy import energy2wavelength, energy2sigma

    energies = [10e3, 30e3, 80e3, 200e3, 300e3]
    sampling = 0.10
    thickness = 1.0
    gpts = (256, 256)
    max_terms = 40

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(energies)))

    # Build Laplacian stencil once (same sampling for all)
    prefactor = 1.0 / (sampling * sampling)
    laplace_fn = _laplace_operator_stencil(8, prefactor, mode="wrap", device="cpu")

    for idx, energy in enumerate(reversed(energies)):  # plot low eV first
        print(f"  Taylor convergence: {energy/1e3:.0f} keV ...")
        pot, probe = make_potential_probe(sampling, energy, gpts=gpts,
                                           slice_thickness=thickness)

        probe_arr = np.asarray(probe.build().array)
        wavelength = energy2wavelength(energy)
        sigma = energy2sigma(energy)
        K0 = 1.0 / wavelength
        transmission_function = pot[0].array[0] * sigma / thickness

        # Run single-slice forward scattering with diagnostics
        exit_wave, diag = _cvdms_forward_scattering(
            probe_arr, transmission_function, laplace_fn, wavelength, thickness,
            max_terms=max_terms, convergence_threshold=1e-6,
            divergence_ratio=5.0, return_diagnostics=True,
        )

        # Extract term amplitudes (indirectly from n_above)
        orders = [r[0] for r in diag["n_above_per_order"]]
        n_above = [r[1] for r in diag["n_above_per_order"]]
        ratios = [r[1] for r in diag["ratios_per_order"]]
        ratio_orders = [r[0] for r in diag["ratios_per_order"]]

        c = colors[idx]
        ax1.semilogy(orders, [max(n, 1) for n in n_above], "o-", color=c,
                     label=f"{energy/1e3:.0f} keV", markersize=5, linewidth=1.5)
        if ratios:
            ax2.semilogy(ratio_orders, ratios, "s--", color=c, markersize=5, linewidth=1.5)

        # Mark convergence
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        print(f"    n_terms={diag['n_terms_used']}, max_amp={diag['max_amplitude']:.2e}, "
              f"overflow={diag['overflow_detected']}, diverged={diag['divergence_truncated']}")

    ax1.set_xlabel("Taylor Series Order n")
    ax1.set_ylabel("Unconverged Pixels (count)")
    ax1.set_title("Pixel Convergence: |term| > cutoff")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Taylor Series Order n")
    ax2.set_ylabel("Term / Accumulated Ratio")
    ax2.set_title("Divergence Ratio per Order")
    ax2.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="divergence_ratio=5")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("CVDMS Taylor Series Convergence Behavior", fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"{output_name}.png"), dpi=300)
    plt.close("all")
    print(f"  Saved {output_name}.png")


def fig_critical_frequency_map(output_name="fig_critical_frequency"):
    """Fig 5: Critical frequency k_critical vs (voltage, sampling) heatmap
    with Nyquist frequency overlay."""
    from abtem.core.energy import energy2wavelength

    energies = np.linspace(5, 300, 50) * 1e3
    samplings = np.linspace(0.03, 0.25, 50)
    dz = 1.0  # typical slice thickness

    # Compute k_critical = sqrt(K0 / (pi * dz))
    K0_vals = np.array([1.0 / energy2wavelength(e) for e in energies])
    k_crit_grid = np.zeros((len(energies), len(samplings)))
    nyquist_grid = np.zeros((len(energies), len(samplings)))

    for i, K0 in enumerate(K0_vals):
        k_crit = np.sqrt(K0 / (np.pi * dz))
        for j, s in enumerate(samplings):
            k_crit_grid[i, j] = k_crit
            nyquist_grid[i, j] = 1.0 / (2.0 * s)

    # Ratio: Nyquist / k_critical. > 1 means frequencies above k_critical exist.
    ratio_grid = nyquist_grid / k_crit_grid

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: k_critical heatmap
    im1 = ax1.pcolormesh(samplings, energies / 1e3, k_crit_grid, shading="auto",
                          cmap="plasma")
    ax1.contour(samplings, energies / 1e3, k_crit_grid, levels=6, colors="white",
                alpha=0.5, linewidths=0.8)
    plt.colorbar(im1, ax=ax1, label=r"$k_{\rm critical}$ [${\rm \AA}^{-1}$]")
    ax1.set_xlabel("Sampling [A]")
    ax1.set_ylabel("Voltage [keV]")
    ax1.set_title(r"Critical Frequency $k_{\rm crit} = \sqrt{K_0 / (\pi\,dz)}$")

    # Panel 2: Ratio Nyquist / k_critical
    im2 = ax2.pcolormesh(samplings, energies / 1e3, ratio_grid, shading="auto",
                          cmap="RdYlBu_r", vmin=0.5, vmax=5.0)
    ax2.contour(samplings, energies / 1e3, ratio_grid,
                levels=[1.0], colors="black", linewidths=2, linestyles="--")
    ax2.contour(samplings, energies / 1e3, ratio_grid,
                levels=[2.0, 3.0, 4.0], colors="gray", alpha=0.5, linewidths=0.8)
    plt.colorbar(im2, ax=ax2, label="Nyquist / k_critical")
    ax2.set_xlabel("Sampling [A]")
    ax2.set_ylabel("Voltage [keV]")
    ax2.set_title(r"Nyquist / $k_{\rm crit}$ (dashed = 1, >1 = risk)")
    ax2.annotate("SAFE", (0.15, 250), fontsize=16, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.6))
    ax2.annotate("CAUTION", (0.06, 50), fontsize=16, color="black", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.6))

    fig.suptitle("CVDMS Stability Boundary: Critical Frequency Analysis", fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"{output_name}.png"), dpi=300)
    plt.close("all")
    print(f"  Saved {output_name}.png")


def fig_intensity_conservation(output_name="fig_intensity_conservation"):
    """Fig 6: Intensity conservation ΔI/I₀ vs thickness for CVDMS vs Fourier."""
    energies = [30e3, 80e3, 200e3]
    sampling = 0.10
    gpts = (256, 256)
    n_slices_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    fig, axes = plt.subplots(1, len(energies), figsize=(6 * len(energies), 5))
    if len(energies) == 1:
        axes = [axes]

    for e_idx, energy in enumerate(energies):
        print(f"  Intensity conservation: {energy/1e3:.0f} keV ...")
        pot, probe = make_potential_probe(sampling, energy, gpts=gpts,
                                           slice_thickness=1.0)
        probe_arr = np.asarray(probe.build().array)
        I0 = float(np.sum(np.abs(probe_arr) ** 2))

        cvdms_dI = []
        fourier_dI = []
        cvdms_max_amp = []

        for n_sl in n_slices_list:
            # CVDMS
            r_c = probe.multislice(pot[:n_sl],
                                    algorithm=CVDMSMultislice(order=1, max_terms=50))
            a_c = np.asarray(r_c.array)
            Ic = float(np.sum(np.abs(a_c) ** 2))
            cvdms_dI.append(abs(Ic - I0) / I0)
            cvdms_max_amp.append(float(np.max(np.abs(a_c))))

            # Fourier
            r_f = probe.multislice(pot[:n_sl],
                                    algorithm=FourierMultislice(order=1))
            a_f = np.asarray(r_f.array)
            If = float(np.sum(np.abs(a_f) ** 2))
            fourier_dI.append(abs(If - I0) / I0)

        thicknesses = [n * 1.0 for n in n_slices_list]
        ax = axes[e_idx]
        ax.semilogy(thicknesses, cvdms_dI, "o-", color="C0", label="CVDMS",
                     markersize=6, linewidth=2)
        ax.semilogy(thicknesses, fourier_dI, "s--", color="C1", label="Fourier",
                     markersize=6, linewidth=2)
        ax.set_xlabel("Thickness [nm]")
        ax.set_ylabel(r"$|\Delta I| / I_0$")
        ax.set_title(f"{energy/1e3:.0f} keV")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Intensity Conservation: CVDMS vs Fourier Multislice", fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"{output_name}.png"), dpi=300)
    plt.close("all")
    print(f"  Saved {output_name}.png")


def fig_thick_sample_stress(output_name="fig_thick_sample_stress"):
    """Fig 7: Thick sample overflow boundary.

    Tests thick samples (up to 50nm) with different slice thicknesses, monitoring
    max amplitude per slice to find the overflow boundary.
    """
    energies = [30e3, 80e3, 200e3, 300e3]
    samplings = [0.05, 0.10, 0.20]
    dz_vals = [0.1, 0.5, 1.0]  # slice thickness [A]
    total_thickness = 500.0    # 50 nm
    gpts = (128, 128)

    fig, axes = plt.subplots(len(energies), len(samplings),
                              figsize=(4 * len(samplings), 3.5 * len(energies)))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(dz_vals)))

    for i, energy in enumerate(energies):
        for j, sampling in enumerate(samplings):
            ax = axes[i][j]
            ax.set_title(f"{energy/1e3:.0f} keV, {sampling:.2f} A", fontsize=12)

            for dz_idx, dz in enumerate(dz_vals):
                n_slices = int(total_thickness / dz)
                pot, probe = make_potential_probe(sampling, energy, gpts=gpts,
                                                   slice_thickness=dz)
                n_avail = min(n_slices, len(pot))
                if n_avail < 5:
                    continue

                max_amps = []
                overflow_at = None

                # Check at key intermediate points
                algo = CVDMSMultislice(order=1, max_terms=50)
                check_fractions = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
                for frac in check_fractions:
                    cp = max(1, int(n_avail * frac))
                    r = probe.multislice(pot[:cp], algorithm=algo)
                    a = np.asarray(r.array)
                    ma = float(np.max(np.abs(a)))
                    max_amps.append(ma)
                    if np.any(np.isinf(a)) or np.any(np.isnan(a)):
                        overflow_at = cp
                        break

                c = colors[dz_idx]
                label = f"dz={dz:.1f}A" if i == 0 and j == 0 else ""
                x_vals = [int(n_avail * f) for f in check_fractions[:len(max_amps)]]
                ax.plot(x_vals, max_amps, "o-", color=c,
                        markersize=4, linewidth=1.5, label=label, alpha=0.8)

                if overflow_at is not None:
                    ax.axvline(x=overflow_at, color=c, linestyle="--", alpha=0.4)

            ax.set_xlabel("Slice index")
            ax.set_ylabel("max |ψ|")
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

    fig.suptitle("CVDMS Thick Sample Stress Test: Amplitude vs Slice Index (50 nm total)",
                 fontsize=18, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"{output_name}.png"), dpi=300)
    plt.close("all")
    print(f"  Saved {output_name}.png")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CVDMS Visualization & CBED Verification")
    print(f"Output: {FIGS_DIR}")
    print("=" * 70)

    print("\n[Fig 4] Taylor convergence per order...")
    fig_taylor_convergence()

    print("\n[Fig 5] Critical frequency map...")
    fig_critical_frequency_map()

    print("\n[Fig 6] Intensity conservation vs thickness...")
    fig_intensity_conservation()

    print("\n[Fig 7] Thick sample stress test...")
    fig_thick_sample_stress()

    print("\n[Fig 1-2] CBED comparison grids...")
    fig_cbed_grid()

    print("\n[Fig 3] CVDMS vs Fourier side-by-side...")
    fig_cbed_side_by_side()

    print("\n" + "=" * 70)
    print(f"All figures saved to {FIGS_DIR}/")
    print("=" * 70)
