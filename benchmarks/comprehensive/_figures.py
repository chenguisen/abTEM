"""
Publication-quality figure generation for the CVDMS benchmark report.

All figures follow a consistent style:
  - 300 DPI, DejaVu Sans
  - Proper colorbars and panel labels
  - Consistent color scheme for algorithms
"""
import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._parameters import ALGORITHM_LABELS, ALGORITHM_COLORS
from ._sweep import SweepEngine, load_cache, cache_paths, make_cache_key
from ._parameters import Baseline, SweepDef, resolve_sweep_params, ALGORITHMS

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": "0.3",
    "axes.labelcolor": "0.2",
    "text.color": "0.2",
    "xtick.color": "0.3",
    "ytick.color": "0.3",
})

OUTPUT_DIR = None
_RESULT_LOOKUP = None  # {(sweep, algo, value): cache_key}


def set_output_dir(path: str):
    global OUTPUT_DIR
    OUTPUT_DIR = path
    os.makedirs(path, exist_ok=True)


def _build_result_lookup(results: list) -> dict:
    """Build {(sweep, algo, value): cache_key} from results list."""
    lookup = {}
    for r in results:
        key = r.get("_cache_key", "")
        if key:
            lookup[(r["sweep"], r["algorithm"], r["value"])] = key
    return lookup


def set_result_lookup(lookup: dict):
    global _RESULT_LOOKUP
    _RESULT_LOOKUP = lookup


def _savefig(fig, name):
    """Save figure to PNG and return base64-encoded string."""
    if OUTPUT_DIR is None:
        raise RuntimeError("Call set_output_dir() first")
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Also return base64 for embedding
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _panel_label(ax, label, x=0.03, y=0.97):
    """Add panel label (A, B, C...) to axis."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8))


def _cb(fig, ax, im):
    """Add colorbar to axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    return cb


# ======================================================================
# Data loading helpers
# ======================================================================
def _load_arrays(cache_dir, sweep_name, algorithm, params):
    """Load cached arrays for a specific (sweep, algo, value) combination."""
    key = make_cache_key(sweep_name, algorithm, params)
    npz_path, _ = cache_paths(cache_dir, sweep_name, key)
    if not os.path.exists(npz_path):
        return None
    return np.load(npz_path)


def _cbed_2d(cbed):
    """Reduce CBED array to 2D for display.

    Cached CBED may be 4D (FP × exit_planes × gx × gy). Sum over
    extra dims to produce a single 2D diffraction pattern.
    Returns None if array is None or all-NaN/inf.
    """
    if cbed is None:
        return None
    arr = np.asarray(cbed)
    while arr.ndim > 2:
        arr = arr.sum(axis=0)
    if not np.isfinite(arr).any():
        return None
    return arr


def _iter_sweep_results(cache_dir, sweep: SweepDef, algorithms=None):
    """Yield (value, label, algorithm, npz_data) for each sweep point.

    Uses stored cache keys from the results summary when available
    (bulletproof against key reconstruction mismatches). Falls back
    to param reconstruction with fast-mode overrides for backward
    compatibility with summaries that lack _cache_key.
    """
    if algorithms is None:
        algorithms = ALGORITHMS

    # Preferred path: stored cache keys from results.
    # If the lookup doesn't match this sweep's values (e.g. thickness
    # values are recalculated by resolve_sweep_params), fall through
    # to the fallback which reconstructs params correctly.
    if _RESULT_LOOKUP is not None:
        found_any = False
        for val in sweep.values:
            for algo in algorithms:
                key = _RESULT_LOOKUP.get((sweep.name, algo, val))
                if key:
                    found_any = True
                    npz_path, _ = cache_paths(cache_dir, sweep.name, key)
                    if os.path.exists(npz_path):
                        yield val, sweep.name, algo, dict(np.load(npz_path))
        if found_any:
            return

    # Fallback: reconstruct params (must match SweepEngine._run_one fast mode)
    baseline = Baseline()
    for val in sweep.values:
        params = resolve_sweep_params(baseline, sweep, val)
        if sweep.name == "sampling":
            from ._parameters import sampling_gpts
            params["_gpts"] = sampling_gpts(params["sampling"])
        elif sweep.full_resolution:
            params["_gpts"] = baseline.gpts
        else:
            params["_gpts"] = (256, 256)
        if sweep.name != "fp":
            params["frozen_phonons"] = 4
        for algo in algorithms:
            data = _load_arrays(cache_dir, sweep.name, algo, params)
            if data is not None:
                yield val, sweep.name, algo, dict(data)


# ======================================================================
# Figure 1: Voltage × Algorithm CBED grid
# ======================================================================
def fig_voltage_cbed_grid(cache_dir: str, sweep: SweepDef) -> str:
    """5×3 panel: rows=voltages, cols=algorithms. Log-scale CBED."""
    nrows, ncols = len(sweep.values), 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))

    for i, val in enumerate(sweep.values):
        for j, algo in enumerate(ALGORITHMS):
            ax = axes[i, j] if nrows > 1 else axes[j]
            data = None
            for v, s, a, d in _iter_sweep_results(cache_dir, sweep, [algo]):
                if v == val and a == algo:
                    data = d
                    break
            cbed = _cbed_2d(data.get("cbed")) if data is not None else None
            if cbed is not None:
                # Downsample for display if > 512
                if max(cbed.shape) > 512:
                    from scipy.ndimage import zoom
                    sy, sx = cbed.shape
                    scale = 512 / max(sy, sx)
                    cbed = zoom(cbed, (scale, scale), order=1)
                im = ax.imshow(cbed, norm=LogNorm(vmin=cbed.max() * 1e-6,
                                                  vmax=cbed.max()),
                               cmap="inferno", origin="lower")
                if i == 0 and j == 2:
                    _cb(fig, ax, im)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

            # Labels
            if i == 0:
                ax.set_title(ALGORITHM_LABELS[algo], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{val/1e3:.0f} keV", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("CBED patterns vs accelerating voltage", fontsize=14, y=1.01)
    plt.tight_layout()
    return _savefig(fig, "fig_01_voltage_cbed_grid")


# ======================================================================
# Figure 2: Voltage line profiles (horizontal cuts through CBED center)
# ======================================================================
def fig_voltage_line_profiles(cache_dir: str, sweep: SweepDef) -> str:
    """Line profiles through CBED center, semilog y, all algorithms overlaid."""
    nrows = 2
    ncols = int(np.ceil(len(sweep.values) / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, val in enumerate(sweep.values):
        ax = axes[idx]
        for algo in ALGORITHMS:
            data = None
            for v, s, a, d in _iter_sweep_results(cache_dir, sweep, [algo]):
                if v == val and a == algo:
                    data = d
                    break
            cbed = _cbed_2d(data.get("cbed")) if data is not None else None
            if cbed is not None:
                center = cbed.shape[0] // 2
                profile = cbed[center, :]
                x = np.arange(len(profile)) - len(profile) // 2
                ax.semilogy(x, profile, lw=1.2, alpha=0.85,
                           color=ALGORITHM_COLORS[algo],
                           label=ALGORITHM_LABELS[algo] if idx == 0 else "")
        ax.set_title(f"{val/1e3:.0f} keV", fontsize=11)
        ax.set_xlabel("Pixel from center")
        if idx == 0:
            ax.set_ylabel("Intensity (log)")
        ax.grid(True, alpha=0.3)

    if len(sweep.values) > 1 and axes[0].lines:
        axes[0].legend(fontsize=9)

    fig.suptitle("CBED horizontal line profiles", fontsize=14, y=1.02)
    plt.tight_layout()
    return _savefig(fig, "fig_02_voltage_line_profiles")


# ======================================================================
# Figure 3: Voltage metrics (NCC, RMSD vs voltage)
# ======================================================================
def fig_voltage_metrics(results: list) -> str:
    """NCC and RMSD vs voltage for CVDMS algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    voltages = sorted(set(r["value"] for r in results if r["sweep"] == "voltage"))
    # Filter results
    v_results = [r for r in results if r["sweep"] == "voltage"]

    for ax, metric, ylabel in zip(axes,
                                   ["ncc_vs_reference", "rmsd_vs_reference"],
                                   ["NCC (vs Fourier)", "RMSD (vs Fourier)"]):
        for algo in ["cvdms_fd", "cvdms_bsc"]:
            xs, ys = [], []
            for v in voltages:
                for r in v_results:
                    if r["algorithm"] == algo and r["value"] == v:
                        m = r.get("metrics", {})
                        if metric in m:
                            xs.append(v / 1e3)
                            ys.append(m[metric])
            if xs:
                ax.plot(xs, ys, "-o", color=ALGORITHM_COLORS[algo], lw=2,
                        label=ALGORITHM_LABELS[algo], ms=6)
        ax.set_xlabel("Voltage (keV)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("CVDMS accuracy vs accelerating voltage", fontsize=14)
    plt.tight_layout()
    return _savefig(fig, "fig_03_voltage_metrics")


# ======================================================================
# Figure 4: FP convergence (NCC vs FP count)
# ======================================================================
def fig_fp_convergence(results: list) -> str:
    """NCC vs frozen phonon count."""
    fp_results = [r for r in results if r["sweep"] == "fp"]
    fp_counts = sorted(set(r["value"] for r in fp_results))

    fig, ax = plt.subplots(figsize=(7, 5))

    for algo in ["cvdms_fd", "cvdms_bsc"]:
        xs, ys = [], []
        for n in fp_counts:
            for r in fp_results:
                if r["algorithm"] == algo and r["value"] == n:
                    m = r.get("metrics", {})
                    # Use intensity conservation as convergence metric
                    ic = m.get("intensity_conservation", None)
                    if ic is not None:
                        xs.append(n)
                        ys.append(ic)
        if xs:
            ax.plot(xs, ys, "-s", color=ALGORITHM_COLORS[algo], lw=2,
                    label=ALGORITHM_LABELS[algo], ms=6)

    ax.set_xlabel("Number of frozen phonon configurations")
    ax.set_ylabel("Intensity conservation |ΔI|/I₀")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    ax.set_title("Frozen phonon convergence", fontsize=13)

    plt.tight_layout()
    return _savefig(fig, "fig_04_fp_convergence")


# ======================================================================
# Figure 5: FP CBED grid
# ======================================================================
def fig_fp_cbed_grid(cache_dir: str, sweep: SweepDef) -> str:
    """Selected FP counts CBED panels."""
    selected = [1, 4, 16, 32]
    ncols = 3
    nrows = len(selected)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))

    for i, fp in enumerate(selected):
        for j, algo in enumerate(ALGORITHMS):
            ax = axes[i, j] if nrows > 1 else axes[j]
            data = None
            for v, s, a, d in _iter_sweep_results(cache_dir, sweep, [algo]):
                if v == fp and a == algo:
                    data = d
                    break
            cbed = _cbed_2d(data.get("cbed")) if data is not None else None
            if cbed is not None:
                if max(cbed.shape) > 400:
                    from scipy.ndimage import zoom
                    scale = 400 / max(cbed.shape)
                    cbed = zoom(cbed, (scale, scale), order=1)
                im = ax.imshow(cbed, norm=LogNorm(vmin=cbed.max() * 1e-6,
                                                 vmax=cbed.max()),
                               cmap="inferno", origin="lower")
                if i == 0 and j == 2:
                    _cb(fig, ax, im)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            if i == 0:
                ax.set_title(ALGORITHM_LABELS[algo], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"FP={fp}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("CBED patterns vs frozen phonon configurations", fontsize=14, y=1.01)
    plt.tight_layout()
    return _savefig(fig, "fig_05_fp_cbed_grid")


# ======================================================================
# Figure 6: Sampling comparison CBED
# ======================================================================
def fig_sampling_comparison(cache_dir: str, sweep: SweepDef) -> str:
    """CBED patterns at different sampling rates."""
    nrows, ncols = len(sweep.values), 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))

    for i, val in enumerate(sweep.values):
        for j, algo in enumerate(ALGORITHMS):
            ax = axes[i, j] if nrows > 1 else axes[j]
            data = None
            for v, s, a, d in _iter_sweep_results(cache_dir, sweep, [algo]):
                if abs(v - val) < 1e-6 and a == algo:
                    data = d
                    break
            cbed = _cbed_2d(data.get("cbed")) if data is not None else None
            if cbed is not None:
                if max(cbed.shape) > 512:
                    from scipy.ndimage import zoom
                    scale = 512 / max(cbed.shape)
                    cbed = zoom(cbed, (scale, scale), order=1)
                im = ax.imshow(cbed, norm=LogNorm(vmin=cbed.max() * 1e-6,
                                                 vmax=cbed.max()),
                               cmap="inferno", origin="lower")
                if i == 0 and j == 2:
                    _cb(fig, ax, im)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            if i == 0:
                ax.set_title(ALGORITHM_LABELS[algo], fontsize=10)
            if j == 0:
                nyquist = f"{1/(2*val):.1f}"
                ax.set_ylabel(f"{val} Å\n(Nyq={nyquist} 1/Å)", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("CBED patterns vs real-space sampling", fontsize=14, y=1.01)
    plt.tight_layout()
    return _savefig(fig, "fig_06_sampling_cbed")


# ======================================================================
# Figure 7: Sampling metrics
# ======================================================================
def fig_sampling_metrics(results: list) -> str:
    """Metrics vs sampling rate."""
    s_results = [r for r in results if r["sweep"] == "sampling"]
    samplings = sorted(set(r["value"] for r in s_results))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, ylabel in zip(axes,
                                   ["symmetry_h", "intensity_conservation"],
                                   ["CBED symmetry score", "Intensity conservation |ΔI|/I₀"]):
        for algo in ["cvdms_fd", "cvdms_bsc"]:
            xs, ys = [], []
            for s in samplings:
                for r in s_results:
                    if r["algorithm"] == algo and r["value"] == s:
                        m = r.get("metrics", {})
                        if metric in m:
                            xs.append(s)
                            ys.append(m[metric])
            if xs:
                ax.plot(xs, ys, "-o", color=ALGORITHM_COLORS[algo], lw=2,
                        label=ALGORITHM_LABELS[algo], ms=6)
        ax.set_xlabel("Sampling (Å)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=9)
        if metric == "symmetry_h":
            ax.axhline(y=0.95, color="gray", ls="--", lw=1, alpha=0.7, label="Threshold")

    fig.suptitle("Effect of sampling rate on CBED quality", fontsize=14)
    plt.tight_layout()
    return _savefig(fig, "fig_07_sampling_metrics")


# ======================================================================
# Figure 8: Thickness CBED
# ======================================================================
def fig_thickness_cbed(cache_dir: str, sweep: SweepDef) -> str:
    """CBED at different thicknesses."""
    nrows, ncols = len(sweep.values), 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))

    for i, val in enumerate(sweep.values):
        for j, algo in enumerate(ALGORITHMS):
            ax = axes[i, j] if nrows > 1 else axes[j]
            data = None
            for v, s, a, d in _iter_sweep_results(cache_dir, sweep, [algo]):
                if v == val and a == algo:
                    data = d
                    break
            cbed = _cbed_2d(data.get("cbed")) if data is not None else None
            if cbed is not None:
                if max(cbed.shape) > 400:
                    from scipy.ndimage import zoom
                    scale = 400 / max(cbed.shape)
                    cbed = zoom(cbed, (scale, scale), order=1)
                im = ax.imshow(cbed, norm=LogNorm(vmin=cbed.max() * 1e-6,
                                                 vmax=cbed.max()),
                               cmap="inferno", origin="lower")
                if i == 0 and j == 2:
                    _cb(fig, ax, im)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            if i == 0:
                ax.set_title(ALGORITHM_LABELS[algo], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{val} nm", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("CBED patterns vs specimen thickness", fontsize=14, y=1.01)
    plt.tight_layout()
    return _savefig(fig, "fig_08_thickness_cbed")


# ======================================================================
# Figure 9: Thickness metrics
# ======================================================================
def fig_thickness_metrics(results: list) -> str:
    """NCC, RMSD, intensity conservation vs thickness."""
    t_results = [r for r in results if r["sweep"] == "thickness"]
    thicknesses = sorted(set(r["value"] for r in t_results))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = ["symmetry_h", "intensity_conservation"]
    labels = ["CBED symmetry score", "Intensity conservation |ΔI|/I₀"]

    for ax, metric, ylabel in zip(axes[:2], metrics, labels):
        for algo in ["cvdms_fd", "cvdms_bsc", "fourier"]:
            xs, ys = [], []
            for t in thicknesses:
                for r in t_results:
                    if r["algorithm"] == algo and r["value"] == t:
                        m = r.get("metrics", {})
                        if metric in m:
                            xs.append(t)
                            ys.append(m[metric])
            if xs:
                ax.plot(xs, ys, "-o", color=ALGORITHM_COLORS.get(algo, "gray"),
                        lw=2, label=ALGORITHM_LABELS.get(algo, algo), ms=6)
        ax.set_xlabel("Thickness (nm)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8)

    # Panel 3: timing
    ax = axes[2]
    for algo in ALGORITHMS:
        xs, ys = [], []
        for t in thicknesses:
            for r in t_results:
                if r["algorithm"] == algo and r["value"] == t:
                    xs.append(t)
                    ys.append(r.get("time", 0))
        if xs:
            ax.plot(xs, ys, "-o", color=ALGORITHM_COLORS.get(algo, "gray"),
                    lw=2, label=ALGORITHM_LABELS.get(algo, algo), ms=6)
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=8)

    fig.suptitle("Effect of specimen thickness on accuracy and performance", fontsize=14)
    plt.tight_layout()
    return _savefig(fig, "fig_09_thickness_metrics")


# ======================================================================
# Figure 10: Slice thickness metrics
# ======================================================================
def fig_slice_thickness_metrics(results: list) -> str:
    """Metrics vs slice thickness."""
    s_results = [r for r in results if r["sweep"] == "slice_thickness"]
    dzs = sorted(set(r["value"] for r in s_results))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, ylabel in zip(axes,
                                   ["symmetry_h", "intensity_conservation"],
                                   ["CBED symmetry score", "Intensity conservation |ΔI|/I₀"]):
        for algo in ["cvdms_fd", "cvdms_bsc", "fourier"]:
            xs, ys = [], []
            for dz in dzs:
                for r in s_results:
                    if r["algorithm"] == algo and abs(r["value"] - dz) < 1e-6:
                        m = r.get("metrics", {})
                        if metric in m:
                            xs.append(dz)
                            ys.append(m[metric])
            if xs:
                ax.plot(xs, ys, "-o", color=ALGORITHM_COLORS.get(algo, "gray"),
                        lw=2, label=ALGORITHM_LABELS.get(algo, algo), ms=6)
        ax.set_xlabel("Slice thickness (Å)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=9)

    fig.suptitle("Effect of slice thickness", fontsize=14)
    plt.tight_layout()
    return _savefig(fig, "fig_10_slice_thickness_metrics")


# ======================================================================
# Figure 11: Intensity conservation bar chart
# ======================================================================
def fig_intensity_conservation(results: list) -> str:
    """|ΔI|/I₀ across all sweeps, grouped bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Group by sweep
    sweeps_names = ["voltage", "sampling", "fp", "thickness", "slice_thickness"]
    sweep_labels = ["Voltage", "Sampling", "Frozen\nphonons", "Thickness", "Slice\nthickness"]

    x = np.arange(len(sweeps_names))
    width = 0.25

    for i, algo in enumerate(["fourier", "cvdms_fd", "cvdms_bsc"]):
        means = []
        for sn in sweeps_names:
            vals = [r.get("metrics", {}).get("intensity_conservation", 0)
                    for r in results if r["sweep"] == sn
                    and r["algorithm"] == algo
                    and r.get("metrics", {}).get("intensity_conservation") is not None]
            means.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * width - width, means, width,
               label=ALGORITHM_LABELS.get(algo, algo),
               color=ALGORITHM_COLORS.get(algo, "gray"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sweep_labels)
    ax.set_ylabel("Mean |ΔI|/I₀")
    ax.set_title("Intensity conservation across parameter sweeps")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _savefig(fig, "fig_11_intensity_conservation")


# ======================================================================
# Figure 12: Validation pass/fail heatmap
# ======================================================================
def fig_validation_heatmap(results: list) -> str:
    """Pass/fail heatmap across all parameter points × algorithms."""
    sweeps_names = ["voltage", "sampling", "fp", "thickness", "slice_thickness"]

    # Build matrix: row = (sweep, value), col = algorithm, value = pass(1)/fail(0)
    rows = []
    row_labels = []
    for sn in sweeps_names:
        vals = sorted(set(r["value"] for r in results if r["sweep"] == sn))
        for v in vals:
            row_labels.append(f"{sn}\n{v}")
            row_data = []
            for algo in ALGORITHMS:
                r_list = [r for r in results if r["sweep"] == sn
                          and r["value"] == v and r["algorithm"] == algo]
                if not r_list:
                    row_data.append(0)
                else:
                    m = r_list[0].get("metrics", {})
                    # Pass if no overflow and either symmetry or NCC is ok
                    ovf = m.get("overflow", True)
                    sym = m.get("symmetry_pass", False)
                    row_data.append(0 if ovf else (1 if sym else 0.5))
            rows.append(row_data)

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(rows) + 1))
    im = ax.imshow(rows, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([ALGORITHM_LABELS[a] for a in ALGORITHMS], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Validation summary: pass (green) / marginal (yellow) / fail (red)",
                 fontsize=11)
    _cb(fig, ax, im)
    plt.tight_layout()
    return _savefig(fig, "fig_12_validation_heatmap")


# ======================================================================
# Figure 13: Performance bar chart
# ======================================================================
def fig_performance(results: list) -> str:
    """Wall-clock time per configuration."""
    sweeps_names = ["voltage", "sampling", "fp", "thickness", "slice_thickness"]

    fig, axes = plt.subplots(1, len(sweeps_names), figsize=(14, 4))

    for i, sn in enumerate(sweeps_names):
        ax = axes[i]
        vals = sorted(set(r["value"] for r in results if r["sweep"] == sn))
        x = np.arange(len(vals))
        width = 0.25

        for j, algo in enumerate(ALGORITHMS):
            times = []
            for v in vals:
                t_vals = [r.get("time", 0) for r in results
                          if r["sweep"] == sn and r["value"] == v
                          and r["algorithm"] == algo]
                times.append(np.mean(t_vals) if t_vals else 0)
            ax.bar(x + j * width - width, times, width,
                   label=ALGORITHM_LABELS.get(algo, algo),
                   color=ALGORITHM_COLORS.get(algo, "gray"), alpha=0.85)

        ax.set_xticks(x)
        # Format tick labels
        if sn == "voltage":
            tick_labels = [f"{v/1e3:.0f}kV" for v in vals]
        elif sn == "thickness":
            tick_labels = [f"{v}nm" for v in vals]
        elif sn == "fp":
            tick_labels = [str(v) for v in vals]
        elif sn == "sampling":
            tick_labels = [f"{v:.2f}" for v in vals]
        elif sn == "slice_thickness":
            tick_labels = [f"{v:.1f}" for v in vals]
        else:
            tick_labels = [str(v) for v in vals]
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=30)
        ax.set_title(sn, fontsize=10)
        if i == 0:
            ax.set_ylabel("Time (s)")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Computation time by sweep and algorithm", fontsize=14)
    fig.legend([ALGORITHM_LABELS[a] for a in ALGORITHMS],
               loc="lower center", ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    return _savefig(fig, "fig_13_performance")


# ======================================================================
# Generate all figures
# ======================================================================
def generate_all(cache_dir: str, results: list) -> dict:
    """Generate all 13 figures and return {fig_name: base64_data} dict.

    Args:
        cache_dir: Path to cache directory
        results: List of result dicts from SweepEngine

    Returns:
        dict mapping figure name -> base64 PNG data URI
    """
    from ._parameters import SWEEPS

    # Build cache key lookup from results so CBED-dependent figures
    # load arrays using the exact keys used during simulation.
    lookup = _build_result_lookup(results)
    if lookup:
        set_result_lookup(lookup)

    figures = {}
    sweep_map = {s.name: s for s in SWEEPS}

    with _timed("fig_01 voltage CBED grid"):
        figures["fig_01"] = fig_voltage_cbed_grid(cache_dir, sweep_map["voltage"])

    with _timed("fig_02 voltage line profiles"):
        figures["fig_02"] = fig_voltage_line_profiles(cache_dir, sweep_map["voltage"])

    with _timed("fig_03 voltage metrics"):
        figures["fig_03"] = fig_voltage_metrics(results)

    with _timed("fig_04 fp convergence"):
        figures["fig_04"] = fig_fp_convergence(results)

    with _timed("fig_05 fp CBED grid"):
        figures["fig_05"] = fig_fp_cbed_grid(cache_dir, sweep_map["fp"])

    with _timed("fig_06 sampling CBED"):
        figures["fig_06"] = fig_sampling_comparison(cache_dir, sweep_map["sampling"])

    with _timed("fig_07 sampling metrics"):
        figures["fig_07"] = fig_sampling_metrics(results)

    with _timed("fig_08 thickness CBED"):
        figures["fig_08"] = fig_thickness_cbed(cache_dir, sweep_map["thickness"])

    with _timed("fig_09 thickness metrics"):
        figures["fig_09"] = fig_thickness_metrics(results)

    with _timed("fig_10 slice thickness metrics"):
        figures["fig_10"] = fig_slice_thickness_metrics(results)

    with _timed("fig_11 intensity conservation"):
        figures["fig_11"] = fig_intensity_conservation(results)

    with _timed("fig_12 validation heatmap"):
        figures["fig_12"] = fig_validation_heatmap(results)

    with _timed("fig_13 performance"):
        figures["fig_13"] = fig_performance(results)

    return figures


class _timed:
    """Context manager for timing figure generation."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        import time
        self.t0 = time.time()
        return self
    def __exit__(self, *args):
        import time
        dt = time.time() - self.t0
        print(f"  [figure] {self.name} → {dt:.1f}s")
