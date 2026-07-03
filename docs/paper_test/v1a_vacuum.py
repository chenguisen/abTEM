#!/usr/bin/env python3
"""V1a: Vacuum propagation flux conservation test.

Tests that the CVDMS propagator is unitary in free space (V=0).
For V=0 the transmission function (sigma * V / dz) is identically zero.

Uses the low-level _cvdms_forward_scattering to isolate the free-space
propagator from potential construction.

Saves NPZ + JSON to docs/data/v1a_vacuum_{stage}.npz/.json.

Acceptance criteria (paper outline §14.2.1):
  |1 - Σ|ψ|²| < 1e-6  at every slice, t up to 400 Å

Stages:
  A: 128×128 grid — rapid iteration
  B: 627×627 grid — target resolution

Usage:
  python v1a_vacuum.py A
  python v1a_vacuum.py B
"""
import sys, os, json, gc
import numpy as np
import cupy as cp
from abtem.core import config as _cfg
from abtem.core.energy import energy2wavelength

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── GPU / FFT configuration (matching notebook convention) ──
_cfg.set({"device": "gpu", "fft": "cupy"})

# ============================================================
# Parameter manifest (self-documenting)
# ============================================================
ENERGY = 30e3                     # Electron energy [eV]
THICKNESS = 400.0                 # Total propagation distance [Å]
SLICE_THICKNESS = 0.4             # Per-slice thickness [Å]
FLOAT_DTYPE = np.complex64

# CVDMS algorithm parameters (paper outline §6.7 defaults)
MAX_TERMS = 30
MAX_INNER = 30
CONVERGENCE_THRESHOLD = 1e-7
DERIVATIVE_ACCURACY = 8
DIVERGENCE_RATIO = 5.0
CHECK_INTERVAL = 2
ANTIALIAS_INNER = True

# Acceptance threshold
FLUX_TOLERANCE = 1e-6             # |1 - Σ|ψ|²| < 1e-6

# Stage selection via command line:  python v1a_vacuum.py [A|B]
STAGE = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
if STAGE == "B":
    GRID_SIZE = 627
    EXTENT = 31.3
    EXIT_PLANES = 40
else:
    GRID_SIZE = 128
    EXTENT = 20.0
    EXIT_PLANES = 20


# ============================================================
# Helpers
# ============================================================
def total_intensity(wave: np.ndarray) -> float:
    """Σ|ψ|²."""
    return float(cp.sum(cp.abs(wave) ** 2))


def check_flux(wave, step, thickness_A, I0, tol=FLUX_TOLERANCE):
    """Assert per-slice flux conservation."""
    I = total_intensity(wave)
    dev = abs(I0 - I)
    assert dev < tol, (
        f"Flux FAILED slice {step} (t={thickness_A:.1f} Å): "
        f"|1 - Σ|ψ|²| = {dev:.2e} > {tol:.1e}"
    )
    return dev


# ============================================================
# Main
# ============================================================
def main():
    n_slices = int(THICKNESS / SLICE_THICKNESS)
    pixel_size = EXTENT / GRID_SIZE
    wavelength_A = energy2wavelength(ENERGY)

    print(f"=== V1a: Vacuum Flux Conservation ===")
    print(f"Grid: {GRID_SIZE}×{GRID_SIZE}  px={pixel_size:.4f} Å/px")
    print(f"Energy: {ENERGY/1e3:.0f} keV  λ={wavelength_A:.4f} Å")
    print(f"Thickness: {THICKNESS} Å  Δz={SLICE_THICKNESS} Å  slices={n_slices}")
    print(f"Tolerance: |1 - Σ|ψ|²| < {FLUX_TOLERANCE:.1e}")
    print()

    # ── Build Laplacian stencil (same path as cvdms_multislice_step) ──
    from abtem.finite_difference import (
        LaplaceOperator, finite_difference_coefficients, _laplace_operator_stencil,
    )

    laplace = LaplaceOperator(accuracy=DERIVATIVE_ACCURACY)

    # Pre-compute stencil coefficients (used by C++ fused kernel)
    stencil_raw = finite_difference_coefficients(2, DERIVATIVE_ACCURACY).astype(np.float32)
    prefactor = 1.0 / (pixel_size * pixel_size)

    # ── Build initial plane wave: ψ₀(x,y) = 1/√N  (Σ|ψ|² = 1) ──
    wave = cp.ones((GRID_SIZE, GRID_SIZE), dtype=FLOAT_DTYPE)
    wave /= cp.sqrt(total_intensity(wave))
    I0 = total_intensity(wave)
    print(f"Initial Σ|ψ|² = {I0:.12f}")

    # ── Transmission function: V=0 → sigma*V/dz = 0 ──
    # The CVDMS code uses sigma * V / dz as the potential argument.
    # For vacuum, this is identically zero everywhere.
    transmission_function = cp.zeros((GRID_SIZE, GRID_SIZE), dtype=FLOAT_DTYPE)

    # ── Build minimal Waves and get stencil ──
    # _cvdms_forward_scattering expects laplace as a callable.
    # For the C++ backend, stencil_raw + prefactor are used instead.
    # We pass both so "auto" can pick the C++ path when on GPU.
    from abtem.cvdms import _cvdms_forward_scattering

    # Get laplace stencil callable (for Python/CuPy fallback)
    # We pass the stencil as a CuPy array that implements the FD convolution
    from abtem.finite_difference import _laplace_operator_stencil
    laplace_stencil = _laplace_operator_stencil(
        DERIVATIVE_ACCURACY, prefactor, mode="wrap", device="gpu"
    )

    # ── Multislice loop with exit-plane recording ──
    record_interval = max(1, n_slices // EXIT_PLANES)
    depths, ratios = [], []
    max_deviation = 0.0

    for step in range(1, n_slices + 1):
        t = step * SLICE_THICKNESS

        wave, diag = _cvdms_forward_scattering(
            wave,
            transmission_function,
            laplace_stencil,
            wavelength_A,
            SLICE_THICKNESS,
            max_terms=MAX_TERMS,
            max_inner=MAX_INNER,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            divergence_ratio=DIVERGENCE_RATIO,
            return_diagnostics=True,
            check_interval=CHECK_INTERVAL,
            prefactor=prefactor,
            stencil_raw=stencil_raw,
            use_fused_kernel=True,
            antialias_inner=ANTIALIAS_INNER,
            sampling=(pixel_size, pixel_size),
        )

        dev = check_flux(wave, step, t, I0)
        max_deviation = max(max_deviation, dev)

        if step % record_interval == 0 or step == n_slices:
            I_ep = total_intensity(wave)
            depths.append(t)
            ratios.append(I_ep / I0)

        if step % 100 == 0 or step == n_slices:
            terms = diag.get("n_terms_used", "?")
            print(f"  slice {step:4d}/{n_slices}  t={t:6.1f} Å  "
                  f"dev={dev:.2e}  terms={terms}")

    # ── Final assertion ──
    print()
    print(f"Max |1 - Σ|ψ|²| = {max_deviation:.2e}")
    passed = max_deviation < FLUX_TOLERANCE
    assert passed, (
        f"FAIL: max deviation {max_deviation:.2e} >= {FLUX_TOLERANCE:.1e}"
    )
    print(f"PASS: Grid {GRID_SIZE}×{GRID_SIZE} — vacuum flux conserved.")
    print(f"  |1 - Σ|ψ|²|_max = {max_deviation:.2e} < {FLUX_TOLERANCE:.1e}")

    # ── Save NPZ + JSON ──
    base = os.path.join(DATA_DIR, f"v1a_vacuum_{STAGE}")
    np.savez(
        base + ".npz",
        depths=np.array(depths),
        I_ratio=np.array(ratios),
        I0=I0,
        max_deviation=max_deviation,
        passed=passed,
    )
    manifest = {
        "script": "v1a_vacuum.py",
        "stage": STAGE,
        "params": {
            "energy_eV": ENERGY,
            "thickness_A": THICKNESS,
            "slice_thickness_A": SLICE_THICKNESS,
            "num_slices": n_slices,
            "grid": [GRID_SIZE, GRID_SIZE],
            "sampling_A_per_px": pixel_size,
            "max_terms": MAX_TERMS,
            "max_inner": MAX_INNER,
            "convergence_threshold": CONVERGENCE_THRESHOLD,
            "derivative_accuracy": DERIVATIVE_ACCURACY,
            "antialias_inner": ANTIALIAS_INNER,
        },
        "results": {
            "I0": float(I0),
            "max_deviation": max_deviation,
            "passed": passed,
        },
    }
    with open(base + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Data saved: {base}.npz + .json")

    # ── Cleanup ──
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
