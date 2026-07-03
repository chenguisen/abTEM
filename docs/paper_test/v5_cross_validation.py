#!/usr/bin/env python3
"""V5: Implementation cross-validation — Python vs C++ CUDA backend.

Tests that the Python (CuPy) and C++ (CUDA fused kernel) implementations
of CVDMS produce identical results (NCC > 1-1e-6).

Also tests FD vs FFT Laplacian for bandlimited potentials.

Saves NPZ + JSON to docs/data/v5_cross_{stage}.npz/.json.

Acceptance criteria (paper outline §14.2.5):
  Python vs C++ NCC > 1-1e-6
  FD vs FFT Laplacian NCC > 1-1e-6

Usage:
  python v5_cross_validation.py A
"""
import sys, os, json, gc
import numpy as np
import cupy as cp
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice
from abtem.core import config as _cfg

_cfg.set({"device": "gpu", "fft": "cupy"})
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ============================================================
# Parameter manifest
# ============================================================
ENERGY = 30e3
SLICE_THICKNESS = 0.4

STAGE = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
if STAGE == "B":
    SUPERCELL_XY = 4; SUPERCELL_Z = 50; SAMPLING = 0.05
else:
    SUPERCELL_XY = 4; SUPERCELL_Z = 25; SAMPLING = 0.20


def to_cpu(arr):
    if hasattr(arr, "get"): return arr.get()
    return arr


def ncc(a, b):
    a_cpu, b_cpu = to_cpu(a).ravel(), to_cpu(b).ravel()
    num = np.abs(np.sum(a_cpu * np.conj(b_cpu)))
    denom = np.sqrt(np.sum(np.abs(a_cpu)**2) * np.sum(np.abs(b_cpu)**2))
    return float(num / denom) if denom > 1e-30 else 0.0


def phase_rms(a, b):
    a_cpu, b_cpu = to_cpu(a).ravel(), to_cpu(b).ravel()
    amp_a, amp_b = np.abs(a_cpu), np.abs(b_cpu)
    thresh = 1e-6 * max(amp_a.max(), amp_b.max())
    mask = (amp_a > thresh) & (amp_b > thresh)
    if mask.sum() < 10: return 0.0
    a_m, b_m = a_cpu[mask], b_cpu[mask]
    cross = np.sum(a_m * np.conj(b_m))
    if abs(cross) < 1e-30: return 0.0
    global_phase = np.angle(cross)
    phase_local = np.angle(np.exp(1j * (np.angle(a_m * np.conj(b_m)) - global_phase)))
    return float(np.sqrt(np.mean(phase_local**2)))


def main():
    print(f"=== V5: Cross-Validation (Stage {STAGE}) ===")
    a = 3.905
    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[a, a, a, 90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)

    potential = abtem.Potential(
        atoms, sampling=SAMPLING, slice_thickness=SLICE_THICKNESS,
        exit_planes=1, projection="finite",
    )
    print(f"Grid: {potential.gpts}  px={potential.sampling[0]:.4f} Å/px")

    # ── V5a: Python vs C++ backend ──
    # Use _cvdms_forward_scattering directly with backend control
    print(f"\n--- V5a: Python vs C++ backend ---")

    wave = abtem.PlaneWave(energy=ENERGY)
    wave.grid.match(potential)

    # Get the first few slices via the low-level API
    from abtem.cvdms import _cvdms_forward_scattering
    from abtem.finite_difference import (
        LaplaceOperator, finite_difference_coefficients, _laplace_operator_stencil,
    )

    wavelength = wave.wavelength
    sigma = abtem.core.energy.energy2sigma(ENERGY)
    px = potential.sampling[0]
    prefactor = 1.0 / (px * px)
    stencil_raw = finite_difference_coefficients(2, 8).astype(np.float32)
    laplace_stencil = _laplace_operator_stencil(8, prefactor, mode="wrap", device="gpu")

    # First slice potential
    pot_array = potential.build(lazy=False)
    V_proj = cp.asarray(pot_array.array[0])
    transmission = cp.asarray(sigma * V_proj / SLICE_THICKNESS, dtype=np.complex64)

    psi0 = cp.ones((potential.gpts[0], potential.gpts[1]), dtype=np.complex64)
    psi0 /= cp.sqrt(cp.sum(cp.abs(psi0)**2))

    # C++ backend
    psi_cpp, _ = _cvdms_forward_scattering(
        psi0.copy(), transmission, laplace_stencil, wavelength, SLICE_THICKNESS,
        max_terms=30, max_inner=30, convergence_threshold=1e-7,
        divergence_ratio=5.0, return_diagnostics=True, check_interval=2,
        prefactor=prefactor, stencil_raw=stencil_raw,
        use_fused_kernel=True, backend="auto",
        antialias_inner=True, sampling=(px, px),
    )

    # Python (CuPy) backend
    psi_py, _ = _cvdms_forward_scattering(
        psi0.copy(), transmission, laplace_stencil, wavelength, SLICE_THICKNESS,
        max_terms=30, max_inner=30, convergence_threshold=1e-7,
        divergence_ratio=5.0, return_diagnostics=True, check_interval=2,
        prefactor=prefactor, stencil_raw=stencil_raw,
        use_fused_kernel=True, backend="cupy",
        antialias_inner=True, sampling=(px, px),
    )

    n_py_cpp = ncc(psi_py, psi_cpp)
    prms_py_cpp = phase_rms(psi_py, psi_cpp)
    passed_py_cpp = n_py_cpp > 1.0 - 1e-6

    print(f"  Python vs C++ NCC: {n_py_cpp:.8f}  phase_RMS: {prms_py_cpp:.2e}")
    print(f"  {'PASS' if passed_py_cpp else 'FAIL'}: V5a — NCC > 1-1e-6")

    # ── V5b: FD vs FFT Laplacian ──
    print(f"\n--- V5b: FD vs FFT Laplacian ---")

    psi_fd, _ = _cvdms_forward_scattering(
        psi0.copy(), transmission, laplace_stencil, wavelength, SLICE_THICKNESS,
        max_terms=30, max_inner=30, convergence_threshold=1e-7,
        divergence_ratio=5.0, return_diagnostics=True, check_interval=2,
        prefactor=prefactor, stencil_raw=stencil_raw,
        use_fused_kernel=True, backend="auto",
        laplace_method="finite-difference",
        antialias_inner=True, sampling=(px, px),
    )

    psi_fft, _ = _cvdms_forward_scattering(
        psi0.copy(), transmission, laplace_stencil, wavelength, SLICE_THICKNESS,
        max_terms=30, max_inner=30, convergence_threshold=1e-7,
        divergence_ratio=5.0, return_diagnostics=True, check_interval=2,
        prefactor=prefactor, stencil_raw=stencil_raw,
        use_fused_kernel=True, backend="auto",
        laplace_method="fft",
        antialias_inner=True, sampling=(px, px),
    )

    n_fd_fft = ncc(psi_fd, psi_fft)
    prms_fd_fft = phase_rms(psi_fd, psi_fft)
    passed_fd_fft = n_fd_fft > 1.0 - 1e-6

    print(f"  FD vs FFT NCC: {n_fd_fft:.8f}  phase_RMS: {prms_fd_fft:.2e}")
    print(f"  {'PASS' if passed_fd_fft else 'FAIL'}: V5b — NCC > 1-1e-6")

    # ── Save ──
    base = os.path.join(DATA_DIR, f"v5_cross_{STAGE}")
    np.savez(base + ".npz",
             ncc_py_cpp=n_py_cpp, phase_rms_py_cpp=prms_py_cpp,
             ncc_fd_fft=n_fd_fft, phase_rms_fd_fft=prms_fd_fft,
             passed_py_cpp=passed_py_cpp, passed_fd_fft=passed_fd_fft)
    with open(base + ".json", "w") as f:
        json.dump({"script": "v5_cross_validation.py", "stage": STAGE,
                   "params": {"energy_eV": ENERGY,
                              "grid": list(potential.gpts),
                              "sampling_A_per_px": SAMPLING,
                              "slice_thickness_A": SLICE_THICKNESS},
                   "results": {
                       "ncc_py_vs_cpp": float(n_py_cpp),
                       "phase_rms_py_vs_cpp": float(prms_py_cpp),
                       "passed_py_cpp": bool(passed_py_cpp),
                       "ncc_fd_vs_fft": float(n_fd_fft),
                       "phase_rms_fd_vs_fft": float(prms_fd_fft),
                       "passed_fd_fft": bool(passed_fd_fft),
                   }}, f, indent=2)
    print(f"Data saved: {base}.npz + .json")

    # Summary
    print(f"\n{'='*60}")
    print(f"V5 Summary:")
    print(f"  Python vs C++:  {'PASS' if passed_py_cpp else 'FAIL'}  NCC={n_py_cpp:.8f}")
    print(f"  FD vs FFT:      {'PASS' if passed_fd_fft else 'FAIL'}  NCC={n_fd_fft:.8f}")


if __name__ == "__main__":
    main()
