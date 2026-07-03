#!/usr/bin/env python3
"""P4: Diagnostic–potential spatial correlation and IPR.

SrTiO3 [001] at 30 keV, probe on Sr column.
Co-registers BSC residual with |∇V| and ∇²V.
Computes IPR(t) for wave localisation measure.

Outputs: docs/data/p4_correlation.npz + .json
"""

import sys, os, json, gc
import numpy as np
import cupy as cp
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice
from abtem.core import config as _cfg
from abtem.core.energy import energy2wavelength, energy2sigma

_cfg.set({"device": "gpu", "fft": "cupy"})
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

ENERGY = 30e3
A_SRTIO3 = 3.905
SUPERCELL_XY, SUPERCELL_Z = 4, 100   # t ≈ 390 Å (approximate 400 Å target)
SAMPLING = 0.05                       # Å/px — fine for gradient analysis
DZ = 0.4
THICKNESS_TARGET = 400                # Å — target for BSC residual
CONVERGENCE_THRESHOLD = 1e-7


def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def pearson_r(a, b):
    """Pearson correlation coefficient between two 2D arrays."""
    a_flat = a.ravel()
    b_flat = b.ravel()
    return np.corrcoef(a_flat, b_flat)[0, 1]


def compute_ipr(wave_2d):
    """Inverse participation ratio: Σ|ψ|⁴ / (Σ|ψ|²)²."""
    intensity = np.abs(wave_2d)**2
    return float((intensity**2).sum() / intensity.sum()**2)


def compute_gradient_fields(V_2d, px_A):
    """Compute |∇V| and ∇²V of projected potential."""
    gy, gx = np.gradient(V_2d, px_A)
    grad_mag = np.sqrt(gx**2 + gy**2)
    lap = np.gradient(gx, px_A, axis=1) + np.gradient(gy, px_A, axis=0)
    return grad_mag, lap


def main():
    print("=== P4: Diagnostic–Potential Correlation ===")

    # ── Build SrTiO3 ──
    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[A_SRTIO3, A_SRTIO3, A_SRTIO3, 90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)
    total_z = atoms.cell[2, 2]
    print(f"Thickness: {total_z:.1f} Å (target ~{THICKNESS_TARGET} Å)")

    # ── Potential fields ──
    pot_static = abtem.Potential(
        atoms, sampling=SAMPLING, slice_thickness=A_SRTIO3,
        exit_planes=1, projection="finite",
    )
    V_data = pot_static.build(lazy=False)
    V_2d = to_cpu(V_data.array[0])
    px_A = pot_static.sampling[0]
    gpts = pot_static.gpts
    print(f"Grid: {gpts}, px={px_A:.4f} Å/px")

    # Compute gradient fields
    grad_mag, lap = compute_gradient_fields(V_2d, px_A)
    V_peak = float(V_2d.max())
    grad_max = float(grad_mag.max())
    lap_peak_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
    lap_at_col = float(lap[lap_peak_idx])
    print(f"V_peak={V_peak:.1f} eV·Å, |∇V|_max={grad_max:.2e}, ∇²V|_col={lap_at_col:.2e}")

    # ── Probe on Sr column ──
    sr_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
    sr_pos_A = (float(sr_idx[1] * px_A), float(sr_idx[0] * px_A))
    print(f"Sr at pixel {sr_idx}, pos ({sr_pos_A[0]:.1f}, {sr_pos_A[1]:.1f}) Å")

    probe = abtem.Probe(
        energy=ENERGY, semiangle_cutoff=10,
        extent=SUPERCELL_XY * A_SRTIO3, gpts=gpts[0],
        device="gpu",
    )
    probe.grid.match(pot_static)
    probe.positions = [(sr_pos_A[0], sr_pos_A[1])]

    n_slices = int(total_z / DZ)
    t400_slice = int(THICKNESS_TARGET / DZ)

    potential = abtem.Potential(
        atoms, sampling=SAMPLING, slice_thickness=DZ,
        exit_planes=1, projection="finite",
    )

    # ── CVDMS with BSC (full operator) ──
    print("CVDMS + BSC...")
    cvdms_bsc = CVDMSMultislice(
        convergence_threshold=CONVERGENCE_THRESHOLD,
        backscattering=True, antialias=True, antialias_inner=True,
    )
    ew_bsc = probe.multislice(potential, algorithm=cvdms_bsc, lazy=False)
    arr_bsc = to_cpu(ew_bsc.array)
    print(f"  BSC wave shape: {arr_bsc.shape}")

    if arr_bsc.ndim == 4:
        arr_bsc = arr_bsc[:, 0, :, :]
    n_planes = len(arr_bsc)

    # IPR vs depth
    z_arr = np.arange(n_planes) * DZ
    ipr_arr = np.array([compute_ipr(arr_bsc[i]) for i in range(n_planes)])
    # Normalize IPR: IPR * area → scale-invariant, ~1 for uniform, >>1 for localized
    area_px = gpts[0] * gpts[1]
    ipr_norm = ipr_arr * area_px
    print(f"  IPR range: {ipr_norm.min():.3f} – {ipr_norm.max():.3f}")

    # BSC residual at t ≈ 400 Å
    idx400 = min(t400_slice - 1, n_planes - 1)
    psi_bsc = arr_bsc[idx400]
    I_bsc = np.abs(psi_bsc)**2

    del ew_bsc, cvdms_bsc
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── CVDMS without BSC (forward only) ──
    print("CVDMS forward-only...")
    cvdms_fwd = CVDMSMultislice(
        convergence_threshold=CONVERGENCE_THRESHOLD,
        backscattering=False, antialias=True, antialias_inner=True,
    )
    ew_fwd = probe.multislice(potential, algorithm=cvdms_fwd, lazy=False)
    arr_fwd = to_cpu(ew_fwd.array)
    print(f"  Forward wave shape: {arr_fwd.shape}")

    if arr_fwd.ndim == 4:
        arr_fwd = arr_fwd[:, 0, :, :]

    psi_fwd = arr_fwd[idx400]
    I_fwd = np.abs(psi_fwd)**2

    # IPR for forward
    ipr_fwd_arr = np.array([compute_ipr(arr_fwd[i]) for i in range(min(n_planes, len(arr_fwd)))])
    ipr_fwd_norm = ipr_fwd_arr * area_px

    del ew_fwd, cvdms_fwd, potential
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── BSC residual analysis ──
    bsc_residual = np.abs(I_bsc - I_fwd)
    bsc_rel = bsc_residual / (I_fwd + 1e-30)
    bsc_rel = np.where(I_fwd > I_fwd.max() * 1e-6, bsc_rel, 0)

    # Pearson correlations
    r_V_bsc = pearson_r(V_2d, bsc_residual)
    r_grad_bsc = pearson_r(grad_mag, bsc_residual)
    r_lap_bsc = pearson_r(lap, bsc_residual)
    r_V_grad = pearson_r(V_2d, grad_mag)

    print(f"\n{'='*60}")
    print("P4 Pearson correlations (BSC residual vs potential fields):")
    print(f"  V   vs BSC residual: r = {r_V_bsc:.4f}")
    print(f"  |∇V| vs BSC residual: r = {r_grad_bsc:.4f}")
    print(f"  ∇²V vs BSC residual: r = {r_lap_bsc:.4f}")
    print(f"  V vs |∇V|: r = {r_V_grad:.4f}")
    print(f"  max BSC residual: {bsc_residual.max():.2e}")
    print(f"  max relative BSC residual: {bsc_rel.max():.2e}")

    # IPR summary
    print(f"\nIPR (normalized):")
    print(f"  BSC:     min={ipr_norm.min():.3f}, max={ipr_norm.max():.3f}, mean={ipr_norm.mean():.3f}")
    print(f"  Forward: min={ipr_fwd_norm.min():.3f}, max={ipr_fwd_norm.max():.3f}, mean={ipr_fwd_norm.mean():.3f}")

    # ── Save ──
    base = os.path.join(DATA_DIR, "p4_correlation")
    np.savez(base + ".npz",
             V_2d=V_2d, grad_mag=grad_mag, lap=lap,
             I_bsc=I_bsc, I_fwd=I_fwd, bsc_residual=bsc_residual,
             z_IPR=z_arr, IPR_bsc=ipr_norm, IPR_fwd=ipr_fwd_norm,
             px_A=px_A, sr_idx=np.array(sr_idx))
    with open(base + ".json", "w") as f:
        json.dump({
            "script": "p4_correlation.py",
            "params": {"energy_eV": ENERGY, "sampling_A_per_px": SAMPLING,
                       "dz_A": DZ, "thickness_A": float(total_z),
                       "supercell_xy": SUPERCELL_XY, "supercell_z": SUPERCELL_Z,
                       "thickness_target_A": THICKNESS_TARGET,
                       "t400_slice": idx400},
            "results": {
                "pearson_r_V_bsc": float(r_V_bsc),
                "pearson_r_grad_bsc": float(r_grad_bsc),
                "pearson_r_lap_bsc": float(r_lap_bsc),
                "pearson_r_V_grad": float(r_V_grad),
                "bsc_residual_max": float(bsc_residual.max()),
                "bsc_rel_max": float(bsc_rel.max()),
                "IPR_bsc_mean": float(ipr_norm.mean()),
                "IPR_fwd_mean": float(ipr_fwd_norm.mean()),
                "V_peak_eVA": float(V_peak),
                "grad_max": float(grad_max),
                "lap_at_col": float(lap_at_col),
            }
        }, f, indent=2)
    print(f"\nData saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
