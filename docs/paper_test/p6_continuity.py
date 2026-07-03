#!/usr/bin/env python3
"""P6: Continuity equation verification — ∂z|ψ|² + ∇⊥·j⊥ = 0.

SrTiO3 [001] at 30 keV, plane-wave illumination.
Runs CVDMS with exit_planes=1 (every slice) to capture ψ at all z-planes.
Computes the continuity residual field and correlates it with the
per-pixel diagnostics (r(R), V(R)) to verify that pixels violating
continuity coincide with the divergent regime.

Outputs: docs/data/p6_continuity.npz + .json
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
A = 3.905
SUPERCELL_XY = 4
SUPERCELL_Z = 25                    # t ≈ 97.6 Å — enough to observe divergence
SAMPLING = 0.10                     # Å/px
DZ = 0.4
CONVERGENCE_THRESHOLD = 1e-7


def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def compute_current(psi, px_A, wavelength, k0):
    """Compute transverse probability current j⊥ = ℑ[ψ*∇⊥ψ] / (2πK₀).

    Returns jx, jy (both real).
    """
    # Gradient via central differences
    gy, gx = np.gradient(psi, px_A)
    jx = np.imag(np.conj(psi) * gx) / (2 * np.pi * k0)
    jy = np.imag(np.conj(psi) * gy) / (2 * np.pi * k0)
    return jx, jy


def compute_divergence(jx, jy, px_A):
    """Compute ∇⊥·j = ∂x jx + ∂y jy."""
    djx_dx = np.gradient(jx, px_A, axis=1)
    djy_dy = np.gradient(jy, px_A, axis=0)
    return djx_dx + djy_dy


def pearson_r(a, b):
    """Pearson correlation coefficient."""
    a_f = a.ravel()
    b_f = b.ravel()
    return float(np.corrcoef(a_f, b_f)[0, 1])


def ncc(a, b):
    a_f = a.ravel(); b_f = b.ravel()
    return float(np.abs(np.dot(a_f.conj(), b_f)) /
                 np.sqrt(np.dot(a_f.conj(), a_f).real * np.dot(b_f.conj(), b_f).real))


def main():
    print("=== P6: Continuity Equation Verification ===")
    wavelength = energy2wavelength(ENERGY)
    k0 = 1.0 / wavelength
    sigma = energy2sigma(ENERGY)

    # ── Build SrTiO3 ──
    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[A, A, A, 90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)
    total_z = atoms.cell[2, 2]
    n_slices = int(total_z / DZ)
    print(f"Thickness: {total_z:.1f} Å  slices: {n_slices}  Δz={DZ} Å")
    print(f"λ={wavelength:.4f} Å  k₀={k0:.1f} Å⁻¹")

    # ── Build potential for static V reference ──
    pot_static = abtem.Potential(
        atoms, sampling=SAMPLING, slice_thickness=A,
        exit_planes=1, projection="finite",
    )
    V_data = pot_static.build(lazy=False)
    V_2d = to_cpu(V_data.array[0])
    gpts = pot_static.gpts
    px_A = pot_static.sampling[0]
    print(f"Grid: {gpts}  px={px_A:.4f} Å/px")

    del V_data
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Run CVDMS with exit_planes=1 (capture every slice) ──
    potential = abtem.Potential(
        atoms, sampling=SAMPLING, slice_thickness=DZ,
        exit_planes=1, projection="finite",
    )

    # Use plane wave for simplicity — continuity should hold for any illumination
    wave = abtem.PlaneWave(energy=ENERGY)
    wave.grid.match(potential)

    print("Running CVDMS with exit_planes=1 (full z-resolved)...")
    cvdms = CVDMSMultislice(
        convergence_threshold=CONVERGENCE_THRESHOLD,
        backscattering=False, antialias=True, antialias_inner=True,
    )
    ew = wave.multislice(potential, algorithm=cvdms, lazy=False)
    arr = to_cpu(ew.array)
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]
    elif arr.ndim == 3:
        pass  # [n_slices, ny, nx]
    print(f"Wave array shape: {arr.shape}")

    del ew, cvdms
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    n_planes = len(arr)
    z_arr = np.arange(n_planes) * DZ

    # ── Compute continuity residual for each adjacent pair ──
    continuity_residual = []
    dz_intensity = []
    div_j = []

    for i in range(n_planes - 1):
        psi_cur = arr[i]
        psi_next = arr[i + 1]

        # ∂z|ψ|² via forward difference
        I_cur = np.abs(psi_cur)**2
        I_next = np.abs(psi_next)**2
        dI_dz = (I_next - I_cur) / DZ

        # ∇⊥·j⊥ at the midpoint (use average ψ or current ψ)
        jx, jy = compute_current(psi_cur, px_A, wavelength, k0)
        div_j_cur = compute_divergence(jx, jy, px_A)

        # Continuity residual: ∂z|ψ|² + ∇⊥·j⊥
        residual = dI_dz + div_j_cur
        continuity_residual.append(residual)
        dz_intensity.append(dI_dz)
        div_j.append(div_j_cur)

    # Stack into arrays
    cont_res = np.stack(continuity_residual, axis=0)  # [n_planes-1, ny, nx]
    dz_I = np.stack(dz_intensity, axis=0)
    div_j_arr = np.stack(div_j, axis=0)

    # ── Statistics ──
    abs_res = np.abs(cont_res)
    max_res = float(abs_res.max())
    mean_res = float(abs_res.mean())
    median_res = float(np.median(abs_res))

    # Fraction of pixels exceeding thresholds
    total_px = abs_res[0].size
    frac_1e4 = float((abs_res.max(axis=0) > 1e-4).sum() / total_px)
    frac_1e6 = float((abs_res.max(axis=0) > 1e-6).sum() / total_px)
    frac_1e8 = float((abs_res.max(axis=0) > 1e-8).sum() / total_px)

    print(f"\n{'='*60}")
    print("P6 Continuity Equation Results:")
    print(f"  |∂z|ψ|² + ∇⊥·j⊥| max  = {max_res:.2e}")
    print(f"  Mean residual            = {mean_res:.2e}")
    print(f"  Median residual          = {median_res:.2e}")
    print(f"  Pixels > 10⁻⁴            = {frac_1e4*100:.2f}%")
    print(f"  Pixels > 10⁻⁶            = {frac_1e6*100:.2f}%")
    print(f"  Pixels > 10⁻⁸            = {frac_1e8*100:.2f}%")

    # ── Correlation with |V| field ──
    # Use the last-plane residual for correlation with V
    last_residual = cont_res[-1]
    # Interpolate V_2d to match residual grid if needed
    r_V_cont = pearson_r(V_2d, last_residual)
    r_V_cont_abs = pearson_r(V_2d, np.abs(last_residual))
    print(f"  Pearson r(V, continuity residual)         = {r_V_cont:.4f}")
    print(f"  Pearson r(V, |continuity residual|)       = {r_V_cont_abs:.4f}")

    # ── Correlation with pixel-wise max residual ──
    max_per_pixel = abs_res.max(axis=0)
    r_V_maxres = pearson_r(V_2d, max_per_pixel)
    print(f"  Pearson r(V, max|residual| per pixel)     = {r_V_maxres:.4f}")

    # ── Save ──
    base = os.path.join(DATA_DIR, "p6_continuity")
    np.savez(base + ".npz",
             V_2d=V_2d,
             cont_residual_last=last_residual,
             max_residual_per_pixel=max_per_pixel,
             z=z_arr,
             px_A=px_A,
             frac_above_1e4=frac_1e4,
             frac_above_1e6=frac_1e6,
             frac_above_1e8=frac_1e8)

    with open(base + ".json", "w") as f:
        json.dump({
            "script": "p6_continuity.py",
            "params": {
                "energy_eV": ENERGY, "sampling_A_per_px": SAMPLING,
                "dz_A": DZ, "supercell_xy": SUPERCELL_XY,
                "supercell_z": SUPERCELL_Z,
                "thickness_A": float(total_z), "n_slices": n_slices,
                "gpts": list(gpts),
            },
            "results": {
                "max_residual": float(max_res),
                "mean_residual": float(mean_res),
                "median_residual": float(median_res),
                "frac_px_above_1e-4": float(frac_1e4),
                "frac_px_above_1e-6": float(frac_1e6),
                "frac_px_above_1e-8": float(frac_1e8),
                "pearson_r_V_residual": float(r_V_cont),
                "pearson_r_V_abs_residual": float(r_V_cont_abs),
                "pearson_r_V_max_residual": float(r_V_maxres),
            }
        }, f, indent=2)
    print(f"\nData saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
