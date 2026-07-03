#!/usr/bin/env python3
"""P3: HOLZ FOLZ ring Δz sweep — CVDMS vs Fourier with convergent probe.

Si [001] at 100 keV, convergent probe (10 mrad) on Si column.
Δz sweep: {0.4, 0.8, 1.0} Å — compares FOLZ ring position and intensity.
The ring position is purely geometric (Ewald sphere); the intensity/depth
of HOLZ features tests commutator sensitivity.

Outputs: docs/data/p3_holz.npz + .json
"""

import sys, os, json, gc
import numpy as np
import cupy as cp
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice, FourierMultislice
from abtem.core import config as _cfg
from abtem.core.energy import energy2wavelength

_cfg.set({"device": "gpu", "fft": "cupy"})
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

ENERGY = 100e3
A_SI = 5.431
C_STAR = 1.0 / A_SI              # 0.1841 Å⁻¹ for [001]
SUPERCELL_XY, SUPERCELL_Z = 4, 50   # t ≈ 272 Å
SAMPLING = 0.04                     # Å/px
CONVERGENCE_THRESHOLD = 1e-7
PROBE_SEMIANGLE = 10                # mrad
DZ_SWEEP = [0.4, 0.8, 1.0]         # Δz in Å
EXIT_STRIDE_FOR_DZ = {0.4: 1, 0.8: 1, 1.0: 1}


def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def radial_profile(intensity_2d, center=None):
    """Compute radial average of a 2D intensity pattern."""
    ny, nx = intensity_2d.shape
    if center is None:
        center = (ny // 2, nx // 2)
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_max = int(r.max()) + 1
    radial = np.zeros(r_max, dtype=np.float64)
    for i in range(r_max):
        mask = (r >= i - 0.5) & (r < i + 0.5)
        if mask.any():
            radial[i] = intensity_2d[mask].mean()
    return np.arange(r_max), radial


def find_folz_ring(q_A, radial_I, k0, c_star, delta_q=1.5):
    """Find FOLZ ring peak in radial profile near analytic g_FOLZ."""
    g_folz_analytic = np.sqrt(2 * k0 * c_star - c_star**2)
    mask = (q_A >= g_folz_analytic - delta_q) & (q_A <= g_folz_analytic + delta_q)
    if not mask.any():
        return g_folz_analytic, 0.0
    window = radial_I[mask]
    q_window = q_A[mask]
    peak_idx = np.argmax(window)
    return float(q_window[peak_idx]), float(window[peak_idx])


def folz_contrast(q_A, radial_I, g_folz, delta_q=0.5):
    """Measure FOLZ ring peak-to-background contrast ratio."""
    inner = (q_A >= g_folz - 1.5) & (q_A <= g_folz - delta_q)
    outer = (q_A >= g_folz + delta_q) & (q_A <= g_folz + 1.5)
    peak_mask = (q_A >= g_folz - delta_q) & (q_A <= g_folz + delta_q)
    bg = (radial_I[inner].mean() + radial_I[outer].mean()) / 2 if inner.any() and outer.any() else 1e-30
    peak = radial_I[peak_mask].max() if peak_mask.any() else 0
    return float(peak / (bg + 1e-40))


def compute_diffraction(psi_exit):
    """Compute diffraction pattern |FFT[ψ]|² from exit wave."""
    if psi_exit.ndim == 3:
        psi = psi_exit[-1]
    elif psi_exit.ndim == 4:
        psi = psi_exit[-1, 0]
    else:
        psi = psi_exit
    dp = np.abs(np.fft.fft2(psi))**2
    return np.fft.fftshift(dp)


def run_method(probe, potential, algorithm, dz_label):
    """Run a multislice method and return exit wave diffraction pattern."""
    ew = probe.multislice(potential, algorithm=algorithm, lazy=False)
    arr = to_cpu(ew.array)
    dp = compute_diffraction(arr)
    del ew
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    return dp


def main():
    print("=== P3: HOLZ FOLZ Ring Δz Sweep (Convergent Probe) ===")

    wavelength = energy2wavelength(ENERGY)
    k0 = 1.0 / wavelength
    g_folz = np.sqrt(2 * k0 * C_STAR - C_STAR**2)
    theta_folz = np.arcsin(g_folz / k0) * 1000
    print(f"Si [001]: a={A_SI:.3f} Å, c*={C_STAR:.4f} Å⁻¹")
    print(f"100 keV: λ={wavelength:.4f} Å, k₀={k0:.2f} Å⁻¹")
    print(f"FOLZ ring analytic: g={g_folz:.3f} Å⁻¹, θ={theta_folz:.1f} mrad")
    print(f"Sampling: {SAMPLING} Å/px → q_Nyquist={1/(2*SAMPLING):.1f} Å⁻¹")

    # ── Build Si [001] crystal ──
    atoms = crystal(
        ["Si"],
        basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
        spacegroup=227,
        cellpar=[A_SI, A_SI, A_SI, 90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)
    total_z = atoms.cell[2, 2]
    print(f"Thickness: {total_z:.1f} Å")

    # ── Reference potential for grid and column finding ──
    pot_ref = abtem.Potential(atoms, sampling=SAMPLING, slice_thickness=A_SI,
                              exit_planes=1, projection="finite")
    V_data = pot_ref.build(lazy=False)
    V_2d = to_cpu(V_data.array[0])
    gpts = pot_ref.gpts
    px_A = pot_ref.sampling[0]
    extent = pot_ref.extent
    print(f"Grid: {gpts}, px={px_A:.4f} Å/px, extent={extent[0]:.1f}×{extent[1]:.1f} Å")

    # Find Si column position
    si_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
    si_pos_A = (float(si_idx[1] * px_A), float(si_idx[0] * px_A))
    print(f"Si column at pixel {si_idx}, pos ({si_pos_A[0]:.1f}, {si_pos_A[1]:.1f}) Å")

    del V_data
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Convergent probe on Si column ──
    probe = abtem.Probe(
        energy=ENERGY, semiangle_cutoff=PROBE_SEMIANGLE,
        extent=SUPERCELL_XY * A_SI, gpts=gpts[0],
        device="gpu",
    )
    probe.grid.match(pot_ref)
    probe.positions = [(si_pos_A[0], si_pos_A[1])]

    # ── Δz sweep ──
    all_results = {}
    dq = None

    print(f"\n{'='*60}")
    print(f"Δz sweep: {DZ_SWEEP}")
    print(f"{'='*60}")

    for dz in DZ_SWEEP:
        n_slices = int(total_z / dz)
        print(f"\n--- Δz = {dz:.1f} Å  (slices={n_slices}) ---")

        potential = abtem.Potential(
            atoms, sampling=SAMPLING, slice_thickness=dz,
            exit_planes=1, projection="finite",
        )

        # CVDMS
        print("  CVDMS...")
        cvdms = CVDMSMultislice(
            convergence_threshold=CONVERGENCE_THRESHOLD,
            backscattering=False, antialias=True, antialias_inner=True,
        )
        dp_c = run_method(probe, potential, cvdms, f"CVDMS_dz{dz}")

        # Fourier
        print("  Fourier...")
        fourier = FourierMultislice(order=1)
        dp_f = run_method(probe, potential, fourier, f"Fourier_dz{dz}")

        if dq is None:
            ny, nx = dp_c.shape
            dq = 1.0 / (nx * px_A)

        # Radial profiles
        r_px, radial_c = radial_profile(dp_c)
        r_px2, radial_f = radial_profile(dp_f)
        q_axis = r_px * dq

        # FOLZ ring position
        g_meas_c, peak_c = find_folz_ring(q_axis, radial_c, k0, C_STAR)
        g_meas_f, peak_f = find_folz_ring(q_axis, radial_f, k0, C_STAR)
        err_c = abs(g_meas_c - g_folz) / g_folz * 100
        err_f = abs(g_meas_f - g_folz) / g_folz * 100

        # FOLZ ring contrast
        contrast_c = folz_contrast(q_axis, radial_c, g_folz)
        contrast_f = folz_contrast(q_axis, radial_f, g_folz)

        # DP NCC
        dp_c_flat = dp_c.ravel()
        dp_f_flat = dp_f.ravel()
        ncc = np.corrcoef(dp_c_flat, dp_f_flat)[0, 1]

        print(f"  FOLZ: analytic={g_folz:.3f} Å⁻¹")
        print(f"    CVDMS:   g={g_meas_c:.3f} ({err_c:.2f}%), contrast={contrast_c:.3f}")
        print(f"    Fourier:  g={g_meas_f:.3f} ({err_f:.2f}%), contrast={contrast_f:.3f}")
        print(f"    DP NCC (C−F) = {ncc:.6f}")

        all_results[f"dz_{dz}"] = {
            "dz_A": dz,
            "g_folz_cvdms": float(g_meas_c),
            "g_folz_fourier": float(g_meas_f),
            "err_cvdms_pct": float(err_c),
            "err_fourier_pct": float(err_f),
            "contrast_cvdms": float(contrast_c),
            "contrast_fourier": float(contrast_f),
            "dp_ncc": float(ncc),
            "q_axis": q_axis.tolist(),
            "radial_cvdms": radial_c.tolist(),
            "radial_fourier": radial_f.tolist(),
        }

        # Save DPs only for key Δz values to limit file size
        if dz in [0.4, 0.8]:
            all_results[f"dz_{dz}"]["dp_cvdms"] = dp_c
            all_results[f"dz_{dz}"]["dp_fourier"] = dp_f

        del potential, dp_c, dp_f
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("P3 Δz Sweep Summary (Convergent Probe CBED):")
    print(f"  Analytic g_FOLZ = {g_folz:.3f} Å⁻¹")
    print(f"  {'Δz':>6s}  {'g_CVDMS':>9s}  {'g_Fourier':>9s}  {'Δg(C−F)':>10s}  {'contrast_C':>10s}  {'contrast_F':>10s}  {'NCC':>8s}")
    for dz in DZ_SWEEP:
        r = all_results[f"dz_{dz}"]
        dg = abs(r["g_folz_cvdms"] - r["g_folz_fourier"])
        print(f"  {dz:>5.1f} Å  {r['g_folz_cvdms']:>9.3f}  {r['g_folz_fourier']:>9.3f}  "
              f"{dg:>10.4f}  {r['contrast_cvdms']:>10.3f}  {r['contrast_fourier']:>10.3f}  {r['dp_ncc']:>8.6f}")

    # ── Save ──
    base = os.path.join(DATA_DIR, "p3_holz")
    save_kwargs = {
        "g_folz_analytic": g_folz,
        "dq_A_per_px": dq,
        "k0": k0,
        "c_star": C_STAR,
        "si_idx": np.array(si_idx),
    }
    for dz in DZ_SWEEP:
        r = all_results[f"dz_{dz}"]
        save_kwargs[f"q_axis_dz{dz}"] = np.array(r["q_axis"])
        save_kwargs[f"radial_c_dz{dz}"] = np.array(r["radial_cvdms"])
        save_kwargs[f"radial_f_dz{dz}"] = np.array(r["radial_fourier"])
        save_kwargs[f"g_c_dz{dz}"] = r["g_folz_cvdms"]
        save_kwargs[f"g_f_dz{dz}"] = r["g_folz_fourier"]
        if "dp_cvdms" in r:
            save_kwargs[f"dp_c_dz{dz}"] = r["dp_cvdms"]
            save_kwargs[f"dp_f_dz{dz}"] = r["dp_fourier"]

    np.savez(base + ".npz", **save_kwargs)

    # Strip numpy arrays from sweep before JSON serialization
    json_sweep = {}
    for dz in DZ_SWEEP:
        r = all_results[f"dz_{dz}"]
        json_sweep[f"dz_{dz}"] = {k: v for k, v in r.items()
                                  if not isinstance(v, np.ndarray)}

    with open(base + ".json", "w") as f:
        json.dump({
            "script": "p3_holz.py",
            "params": {"energy_eV": ENERGY, "sampling_A_per_px": SAMPLING,
                       "thickness_A": float(total_z),
                       "supercell_xy": SUPERCELL_XY, "supercell_z": SUPERCELL_Z,
                       "dz_sweep": DZ_SWEEP, "probe_semiangle_mrad": PROBE_SEMIANGLE,
                       "material": "Si", "orientation": "[001]",
                       "a_A": float(A_SI), "c_star_per_A": float(C_STAR)},
            "results": {"g_FOLZ_analytic_A": float(g_folz)},
            "sweep": json_sweep,
        }, f, indent=2)
    print(f"\nData saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
