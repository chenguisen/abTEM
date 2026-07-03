#!/usr/bin/env python3
"""C2: Thickness-dependent accuracy sweep — CVDMS vs Fourier agreement mapping.

SrTiO3 [001] @ 30 keV, probe on Sr column, Δz = 0.4 Å fixed.
Sweeps thickness t ∈ [20, 625] Å via supercell_z ∈ {5, 10, 20, 40, 80, 160}.
Measures NCC, phase RMS, on-column |ψ|² correlation, and DP NCC vs t.

Outputs: docs/data/c2_thickness.npz + .json
"""

import sys, os, json, gc
import numpy as np
import cupy as cp
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice, FourierMultislice
from abtem.core import config as _cfg
from abtem.core.energy import energy2wavelength, energy2sigma

_cfg.set({"device": "gpu", "fft": "cupy"})
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

ENERGY = 30e3
A = 3.905
SUPERCELL_XY = 4
SAMPLING = 0.10                       # Å/px
DZ = 0.4
SUPERCELL_Z_SWEEP = [5, 10, 20, 40, 80, 160]
CONVERGENCE_THRESHOLD = 1e-7


def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def ncc(a, b):
    """Normalized cross-correlation of two complex arrays."""
    a_f = a.ravel(); b_f = b.ravel()
    return float(np.abs(np.dot(a_f.conj(), b_f)) /
                 np.sqrt(np.dot(a_f.conj(), a_f).real * np.dot(b_f.conj(), b_f).real))


def phase_rms(a, b):
    """RMS phase difference between two complex wave functions (radians)."""
    phase_diff = np.angle(a * np.conj(b))
    return float(np.sqrt(np.mean(phase_diff**2)))


def amp_rms_error(a, b):
    """RMS relative amplitude error."""
    amp_a = np.abs(a); amp_b = np.abs(b)
    return float(np.sqrt(np.mean(((amp_a - amp_b) / (amp_a.mean() + 1e-30))**2)))


def compute_dp(psi):
    """Compute diffraction pattern from exit wave."""
    dp = np.abs(np.fft.fft2(psi))**2
    return np.fft.fftshift(dp)


def main():
    print("=== C2: Thickness-dependent Accuracy Sweep ===")
    print(f"SrTiO3 [001] @ {ENERGY/1000:.0f} keV, Δz={DZ} Å, "
          f"Δx={SAMPLING} Å/px, probe 10 mrad on Sr")

    # ── Reference: analytic ξ_ch ──
    sigma = energy2sigma(ENERGY)
    atoms_ref = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[A, A, A, 90, 90, 90],
    )
    atoms_ref *= (SUPERCELL_XY, SUPERCELL_XY, 1)
    ref_pot = abtem.Potential(atoms_ref, sampling=SAMPLING, slice_thickness=A,
                               exit_planes=1, projection="finite")
    V_data = ref_pot.build(lazy=False)
    V_2d = to_cpu(V_data.array[0])
    gpts = ref_pot.gpts
    px_A = ref_pot.sampling[0]

    # Find Sr column
    sr_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
    sr_pos_A = (float(sr_idx[1] * px_A), float(sr_idx[0] * px_A))
    print(f"Sr at pixel {sr_idx}, pos ({sr_pos_A[0]:.1f}, {sr_pos_A[1]:.1f}) Å")
    print(f"Grid: {gpts}")

    del V_data, ref_pot, atoms_ref
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Thickness sweep ──
    all_results = {}
    print(f"\n{'='*60}")
    print(f"Thickness sweep: SUPERCELL_Z = {SUPERCELL_Z_SWEEP}")
    print(f"{'='*60}")

    for sc_z in SUPERCELL_Z_SWEEP:
        atoms = crystal(
            ["Sr", "Ti", "O"],
            basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
            spacegroup=221, cellpar=[A, A, A, 90, 90, 90],
        )
        atoms *= (SUPERCELL_XY, SUPERCELL_XY, sc_z)
        total_z = atoms.cell[2, 2]
        n_slices = int(total_z / DZ)
        print(f"\n--- t = {total_z:.1f} Å  (sc_z={sc_z}, slices={n_slices}) ---")

        probe = abtem.Probe(
            energy=ENERGY, semiangle_cutoff=10,
            extent=SUPERCELL_XY * A, gpts=gpts[0],
            device="gpu",
        )
        potential = abtem.Potential(
            atoms, sampling=SAMPLING, slice_thickness=DZ,
            exit_planes=1, projection="finite",
        )
        probe.grid.match(potential)
        probe.positions = [(sr_pos_A[0], sr_pos_A[1])]

        # CVDMS
        print("  CVDMS...")
        cvdms = CVDMSMultislice(
            convergence_threshold=CONVERGENCE_THRESHOLD,
            backscattering=False, antialias=True, antialias_inner=True,
        )
        ew_c = probe.multislice(potential, algorithm=cvdms, lazy=False)
        arr_c = to_cpu(ew_c.array)
        if arr_c.ndim == 4: arr_c = arr_c[:, 0, :, :]
        psi_c = arr_c[-1]
        del ew_c, cvdms
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

        # Fourier
        print("  Fourier...")
        fourier = FourierMultislice(order=1)
        ew_f = probe.multislice(potential, algorithm=fourier, lazy=False)
        arr_f = to_cpu(ew_f.array)
        if arr_f.ndim == 4: arr_f = arr_f[:, 0, :, :]
        psi_f = arr_f[-1]
        del ew_f, fourier
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

        # Metrics
        ncc_val = ncc(psi_c, psi_f)
        phrms = phase_rms(psi_c, psi_f)
        amprms = amp_rms_error(psi_c, psi_f)
        dp_c = compute_dp(psi_c); dp_f = compute_dp(psi_f)
        dp_ncc = ncc(dp_c, dp_f)

        # On-column intensity
        cy, cx = gpts[0] // 2, gpts[1] // 2
        I_c = np.abs(psi_c)**2; I_f = np.abs(psi_f)**2
        I_col_ncc = float(np.corrcoef(
            np.abs(arr_c[:, cy, cx])**2 if arr_c.ndim == 3 else np.abs(arr_c[:, 0, cy, cx])**2,
            np.abs(arr_f[:, cy, cx])**2)[0, 1])

        print(f"  NCC = {ncc_val:.8f}  (1−NCC = {1-ncc_val:.2e})")
        print(f"  Phase RMS = {phrms:.2e} rad  (={phrms*1000:.2f} mrad)")
        print(f"  Amp RMS error = {amprms:.2e}")
        print(f"  DP NCC = {dp_ncc:.8f}")

        all_results[f"t_{total_z:.0f}"] = {
            "thickness_A": float(total_z),
            "supercell_z": sc_z, "n_slices": n_slices,
            "ncc": float(ncc_val),
            "one_minus_ncc": float(1 - ncc_val),
            "phase_rms_rad": float(phrms),
            "phase_rms_mrad": float(phrms * 1000),
            "amp_rms_error": float(amprms),
            "dp_ncc": float(dp_ncc),
            "I_col_ncc": float(I_col_ncc) if np.isfinite(I_col_ncc) else 0.0,
        }

        del atoms, probe, potential, arr_c, arr_f, psi_c, psi_f, dp_c, dp_f
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("C2 Thickness Sweep Summary:")
    print(f"  {'t (Å)':>8s}  {'NCC':>10s}  {'1−NCC':>10s}  "
          f"{'Phase RMS':>11s}  {'Amp RMS':>9s}  {'DP NCC':>10s}")
    thicknesses = sorted([r["thickness_A"] for r in all_results.values()])
    for t in thicknesses:
        r = all_results[f"t_{t:.0f}"]
        print(f"  {t:>8.1f}  {r['ncc']:>10.8f}  {r['one_minus_ncc']:>10.2e}  "
              f"{r['phase_rms_mrad']:>9.2f} mrad  {r['amp_rms_error']:>9.2e}  "
              f"{r['dp_ncc']:>10.8f}")

    # ── Save ──
    base = os.path.join(DATA_DIR, "c2_thickness")
    np.savez(base + ".npz",
             thickness_A=np.array(thicknesses),
             ncc=np.array([all_results[f"t_{t:.0f}"]["ncc"] for t in thicknesses]),
             one_minus_ncc=np.array([all_results[f"t_{t:.0f}"]["one_minus_ncc"] for t in thicknesses]),
             phase_rms_mrad=np.array([all_results[f"t_{t:.0f}"]["phase_rms_mrad"] for t in thicknesses]),
             amp_rms_error=np.array([all_results[f"t_{t:.0f}"]["amp_rms_error"] for t in thicknesses]),
             dp_ncc=np.array([all_results[f"t_{t:.0f}"]["dp_ncc"] for t in thicknesses]))
    with open(base + ".json", "w") as f:
        json.dump({
            "script": "c2_thickness.py",
            "params": {"energy_eV": ENERGY, "sampling_A_per_px": SAMPLING,
                       "dz_A": DZ, "supercell_xy": SUPERCELL_XY,
                       "supercell_z_sweep": SUPERCELL_Z_SWEEP},
            "sweep": all_results,
        }, f, indent=2)
    print(f"\nData saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
