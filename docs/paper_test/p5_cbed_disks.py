#!/usr/bin/env python3
"""P5: CBED disk-specific intensity analysis — CVDMS vs Fourier vs thickness.

SrTiO3 [001] at 30 keV, convergent probe on Sr column.
Computes per-disk integrated intensities for {200}, {220}, {420} reflections
and compares CVDMS (Δz=0.4 Å) against Δz→0 Fourier reference.

Uses 8×8 supercell for sufficient DP resolution (~0.032 Å⁻¹ per pixel).
Probe semiangle = 20 mrad for adequately sampled CBED disks.

Outputs: docs/data/p5_cbed_disks.npz + .json
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
SUPERCELL_XY = 8                       # larger for DP resolution
SAMPLING = 0.10                        # Å/px
DZ = 0.4
DZ_REF = 0.2                           # Δz→0 reference (Fourier)
CONVERGENCE_THRESHOLD = 1e-7
PROBE_SEMIANGLE = 20                   # mrad — for larger CBED disks
THICKNESSES_Z = [10, 25, 50, 100]      # supercell_z for thickness sweep


def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def compute_dp(psi):
    """Diffraction pattern |FFT[ψ]|², shifted."""
    dp = np.abs(np.fft.fft2(psi))**2
    return np.fft.fftshift(dp)


def integrate_disk(dp, qx, qy, g_vec, radius):
    """Integrate intensity within circular aperture of `radius` around `g_vec`.

    Parameters
    ----------
    dp : 2D array
        Diffraction pattern (FFT-shifted, DC at centre).
    qx, qy : 1D arrays
        Reciprocal-space coordinates (Å⁻¹), FFT-shifted.
    g_vec : tuple (gx, gy)
        Reciprocal lattice vector in Å⁻¹.
    radius : float
        Integration aperture radius in Å⁻¹ (typically α/λ for CBED disk).

    Returns
    -------
    I_disk : float
        Integrated intensity within the disk.
    I_total : float
        Total intensity of the DP (for normalisation).
    """
    gx, gy = g_vec
    QX, QY = np.meshgrid(qx, qy, indexing="ij")
    dist = np.sqrt((QX - gx)**2 + (QY - gy)**2)
    mask = dist <= radius
    I_disk = float(dp[mask].sum())
    I_total = float(dp.sum())
    return I_disk, I_total


def get_reciprocal_axes(gpts, px_A):
    """Return FFT-shifted reciprocal-space coordinate axes (Å⁻¹)."""
    nx, ny = gpts
    dq = 1.0 / (nx * px_A)
    qx = np.fft.fftshift(np.fft.fftfreq(nx, px_A))
    qy = np.fft.fftshift(np.fft.fftfreq(ny, px_A))
    return qx, qy, dq


def main():
    print("=== P5: CBED Disk-Specific Intensity Analysis ===")
    wavelength = energy2wavelength(ENERGY)
    k0 = 1.0 / wavelength
    sigma = energy2sigma(ENERGY)
    disk_radius = PROBE_SEMIANGLE * 1e-3 / wavelength  # α/λ in Å⁻¹

    # Reciprocal lattice vectors for SrTiO3 [001] (cubic, a=3.905 Å)
    # {hkl} disks in DP at g = √(h²+k²+l²)/a
    # For [001] zone axis, the ZOLZ reflections are {hk0}
    g_200 = (2.0 / A, 0.0)       # 0.512 Å⁻¹
    g_020 = (0.0, 2.0 / A)       # 0.512 Å⁻¹
    g_220 = (2.0 / A, 2.0 / A)   # 0.724 Å⁻¹
    g_420 = (4.0 / A, 2.0 / A)   # 1.145 Å⁻¹
    # Use symmetrically equivalent disks for better statistics
    g_220_2 = (2.0 / A, -2.0 / A)
    g_420_2 = (4.0 / A, -2.0 / A)

    disks = {
        "{200}": [g_200, g_020],
        "{220}": [g_220, g_220_2],
        "{420}": [g_420, g_420_2],
    }

    print(f"Energy: {ENERGY/1000:.0f} keV  λ={wavelength:.4f} Å  k₀={k0:.1f} Å⁻¹")
    print(f"Sampling: {SAMPLING} Å/px  supercell_xy={SUPERCELL_XY}")
    print(f"Probe semiangle: {PROBE_SEMIANGLE} mrad  disk radius: {disk_radius:.4f} Å⁻¹")
    print(f"Δz = {DZ} Å (CVDMS), Δz_ref = {DZ_REF} Å (Fourier)")
    for name, g_list in disks.items():
        print(f"  {name}: g = ({g_list[0][0]:.4f}, {g_list[0][1]:.4f}) Å⁻¹")

    # ── Build reference (single unit cell) for column finding ──
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

    sr_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
    sr_pos_A = (float(sr_idx[1] * px_A), float(sr_idx[0] * px_A))
    print(f"Sr at pixel {sr_idx}, pos ({sr_pos_A[0]:.1f}, {sr_pos_A[1]:.1f}) Å")
    print(f"Grid: {gpts}  dq = {1.0/(gpts[0]*px_A):.4f} Å⁻¹")
    print(f"CBED disk diameter: {2*disk_radius:.4f} Å⁻¹ ≈ {2*disk_radius/(1.0/(gpts[0]*px_A)):.1f} px")

    qx, qy, dq = get_reciprocal_axes(gpts, px_A)

    del V_data, ref_pot, atoms_ref
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Thickness sweep ──
    all_results = {}

    for sc_z in THICKNESSES_Z:
        atoms = crystal(
            ["Sr", "Ti", "O"],
            basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
            spacegroup=221, cellpar=[A, A, A, 90, 90, 90],
        )
        atoms *= (SUPERCELL_XY, SUPERCELL_XY, sc_z)
        total_z = atoms.cell[2, 2]
        n_slices = int(total_z / DZ)
        n_slices_ref = int(total_z / DZ_REF)
        print(f"\n{'='*60}")
        print(f"t = {total_z:.1f} Å  (sc_z={sc_z}, slices: {n_slices} CVDMS, {n_slices_ref} ref)")

        probe = abtem.Probe(
            energy=ENERGY, semiangle_cutoff=PROBE_SEMIANGLE,
            extent=SUPERCELL_XY * A, gpts=gpts[0],
            device="gpu",
        )

        # ── CVDMS (production Δz) ──
        pot_cvdms = abtem.Potential(
            atoms, sampling=SAMPLING, slice_thickness=DZ,
            exit_planes=1, projection="finite",
        )
        probe_c = abtem.Probe(
            energy=ENERGY, semiangle_cutoff=PROBE_SEMIANGLE,
            extent=SUPERCELL_XY * A, gpts=gpts[0],
            device="gpu",
        )
        probe_c.grid.match(pot_cvdms)
        probe_c.positions = [(sr_pos_A[0], sr_pos_A[1])]

        print("  CVDMS...")
        cvdms = CVDMSMultislice(
            convergence_threshold=CONVERGENCE_THRESHOLD,
            backscattering=False, antialias=True, antialias_inner=True,
        )
        ew_c = probe_c.multislice(pot_cvdms, algorithm=cvdms, lazy=False)
        arr_c = to_cpu(ew_c.array)
        if arr_c.ndim == 4:
            arr_c = arr_c[:, 0, :, :]
        psi_c = arr_c[-1]
        dp_c = compute_dp(psi_c)

        del ew_c, cvdms, pot_cvdms, probe_c
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

        # ── Fourier reference (Δz→0) ──
        pot_ref_dz = abtem.Potential(
            atoms, sampling=SAMPLING, slice_thickness=DZ_REF,
            exit_planes=1, projection="finite",
        )
        probe_ref = abtem.Probe(
            energy=ENERGY, semiangle_cutoff=PROBE_SEMIANGLE,
            extent=SUPERCELL_XY * A, gpts=gpts[0],
            device="gpu",
        )
        probe_ref.grid.match(pot_ref_dz)
        probe_ref.positions = [(sr_pos_A[0], sr_pos_A[1])]

        print("  Fourier reference...")
        fourier = FourierMultislice(order=1)
        ew_f = probe_ref.multislice(pot_ref_dz, algorithm=fourier, lazy=False)
        arr_f = to_cpu(ew_f.array)
        if arr_f.ndim == 4:
            arr_f = arr_f[:, 0, :, :]
        psi_f = arr_f[-1]
        dp_f = compute_dp(psi_f)

        del ew_f, fourier, pot_ref_dz, probe_ref
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

        # ── DP NCC (overall) ──
        dp_ncc = float(np.corrcoef(dp_c.ravel(), dp_f.ravel())[0, 1])

        # ── Per-disk integration ──
        disk_results = {}
        for disk_name, g_list in disks.items():
            I_c_total = 0.0
            I_f_total = 0.0
            I_total_c = 0.0
            I_total_f = 0.0
            for g_vec in g_list:
                I_c_disk, I_tot_c = integrate_disk(dp_c, qx, qy, g_vec, disk_radius)
                I_f_disk, I_tot_f = integrate_disk(dp_f, qx, qy, g_vec, disk_radius)
                I_c_total += I_c_disk
                I_f_total += I_f_disk
                I_total_c = I_tot_c
                I_total_f = I_tot_f
            # Normalise as fraction of total DP intensity
            frac_c = I_c_total / (I_total_c + 1e-30)
            frac_f = I_f_total / (I_total_f + 1e-30)
            rel_dev = abs(frac_c - frac_f) / (frac_f + 1e-30) * 100  # % deviation

            print(f"  {disk_name}: I_c={frac_c:.6e} I_f={frac_f:.6e}  deviation={rel_dev:.3f}%")
            disk_results[disk_name] = {
                "I_cvdms_norm": float(frac_c),
                "I_ref_norm": float(frac_f),
                "rel_deviation_pct": float(rel_dev),
            }

        all_results[f"t_{total_z:.0f}"] = {
            "thickness_A": float(total_z),
            "supercell_z": sc_z,
            "n_slices_cvdms": n_slices,
            "n_slices_ref": n_slices_ref,
            "dp_ncc": float(dp_ncc),
            "disks": disk_results,
        }

        del atoms, arr_c, arr_f, psi_c, psi_f, dp_c, dp_f
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("P5 CBED Disk Intensity Summary")
    print(f"  Disk radius: {disk_radius:.4f} Å⁻¹ ({PROBE_SEMIANGLE} mrad probe)")
    header = f"  {'t (Å)':>8s}  {'DP NCC':>10s}"
    for name in disks:
        header += f"  {name + ' dev%':>12s}"
    print(header)
    for t_key in sorted(all_results.keys(), key=lambda k: all_results[k]["thickness_A"]):
        r = all_results[t_key]
        line = f"  {r['thickness_A']:>8.1f}  {r['dp_ncc']:>10.6f}"
        for name in disks:
            line += f"  {r['disks'][name]['rel_deviation_pct']:>12.3f}"
        print(line)

    # Report max deviation across all disks and thicknesses up to 400 Å
    max_dev = 0.0
    for t_key, r in all_results.items():
        if r["thickness_A"] <= 400:
            for name in disks:
                max_dev = max(max_dev, r["disks"][name]["rel_deviation_pct"])
    print(f"\n  Max disk deviation at t ≤ 400 Å: {max_dev:.2f}%")

    # ── Save ──
    base = os.path.join(DATA_DIR, "p5_cbed_disks")
    # JSON-serialisable
    json_results = {}
    for t_key, r in all_results.items():
        json_results[t_key] = {
            "thickness_A": r["thickness_A"],
            "supercell_z": r["supercell_z"],
            "n_slices_cvdms": r["n_slices_cvdms"],
            "n_slices_ref": r["n_slices_ref"],
            "dp_ncc": r["dp_ncc"],
            "disks": {k: dict(v) for k, v in r["disks"].items()},
        }
    with open(base + ".json", "w") as f:
        json.dump({
            "script": "p5_cbed_disks.py",
            "params": {
                "energy_eV": ENERGY, "sampling_A_per_px": SAMPLING,
                "dz_A": DZ, "dz_ref_A": DZ_REF,
                "supercell_xy": SUPERCELL_XY,
                "probe_semiangle_mrad": PROBE_SEMIANGLE,
                "disk_radius_per_A": float(disk_radius),
                "gpts": list(gpts),
                "thicknesses_z": THICKNESSES_Z,
            },
            "max_deviation_pct_t400": float(max_dev),
            "sweep": json_results,
        }, f, indent=2)
    print(f"\nData saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
