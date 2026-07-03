#!/usr/bin/env python3
"""C4: Antialiasing on/off comparison — bandwidth explosion negative control.

SrTiO3 [001] @ 30 keV, probe on Sr column.
Runs CVDMS with and without antialiasing (antialias + antialias_inner).
Measures exit wave quality vs a Δz=0.2 Å reference (finest practical slicing).

Outputs: docs/data/c4_antialias.npz + .json
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

ENERGY = 30e3
A = 3.905
SUPERCELL_XY, SUPERCELL_Z = 4, 50    # t ≈ 195 Å
SAMPLING = 0.05                       # Å/px — fine, triggers bandwidth issues
DZ = 0.4
DZ_REF = 0.2                          # reference Δz
CONVERGENCE_THRESHOLD = 1e-7


def to_cpu(arr):
    if hasattr(arr, "get"): return arr.get()
    return np.asarray(arr)


def ncc(a, b):
    a_f = a.ravel(); b_f = b.ravel()
    return float(np.abs(np.dot(a_f.conj(), b_f)) /
                 np.sqrt(np.dot(a_f.conj(), a_f).real * np.dot(b_f.conj(), b_f).real))


def build_atoms():
    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[A, A, A, 90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)
    return atoms


def main():
    print("=== C4: Antialiasing On/Off Comparison ===")

    atoms = build_atoms()
    total_z = atoms.cell[2, 2]
    print(f"Thickness: {total_z:.1f} Å, Δz={DZ} Å, Δx={SAMPLING} Å/px")

    # Grid and column position
    ref_pot = abtem.Potential(atoms, sampling=SAMPLING, slice_thickness=A,
                               exit_planes=1, projection="finite")
    V_data = ref_pot.build(lazy=False)
    V_2d = to_cpu(V_data.array[0])
    gpts = ref_pot.gpts; px_A = ref_pot.sampling[0]
    sr_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
    sr_pos_A = (float(sr_idx[1] * px_A), float(sr_idx[0] * px_A))
    print(f"Grid: {gpts}, Sr at ({sr_pos_A[0]:.1f}, {sr_pos_A[1]:.1f}) Å")

    del V_data, ref_pot
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # Common probe
    probe = abtem.Probe(
        energy=ENERGY, semiangle_cutoff=10,
        extent=SUPERCELL_XY * A, gpts=gpts[0],
        device="gpu",
    )
    probe.positions = [(sr_pos_A[0], sr_pos_A[1])]

    # ── Reference: Δz=0.2 Å with AA ──
    print(f"\nReference: CVDMS Δz={DZ_REF} Å with AA...")
    potential_ref = abtem.Potential(
        atoms, sampling=SAMPLING, slice_thickness=DZ_REF,
        exit_planes=1, projection="finite",
    )
    probe_ref = abtem.Probe(energy=ENERGY, semiangle_cutoff=10,
                            extent=SUPERCELL_XY * A, gpts=gpts[0], device="gpu")
    probe_ref.grid.match(potential_ref)
    probe_ref.positions = [(sr_pos_A[0], sr_pos_A[1])]
    cvdms_ref = CVDMSMultislice(
        convergence_threshold=CONVERGENCE_THRESHOLD,
        backscattering=False, antialias=True, antialias_inner=True,
    )
    ew_ref = probe_ref.multislice(potential_ref, algorithm=cvdms_ref, lazy=False)
    arr_ref = to_cpu(ew_ref.array)
    if arr_ref.ndim == 4: arr_ref = arr_ref[:, 0, :, :]
    psi_ref = arr_ref[-1]
    del ew_ref, cvdms_ref, probe_ref, potential_ref
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Test: Δz=0.4 Å with AA on ──
    print("CVDMS Δz=0.4 Å AA=ON...")
    potential = abtem.Potential(
        atoms, sampling=SAMPLING, slice_thickness=DZ,
        exit_planes=1, projection="finite",
    )
    probe_aa = abtem.Probe(energy=ENERGY, semiangle_cutoff=10,
                           extent=SUPERCELL_XY * A, gpts=gpts[0], device="gpu")
    probe_aa.grid.match(potential)
    probe_aa.positions = [(sr_pos_A[0], sr_pos_A[1])]

    cvdms_aa = CVDMSMultislice(
        convergence_threshold=CONVERGENCE_THRESHOLD,
        backscattering=False, antialias=True, antialias_inner=True,
    )
    ew_aa = probe_aa.multislice(potential, algorithm=cvdms_aa, lazy=False)
    arr_aa = to_cpu(ew_aa.array)
    if arr_aa.ndim == 4: arr_aa = arr_aa[:, 0, :, :]
    psi_aa = arr_aa[-1]
    del ew_aa, cvdms_aa, probe_aa
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Test: Δz=0.4 Å with AA off ──
    print("CVDMS Δz=0.4 Å AA=OFF...")
    probe_noaa = abtem.Probe(energy=ENERGY, semiangle_cutoff=10,
                             extent=SUPERCELL_XY * A, gpts=gpts[0], device="gpu")
    probe_noaa.grid.match(potential)
    probe_noaa.positions = [(sr_pos_A[0], sr_pos_A[1])]

    overflow_noaa = False
    cvdms_noaa = CVDMSMultislice(
        convergence_threshold=CONVERGENCE_THRESHOLD,
        backscattering=False, antialias=False, antialias_inner=False,
    )
    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ew_noaa = probe_noaa.multislice(potential, algorithm=cvdms_noaa, lazy=False)
            if any("overflow" in str(x.message).lower() for x in w):
                overflow_noaa = True
                print("  ⚠ CVDMS overflow detected (AA=OFF) — bandwidth explosion confirmed!")
    except Exception:
        overflow_noaa = True
        print("  ⚠ CVDMS overflow (AA=OFF)")

    arr_noaa = to_cpu(ew_noaa.array)
    if arr_noaa.ndim == 4: arr_noaa = arr_noaa[:, 0, :, :]
    psi_noaa = arr_noaa[-1]
    # Check for inf/nan
    has_inf = np.isinf(psi_noaa).any()
    has_nan = np.isnan(psi_noaa).any()
    if has_inf or has_nan:
        overflow_noaa = True
        print(f"  ⚠ Wave function contains {'inf' if has_inf else ''}{'+' if has_inf and has_nan else ''}{'nan' if has_nan else ''}")
    del ew_noaa, cvdms_noaa, probe_noaa, potential
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Metrics for AA=ON ──
    ncc_aa = ncc(psi_aa, psi_ref)
    phase_err_aa = np.angle(psi_aa * np.conj(psi_ref))
    phase_rms_aa = float(np.sqrt(np.mean(phase_err_aa**2)))
    amp_aa = np.abs(psi_aa); amp_ref = np.abs(psi_ref)
    amp_rms_aa = float(np.sqrt(np.mean(((amp_aa - amp_ref)/amp_ref.mean())**2)))
    fft_aa = np.fft.fftshift(np.fft.fft2(psi_aa))
    fft_ref = np.fft.fftshift(np.fft.fft2(psi_ref))

    # ── Metrics for AA=OFF (may be invalid due to overflow) ──
    noaa_valid = not overflow_noaa and not np.isinf(psi_noaa).any() and not np.isnan(psi_noaa).any()
    if noaa_valid:
        ncc_noaa = ncc(psi_noaa, psi_ref)
        ncc_aa_noaa = ncc(psi_aa, psi_noaa)
        phase_err_noaa = np.angle(psi_noaa * np.conj(psi_ref))
        phase_rms_noaa = float(np.sqrt(np.mean(phase_err_noaa**2)))
        amp_noaa = np.abs(psi_noaa)
        amp_rms_noaa = float(np.sqrt(np.mean(((amp_noaa - amp_ref)/amp_ref.mean())**2)))
        fft_noaa = np.fft.fftshift(np.fft.fft2(psi_noaa))
    else:
        ncc_noaa = 0.0; ncc_aa_noaa = 0.0
        phase_rms_noaa = float('inf'); amp_rms_noaa = float('inf')
        phase_err_noaa = np.zeros_like(psi_noaa.real)
        amp_noaa = np.zeros_like(psi_noaa.real)
        fft_noaa = np.zeros_like(fft_ref)

    # ── Radial power spectrum ──
    def radial_power(fft_data, n_bins=50):
        ny, nx = fft_data.shape
        y, x = np.indices((ny, nx))
        r = np.sqrt((x - nx//2)**2 + (y - ny//2)**2)
        r_max = int(r.max())
        bins = np.linspace(0, r_max, n_bins + 1)
        power = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (r >= bins[i]) & (r < bins[i+1])
            if mask.any():
                power[i] = float((np.abs(fft_data[mask])**2).mean())
        return (bins[:-1] + bins[1:]) / 2, power

    r_freq, pwr_aa = radial_power(fft_aa)
    _, pwr_ref = radial_power(fft_ref)
    _, pwr_noaa = radial_power(fft_noaa) if noaa_valid else (r_freq, np.zeros_like(pwr_aa))

    # High-frequency power (> 2/3 Nyquist)
    nyquist_px = fft_aa.shape[0] // 2
    high_freq_mask = r_freq > 2/3 * nyquist_px
    hf_pwr_aa = float(pwr_aa[high_freq_mask].sum()) if high_freq_mask.any() and pwr_aa.sum() > 0 else 0
    hf_pwr_ref = float(pwr_ref[high_freq_mask].sum()) if high_freq_mask.any() and pwr_ref.sum() > 0 else 0
    hf_pwr_noaa = float(pwr_noaa[high_freq_mask].sum()) if noaa_valid and high_freq_mask.any() else 0

    print(f"\n{'='*60}")
    print("C4 Results:")
    print(f"  NCC vs ref (AA=ON):  {ncc_aa:.8f}  (1−NCC={1-ncc_aa:.2e})")
    if noaa_valid:
        print(f"  NCC vs ref (AA=OFF): {ncc_noaa:.8f}  (1−NCC={1-ncc_noaa:.2e})")
        print(f"  NCC (AA-ON vs AA-OFF): {ncc_aa_noaa:.8f}")
    print(f"  Phase RMS vs ref (AA=ON):  {phase_rms_aa*1000:.2f} mrad")
    if noaa_valid:
        print(f"  Phase RMS vs ref (AA=OFF): {phase_rms_noaa*1000:.2f} mrad")
    print(f"  Amp RMS vs ref (AA=ON):  {amp_rms_aa:.4f}")
    if noaa_valid:
        print(f"  Amp RMS vs ref (AA=OFF): {amp_rms_noaa:.4f}")
    print(f"  AA=OFF overflow: {overflow_noaa} {'(bandwidth explosion confirmed!)' if overflow_noaa else ''}")
    if pwr_ref.sum() > 0:
        print(f"  High-freq power ratio: ref={hf_pwr_ref/pwr_ref.sum():.2e}, AA=ON={hf_pwr_aa/pwr_aa.sum():.2e}" +
              (f", AA=OFF={hf_pwr_noaa/pwr_noaa.sum():.2e}" if noaa_valid else ", AA=OFF=OVERFLOW"))

    # ── Save ──
    base = os.path.join(DATA_DIR, "c4_antialias")
    hf_ratio_ref = hf_pwr_ref/pwr_ref.sum() if pwr_ref.sum() > 0 else 0
    hf_ratio_aa = hf_pwr_aa/pwr_aa.sum() if pwr_aa.sum() > 0 else 0
    hf_ratio_noaa = hf_pwr_noaa/pwr_noaa.sum() if noaa_valid and pwr_noaa.sum() > 0 else float('nan')

    np.savez(base + ".npz",
             psi_aa=psi_aa, psi_noaa=psi_noaa, psi_ref=psi_ref,
             amp_aa=amp_aa, amp_noaa=amp_noaa, amp_ref=amp_ref,
             phase_err_aa=phase_err_aa, phase_err_noaa=phase_err_noaa,
             r_freq=r_freq, pwr_aa=pwr_aa, pwr_noaa=pwr_noaa, pwr_ref=pwr_ref,
             ncc_aa=ncc_aa, ncc_noaa=ncc_noaa,
             phase_rms_aa_mrad=phase_rms_aa*1000,
             phase_rms_noaa_mrad=phase_rms_noaa*1000 if noaa_valid else 0,
             overflow_noaa=overflow_noaa)
    with open(base + ".json", "w") as f:
        json.dump({
            "script": "c4_antialias.py",
            "params": {"energy_eV": ENERGY, "sampling_A_per_px": SAMPLING,
                       "dz_A": DZ, "dz_ref_A": DZ_REF, "thickness_A": float(total_z),
                       "supercell_xy": SUPERCELL_XY, "supercell_z": SUPERCELL_Z},
            "results": {
                "ncc_aa_vs_ref": float(ncc_aa),
                "ncc_noaa_vs_ref": float(ncc_noaa) if noaa_valid else None,
                "phase_rms_aa_mrad": float(phase_rms_aa*1000),
                "phase_rms_noaa_mrad": float(phase_rms_noaa*1000) if noaa_valid else None,
                "amp_rms_aa": float(amp_rms_aa),
                "amp_rms_noaa": float(amp_rms_noaa) if noaa_valid else None,
                "hf_power_ratio_aa": float(hf_ratio_aa),
                "hf_power_ratio_noaa": float(hf_ratio_noaa) if noaa_valid else None,
                "hf_power_ratio_ref": float(hf_ratio_ref),
                "aa_off_overflow": bool(overflow_noaa),
            },
        }, f, indent=2)
    print(f"Data saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
