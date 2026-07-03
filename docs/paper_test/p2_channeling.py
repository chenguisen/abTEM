#!/usr/bin/env python3
"""P2: Channeling Pendellösung — CVDMS vs Fourier multislice, Δz sweep.

SrTiO3 [001] at 30 keV, focused probe on Sr column.
Sweeps Δz ∈ {0.2, 0.4, 0.8, 1.0} Å at fixed large thickness (~780 Å).
Extracts on-column Pendellösung period via autocorrelation.
Finds commutator-sensitive Δz where CVDMS and Fourier diverge.

Outputs: docs/data/p2_channeling.npz + .json
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
SUPERCELL_XY, SUPERCELL_Z = 4, 200   # t ≈ 781 Å
SAMPLING = 0.10                       # Å/px
CONVERGENCE_THRESHOLD = 1e-7
DZ_SWEEP = [0.2, 0.4, 0.8, 1.0]     # Δz in Å — ≤1 Å for physical validity
# Keep effective depth sampling consistent at ~1.6–2.0 Å
EXIT_STRIDE_FOR_DZ = {0.2: 8, 0.4: 4, 0.8: 2, 1.0: 2}


def to_cpu(arr):
    if hasattr(arr, "get"): return arr.get()
    return np.asarray(arr)


def extract_period_acf(z, intensity):
    """Extract period from autocorrelation of detrended on-column |ψ|²."""
    i_cpu = np.asarray(intensity, dtype=np.float64)
    trend = np.polyfit(z, i_cpu, 1)
    detrended = i_cpu - np.polyval(trend, z)
    detrended -= detrended.mean()
    acf = np.correlate(detrended, detrended, mode="full")
    acf = acf[len(acf)//2:]
    dz = z[1] - z[0]
    min_lag = max(1, int(20.0 / dz))
    max_lag = min(len(acf) - 1, int(200.0 / dz))
    if min_lag >= max_lag:
        return np.inf, 0.0, acf[0] if len(acf) > 0 else 0.0
    acf_window = acf[min_lag:max_lag]
    peak_lag = np.argmax(acf_window) + min_lag
    period = peak_lag * dz
    amplitude = acf[peak_lag] / acf[0]
    return period, amplitude


def run_two_methods(probe, potential, dz, exit_stride):
    """Run CVDMS and Fourier, return (z, I_c, I_f, period_c, period_f, amp_c, amp_f)."""
    dz_eff = dz * exit_stride
    gpts = potential.gpts
    cy = gpts[0] // 2
    cx = gpts[1] // 2

    # CVDMS
    cvdms = CVDMSMultislice(
        convergence_threshold=CONVERGENCE_THRESHOLD,
        backscattering=False, antialias=True, antialias_inner=True,
    )
    ew_c = probe.multislice(potential, algorithm=cvdms, lazy=False)
    arr_c = to_cpu(ew_c.array)
    if arr_c.ndim == 4:
        arr_c = arr_c[:, 0, :, :]
    z = np.arange(len(arr_c)) * dz_eff
    I_c = np.abs(arr_c[:, cy, cx])**2
    period_c, amp_c = extract_period_acf(z, I_c)
    del ew_c, cvdms
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # Fourier
    fourier = FourierMultislice(order=1)
    ew_f = probe.multislice(potential, algorithm=fourier, lazy=False)
    arr_f = to_cpu(ew_f.array)
    if arr_f.ndim == 4:
        arr_f = arr_f[:, 0, :, :]
    I_f = np.abs(arr_f[:, cy, cx])**2
    period_f, amp_f = extract_period_acf(z, I_f)
    del ew_f, fourier
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    return z, I_c, I_f, period_c, period_f, amp_c, amp_f


def main():
    print("=== P2: Channeling Pendellösung Δz Sweep ===")

    a = A
    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[a, a, a, 90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)
    total_z = atoms.cell[2, 2]
    print(f"Thickness: {total_z:.1f} Å  px={SAMPLING:.3f} Å/px  supercell_z={SUPERCELL_Z}")

    # ── Analytic ξ_ch ──
    sigma = energy2sigma(ENERGY)
    ref_pot = abtem.Potential(atoms, sampling=SAMPLING, slice_thickness=a,
                               exit_planes=1, projection="finite")
    V_data = ref_pot.build(lazy=False)
    V_2d = to_cpu(V_data.array[0] if hasattr(V_data.array, "__len__") else V_data.array)
    gpts = ref_pot.gpts
    V_k = np.fft.fft2(V_2d)
    V_k_abs = np.abs(V_k); V_k_abs[0, 0] = 0
    dom_idx = np.unravel_index(V_k_abs.argmax(), V_k_abs.shape)
    V_g = float(V_k_abs[dom_idx] / V_2d.size)
    xi_ch = np.pi / (sigma * V_g)
    print(f"V_g = {V_g:.1f} eV·Å,  ξ_ch(bulk) = {xi_ch:.0f} Å")

    # Sr column
    sr_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
    px_A = ref_pot.sampling[0]
    sr_pos_A = (float(sr_idx[1] * px_A), float(sr_idx[0] * px_A))
    print(f"Sr at pixel {sr_idx}, pos ({sr_pos_A[0]:.1f}, {sr_pos_A[1]:.1f}) Å")
    print(f"Grid: {gpts}, extent: {ref_pot.extent}")

    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Probe on Sr column ──
    probe = abtem.Probe(
        energy=ENERGY, semiangle_cutoff=10,
        extent=SUPERCELL_XY * a, gpts=gpts[0],
        device="gpu",
    )
    probe.grid.match(ref_pot)
    probe.positions = [(sr_pos_A[0], sr_pos_A[1])]

    # ── Δz sweep ──
    all_results = {}
    print(f"\n{'='*60}")
    print(f"Δz sweep: {DZ_SWEEP}")
    print(f"{'='*60}")

    for dz in DZ_SWEEP:
        exit_stride = EXIT_STRIDE_FOR_DZ[dz]
        dz_eff = dz * exit_stride
        n_slices = int(total_z / dz)
        print(f"\n--- Δz = {dz:.1f} Å  (exit_stride={exit_stride}, dz_eff={dz_eff:.1f} Å, slices={n_slices}) ---")

        potential = abtem.Potential(
            atoms, sampling=SAMPLING, slice_thickness=dz,
            exit_planes=exit_stride, projection="finite",
        )

        z, I_c, I_f, period_c, period_f, amp_c, amp_f = run_two_methods(
            probe, potential, dz, exit_stride)

        delta_xi = abs(period_c - period_f)
        err_c = abs(period_c - xi_ch) / xi_ch * 100 if np.isfinite(period_c) else np.inf
        err_f = abs(period_f - xi_ch) / xi_ch * 100 if np.isfinite(period_f) else np.inf

        print(f"  CVDMS:   ξ = {period_c:.1f} Å  (ACF amp = {amp_c:.3f}, err vs bulk = {err_c:.1f}%)")
        print(f"  Fourier:  ξ = {period_f:.1f} Å  (ACF amp = {amp_f:.3f}, err vs bulk = {err_f:.1f}%)")
        print(f"  Δξ(C−F) = {delta_xi:.2f} Å")

        all_results[f"dz_{dz}"] = {
            "dz_A": dz,
            "z": z.tolist(),
            "I_cvdms": I_c.tolist(),
            "I_fourier": I_f.tolist(),
            "period_cvdms_A": float(period_c),
            "period_fourier_A": float(period_f),
            "acf_amp_cvdms": float(amp_c),
            "acf_amp_fourier": float(amp_f),
            "delta_xi_CF_A": float(delta_xi),
            "err_cvdms_vs_bulk_pct": float(err_c),
            "err_fourier_vs_bulk_pct": float(err_f),
        }
        del potential
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("P2 Δz Sweep Summary:")
    print(f"  Bulk ξ_ch = {xi_ch:.0f} Å  (from V_g = {V_g:.1f} eV·Å)")
    print(f"  {'Δz':>6s}  {'ξ_CVDMS':>8s}  {'ξ_Fourier':>8s}  {'Δξ(C-F)':>10s}  {'ACF_C':>7s}  {'ACF_F':>7s}")
    for dz in DZ_SWEEP:
        r = all_results[f"dz_{dz}"]
        print(f"  {dz:>5.1f} Å  {r['period_cvdms_A']:>8.1f}  {r['period_fourier_A']:>8.1f}  "
              f"{r['delta_xi_CF_A']:>10.2f}  {r['acf_amp_cvdms']:>7.3f}  {r['acf_amp_fourier']:>7.3f}")

    # ── Save ──
    base = os.path.join(DATA_DIR, "p2_channeling")
    np.savez(base + ".npz",
             xi_ch_analytic=xi_ch, V_g=V_g,
             dz_sweep=np.array(DZ_SWEEP),
             **{f"z_dz{dz}": np.array(all_results[f"dz_{dz}"]["z"]) for dz in DZ_SWEEP},
             **{f"I_c_dz{dz}": np.array(all_results[f"dz_{dz}"]["I_cvdms"]) for dz in DZ_SWEEP},
             **{f"I_f_dz{dz}": np.array(all_results[f"dz_{dz}"]["I_fourier"]) for dz in DZ_SWEEP},
             **{f"xi_c_dz{dz}": all_results[f"dz_{dz}"]["period_cvdms_A"] for dz in DZ_SWEEP},
             **{f"xi_f_dz{dz}": all_results[f"dz_{dz}"]["period_fourier_A"] for dz in DZ_SWEEP},
             sr_idx=np.array(sr_idx))
    with open(base + ".json", "w") as f:
        json.dump({
            "script": "p2_channeling.py",
            "params": {"energy_eV": ENERGY, "sampling_A_per_px": SAMPLING,
                       "thickness_A": float(total_z),
                       "supercell_xy": SUPERCELL_XY, "supercell_z": SUPERCELL_Z,
                       "dz_sweep": DZ_SWEEP},
            "results": {"xi_ch_analytic_A": float(xi_ch), "V_g_eVA": float(V_g)},
            "sweep": all_results,
        }, f, indent=2)
    print(f"\nData saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
