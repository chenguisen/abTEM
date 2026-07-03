#!/usr/bin/env python3
"""P2 cross-material: Channeling Pendellösung Δz sweep for Si and Au.

Extends P2 beyond SrTiO3 to test universality of commutator-driven
ACF amplitude enhancement in CVDMS vs Fourier.

Si [001] @ 100 keV, Au [001] @ 300 keV.
Δz sweep: {0.4, 0.8, 1.0} Å at fixed thickness (~400 Å).

Outputs: docs/data/p2_si.npz/.json, p2_au.npz/.json
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

# ── Material definitions ──
MATERIALS = {
    "Si": {
        "energy": 100e3,
        "a": 5.431,
        "spacegroup": 227,
        "symbols": ["Si"],
        "basis": [(0, 0, 0), (0.25, 0.25, 0.25)],
        "supercell_xy": 4, "supercell_z": 80,   # t ≈ 434 Å
        "sampling": 0.10,  # Å/px
    },
    "Au": {
        "energy": 300e3,
        "a": 4.078,
        "spacegroup": 225,
        "symbols": ["Au"],
        "basis": [(0, 0, 0)],
        "supercell_xy": 4, "supercell_z": 100,  # t ≈ 408 Å
        "sampling": 0.10,  # Å/px
    },
}

DZ_SWEEP = [0.4, 0.8, 1.0]
EXIT_STRIDE_FOR_DZ = {0.4: 2, 0.8: 1, 1.0: 1}
CONVERGENCE_THRESHOLD = 1e-7


def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
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
        return np.inf, 0.0
    acf_window = acf[min_lag:max_lag]
    peak_lag = np.argmax(acf_window) + min_lag
    period = peak_lag * dz
    amplitude = acf[peak_lag] / acf[0]
    return period, amplitude


def run_two_methods(probe, potential, dz, exit_stride, gpts):
    """Run CVDMS and Fourier, return (z, I_c, I_f, period_c, period_f, amp_c, amp_f)."""
    dz_eff = dz * exit_stride
    cy, cx = gpts[0] // 2, gpts[1] // 2

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


def process_material(name, params):
    print(f"\n{'='*60}")
    print(f"P2 Cross-Material: {name} [001] @ {params['energy']/1000:.0f} keV")
    print(f"{'='*60}")

    energy = params["energy"]
    a = params["a"]
    supercell_xy = params["supercell_xy"]
    supercell_z = params["supercell_z"]
    sampling = params["sampling"]

    atoms = crystal(
        params["symbols"],
        basis=params["basis"],
        spacegroup=params["spacegroup"],
        cellpar=[a, a, a, 90, 90, 90],
    )
    atoms *= (supercell_xy, supercell_xy, supercell_z)
    total_z = atoms.cell[2, 2]
    print(f"Thickness: {total_z:.1f} Å  px={sampling:.3f} Å/px  supercell_z={supercell_z}")

    # Analytic ξ_ch
    sigma = energy2sigma(energy)
    ref_pot = abtem.Potential(atoms, sampling=sampling, slice_thickness=a,
                               exit_planes=1, projection="finite")
    V_data = ref_pot.build(lazy=False)
    V_2d = to_cpu(V_data.array[0])
    gpts = ref_pot.gpts

    V_k = np.fft.fft2(V_2d)
    V_k_abs = np.abs(V_k); V_k_abs[0, 0] = 0
    dom_idx = np.unravel_index(V_k_abs.argmax(), V_k_abs.shape)
    V_g = float(V_k_abs[dom_idx] / V_2d.size)
    xi_ch = np.pi / (sigma * V_g)
    print(f"V_g = {V_g:.1f} eV·Å,  ξ_ch(bulk) = {xi_ch:.0f} Å")
    print(f"Grid: {gpts}, extent: {ref_pot.extent}")

    # Find column position
    col_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
    px_A = ref_pot.sampling[0]
    col_pos_A = (float(col_idx[1] * px_A), float(col_idx[0] * px_A))
    print(f"Column at pixel {col_idx}, pos ({col_pos_A[0]:.1f}, {col_pos_A[1]:.1f}) Å")

    del V_data
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # Probe on column
    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=10,
        extent=supercell_xy * a, gpts=gpts[0],
        device="gpu",
    )
    probe.grid.match(ref_pot)
    probe.positions = [(col_pos_A[0], col_pos_A[1])]

    # Δz sweep
    all_results = {}
    for dz in DZ_SWEEP:
        exit_stride = EXIT_STRIDE_FOR_DZ[dz]
        dz_eff = dz * exit_stride
        n_slices = int(total_z / dz)
        print(f"\n--- Δz = {dz:.1f} Å  (exit_stride={exit_stride}, dz_eff={dz_eff:.1f} Å, slices={n_slices}) ---")

        potential = abtem.Potential(
            atoms, sampling=sampling, slice_thickness=dz,
            exit_planes=exit_stride, projection="finite",
        )

        z, I_c, I_f, period_c, period_f, amp_c, amp_f = run_two_methods(
            probe, potential, dz, exit_stride, gpts)

        delta_xi = abs(period_c - period_f)
        print(f"  CVDMS:   ξ = {period_c:.1f} Å  (ACF amp = {amp_c:.3f})")
        print(f"  Fourier:  ξ = {period_f:.1f} Å  (ACF amp = {amp_f:.3f})")
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
        }
        del potential
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # Summary
    print(f"\n{'='*60}")
    print(f"{name} Δz Sweep Summary (bulk ξ_ch = {xi_ch:.0f} Å):")
    print(f"  {'Δz':>6s}  {'ξ_CVDMS':>8s}  {'ξ_Fourier':>8s}  {'Δξ(C−F)':>10s}  {'ACF_C':>7s}  {'ACF_F':>7s}")
    for dz in DZ_SWEEP:
        r = all_results[f"dz_{dz}"]
        print(f"  {dz:>5.1f} Å  {r['period_cvdms_A']:>8.1f}  {r['period_fourier_A']:>8.1f}  "
              f"{r['delta_xi_CF_A']:>10.2f}  {r['acf_amp_cvdms']:>7.3f}  {r['acf_amp_fourier']:>7.3f}")

    # Save
    fname = name.lower()
    base = os.path.join(DATA_DIR, f"p2_{fname}")
    np.savez(base + ".npz",
             xi_ch_analytic=xi_ch, V_g=V_g,
             dz_sweep=np.array(DZ_SWEEP),
             **{f"z_dz{dz}": np.array(all_results[f"dz_{dz}"]["z"]) for dz in DZ_SWEEP},
             **{f"I_c_dz{dz}": np.array(all_results[f"dz_{dz}"]["I_cvdms"]) for dz in DZ_SWEEP},
             **{f"I_f_dz{dz}": np.array(all_results[f"dz_{dz}"]["I_fourier"]) for dz in DZ_SWEEP},
             **{f"xi_c_dz{dz}": all_results[f"dz_{dz}"]["period_cvdms_A"] for dz in DZ_SWEEP},
             **{f"xi_f_dz{dz}": all_results[f"dz_{dz}"]["period_fourier_A"] for dz in DZ_SWEEP},
             col_idx=np.array(col_idx))
    with open(base + ".json", "w") as f:
        json.dump({
            "script": "p2_cross_material.py",
            "material": name,
            "params": {"energy_eV": energy, "sampling_A_per_px": sampling,
                       "thickness_A": float(total_z),
                       "supercell_xy": supercell_xy, "supercell_z": supercell_z,
                       "dz_sweep": DZ_SWEEP, "a_A": a},
            "results": {"xi_ch_analytic_A": float(xi_ch), "V_g_eVA": float(V_g)},
            "sweep": all_results,
        }, f, indent=2)
    print(f"Data saved: {base}.npz + .json")

    return {"name": name, "xi_ch": xi_ch, "V_g": V_g, "results": all_results,
            "gpts": gpts, "total_z": total_z}


def main():
    print("=== P2 Cross-Material: Channeling Pendellösung Δz Sweep ===")
    for name, params in MATERIALS.items():
        process_material(name, params)
    print("\n=== P2 Cross-Material Complete ===")


if __name__ == "__main__":
    main()
