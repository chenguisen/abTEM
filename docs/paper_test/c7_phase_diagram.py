#!/usr/bin/env python3
"""C7: (ρ,η) phase diagram — dimensionless stability map.

Uses P1 material parameters and runs CVDMS at sampled (ρ,η) points
to classify convergent / conditional / divergent regimes.

ρ = Δz / ℓ_mfp  (scattering strength per slice)
η = r_F / w_col  (diffraction-to-column-width ratio)

Outputs: docs/data/c7_phase.npz + .json
"""

import sys, os, json, gc
import numpy as np
import cupy as cp
from scipy.ndimage import center_of_mass
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice
from abtem.core import config as _cfg
from abtem.core.energy import energy2wavelength, energy2sigma

_cfg.set({"device": "gpu", "fft": "cupy"})
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CONVERGENCE_THRESHOLD = 1e-7
THICKNESS = 200  # Å — fixed thickness for phase diagram
SAMPLING = 0.10  # Å/px


def to_cpu(arr):
    if hasattr(arr, "get"): return arr.get()
    return np.asarray(arr)


# Material definitions
MATERIALS = {
    "SrTiO3": {"a": 3.905, "sg": 221, "symbols": ["Sr", "Ti", "O"],
               "basis": [(0,0,0), (0.5,0.5,0.5), (0.5,0.5,0)]},
    "Si": {"a": 5.431, "sg": 227, "symbols": ["Si"],
           "basis": [(0,0,0), (0.25,0.25,0.25)]},
    "Au": {"a": 4.078, "sg": 225, "symbols": ["Au"],
           "basis": [(0,0,0)]},
}

VOLTAGES = [30e3, 100e3, 300e3]
DZ_VALUES = [0.2, 0.4, 0.8, 1.0, 2.0]  # Including Δz > 1 Å deliberately


def compute_material_params(name, params, voltage):
    """Compute V_rms, w_col, ℓ_mfp, r_F for a material-voltage pair."""
    a = params["a"]
    sigma = energy2sigma(voltage)
    wavelength = energy2wavelength(voltage)

    atoms = crystal(params["symbols"], basis=params["basis"],
                    spacegroup=params["sg"],
                    cellpar=[a, a, a, 90, 90, 90])
    atoms *= (4, 4, 1)

    pot = abtem.Potential(atoms, sampling=SAMPLING, slice_thickness=a,
                          exit_planes=1, projection="finite")
    V_data = pot.build(lazy=False)
    V_2d = to_cpu(V_data.array[0])
    px_A = pot.sampling[0]

    V_rms = float(np.std(V_2d))
    V_peak = float(V_2d.max())
    col_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
    gpts = tuple(pot.gpts)

    # Column width: FWHM of V around peak
    cy, cx = col_idx
    profile_x = V_2d[cy, :]
    half_max = V_peak / 2
    above = np.where(profile_x > half_max)[0]
    w_col = float((above[-1] - above[0]) * px_A) if len(above) > 1 else a/4

    del V_data, pot, atoms
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    ell_mfp = 1.0 / (sigma * V_rms) if V_rms > 0 else float('inf')

    return {"V_rms": V_rms, "V_peak": V_peak, "w_col_A": w_col,
            "sigma": sigma, "wavelength_A": wavelength,
            "ell_mfp_A": ell_mfp, "col_idx": col_idx, "gpts": gpts,
            "extent": float(a * 4)}


def run_and_classify(name, mat_params, voltage, params, dz):
    """Run CVDMS at given Δz and classify regime."""
    a = mat_params["a"]
    supercell_z = int(THICKNESS / a) + 1

    atoms = crystal(mat_params["symbols"], basis=mat_params["basis"],
                    spacegroup=mat_params["sg"],
                    cellpar=[a, a, a, 90, 90, 90])
    atoms *= (4, 4, supercell_z)
    total_z = atoms.cell[2, 2]

    col_idx = params["col_idx"]
    px_A = SAMPLING
    col_pos_A = (float(col_idx[1] * px_A), float(col_idx[0] * px_A))

    probe = abtem.Probe(energy=voltage, semiangle_cutoff=10,
                        extent=4*a, gpts=params["gpts"][0], device="gpu")
    potential = abtem.Potential(atoms, sampling=SAMPLING, slice_thickness=dz,
                                exit_planes=1, projection="finite")
    probe.grid.match(potential)
    probe.positions = [(col_pos_A[0], col_pos_A[1])]

    n_slices = int(total_z / dz)

    try:
        import warnings
        overflow = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cvdms = CVDMSMultislice(
                convergence_threshold=CONVERGENCE_THRESHOLD,
                backscattering=False, antialias=True, antialias_inner=True,
            )
            ew = probe.multislice(potential, algorithm=cvdms, lazy=False)
            for x in w:
                if "overflow" in str(x.message).lower():
                    overflow = True

        arr = to_cpu(ew.array)
        if arr.ndim == 4: arr = arr[:, 0, :, :]
        psi_exit = arr[-1]

        has_inf = np.isinf(psi_exit).any()
        has_nan = np.isnan(psi_exit).any()

        if has_inf or has_nan or overflow:
            regime = "divergent"
        else:
            # Check I/I₀ ratio
            I_exit = np.abs(psi_exit)**2
            I_entrance = np.abs(arr[0])**2
            I_ratio = float(I_exit.sum() / (I_entrance.sum() + 1e-30))

            if 0.99 < I_ratio < 1.01:
                regime = "convergent"
            elif 0.9 < I_ratio < 1.1:
                regime = "conditional"
            else:
                regime = "divergent"

        del ew, cvdms
    except Exception:
        regime = "divergent"
        I_ratio = 0

    del atoms, probe, potential
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    return {"regime": regime, "n_slices": n_slices, "dz": dz,
            "thickness": float(total_z)}


def main():
    print("=== C7: (ρ,η) Phase Diagram ===")

    # First compute all material params
    all_params = {}
    for name, mat in MATERIALS.items():
        for V in VOLTAGES:
            key = f"{name}_{V/1000:.0f}keV"
            print(f"Computing params: {key}...")
            all_params[key] = compute_material_params(name, mat, V)
            cp.get_default_memory_pool().free_all_blocks()

    # Now run at selected (ρ,η) points
    results = []
    total = len(MATERIALS) * len(VOLTAGES) * len(DZ_VALUES)
    count = 0
    for name, mat in MATERIALS.items():
        for V in VOLTAGES:
            key = f"{name}_{V/1000:.0f}keV"
            p = all_params[key]
            for dz in DZ_VALUES:
                count += 1
                r_F = np.sqrt(p["wavelength_A"] * dz)
                rho = dz / p["ell_mfp_A"] if p["ell_mfp_A"] > 0 else float('inf')
                eta = r_F / p["w_col_A"] if p["w_col_A"] > 0 else float('inf')

                # Skip if ρ < 0.01 (essentially no scattering) or ρ > 5 (definitely divergent)
                if rho > 5:
                    regime_info = {"regime": "divergent", "dz": dz,
                                   "n_slices": 0, "thickness": THICKNESS}
                elif rho < 0.01:
                    regime_info = {"regime": "convergent", "dz": dz,
                                   "n_slices": 0, "thickness": THICKNESS}
                else:
                    print(f"[{count}/{total}] {key} Δz={dz:.1f} Å (ρ={rho:.3f}, η={eta:.3f})...")
                    regime_info = run_and_classify(name, mat, V, p, dz)

                results.append({
                    "material": name, "voltage_keV": V/1000,
                    "dz_A": dz, "rho": float(rho), "eta": float(eta),
                    "r_F_A": float(r_F), "ell_mfp_A": float(p["ell_mfp_A"]),
                    "w_col_A": float(p["w_col_A"]),
                    **regime_info,
                })
                print(f"  → {regime_info['regime']}")

    # Summary
    regimes = {"convergent": 0, "conditional": 0, "divergent": 0}
    for r in results: regimes[r["regime"]] += 1
    print(f"\n{'='*60}")
    print(f"Phase diagram complete: {regimes}")

    # Find approximate phase boundary ρ_c
    conv_rhos = [r["rho"] for r in results if r["regime"] == "convergent"]
    div_rhos = [r["rho"] for r in results if r["regime"] == "divergent"]
    if conv_rhos and div_rhos:
        rho_boundary = (max(conv_rhos) + min(div_rhos)) / 2
        print(f"Estimated ρ_c ≈ {rho_boundary:.3f}")

    # Save
    base = os.path.join(DATA_DIR, "c7_phase")
    np.savez(base + ".npz",
             rho=np.array([r["rho"] for r in results]),
             eta=np.array([r["eta"] for r in results]))
    with open(base + ".json", "w") as f:
        json.dump({
            "script": "c7_phase_diagram.py",
            "params": {"thickness_target_A": THICKNESS,
                       "sampling_A_per_px": SAMPLING,
                       "dz_values": DZ_VALUES},
            "results": results,
        }, f, indent=2)
    print(f"Data saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
