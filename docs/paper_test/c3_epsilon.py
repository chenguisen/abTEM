#!/usr/bin/env python3
"""C3: Convergence threshold sensitivity — cross-material ε sweep.

3 materials (Si @ 100keV, SrTiO3 @ 30keV, Au @ 300keV), Δz=0.4 Å, t≈200 Å.
Sweeps ε ∈ {1e-4, ..., 1e-9}. Measures I/I0, NCC vs ε=1e-9 ref.

Outputs: docs/data/c3_epsilon.npz + .json
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

SAMPLING = 0.10  # Å/px
SUPERCELL_XY = 4
DZ = 0.4  # Å

MATERIALS = [
    {"name": "SrTiO3", "a": 3.905, "sg": 221, "symbols": ["Sr", "Ti", "O"],
     "basis": [(0,0,0), (0.5,0.5,0.5), (0.5,0.5,0)],
     "energy": 30e3, "supercell_z": 51},   # t ≈ 199 Å
    {"name": "Si", "a": 5.431, "sg": 227, "symbols": ["Si"],
     "basis": [(0,0,0), (0.25,0.25,0.25)],
     "energy": 100e3, "supercell_z": 37},   # t ≈ 201 Å
    {"name": "Au", "a": 4.078, "sg": 225, "symbols": ["Au"],
     "basis": [(0,0,0)],
     "energy": 300e3, "supercell_z": 49},   # t ≈ 200 Å
]

EPS_VALUES = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]


def to_cpu(arr):
    if hasattr(arr, "get"): return arr.get()
    return np.asarray(arr)


def ncc(a, b):
    a_f = a.ravel(); b_f = b.ravel()
    denom = np.sqrt(np.dot(a_f.conj(), a_f).real * np.dot(b_f.conj(), b_f).real)
    return float(np.abs(np.dot(a_f.conj(), b_f)) / denom) if denom > 0 else 0.0


def main():
    print("=== C3: Convergence Threshold Sensitivity ===")

    all_results = {}

    for mat in MATERIALS:
        name = mat["name"]
        E = mat["energy"]
        sc_z = mat["supercell_z"]
        print(f"\n{'='*60}")
        print(f"{name} @ {E/1000:.0f} keV, Δz={DZ} Å, supercell_z={sc_z}")

        atoms = crystal(mat["symbols"], basis=mat["basis"],
                        spacegroup=mat["sg"],
                        cellpar=[mat["a"]] * 3 + [90, 90, 90])
        atoms *= (SUPERCELL_XY, SUPERCELL_XY, sc_z)
        total_z = atoms.cell[2, 2]
        n_slices = int(total_z / DZ)

        # Find column position
        ref_pot = abtem.Potential(atoms, sampling=SAMPLING, slice_thickness=mat["a"],
                                   exit_planes=1, projection="finite")
        V_data = ref_pot.build(lazy=False)
        V_2d = to_cpu(V_data.array[0])
        gpts = ref_pot.gpts
        px_A = ref_pot.sampling[0]
        col_idx = np.unravel_index(V_2d.argmax(), V_2d.shape)
        col_pos_A = (float(col_idx[1] * px_A), float(col_idx[0] * px_A))
        print(f"  Grid: {gpts}, column at ({col_pos_A[0]:.1f}, {col_pos_A[1]:.1f}) Å, "
              f"t={total_z:.1f} Å, slices={n_slices}")

        del V_data, ref_pot
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

        # Reference: ε = 1e-9
        print(f"  Reference ε=1e-9...")
        probe_ref = abtem.Probe(energy=E, semiangle_cutoff=10,
                                 extent=SUPERCELL_XY * mat["a"], gpts=gpts[0],
                                 device="gpu")
        potential = abtem.Potential(atoms, sampling=SAMPLING, slice_thickness=DZ,
                                     exit_planes=1, projection="finite")
        probe_ref.grid.match(potential)
        probe_ref.positions = [(col_pos_A[0], col_pos_A[1])]

        cvdms_ref = CVDMSMultislice(convergence_threshold=1e-9,
                                     backscattering=False,
                                     antialias=True, antialias_inner=True)
        ew_ref = probe_ref.multislice(potential, algorithm=cvdms_ref, lazy=False)
        arr_ref = to_cpu(ew_ref.array)
        if arr_ref.ndim == 4: arr_ref = arr_ref[:, 0, :, :]
        psi_ref = arr_ref[-1]
        I_ref = float((np.abs(psi_ref)**2).sum())
        I0_ref = float((np.abs(arr_ref[0])**2).sum())
        del ew_ref, cvdms_ref, probe_ref
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

        # ε sweep
        results = []
        for eps in EPS_VALUES:
            print(f"  ε={eps:.0e}...", end=" ")
            probe = abtem.Probe(energy=E, semiangle_cutoff=10,
                                 extent=SUPERCELL_XY * mat["a"], gpts=gpts[0],
                                 device="gpu")
            probe.grid.match(potential)
            probe.positions = [(col_pos_A[0], col_pos_A[1])]

            try:
                import warnings
                overflow = False
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    cvdms = CVDMSMultislice(convergence_threshold=eps,
                                             backscattering=False,
                                             antialias=True, antialias_inner=True)
                    ew = probe.multislice(potential, algorithm=cvdms, lazy=False)
                    for x in w:
                        if "overflow" in str(x.message).lower():
                            overflow = True

                arr = to_cpu(ew.array)
                if arr.ndim == 4: arr = arr[:, 0, :, :]
                psi = arr[-1]
                I_exit = float((np.abs(psi)**2).sum())
                I_entrance = float((np.abs(arr[0])**2).sum())
                I_ratio = I_exit / I_entrance if I_entrance > 0 else 0
                ncc_val = ncc(psi, psi_ref)
                has_inf = np.isinf(psi).any()
                has_nan = np.isnan(psi).any()
                del ew, cvdms
            except Exception as e:
                I_ratio = 0; ncc_val = 0
                overflow = True; has_inf = has_nan = True
                print(f"ERROR: {e}")

            r = {"eps": float(eps), "I_ratio": I_ratio, "ncc": ncc_val,
                 "overflow": bool(overflow), "has_inf": bool(has_inf),
                 "has_nan": bool(has_nan)}
            results.append(r)
            status = "OVERFLOW" if overflow else ("OK" if ncc_val > 0.99999 else f"NCC={ncc_val:.6f}")
            print(f"I/I0={I_ratio:.8f} NCC={ncc_val:.8f} {status}")

            del probe, arr
            gc.collect(); cp.get_default_memory_pool().free_all_blocks()

        del atoms, potential
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()

        all_results[name] = {
            "energy_eV": float(E), "thickness_A": float(total_z),
            "n_slices": n_slices, "gpts": list(gpts),
            "I0_ref": I0_ref, "I_ref": I_ref,
            "sweep": results,
        }

    # Summary
    print(f"\n{'='*70}")
    print("C3 Summary: NCC vs ε=1e-9 reference")
    print(f"  {'ε':>10s}  {'SrTiO3 NCC':>14s}  {'Si NCC':>14s}  {'Au NCC':>14s}")
    for eps in EPS_VALUES:
        ncc_vals = []
        for mat in MATERIALS:
            r = [x for x in all_results[mat["name"]]["sweep"] if x["eps"] == eps][0]
            ncc_vals.append(r["ncc"])
        print(f"  {eps:>10.0e}  {ncc_vals[0]:>14.8f}  {ncc_vals[1]:>14.8f}  {ncc_vals[2]:>14.8f}")

    # Find ε where NCC > 1-1e-6 for all materials
    for mat in MATERIALS:
        name = mat["name"]
        for r in all_results[name]["sweep"]:
            if r["ncc"] > 1 - 1e-6 and not r["overflow"]:
                print(f"  {name}: NCC > 1-1e-6 at ε ≤ {r['eps']:.0e}")
                break

    # Save
    base = os.path.join(DATA_DIR, "c3_epsilon")
    np.savez(base + ".npz",
             eps=np.array(EPS_VALUES),
             ncc_srtio3=np.array([r["ncc"] for r in all_results["SrTiO3"]["sweep"]]),
             ncc_si=np.array([r["ncc"] for r in all_results["Si"]["sweep"]]),
             ncc_au=np.array([r["ncc"] for r in all_results["Au"]["sweep"]]))
    with open(base + ".json", "w") as f:
        json.dump({"script": "c3_epsilon.py",
                   "params": {"dz_A": DZ, "sampling_A_per_px": SAMPLING,
                              "supercell_xy": SUPERCELL_XY,
                              "eps_values": EPS_VALUES},
                   "results": all_results}, f, indent=2)
    print(f"\nData saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
