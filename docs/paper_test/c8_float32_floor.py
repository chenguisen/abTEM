#!/usr/bin/env python3
"""C8: Float32 precision floor — quantify numerical amplification in CVDMS.

float32 cancellation in the K-series sum causes I/I0 > 1 (non-unitary).
This maps the (Δz, t) region where float32 error exceeds 1% in I/I0.

SrTiO3 [001] @ 30 keV, probe on Sr column.
Sweeps Δz ∈ {0.2, 0.4, 0.8, 1.0, 1.5, 2.0} × t ∈ {20, 50, 100, 200, 300} Å.

Outputs: docs/data/c8_float32.npz + .json
"""

import sys, os, json, gc
import numpy as np
import cupy as cp
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice
from abtem.core import config as _cfg
from abtem.core.energy import energy2wavelength

_cfg.set({"device": "gpu", "fft": "cupy"})
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

ENERGY = 30e3
A = 3.905
SUPERCELL_XY = 4
SAMPLING = 0.10  # Å/px — moderate, antialias not needed
CONVERGENCE_THRESHOLD = 1e-7

DZ_SWEEP = [0.2, 0.4, 0.8, 1.0, 1.5, 2.0]
SUPERCELL_Z_SWEEP = [5, 13, 26, 51, 77]  # target t ≈ {20, 50, 100, 200, 300} Å


def to_cpu(arr):
    if hasattr(arr, "get"): return arr.get()
    return np.asarray(arr)


def main():
    print("=== C8: Float32 Precision Floor ===")
    print(f"SrTiO3 [001] @ {ENERGY/1000:.0f} keV, Δx={SAMPLING} Å/px")

    # Find Sr column position
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
    print(f"Sr at ({sr_pos_A[0]:.1f}, {sr_pos_A[1]:.1f}) Å, Grid: {gpts}")

    del V_data, ref_pot, atoms_ref
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    results = []
    total = len(DZ_SWEEP) * len(SUPERCELL_Z_SWEEP)
    count = 0

    for dz in DZ_SWEEP:
        for sc_z in SUPERCELL_Z_SWEEP:
            count += 1
            atoms = crystal(
                ["Sr", "Ti", "O"],
                basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                spacegroup=221, cellpar=[A, A, A, 90, 90, 90],
            )
            atoms *= (SUPERCELL_XY, SUPERCELL_XY, sc_z)
            total_z = atoms.cell[2, 2]
            n_slices = int(total_z / dz)

            probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=10,
                                extent=SUPERCELL_XY * A, gpts=gpts[0],
                                device="gpu")
            potential = abtem.Potential(atoms, sampling=SAMPLING,
                                        slice_thickness=dz, exit_planes=1,
                                        projection="finite")
            probe.grid.match(potential)
            probe.positions = [(sr_pos_A[0], sr_pos_A[1])]

            print(f"[{count}/{total}] Δz={dz:.1f} Å, t={total_z:.0f} Å "
                  f"(sc_z={sc_z}, slices={n_slices})...", end=" ")

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

                psi_entrance = arr[0]
                psi_exit = arr[-1]

                I_entrance = float((np.abs(psi_entrance)**2).sum())
                I_exit = float((np.abs(psi_exit)**2).sum())
                I_ratio = I_exit / I_entrance if I_entrance > 0 else 0

                has_inf = np.isinf(psi_exit).any()
                has_nan = np.isnan(psi_exit).any()

                # Measure total wave function power per slice to detect growth
                slice_powers = [(float((np.abs(arr[i])**2).sum()) / I_entrance)
                                for i in range(arr.shape[0])]
                max_power_ratio = max(slice_powers)
                min_power_ratio = min(slice_powers)

                del ew, cvdms
            except Exception as e:
                I_ratio = 0
                overflow = True
                has_inf = has_nan = True
                max_power_ratio = min_power_ratio = float('nan')
                print(f"  ERROR: {e}")

            r = {
                "dz_A": dz, "thickness_A": float(total_z),
                "supercell_z": sc_z, "n_slices": n_slices,
                "I_ratio": I_ratio,
                "I_excess": I_ratio - 1.0,  # < 0 means numerical diffusion
                "max_power_ratio": max_power_ratio,
                "min_power_ratio": min_power_ratio,
                "overflow": bool(overflow),
                "has_inf": bool(has_inf), "has_nan": bool(has_nan),
            }
            results.append(r)
            print(f"I/I0 = {I_ratio:.8f} "
                  f"({'overflow!' if overflow else 'OK' if abs(I_ratio - 1) < 0.01 else 'WARN'})")

            del atoms, probe, potential, arr
            gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    # Summary
    print(f"\n{'='*70}")
    print("C8 Float32 Precision Floor Summary:")
    print(f"  {'Δz':>5s}  {'t (Å)':>8s}  {'slices':>7s}  {'I/I0':>12s}  {'I/I0−1':>12s}  Status")

    good = [r for r in results if abs(r["I_ratio"] - 1) < 0.01 and not r["overflow"]]
    warn = [r for r in results if 0.01 <= abs(r["I_ratio"] - 1) < 0.1 and not r["overflow"]]
    bad = [r for r in results if abs(r["I_ratio"] - 1) >= 0.1 or r["overflow"]]

    for r in results:
        status = "OVERFLOW" if r["overflow"] else \
                 "BAD" if abs(r["I_ratio"] - 1) >= 0.1 else \
                 "WARN" if abs(r["I_ratio"] - 1) >= 0.01 else "OK"
        print(f"  {r['dz_A']:>5.1f}  {r['thickness_A']:>8.1f}  {r['n_slices']:>7d}  "
              f"{r['I_ratio']:>12.8f}  {r['I_excess']:>+12.2e}  {status}")

    print(f"\nGood (|I/I0−1|<1%): {len(good)} | Warn (1-10%): {len(warn)} | Bad (>10%): {len(bad)}")

    # Save
    base = os.path.join(DATA_DIR, "c8_float32")
    np.savez(base + ".npz",
             dz=np.array([r["dz_A"] for r in results]),
             thickness=np.array([r["thickness_A"] for r in results]),
             I_ratio=np.array([r["I_ratio"] for r in results]),
             I_excess=np.array([r["I_excess"] for r in results]))
    with open(base + ".json", "w") as f:
        json.dump({
            "script": "c8_float32_floor.py",
            "params": {"energy_eV": ENERGY, "sampling_A_per_px": SAMPLING,
                       "convergence_threshold": CONVERGENCE_THRESHOLD,
                       "dz_sweep": DZ_SWEEP,
                       "supercell_z_sweep": SUPERCELL_Z_SWEEP},
            "results": results,
            "summary": {"n_good": len(good), "n_warn": len(warn), "n_bad": len(bad)}
        }, f, indent=2)
    print(f"Data saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
