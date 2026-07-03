#!/usr/bin/env python3
"""P1-DW: Debye-Waller broadened V_peak verification.

Computes DW-broadened projected potentials for SrTiO3, Si, Au at 300 K
and verifies V_peak values stated in paper Introduction (Section 1.4).

DW factor: exp(-B q²/4) applied in reciprocal space.
Static V(R) → FFT → multiply by exp(-B q²/4) → IFFT → V^(B)(R).

Outputs: docs/data/p1_dw_broadening.npz + .json
"""

import sys, os, json, gc
import numpy as np
from ase.spacegroup import crystal
import abtem
from abtem.core import config as _cfg

_cfg.set({"device": "gpu", "fft": "cupy"})
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SAMPLING = 0.02  # Å/px — fine for accurate V_peak
SUPERCELL_XY = 4

MATERIALS = {
    "SrTiO3": {
        "symbols": ["Sr", "Ti", "O"],
        "basis": [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        "spacegroup": 221, "a": 3.905,
        "B": {"Sr": 0.62, "Ti": 0.51, "O": 0.86},  # Å² at 300 K
        "V_peak_paper": 1060,  # eV·Å, stated in paper
    },
    "Si": {
        "symbols": ["Si"],
        "basis": [(0, 0, 0), (0.25, 0.25, 0.25)],
        "spacegroup": 227, "a": 5.431,
        "B": {"Si": 0.46},
        "V_peak_paper": 470,
    },
    "Au": {
        "symbols": ["Au"],
        "basis": [(0, 0, 0)],
        "spacegroup": 225, "a": 4.078,
        "B": {"Au": 0.66},
        "V_peak_paper": 2300,
    },
}


def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def apply_dw_broadening(V_2d, px_A, B_iso):
    """Apply isotropic Debye-Waller factor exp(-B q²/4) in reciprocal space.

    Parameters
    ----------
    V_2d : 2D array
        Static projected potential (eV·Å).
    px_A : float
        Pixel size in Å.
    B_iso : float
        Isotropic DW factor B in Å².

    Returns
    -------
    V_dw : 2D array
        DW-broadened projected potential.
    """
    V_cpu = to_cpu(V_2d)
    ny, nx = V_cpu.shape
    V_k = np.fft.fft2(V_cpu)
    kx = np.fft.fftfreq(nx, px_A)
    ky = np.fft.fftfreq(ny, px_A)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    q2 = KX**2 + KY**2
    dw_factor = np.exp(-B_iso * q2 / 4.0)
    V_k_dw = V_k * dw_factor
    V_dw = np.fft.ifft2(V_k_dw).real
    return V_dw


def build_static_potential(mat_key):
    """Build static (no DW) projected potential for one unit cell in xy."""
    mat = MATERIALS[mat_key]
    atoms = crystal(
        mat["symbols"], basis=mat["basis"],
        spacegroup=mat["spacegroup"],
        cellpar=[mat["a"]] * 3 + [90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, 1)
    potential = abtem.Potential(
        atoms, sampling=SAMPLING,
        slice_thickness=mat["a"],
        exit_planes=1, projection="finite",
    )
    V_data = potential.build(lazy=False)
    V_2d = V_data.array[0] if hasattr(V_data.array, "__len__") else V_data.array
    return to_cpu(V_2d), potential.sampling[0], potential.gpts


def main():
    print("=== P1-DW: Debye-Waller Broadened V_peak Verification ===\n")

    results = {}
    for name in ["Si", "SrTiO3", "Au"]:
        mat = MATERIALS[name]
        print(f"--- {name} ---")
        V_static, px_A, gpts = build_static_potential(name)
        V_static_peak = float(V_static.max())

        # Get effective isotropic B for the dominant scatterer
        # For multi-element materials, use the B of the heaviest element
        # This is approximate — full treatment needs species-resolved DW
        if name == "SrTiO3":
            # Weighted average B
            B_eff = mat["B"]["Sr"]  # Sr dominates the column potential
        else:
            B_eff = list(mat["B"].values())[0]

        print(f"  Static V_peak = {V_static_peak:.0f} eV·Å")
        print(f"  B_eff = {B_eff} Å²")

        V_dw = apply_dw_broadening(V_static, px_A, B_eff)
        V_dw_peak = float(V_dw.max())
        reduction_pct = (1 - V_dw_peak / V_static_peak) * 100
        print(f"  DW-broadened V_peak = {V_dw_peak:.0f} eV·Å  (reduction: {reduction_pct:.1f}%)")

        paper_val = mat["V_peak_paper"]
        rel_diff = abs(V_dw_peak - paper_val) / paper_val * 100
        print(f"  Paper value = {paper_val} eV·Å  (relative diff: {rel_diff:.1f}%)")

        results[name] = {
            "V_static_peak_eVA": float(V_static_peak),
            "V_dw_peak_eVA": float(V_dw_peak),
            "B_eff_A2": B_eff,
            "reduction_pct": float(reduction_pct),
            "paper_value_eVA": paper_val,
            "rel_diff_pct": float(rel_diff),
        }

        gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print("Summary: DW-broadened V_peak verification")
    print(f"  {'Material':>10s}  {'Static':>8s}  {'DW':>8s}  {'Paper':>8s}  {'Diff%':>8s}")
    for name, r in results.items():
        print(f"  {name:>10s}  {r['V_static_peak_eVA']:>8.0f}  {r['V_dw_peak_eVA']:>8.0f}  "
              f"{r['paper_value_eVA']:>8.0f}  {r['rel_diff_pct']:>7.1f}%")

    # Save
    base = os.path.join(DATA_DIR, "p1_dw_broadening")
    np.savez(base + ".npz")
    with open(base + ".json", "w") as f:
        json.dump({
            "script": "p1_dw_broadening.py",
            "params": {"sampling_A_per_px": SAMPLING, "supercell_xy": SUPERCELL_XY,
                       "temperature_K": 300},
            "results": results,
        }, f, indent=2)
    print(f"\nData saved: {base}.json")


if __name__ == "__main__":
    main()
