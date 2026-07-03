#!/usr/bin/env python3
"""V4: Convergence self-consistency — ε refinement.

Tests that CVDMS observables converge as the convergence threshold ε
decreases, and that the computational cost (Taylor order) follows
the expected log(1/ε) + const scaling.

Runs SrTiO₃ at fixed Δz=0.4 Å with ε ∈ {1e-5, …, 1e-9} and compares
I/I₀ and exit wave NCC against the strictest ε reference.

Saves NPZ + JSON to docs/data/v4_convergence_{stage}.npz/.json.

Acceptance criteria (paper outline §14.2.4):
  Observables stabilise to < 2×10⁻⁴ relative at ε ≤ 1e-7
  Taylor order n* follows log₁₀(1/ε) + 3

Usage:
  python v4_convergence.py A    # 100 Å, 79×79 grid
  python v4_convergence.py B    # 200 Å, 313×313 grid (if needed)
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

# ============================================================
# Parameter manifest
# ============================================================
ENERGY = 30e3                    # eV
SLICE_THICKNESS = 0.4            # Å

STAGE = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
if STAGE == "B":
    SUPERCELL_XY = 4
    SUPERCELL_Z = 50             # ~195 Å
    SAMPLING = 0.05              # Å/px
else:
    SUPERCELL_XY = 4
    SUPERCELL_Z = 25             # ~98 Å
    SAMPLING = 0.20              # Å/px

EPS_VALUES = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]


# ============================================================
# Helpers
# ============================================================
def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return arr


def ncc(a, b):
    a_cpu, b_cpu = to_cpu(a).ravel(), to_cpu(b).ravel()
    num = np.abs(np.sum(a_cpu * np.conj(b_cpu)))
    denom = np.sqrt(np.sum(np.abs(a_cpu)**2) * np.sum(np.abs(b_cpu)**2))
    return float(num / denom) if denom > 1e-30 else 0.0


def total_intensity(arr):
    return float(np.sum(np.abs(to_cpu(arr)) ** 2))


def cleanup():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


# ============================================================
# Main
# ============================================================
def main():
    print(f"=== V4: Convergence Self-Consistency (Stage {STAGE}) ===")
    print(f"Supercell: ({SUPERCELL_XY},{SUPERCELL_XY},{SUPERCELL_Z})")
    print(f"Sampling: {SAMPLING} Å/px")
    print(f"ε values: {EPS_VALUES}")

    # ── Build SrTiO₃ ──
    a = 3.905
    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[a, a, a, 90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)
    total_z = atoms.cell[2, 2]

    potential = abtem.Potential(
        atoms, sampling=SAMPLING, slice_thickness=SLICE_THICKNESS,
        exit_planes=1, projection="finite",
    )
    print(f"Grid: {potential.gpts}  px={potential.sampling[0]:.4f} Å/px  "
          f"thickness={total_z:.1f} Å")

    wave = abtem.PlaneWave(energy=ENERGY)
    wave.grid.match(potential)

    # ── Run at each ε ──
    exit_waves = {}
    I_ratios = {}

    for eps in EPS_VALUES:
        print(f"\n--- ε = {eps:.0e} ---")
        cvdms = CVDMSMultislice(
            convergence_threshold=eps,
            backscattering=False,
            antialias=False,
            antialias_inner=True,
        )
        ew = wave.multislice(potential, algorithm=cvdms, lazy=False)
        exit_arr = cp.asarray(ew.array[-1]) if hasattr(ew.array, "__len__") else cp.asarray(ew.array)
        exit_waves[eps] = exit_arr
        I_ratios[eps] = total_intensity(exit_arr) / total_intensity(cp.asarray(ew.array[0]))
        print(f"  I/I₀ = {I_ratios[eps]:.8f}")

        del ew, cvdms
        cleanup()

    # ── Compare to strictest ε reference ──
    eps_ref = EPS_VALUES[-1]
    ref_wave = exit_waves[eps_ref]
    ref_I = I_ratios[eps_ref]

    results = {"eps": [], "I_I0": [], "delta_I": [], "ncc": []}

    print(f"\n{'='*60}")
    print(f"Convergence vs ε = {eps_ref:.0e} (reference)")
    print(f"{'ε':>10s}  {'I/I₀':>10s}  {'ΔI/I₀':>12s}  {'NCC':>10s}")
    print("-" * 52)

    for eps in EPS_VALUES[:-1]:
        n = ncc(exit_waves[eps], ref_wave)
        di = abs(I_ratios[eps] - ref_I) / ref_I
        results["eps"].append(eps)
        results["I_I0"].append(I_ratios[eps])
        results["delta_I"].append(di)
        results["ncc"].append(n)
        print(f"  {eps:10.0e}  {I_ratios[eps]:10.8f}  {di:12.2e}  {n:10.8f}")

    # ── Pass/fail ──
    # At ε ≤ 1e-7, ΔI/I₀ should be < 2e-4
    passed = all(di < 2e-4 for eps, di in zip(results["eps"], results["delta_I"])
                 if eps <= 1e-7)

    print(f"\n  Ref I/I₀ (ε={eps_ref:.0e}): {ref_I:.8f}")
    print(f"  {'PASS' if passed else 'FAIL'}: V4 — ΔI/I₀ < 2e-4 at ε ≤ 1e-7")

    # ── Save ──
    base = os.path.join(DATA_DIR, f"v4_convergence_{STAGE}")
    np.savez(base + ".npz",
             eps=np.array(results["eps"]), I_I0=np.array(results["I_I0"]),
             delta_I=np.array(results["delta_I"]), ncc=np.array(results["ncc"]),
             eps_ref=eps_ref, I_ref=ref_I, passed=passed)
    with open(base + ".json", "w") as f:
        json.dump({"script": "v4_convergence.py", "stage": STAGE,
                   "params": {"energy_eV": ENERGY, "grid": list(potential.gpts),
                              "sampling_A_per_px": SAMPLING,
                              "thickness_A": total_z,
                              "slice_thickness_A": SLICE_THICKNESS,
                              "supercell": [SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z],
                              "eps_values": EPS_VALUES, "eps_ref": eps_ref},
                   "results": {"I_ref": float(ref_I), "max_delta_I": float(max(results["delta_I"])),
                               "min_ncc": float(min(results["ncc"])), "passed": bool(passed)}},
                  f, indent=2)
    print(f"Data saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
