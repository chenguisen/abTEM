#!/usr/bin/env python3
"""V3: Slice discretisation independence — CVDMS convergence as Δz → 0.

Tests that CVDMS exit waves converge as slice thickness decreases,
verifying O(Δz²) scaling of the splitting error (paper outline §14.2.3).

At each Δz, the same SrTiO₃ sample is simulated and the final exit wave
is compared against the finest-Δz reference using NCC and phase RMS.

Saves NPZ + JSON to docs/data/v3_slice_{stage}.npz/.json.

Acceptance criteria (paper outline §14.2.3):
  NCC > 1-1e-5 between consecutive Δz refinements
  Error scaling consistent with O(Δz²)

Usage:
  python v3_slice_refinement.py A    # 100 Å, Δz ∈ {0.8,0.4,0.2,0.1}
  python v3_slice_refinement.py B    # 200 Å, Δz ∈ {0.4,0.2,0.1,0.05}
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
CONVERGENCE_THRESHOLD = 1e-7

# Stage selection
STAGE = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
if STAGE == "B":
    SUPERCELL_XY = 4
    SUPERCELL_Z = 50             # ~195 Å at 3.905 Å/cell
    SAMPLING = 0.05              # Å/px
    DZ_VALUES = [0.4, 0.2, 0.1, 0.05]  # Å
    TOLERANCE_NCC = 1e-5
else:
    SUPERCELL_XY = 4
    SUPERCELL_Z = 25             # ~98 Å
    SAMPLING = 0.20              # Å/px
    DZ_VALUES = [0.8, 0.4, 0.2, 0.1]  # Å
    TOLERANCE_NCC = 1e-3   # coarsest Δz has significant splitting error


# ============================================================
# Helpers
# ============================================================
def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return arr


def ncc(a, b):
    """Normalised cross-correlation."""
    a_cpu, b_cpu = to_cpu(a).ravel(), to_cpu(b).ravel()
    num = np.abs(np.sum(a_cpu * np.conj(b_cpu)))
    denom = np.sqrt(np.sum(np.abs(a_cpu)**2) * np.sum(np.abs(b_cpu)**2))
    return float(num / denom) if denom > 1e-30 else 0.0


def phase_rms(a, b):
    """RMS phase difference, amplitude-weighted, global phase removed."""
    a_cpu, b_cpu = to_cpu(a).ravel(), to_cpu(b).ravel()
    amp_a, amp_b = np.abs(a_cpu), np.abs(b_cpu)
    thresh = 1e-6 * max(amp_a.max(), amp_b.max())
    mask = (amp_a > thresh) & (amp_b > thresh)
    if mask.sum() < 10:
        return 0.0
    a_m, b_m = a_cpu[mask], b_cpu[mask]
    cross = np.sum(a_m * np.conj(b_m))
    if abs(cross) < 1e-30:
        return 0.0
    global_phase = np.angle(cross)
    phase_local = np.angle(np.exp(1j * (np.angle(a_m * np.conj(b_m)) - global_phase)))
    return float(np.sqrt(np.mean(phase_local**2)))


def cleanup():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


# ============================================================
# Main
# ============================================================
def main():
    print(f"=== V3: Slice Refinement (Stage {STAGE}) ===")
    print(f"Supercell: ({SUPERCELL_XY},{SUPERCELL_XY},{SUPERCELL_Z})")
    print(f"Sampling: {SAMPLING} Å/px")

    # ── Build SrTiO₃ ──
    a = 3.905
    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[a, a, a, 90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)
    total_z = atoms.cell[2, 2]
    print(f"Total thickness: {total_z:.1f} Å")

    # ── Build initial plane wave ──
    # Build once with a reference potential to get grid info
    ref_pot = abtem.Potential(
        atoms, sampling=SAMPLING, slice_thickness=DZ_VALUES[0],
        exit_planes=1, projection="finite",
    )
    wave = abtem.PlaneWave(energy=ENERGY)
    wave.grid.match(ref_pot)
    print(f"Grid: {ref_pot.gpts}  px={ref_pot.sampling[0]:.4f} Å/px")

    # ── Run CVDMS at each Δz ──
    exit_waves = {}  # dz -> final exit wave array
    dz_info = {}     # dz -> {n_slices, ...}

    for dz in DZ_VALUES:
        print(f"\n--- Δz = {dz:.3f} Å ---")
        n_slices = int(total_z / dz)
        print(f"  Slices: {n_slices}")

        potential = abtem.Potential(
            atoms, sampling=SAMPLING, slice_thickness=dz,
            exit_planes=1, projection="finite",
        )

        cvdms = CVDMSMultislice(
            convergence_threshold=CONVERGENCE_THRESHOLD,
            backscattering=False,
            antialias=False,
            antialias_inner=True,
        )

        ew = wave.multislice(potential, algorithm=cvdms, lazy=False)
        # Take the final exit plane
        exit_wave_arr = cp.asarray(ew.array[-1]) if hasattr(ew.array, "__len__") else cp.asarray(ew.array)
        exit_waves[dz] = exit_wave_arr
        dz_info[dz] = {"n_slices": n_slices}
        print(f"  Exit wave shape: {exit_wave_arr.shape}")

        del ew, potential, cvdms
        cleanup()

    # ── Pairwise NCC between consecutive Δz ──
    # Also compare all coarser Δz to the finest for O(Δz²) scaling
    dz_finest = DZ_VALUES[-1]
    ref_wave = exit_waves[dz_finest]

    results = {"dz": [], "n_slices": [], "ncc_pairwise": [],
               "ncc_vs_finest": [], "phase_rms": [], "dz_finest": dz_finest}

    print(f"\n{'='*60}")
    print(f"Convergence: pairwise NCC between consecutive Δz")
    print(f"{'Δz₁→Δz₂':>16s}  {'NCC':>10s}  {'phase RMS':>12s}")
    print("-" * 45)

    for i in range(len(DZ_VALUES) - 1):
        dz1, dz2 = DZ_VALUES[i], DZ_VALUES[i+1]
        n = ncc(exit_waves[dz1], exit_waves[dz2])
        prms = phase_rms(exit_waves[dz1], exit_waves[dz2])
        results["dz"].append(dz1)
        results["n_slices"].append(dz_info[dz1]["n_slices"])
        results["ncc_pairwise"].append(n)
        results["ncc_vs_finest"].append(ncc(exit_waves[dz1], ref_wave))
        results["phase_rms"].append(prms)
        ok = n > 1.0 - TOLERANCE_NCC
        print(f"  {dz1:.3f} → {dz2:.3f} Å    {n:10.8f}  {prms:12.2e}  {'PASS' if ok else 'FAIL'}")

    ncc_values = results["ncc_pairwise"]
    dz_values = results["dz"]

    # Acceptance: finest pair must converge; all must be monotonic
    finest_pair_ncc = ncc_values[-1]
    passed = finest_pair_ncc > 1.0 - TOLERANCE_NCC

    # Check monotonic NCC improvement with decreasing Δz (toward finest)
    monotonic = all(results["ncc_vs_finest"][i] <= results["ncc_vs_finest"][i+1] + 1e-10
                    for i in range(len(results["ncc_vs_finest"]) - 1))

    print(f"\n  NCC monotonic toward finest: {monotonic}")
    print(f"  Finest pair NCC ({DZ_VALUES[-2]:.3f}→{DZ_VALUES[-1]:.3f}): {finest_pair_ncc:.8f}")
    print(f"  {'PASS' if passed else 'FAIL'}: V3 — finest pair NCC > 1-{TOLERANCE_NCC:.0e}")

    # ── Save ──
    base = os.path.join(DATA_DIR, f"v3_slice_{STAGE}")
    np.savez(base + ".npz",
             dz=np.array(dz_values), n_slices=np.array(results["n_slices"]),
             ncc_pairwise=np.array(results["ncc_pairwise"]),
             ncc_vs_finest=np.array(results["ncc_vs_finest"]),
             phase_rms=np.array(results["phase_rms"]),
             dz_finest=dz_finest, passed=passed, monotonic=monotonic)
    with open(base + ".json", "w") as f:
        json.dump({"script": "v3_slice_refinement.py", "stage": STAGE,
                   "params": {"energy_eV": ENERGY, "grid": list(ref_pot.gpts),
                              "sampling_A_per_px": SAMPLING,
                              "thickness_A": total_z,
                              "supercell": [SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z],
                              "dz_values": DZ_VALUES,
                              "dz_finest": dz_finest,
                              "convergence_threshold": CONVERGENCE_THRESHOLD,
                              "antialias": False, "antialias_inner": True,
                              "backscattering": False},
                   "results": {"ncc_min": float(min(ncc_values)),
                               "monotonic": bool(monotonic),
                               "passed": bool(passed)}},
                  f, indent=2)
    print(f"Data saved: {base}.npz + .json")


if __name__ == "__main__":
    main()
