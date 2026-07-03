#!/usr/bin/env python3
"""V2: 100-unit-cell outer Taylor divergence reproduction.

Runs CVDMS forward propagation through 100 SrTiO3 unit cells (~390 Å)
and monitors I/I₀ at every exit plane to detect float32 accumulation
overflow / divergence.

Stages:
  A: coarse sampling (0.20 Å/px), thin — should pass cleanly
  B: fine sampling (0.05 Å/px), 100 cells — expected divergence

Usage:
  python v2_divergence.py A
  python v2_divergence.py B
  python v2_divergence.py A B
"""
import sys, os, json, gc, warnings
import numpy as np
import cupy as cp
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice
from abtem.core import config as _cfg

_cfg.set({"device": "gpu", "fft": "cupy"})

ENERGY = 30e3
SLICE_THICKNESS = 0.4
CONVERGENCE_THRESHOLD = 1e-7
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return arr


def total_intensity(arr):
    return float(np.sum(np.abs(to_cpu(arr)) ** 2))


def cleanup():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def run_stage(stage):
    if stage == "B":
        supercell_xy = 8
        supercell_z = 100       # ~390 Å
        sampling = 0.05
        exit_planes = 80
    else:
        supercell_xy = 4
        supercell_z = 100       # same thickness, coarser grid
        sampling = 0.20
        exit_planes = 40

    print(f"=== V2: 100-cell Divergence (Stage {stage}) ===")
    print(f"Supercell: ({supercell_xy},{supercell_xy},{supercell_z})")
    print(f"Sampling: {sampling} Å/px")

    a = 3.905
    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[a, a, a, 90, 90, 90],
    )
    atoms *= (supercell_xy, supercell_xy, supercell_z)
    total_z = atoms.cell[2, 2]
    n_slices = int(total_z / SLICE_THICKNESS)
    print(f"Thickness: {total_z:.1f} Å  slices: {n_slices}")

    potential = abtem.Potential(
        atoms,
        sampling=sampling,
        slice_thickness=SLICE_THICKNESS,
        exit_planes=exit_planes,
        projection="finite",
    )
    print(f"Grid: {potential.gpts}  px={potential.sampling[0]:.4f} Å/px")
    print(f"Exit planes: {potential.num_exit_planes}")

    wave = abtem.PlaneWave(energy=ENERGY)
    wave.grid.match(potential)

    cvdms = CVDMSMultislice(
        convergence_threshold=CONVERGENCE_THRESHOLD,
        backscattering=True,
        calculate_backscattered=True,
        antialias=True,
        antialias_inner=True,
    )

    overflow_warnings = []

    print("Running CVDMS forward propagation...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exit_wave = wave.multislice(potential, algorithm=cvdms, lazy=False)

        for warning in w:
            msg = str(warning.message)
            if "overflow" in msg.lower() or "diverg" in msg.lower():
                overflow_warnings.append(msg)
                print(f"  [WARNING] {msg}")

    print(f"Exit wave shape: {exit_wave.shape}")

    I0 = total_intensity(exit_wave.array[0])
    ratios = []
    depths = []
    max_deviation = 0.0
    above_one = False
    non_monotonic = False

    print(f"\n{'EP':>4s}  {'Depth':>7s}  {'I/I0':>10s}  {'|1-I/I0|':>12s}")
    print("-" * 44)

    for ep in range(exit_wave.shape[0]):
        I_ep = total_intensity(exit_wave.array[ep])
        ratio = I_ep / I0
        dev = abs(1.0 - ratio)
        max_deviation = max(max_deviation, dev)
        depth = ep * (total_z / (exit_wave.shape[0] - 1)) if exit_wave.shape[0] > 1 else total_z
        if ratio > 1.0:
            above_one = True
        if ep > 0 and ratio > ratios[-1]:
            non_monotonic = True
        ratios.append(ratio)
        depths.append(depth)
        flag = ""
        if ratio > 1.0:
            flag = "  >1!"
        if ep > 0 and ratio > ratios[-2]:
            flag += "  non-monotonic"
        print(f"{ep:>4d}  {depth:>6.1f}Å  {ratio:10.8f}  {dev:12.2e}{flag}")

    print(f"\nMax |1 - I/I₀| = {max_deviation:.2e}")
    diverged = above_one or non_monotonic or len(overflow_warnings) > 0
    print(f"I/I₀ > 1: {above_one}  |  non-monotonic: {non_monotonic}  |  overflow: {len(overflow_warnings) > 0}")
    print(f"Divergence: {diverged}")

    loss_pct = (1.0 - ratios[-1]) * 100

    base = os.path.join(DATA_DIR, f"v2_divergence_{stage}")
    np.savez(
        base + ".npz",
        depths=np.array(depths),
        I_ratio=np.array(ratios),
        I0=I0,
        max_deviation=max_deviation,
        loss_pct=loss_pct,
        diverged=diverged,
        above_one=above_one,
        non_monotonic=non_monotonic,
    )
    manifest = {
        "script": "v2_divergence.py",
        "stage": stage,
        "params": {
            "energy_eV": ENERGY,
            "slice_thickness_A": SLICE_THICKNESS,
            "convergence_threshold": CONVERGENCE_THRESHOLD,
            "supercell": [supercell_xy, supercell_xy, supercell_z],
            "grid": list(potential.gpts),
            "sampling_A_per_px": sampling,
            "num_exit_planes": exit_planes,
            "thickness_A": total_z,
            "num_slices": n_slices,
            "antialias": True,
            "antialias_inner": True,
            "backscattering": True,
        },
        "results": {
            "I0": I0,
            "max_deviation": max_deviation,
            "loss_pct": loss_pct,
            "diverged": diverged,
            "above_one": above_one,
            "non_monotonic": non_monotonic,
            "overflow_warnings": overflow_warnings,
        },
    }
    with open(base + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Data saved: {base}.npz + .json")

    del exit_wave
    cleanup()
    return diverged


if __name__ == "__main__":
    stages = sys.argv[1:] if len(sys.argv) > 1 else ["A", "B"]
    for s in stages:
        s_upper = s.upper()
        if s_upper not in ("A", "B"):
            print(f"Unknown stage: {s}, skipping")
            continue
        run_stage(s_upper)
