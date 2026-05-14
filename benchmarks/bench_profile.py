"""
Combined optimization profile: tiled kernel + backscattering GPU fix.
"""
import time
import numpy as np
import cupy as cp
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice

abtem.config.set({"device": "gpu", "fft": "cupy", "diagnostics.task_progress": False})

a = 3.905
atoms = crystal(
    ["Sr", "Ti", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
    spacegroup=221,
    cellpar=[a, a, a, 90, 90, 90],
)
atoms *= (4, 4, 10)

frozen_phonons = abtem.FrozenPhonons(atoms, 1, {"Sr": 0.164356, "Ti": 0.116584, "O": 0.148198})
potential = abtem.Potential(
    frozen_phonons,
    sampling=0.05,
    projection="finite",
    slice_thickness=0.4,
    exit_planes=10,
)
gx, gy = potential.gpts
sz = potential.slice_thickness[0]
n_slices = int(atoms.cell[2, 2] / sz)
print(f"Grid: {gx}x{gy}, slices: {n_slices}")

wave = abtem.Probe(energy=30e3, semiangle_cutoff=35)
wave.grid.match(potential)

cvdms = CVDMSMultislice(
    order=1, convergence_threshold=1e-6, max_terms=50,
    use_fused_kernel=True, backscattering=True,
)

t0 = time.time()
measurements = wave.multislice(potential, algorithm=cvdms).diffraction_patterns(max_angle="cutoff")
measurements = measurements.mean(0)
measurements.compute()
total_time = time.time() - t0

print(f"\n=== Combined (tiled kernel + backscattering GPU fix) ===")
print(f"Total: {total_time:.1f}s")
print(f"Grid: {gx}x{gy}, slices: {n_slices}")
print(f"Avg per slice: {total_time/n_slices*1000:.1f}ms")
