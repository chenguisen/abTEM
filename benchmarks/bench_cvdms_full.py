"""
Full CVDMS verification: run SrTiO3 (1 frozen phonon) with tiled kernel.
Measures performance without needing separate non-tiled comparison.
"""
import time
import numpy as np
import cupy as cp
from ase.build import bulk
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice

abtem.config.set({"device": "gpu", "fft": "cupy", "diagnostics.task_progress": False})

# ---- Build SrTiO3 (perovskite, Pm-3m, #221) ----
a = 3.905
atoms = crystal(
    ["Sr", "Ti", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
    spacegroup=221,
    cellpar=[a, a, a, 90, 90, 90],
)
atoms *= (8, 8, 20)

frozen_phonons = abtem.FrozenPhonons(atoms, 1, {"Sr": 0.164356, "Ti": 0.116584, "O": 0.148198})
potential = abtem.Potential(
    frozen_phonons,
    sampling=0.05,
    projection="finite",
    slice_thickness=0.4,
    exit_planes=20,
)

gx, gy = potential.gpts
total_z = atoms.cell[2, 2]
sz = potential.slice_thickness[0]
n_slices = int(total_z / sz)
lx, ly = potential.extent
print(f"Grid: {gx}x{gy}, extent: {lx:.1f}x{ly:.1f}A")
print(f"Slices: {n_slices}, total thickness: {total_z:.1f}A")

wave = abtem.Probe(energy=30e3, semiangle_cutoff=35)
wave.grid.match(potential)

# Run with tiled kernel (auto-selected)
cvdms = CVDMSMultislice(
    order=1, convergence_threshold=1e-6, max_terms=50,
    use_fused_kernel=True, backscattering=True,
)

t0 = time.time()
measurements = wave.multislice(potential, algorithm=cvdms).diffraction_patterns(max_angle="cutoff")
measurements = measurements.mean(0)
measurements.compute()
total_time = time.time() - t0

print(f"\nTotal simulation time: {total_time:.1f}s")
print(f"Per-slice: {total_time/n_slices*1000:.1f}ms")

# Show result
result = measurements[-1]
arr = result.array
if hasattr(arr, 'get'):
    arr = arr.get()
print(f"CBED pattern: {arr.shape}")
print("PASS: Simulation completed successfully")
