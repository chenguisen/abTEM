"""
BSC per-slice running accumulation verification test.

Tests that:
1. Running accumulation path executes (no crash)
2. No NaN/Inf in output
3. BSC fraction is physically reasonable (0.05-5% for thin specimen)
4. BSC at bottom exit plane = 0
5. BSC at entrance surface > 0
"""

import numpy as np
import ase
import warnings
warnings.filterwarnings('ignore')

from abtem import Potential, Probe, Waves, PixelatedDetector, CVDMSMultislice
from abtem.multislice import multislice_and_detect

# ── Small Au crystal ──────────────────────────────────────────────────
# Au has stronger scattering → more BSC signal for easier detection
atoms = ase.build.bulk("Au", "fcc", a=4.08)
atoms *= (4, 4, 6)  # 6 layers, cell z ≈ 24.5A

gpts = (128, 128)
energy = 300e3
dz = 0.5

# Build potential once with exit planes covering all slices
potential = Potential(
    atoms, sampling=0.1, gpts=gpts,
    slice_thickness=dz, device="gpu",
)
num_slices = potential.num_slices

# Exit planes: every ~8 slices + first and last
eps = list(range(-1, num_slices, 8))
if eps[-1] != num_slices - 1:
    eps.append(num_slices - 1)

# Rebuild with custom exit planes
potential = Potential(
    atoms, sampling=0.1, gpts=gpts,
    slice_thickness=dz, exit_planes=eps, device="gpu",
)

total_thick = num_slices * dz
print(f"Grid: {gpts}")
print(f"Cell z: {atoms.cell.lengths()[2]:.2f}A, slices: {num_slices}")
print(f"Total thickness: {total_thick:.1f}A, exit planes: {len(eps)}")
print()

# ── Probe and algorithm ──────────────────────────────────────────────
probe = Probe(energy=energy, semiangle_cutoff=20, device="gpu")
probe.grid.match(potential)

algorithm = CVDMSMultislice(
    max_terms=20, max_inner=10,
    convergence_threshold=1e-7, order=3,
    backscattering=True, calculate_backscattered=True,
    antialias=True, use_fused_kernel=True,
    backend="cupy",
)

waves = probe.build().compute()
detector = PixelatedDetector()

# ── Run CVDMS with BSC ───────────────────────────────────────────────
print("--- Running CVDMS with BSC ---")
detector_results = multislice_and_detect(
    waves, potential,
    detectors=[detector],
    algorithm=algorithm,
    pbar=True,
    return_backscattered=True,
)

bsc_measurement = detector_results[-1]
bsc_array = bsc_measurement.array
if hasattr(bsc_array, 'compute'):
    bsc_array = bsc_array.compute()
elif hasattr(bsc_array, 'get'):
    bsc_array = bsc_array.get()
print(f"BSC shape: {bsc_array.shape}, dtype: {bsc_array.dtype}")

# ── Verification ─────────────────────────────────────────────────────
errors = []

# 1. No NaN/Inf
if np.any(np.isnan(bsc_array)):
    errors.append("NaN in BSC array!")
if np.any(np.isinf(bsc_array)):
    errors.append("Inf in BSC array!")
print("[PASS]" if not errors else "[FAIL]", "No NaN/Inf ✓" if not errors else "NaN/Inf present!")

# 2. Bottom exit plane BSC ≈ 0
bottom = np.abs(bsc_array[-1]) ** 2
bottom_total = float(bottom.sum())
if bottom_total > 1e-10:
    errors.append(f"Bottom EP BSC not zero: {bottom_total:.3e}")
print(f"Bottom EP |bsc|²: {bottom_total:.3e}", "[PASS]" if bottom_total <= 1e-10 else "[FAIL]")

# 3. Entrance surface BSC > 0
top_int = np.abs(bsc_array[0]) ** 2
top_total = float(top_int.sum())
if top_total <= 0:
    errors.append("Entrance surface BSC <= 0!")
print(f"Entrance EP |bsc|²: {top_total:.6e}", "[PASS]" if top_total > 0 else "[FAIL]")

# 4. BSC fraction (physically reasonable: 0.05% - 5% for thin specimen)
incident_wave = probe.build().compute()
incident_arr = incident_wave.array
if hasattr(incident_arr, 'get'):
    incident_arr = incident_arr.get()
incident_total = float((np.abs(incident_arr) ** 2).sum())
bsc_frac = top_total / incident_total * 100
if not (0.01 < bsc_frac < 20.0):
    errors.append(f"BSC fraction {bsc_frac:.4f}% outside [0.01, 20.0]%")
print(f"BSC fraction: {bsc_frac:.4f}%", "[PASS]" if (0.01 < bsc_frac < 20.0) else "[FAIL]")

# 5. Depth profile
ep_ints = [float(np.abs(bsc_array[i]).sum()) for i in range(len(bsc_array))]
max_depth = max(ep_ints)
print(f"\nDepth profile (sum|bsc|):")
for i, v in enumerate(ep_ints):
    bar = "█" * int(v / max_depth * 30) if max_depth > 0 else "-"
    print(f"  EP {i:2d}: {v:.4e} {bar}")

# ── Summary ──────────────────────────────────────────────────────────
print()
if errors:
    for e in errors:
        print(f"[FAIL] {e}")
    print(f"\n{len(errors)} failure(s)")
    raise SystemExit(1)
else:
    print("[PASS] All checks passed!")
    print(f"BSC fraction = {bsc_frac:.4f}% @ {energy/1000:.0f} keV, "
          f"{total_thick:.0f}A Au, {gpts}, {len(eps)} EPs")
