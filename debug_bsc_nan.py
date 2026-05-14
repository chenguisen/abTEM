"""
Diagnose NaN in BSC back-propagation at 300 keV (full 235-slice specimen).
"""
import numpy as np
import cupy as cp
import abtem
from abtem import PlaneWave, Potential
from abtem.multislice import CVDMSMultislice
from abtem.core import config as _cfg
_cfg.set({"device": "gpu", "fft": "cupy"})

from ase.spacegroup import crystal

# ── Build SrTiO3 (FULL thickness = 30 unit cells) ──────────────────
atoms = crystal(
    ["Sr", "Ti", "O"],
    basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
    spacegroup=221,
    cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
)
atoms *= (6, 6, 30)

EXIT_PLANES = 30
sampling = 0.05
slice_thickness = 0.5
convergence_threshold = 1e-8

single_potential = Potential(
    atoms,
    sampling=sampling,
    slice_thickness=slice_thickness,
    exit_planes=EXIT_PLANES,
)

plane_wave = PlaneWave(energy=300e3)

# Collect info
potential_slices = list(single_potential.generate_slices())
num_slices = len(potential_slices)
exit_planes = list(single_potential.exit_planes)
num_exit_planes = single_potential.num_exit_planes
print(f"Total slices: {num_slices}")
print(f"Exit planes: {exit_planes}")
print(f"Num exit planes: {num_exit_planes}")

def _to_np(arr):
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)

# ── STEP 1: Check per-slice BSC arrays (capture before back-prop) ──
print("\n" + "=" * 60)
print("STEP 1: Capture per-slice BSC data before back-propagation")
print("=" * 60)

import abtem.multislice
original_bp = abtem.multislice._back_propagate_bsc_impl

captured_bsc = []
def capture_bsc(backscattered_waves, potential_slices, exit_planes,
                 multislice_step, per_slice_bsc_arrays=None):
    global captured_bsc
    if per_slice_bsc_arrays is not None:
        captured_bsc = list(per_slice_bsc_arrays)  # copy list
        print(f"  Captured {len(captured_bsc)} per-slice BSC arrays")
    return original_bp(backscattered_waves, potential_slices, exit_planes,
                        multislice_step, per_slice_bsc_arrays=per_slice_bsc_arrays)

abtem.multislice._back_propagate_bsc_impl = capture_bsc

print("Running CVDMS + BSC (full thickness)...")
result = plane_wave.multislice(
    single_potential,
    algorithm=CVDMSMultislice(
        backscattering=True,
        calculate_backscattered=True,
        convergence_threshold=convergence_threshold,
    ),
    return_backscattered=True,
    lazy=False,
)

exit_wave_bsc = result[0]
bsc_wave = result[-1]

# Restore original
abtem.multislice._back_propagate_bsc_impl = original_bp

print(f"\nExit wave has NaN: {np.any(np.isnan(_to_np(exit_wave_bsc.array)))}")
print(f"BSC wave has NaN: {np.any(np.isnan(_to_np(bsc_wave.array)))}")

# NaN per exit plane
for ep in range(len(bsc_wave)):
    arr = _to_np(bsc_wave._array[ep])
    has_nan = np.any(np.isnan(arr))
    print(f"  EP {ep}: NaN={has_nan} max|arr|={float(np.max(np.abs(arr))):.4e}")

# ── Check captured per-slice BSC arrays for NaN ──────────────────
print(f"\nChecking {len(captured_bsc)} per-slice BSC arrays:")
nan_slices = []
bad_slices = []
for i, arr in enumerate(captured_bsc):
    arr_cpu = _to_np(arr)
    if np.any(np.isnan(arr_cpu.real)) or np.any(np.isnan(arr_cpu.imag)):
        nan_slices.append(i)
    if np.any(np.isinf(arr_cpu.real)) or np.any(np.isinf(arr_cpu.imag)):
        bad_slices.append(i)
    if np.any(np.abs(arr_cpu) > 10.0):
        bad_slices.append(i)

if nan_slices:
    print(f"  ❌ BSC NaN at slices: {nan_slices[:20]}{'...' if len(nan_slices) > 20 else ''}")
else:
    print(f"  ✅ No BSC NaN in any slice")

if bad_slices:
    print(f"  ⚠️  BSC bad values at slices: {bad_slices[:20]}{'...' if len(bad_slices) > 20 else ''}")

# Compute cumulative sum of BSC arrays (bottom-up)
print("\nCumulative BSC sum (bottom-up, first 10 and last 10):")
cumulative = np.zeros_like(_to_np(captured_bsc[0]), dtype=np.complex128)
for i in range(num_slices - 1, -1, -1):
    cumulative += _to_np(captured_bsc[i]).astype(np.complex128)
    if (num_slices - i) <= 10 or i < 10:
        cmax = float(np.max(np.abs(cumulative)))
        print(f"  After sl={i:3d} (accumulating upward): |cum|_max = {cmax:.4e}")

# Total sum
total_sum = cumulative
print(f"\nTotal cumulative BSC: max|sum| = {float(np.max(np.abs(total_sum))):.4e}")

del result, exit_wave_bsc, bsc_wave, cumulative, total_sum
cp.get_default_memory_pool().free_all_blocks()
