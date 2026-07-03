"""Diagnostic: examine per-slice BSC data and V values from V1c Stage A."""
import sys, gc
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
SUPERCELL_XY = 4
SUPERCELL_Z = 10
EXIT_PLANES = 8

a = 3.905
atoms = crystal(
    ["Sr", "Ti", "O"],
    basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
    spacegroup=221,
    cellpar=[a, a, a, 90, 90, 90],
)
atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)

potential = abtem.Potential(
    atoms, gpts=(128, 128),
    slice_thickness=SLICE_THICKNESS,
    exit_planes=EXIT_PLANES, projection="finite",
)

print(f"Exit planes: {potential.exit_planes}")
print(f"num_exit_planes: {potential.num_exit_planes}")
print(f"num_slices: {potential.num_slices}")

# Examine V values (transmission functions)
from abtem.core.energy import energy2sigma
sigma = energy2sigma(ENERGY)

potential_slices = list(potential.generate_slices())
print(f"\nActual num_slices from generate_slices: {len(potential_slices)}")

V_max_list = []
for i, sl in enumerate(potential_slices):
    tf = sl.array[0] * sigma / float(sl.thickness)
    if hasattr(tf, 'get'):
        tf = tf.get()
    V_max_list.append(float(np.max(np.abs(tf))))

print(f"V_max (transmission function |sigma*V/dz|) range: "
      f"{min(V_max_list):.4f} to {max(V_max_list):.4f}")
print(f"V_max per slice (first 10): {V_max_list[:10]}")

# Run simulation and capture per-slice BSC
import abtem.multislice
original_bp = abtem.multislice._back_propagate_bsc_impl

captured_bsc = []
captured_ep = [None]

def capture_bsc(backscattered_waves, potential_slices, exit_planes,
                 multislice_step, per_slice_bsc_arrays=None):
    global captured_bsc
    if per_slice_bsc_arrays is not None:
        # Unwrap
        if (isinstance(per_slice_bsc_arrays, list) and
            len(per_slice_bsc_arrays) == 1 and
            isinstance(per_slice_bsc_arrays[0], list)):
            captured_bsc = list(per_slice_bsc_arrays[0])
        else:
            captured_bsc = list(per_slice_bsc_arrays)
        print(f"Captured {len(captured_bsc)} per-slice BSC arrays")

    # Run the original back-propagation (which will crash)
    try:
        result = original_bp(backscattered_waves, potential_slices, exit_planes,
                            multislice_step, per_slice_bsc_arrays=per_slice_bsc_arrays)
        return result
    except Exception as e:
        print(f"Back-propagation failed: {e}")
        return backscattered_waves

abtem.multislice._back_propagate_bsc_impl = capture_bsc

wave = abtem.PlaneWave(energy=ENERGY)
wave.grid.match(potential)

cvdms = CVDMSMultislice(
    convergence_threshold=CONVERGENCE_THRESHOLD,
    backscattering=True,
    calculate_backscattered=True,
)

print("\nRunning CVDMS + BSC...")
try:
    result = wave.multislice(
        potential, algorithm=cvdms,
        return_backscattered=True, lazy=False,
    )
except Exception as e:
    print(f"Simulation failed: {e}")

abtem.multislice._back_propagate_bsc_impl = original_bp

# Analyze captured BSC
print(f"\nAnalyzing {len(captured_bsc)} captured BSC arrays:")
for i, arr in enumerate(captured_bsc):
    arr_cpu = arr.get() if hasattr(arr, 'get') else np.asarray(arr)
    bsc_max = float(np.max(np.abs(arr_cpu)))
    bsc_sum = float(np.sum(np.abs(arr_cpu)**2))
    has_nan = np.any(np.isnan(arr_cpu))
    has_inf = np.any(np.isinf(arr_cpu))
    if bsc_max > 1e-3 or has_nan or has_inf:
        print(f"  sl {i:3d}: max|BSC|={bsc_max:.4e}  Σ|BSC|²={bsc_sum:.4e}  "
              f"NaN={has_nan} Inf={has_inf}")

# Also get the BSC from the exit plane WavesDetector
print("\nDone.")
gc.collect()
cp.get_default_memory_pool().free_all_blocks()
