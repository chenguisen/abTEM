"""Diagnostic: test BSCBackPropEngine.compute_accumulate in isolation."""
import numpy as np
import cupy as cp
from abtem.core import config as _cfg
_cfg.set({"device": "gpu", "fft": "cupy"})

# Simulate the data that _back_propagate_bsc_impl would pass to C++
nx, ny = 128, 128
num_slices = 39
num_exit_planes = 8

# Fake BSC data (small random values to avoid overflow)
bsc_re_list = []
bsc_im_list = []
for i in range(num_slices):
    arr = cp.random.randn(nx, ny).astype(cp.float32) * 1e-6
    bsc_re_list.append(cp.ascontiguousarray(arr))
    bsc_im_list.append(cp.ascontiguousarray(cp.zeros_like(arr)))

# Fake V (transmission functions)
V_list = []
for i in range(num_slices):
    arr = cp.random.randn(nx, ny).astype(cp.float32) * 0.01
    V_list.append(cp.ascontiguousarray(arr))

# Fake exit plane buffers
ep_re_list = []
ep_im_list = []
for i in range(num_exit_planes):
    arr = cp.zeros((nx, ny), dtype=cp.float32)
    ep_re_list.append(cp.ascontiguousarray(arr))
    ep_im_list.append(cp.ascontiguousarray(cp.zeros_like(arr)))

# Exit plane indices (matching potential.exit_planes pattern)
# With 39 slices and 8 exit planes spacing: range(7, 39, 8) = [7,15,23,31,39?→38]
# _validate_exit_planes(8, 39):
#   range(8-1, 39, 8) = [7, 15, 23, 31]
#   append 38
#   prepend -1
#   = (-1, 7, 15, 23, 31, 38) → len=6
ep_indices = [int(x) for x in (-1, 7, 15, 23, 31, 38)]
actual_num_ep = len(ep_indices)

# Adjust ep lists to match
ep_re_list = ep_re_list[:actual_num_ep]
ep_im_list = ep_im_list[:actual_num_ep]

print(f"nx={nx}, ny={ny}, count={nx*ny}")
print(f"num_slices={num_slices}, num_exit_planes={actual_num_ep}")
print(f"ep_indices={ep_indices}")
print(f"bsc_re shapes: {[a.shape for a in bsc_re_list[:3]]}...")
print(f"V shapes: {[a.shape for a in V_list[:3]]}...")
print(f"ep_re shapes: {[a.shape for a in ep_re_list]}")

# Compute laplace prefactor
pixel_size = 0.3124  # ~40Å / 128
laplace_prefactor = 1.0 / (pixel_size * pixel_size)
wavelength = 0.00698  # 30 keV
dz = 0.4

print(f"\nCalling BSCBackPropEngine.compute_accumulate...")
print(f"  laplace_prefactor={laplace_prefactor:.2f}, wavelength={wavelength:.4f}, dz={dz}")

from _cvdms_backend import BSCBackPropEngine

engine = BSCBackPropEngine()
engine.compute_accumulate(
    bsc_re_list, bsc_im_list, V_list,
    ep_re_list, ep_im_list, ep_indices,
    nx, ny, wavelength, dz,
    convergence_threshold=1e-7,
    max_terms=50, max_inner=100,
    laplace_prefactor=laplace_prefactor,
    accuracy=8,
    use_conj=True,
)

print("compute_accumulate returned successfully!")

# Check results
for ep_idx in range(actual_num_ep):
    ep_val = ep_re_list[ep_idx] + 1.0j * ep_im_list[ep_idx]
    max_abs = float(cp.max(cp.abs(ep_val)))
    print(f"  EP {ep_idx} (idx={ep_indices[ep_idx]}): max|val| = {max_abs:.4e}")

print("\nPASS: BSCBackPropEngine works in isolation.")
