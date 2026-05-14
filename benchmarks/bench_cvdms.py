"""
CVDMS benchmark: realistic slow-converging data to stress kernel perf.
"""

import math
import time
import numpy as np
import cupy as cp

H, W = 627, 627
BATCH = 1
E = 30e3
wavelength = 1.0 / math.sqrt(2 * E * 511.0e3 / (511.0e3 + E))
THRESHOLD = 1e-6
MAX_ITER = 50
dx = dy = 0.05
prefactor = 1.0 / (dx * dy)

from abtem.finite_difference import finite_difference_coefficients
stencil_raw = finite_difference_coefficients(2, 8).astype(np.float32)
sn = len(stencil_raw) // 2

# Realistic data: stronger potential = more K-series iterations
np.random.seed(42)
cp.random.seed(42)
waves = (cp.random.randn(BATCH, H, W) + 1j * cp.random.randn(BATCH, H, W)).astype(cp.complex64)
# Stronger potential (3x) to force more inner iterations
V = cp.random.randn(H, W).astype(cp.float32) * 0.3

K0 = 1.0 / wavelength
inv_4piK0 = 1.0 / (4.0 * math.pi * K0)
lap_factor = float(prefactor * inv_4piK0)

waves_flat = waves.reshape(BATCH, H, W)
V_c = cp.ascontiguousarray(V, dtype=cp.float32).reshape(H, W)
sc = cp.asarray(stencil_raw, dtype=cp.float32)

from abtem.cvdms_kernels import _get_k_iteration_kernel, _get_k_iteration_kernel_tiled

kern_untiled = _get_k_iteration_kernel(lap_factor, inv_4piK0, THRESHOLD)
kern_tiled = _get_k_iteration_kernel_tiled(lap_factor, inv_4piK0, THRESHOLD, sn)

grid_nt = (int(math.ceil(W / 16)), int(math.ceil(H / 16)), BATCH)
grid_t = (int(math.ceil(W / 32)), int(math.ceil(H / 16)), BATCH)
shared_bytes = 2 * (16 + 2 * sn) * (32 + 2 * sn) * 4

print(f"Grid: {H}x{W}, sn={sn}, tile=32x16, shared={shared_bytes}B")

def run_full_series(kernel, grid, block, shared_mem=0):
    cur_re = cp.ascontiguousarray(waves_flat.real.reshape(-1))
    cur_im = cp.ascontiguousarray(waves_flat.imag.reshape(-1))
    nxt_re = cp.empty_like(cur_re)
    nxt_im = cp.empty_like(cur_im)
    kseries_re = cp.zeros_like(cur_re)
    kseries_im = cp.zeros_like(cur_im)
    n_above_dev = cp.zeros(1, dtype=cp.int32)
    overflowed_dev = cp.zeros(1, dtype=cp.int32)
    prev_n_above = None
    n_iters = 0

    for n in range(1, MAX_ITER + 1):
        n_above_dev.fill(0)
        overflowed_dev.fill(0)
        kernel(grid, block, (
            cur_re, cur_im, nxt_re, nxt_im, kseries_re, kseries_im,
            V_c, n_above_dev, overflowed_dev, H, W,
            np.int64(H * W), sc, sn, n,
        ), shared_mem=shared_mem)
        n_iters = n

        if overflowed_dev.item() > 0:
            break
        n_above = int(n_above_dev.item())
        if prev_n_above is not None and n_above >= prev_n_above:
            break
        prev_n_above = n_above
        if n_above == 0:
            break
        cur_re, cur_im, nxt_re, nxt_im = nxt_re, nxt_im, cur_re, cur_im

    n_pixels = BATCH * H * W
    result = cp.empty(n_pixels, dtype=cp.complex64)
    result.real[:] = kseries_re
    result.imag[:] = kseries_im
    return result.reshape(BATCH, H, W), n_iters

# Extended warmup
for _ in range(10):
    run_full_series(kern_untiled, grid_nt, (16, 16, 1))
    run_full_series(kern_tiled, grid_t, (32, 16, 1), shared_bytes)
cp.cuda.Stream.null.synchronize()

# Benchmark with CUDA events for precise timing
n_trials = 20

def benchmark(kernel, grid, block, shared_mem, label):
    times = []
    iters_list = []
    result = None

    for i in range(n_trials):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        r, it = run_full_series(kernel, grid, block, shared_mem)
        end_event.record()
        end_event.synchronize()

        elapsed = cp.cuda.get_elapsed_time(start_event, end_event)  # ms
        times.append(elapsed)
        iters_list.append(it)
        if i == 0:
            result = r.copy()

    avg = np.mean(times)
    print(f"  Mean: {avg:.3f} ms, Min: {np.min(times):.3f} ms, Max: {np.max(times):.3f} ms")
    print(f"  Iterations: {np.mean(iters_list):.1f} (min={min(iters_list)}, max={max(iters_list)})")
    return result, np.array(times), iters_list

print("\n--- Non-tiled (16x16, no shared mem) ---")
ref_nt, t_nt, it_nt = benchmark(kern_untiled, grid_nt, (16, 16, 1), 0, "non-tiled")

print("\n--- Tiled (32x16, shared mem) ---")
ref_t, t_t, it_t = benchmark(kern_tiled, grid_t, (32, 16, 1), shared_bytes, "tiled")

print(f"\n  Speedup: {np.mean(t_nt)/np.mean(t_t):.2f}x")

# Verification
diff = cp.abs(ref_t - ref_nt)
print(f"\n--- Numerical verification ---")
print(f"  Max diff: {float(cp.max(diff)):.2e}")
print(f"  Mean diff: {float(cp.mean(diff)):.2e}")
if float(cp.max(diff)) < 1e-5:
    print("  ✓ PASS")
else:
    print("  ✗ FAIL")
