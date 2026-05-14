"""
Fused CUDA kernels for CVDMS multislice.

Replaces the inner K-series Python loop with per-iteration fused CUDA kernels.
Each kernel combines Laplacian, K-operator, scaling, and accumulation into a
single launch, eliminating intermediate scratch buffer traffic and D2H syncs.

Design:
- Per-iteration kernel (not fully fused): one K-iteration per kernel launch
- Global memory ping-pong for inter-iteration wave state (avoiding inter-block sync)
- Block-level convergence counter via shared memory + atomic reduction
- Per-pixel early termination tracked atomically
"""

from __future__ import annotations

import math

import cupy as cp
import numpy as np

_kernel_cache = {}


def _get_convergence_check_kernel(threshold: float = 1e-6) -> cp.RawKernel:
    """Get or compile the on-device convergence check kernel."""
    # CUDA workaround: embed threshold as compile-time constant to avoid
    # CuPy 13.6.0 compiler bug where float kernel params get optimized to ~0.
    cache_key = f"convergence_check_{threshold}"
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    kernel_src = """
extern "C" __global__
void convergence_check(
    const float* __restrict__ working,
    const float* __restrict__ exit_wave,
    int* __restrict__ n_above,
    int* __restrict__ overflowed,
    float* __restrict__ sum_working,
    float* __restrict__ sum_exit,
    int H,
    int W,
    long long stride
) {
    // Compile-time constant (CUDA workaround)
    const float threshold = __THR__f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    __shared__ int smem_n_above;
    __shared__ int smem_overflowed;
    __shared__ float smem_sum_w;
    __shared__ float smem_sum_e;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        smem_n_above = 0;
        smem_overflowed = 0;
        smem_sum_w = 0.0f;
        smem_sum_e = 0.0f;
    }
    __syncthreads();

    if (col < W && row < H) {
        long long base = (long long)batch * stride;
        long long idx = base + (long long)row * W + col;
        long long i2 = idx * 2;  // complex64 has interleaved re/im

        float wre = working[i2];
        float wim = working[i2 + 1];
        float ere = exit_wave[i2];
        float eim = exit_wave[i2 + 1];

        // Overflow detection
        if (isinf(wre) || isnan(wre) || isinf(wim) || isnan(wim) ||
            isinf(ere) || isnan(ere) || isinf(eim) || isnan(eim)) {
            atomicAdd(&smem_overflowed, 1);
        }

        // Convergence: |working| > threshold
        float w_mag = sqrt(wre * wre + wim * wim);
        if (w_mag > threshold) {
            atomicAdd(&smem_n_above, 1);
        }

        // Divergence sums
        atomicAdd(&smem_sum_w, w_mag);
        atomicAdd(&smem_sum_e, sqrt(ere * ere + eim * eim));
    }

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (smem_overflowed > 0) {
            atomicOr(overflowed, 1);
        }
        atomicAdd(n_above, smem_n_above);
        atomicAdd(sum_working, smem_sum_w);
        atomicAdd(sum_exit, smem_sum_e);
    }
}
""".replace("__THR__", f"{threshold}")

    kernel = cp.RawKernel(kernel_src, "convergence_check")
    _kernel_cache[cache_key] = kernel
    return kernel


def compute_convergence_check(
    working: cp.ndarray,
    exit_wave: cp.ndarray,
    threshold: float,
) -> tuple[int, int, float, float]:
    """
    Compute convergence metrics on-device in a single kernel launch.

    Replaces 4 separate GPU reduction + D2H sync operations with a single
    kernel that computes all metrics using shared-memory atomics.

    Parameters
    ----------
    working : cp.ndarray
        Current K-series term (complex64), shape (batch, H, W) or (H, W).
    exit_wave : cp.ndarray
        Accumulated exit wave (complex64), same shape as working.
    threshold : float
        Per-pixel convergence threshold.

    Returns
    -------
    n_above : int
        Number of pixels with |working| > threshold.
    overflowed : int
        1 if any pixel has inf/nan, 0 otherwise.
    sum_working : float
        Sum of |working| across all pixels.
    sum_exit : float
        Sum of |exit_wave| across all pixels.
    """
    xp = cp

    orig = working.shape
    if working.ndim == 2:
        H, W = orig
        batch = 1
    else:
        H, W = orig[-2], orig[-1]
        batch = working.size // (H * W)

    stride = H * W

    n_above_dev = xp.zeros(1, dtype=xp.int32)
    overflowed_dev = xp.zeros(1, dtype=xp.int32)
    sum_working_dev = xp.zeros(1, dtype=xp.float32)
    sum_exit_dev = xp.zeros(1, dtype=xp.float32)

    kernel = _get_convergence_check_kernel(float(threshold))

    block_x = 16
    block_y = 16
    grid_x = int(math.ceil(W / block_x))
    grid_y = int(math.ceil(H / block_y))
    grid = (grid_x, grid_y, batch)
    block = (block_x, block_y, 1)

    kernel(
        grid, block,
        (
            xp.ascontiguousarray(working, dtype=xp.complex64),
            xp.ascontiguousarray(exit_wave, dtype=xp.complex64),
            n_above_dev,
            overflowed_dev,
            sum_working_dev,
            sum_exit_dev,
            H, W,
            np.int64(stride),
        ),
    )

    return (
        int(n_above_dev.item()),
        int(overflowed_dev.item()),
        float(sum_working_dev.item()),
        float(sum_exit_dev.item()),
    )


def _get_k_iteration_kernel_tiled(lap_factor: float, inv_4piK0: float, threshold: float = 1e-6, sn: int = 4) -> cp.RawKernel:
    """Tiled kernel using shared memory for Laplacian stencil reuse.

    Each block cooperatively loads a tile+halo region into __shared__,
    eliminating redundant global reads for the 2D Laplacian stencil.
    On RTX 3080 (sm_86) this reduces global traffic ~5-10x for the
    memory-bound stencil portion.

    Tile dimensions: 32x16 = 512 threads, good occupancy on Ampere.
    Shared memory: 2 * (TY+2*sn) * (TX+2*sn) * 4 bytes, allocated at launch.

    Numerically identical to the non-tiled kernel — same coefficients,
    same operations, same coefficient cascade. Only memory access pattern
    differs (shared vs. global for the stencil neighborhood).
    """
    cache_key = f"k_iteration_tiled_{lap_factor}_{inv_4piK0}_{threshold}_sn{sn}"
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    TX = 32
    TY = 16

    kernel_src = f"""
extern "C" __global__
void k_iteration_tiled(
    const float* __restrict__ cur_re,
    const float* __restrict__ cur_im,
    float* __restrict__ next_re,
    float* __restrict__ next_im,
    float* __restrict__ kseries_re,
    float* __restrict__ kseries_im,
    const float* __restrict__ V,
    int* __restrict__ n_above,
    int* __restrict__ overflowed,
    int H,
    int W,
    long long stride,
    const float* __restrict__ sc,
    int _sn_arg,
    int iter_n
) {{
    // Compile-time constants
    const float lap_factor = {lap_factor}f;
    const float inv_4piK0 = {inv_4piK0}f;
    const float threshold = {threshold}f;
    const int sn = {sn};

    const int TX = {TX};
    const int TY = {TY};

    // Dynamic shared memory for re + im tiles (allocated at launch)
    extern __shared__ float shared[];
    const int sx = TX + 2 * sn;
    const int sy = TY + 2 * sn;
    float* tile_re = shared;
    float* tile_im = shared + sy * sx;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int batch = blockIdx.z;
    long long base = (long long)batch * stride;
    int col0 = bx * TX;
    int row0 = by * TY;

    // Cooperative tile+halo loading from global to shared.
    // Use conditional add/sub instead of modulo for RTX 3080 (sm_86):
    // integer DIV is ~20 cycles, branch is ~4 cycles when not taken.
    for (int i = threadIdx.y; i < sy; i += blockDim.y) {{
        for (int j = threadIdx.x; j < sx; j += blockDim.x) {{
            int g_row = row0 + i - sn;
            if (g_row < 0) g_row += H;
            else if (g_row >= H) g_row -= H;
            int g_col = col0 + j - sn;
            if (g_col < 0) g_col += W;
            else if (g_col >= W) g_col -= W;
            long long idx = base + (long long)g_row * W + g_col;
            tile_re[i * sx + j] = cur_re[idx];
            tile_im[i * sx + j] = cur_im[idx];
        }}
    }}
    __syncthreads();

    int col = col0 + threadIdx.x;
    int row = row0 + threadIdx.y;

    // Block-level convergence counter
    __shared__ int smem_n_above;
    if (threadIdx.x == 0 && threadIdx.y == 0) {{
        smem_n_above = 0;
    }}
    __syncthreads();

    if (threadIdx.x < TX && threadIdx.y < TY && col < W && row < H) {{
        long long idx = base + (long long)row * W + col;
        int ti = (threadIdx.y + sn) * sx;
        int tj = threadIdx.x + sn;

        // ---- 2D Laplacian from shared memory ----
        float lap_re = 0.0f, lap_im = 0.0f;
        for (int k = -sn; k <= sn; k++) {{
            float ck = sc[k + sn];
            // Symmetric stencil: vertical (k*SX) + horizontal (k)
            lap_re += ck * (tile_re[ti + k * sx + tj] + tile_re[ti + tj + k]);
            lap_im += ck * (tile_im[ti + k * sx + tj] + tile_im[ti + tj + k]);
        }}

        // ---- K-operator: K(w) = V * w + laplacian(w) / (4 * pi * K0) ----
        float v = V[(long long)row * W + col];
        float kw_re = v * tile_re[ti + tj] + lap_re * lap_factor;
        float kw_im = v * tile_im[ti + tj] + lap_im * lap_factor;

        // ---- Overflow detection ----
        if (isnan(kw_re) || isinf(kw_re) || isnan(kw_im) || isinf(kw_im)) {{
            atomicOr(overflowed, 1);
        }} else {{
            // ---- K-series coefficient ----
            float s;
            if (iter_n == 1) {{
                s = 1.0f;
            }} else {{
                s = (1.5f - (float)iter_n) * 4.0f * inv_4piK0 / (float)iter_n;
            }}

            float s_re = kw_re * s;
            float s_im = kw_im * s;

            // ---- Convergence check on the SCALED term ----
            float mag = sqrt(s_re * s_re + s_im * s_im);
            if (mag > threshold) {{
                atomicAdd(&smem_n_above, 1);
            }}

            // ---- Accumulate to k_series ----
            kseries_re[idx] += s_re;
            kseries_im[idx] += s_im;

            // ---- Store SCALED wave for coefficient cascade ----
            next_re[idx] = s_re;
            next_im[idx] = s_im;
        }}
    }}

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {{
        atomicAdd(n_above, smem_n_above);
    }}
}}
"""

    kernel = cp.RawKernel(kernel_src, "k_iteration_tiled")
    _kernel_cache[cache_key] = kernel
    return kernel


def _get_k_iteration_kernel(lap_factor: float, inv_4piK0: float, threshold: float = 1e-6) -> cp.RawKernel:
    """Get or compile the per-iteration fused K-series kernel."""
    # Include constants in cache key since they're embedded as compile-time literals
    # (CUDA workaround: float kernel params get optimized to ~0 by CuPy 13.6.0 compiler)
    cache_key = f"k_iteration_fused_{lap_factor}_{inv_4piK0}_{threshold}"
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    kernel_src = """
extern "C" __global__
void k_iteration_fused(
    const float* __restrict__ cur_re,
    const float* __restrict__ cur_im,
    float* __restrict__ next_re,
    float* __restrict__ next_im,
    float* __restrict__ kseries_re,
    float* __restrict__ kseries_im,
    const float* __restrict__ V,
    int* __restrict__ n_above,
    int* __restrict__ overflowed,
    int H,
    int W,
    long long stride,
    const float* __restrict__ sc,
    int sn,
    int iter_n
) {
    // Compile-time constants (CUDA workaround: embedded as literals to avoid
    // optimizer bug where function-parameter floats zero out multiply results)
    const float lap_factor = __LAPF__f;
    const float inv_4piK0 = __INV4PIK0__f;
    const float threshold = __THR__f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    // Block-level convergence counter (reduces global atomic contention)
    __shared__ int smem_n_above;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        smem_n_above = 0;
    }
    __syncthreads();

    if (col < W && row < H) {
        long long base = (long long)batch * stride;
        long long idx = base + (long long)row * W + col;

        float wre = cur_re[idx];
        float wim = cur_im[idx];

        // ---- 2D Laplacian (separable, periodic boundary) ----
        // Conditional add/sub for periodic wrap (faster than modulo on sm_86).
        float lap_re = 0.0f, lap_im = 0.0f;
        for (int k = -sn; k <= sn; k++) {
            float ck = sc[k + sn];
            int rr = row + k;
            if (rr < 0) rr += H;
            else if (rr >= H) rr -= H;
            int cc = col + k;
            if (cc < 0) cc += W;
            else if (cc >= W) cc -= W;
            long long iv = base + (long long)rr * W + col;
            long long ih = base + (long long)row * W + cc;
            lap_re += ck * (cur_re[iv] + cur_re[ih]);
            lap_im += ck * (cur_im[iv] + cur_im[ih]);
        }

        // ---- K-operator: K(w) = V * w + laplacian(w) / (4 * pi * K0) ----
        float v = V[(long long)row * W + col];
        float kw_re = v * wre + lap_re * lap_factor;
        float kw_im = v * wim + lap_im * lap_factor;

        // ---- Overflow detection ----
        if (isnan(kw_re) || isinf(kw_re) || isnan(kw_im) || isinf(kw_im)) {
            atomicOr(overflowed, 1);
        } else {
            // ---- K-series coefficient ----
            float s;
            if (iter_n == 1) {
                s = 1.0f;
            } else {
                // c_n = (0.5 - n + 1) * lambda / (pi * n)
                //     = (1.5 - n) * 4 * inv_4piK0 / n
                s = (1.5f - (float)iter_n) * 4.0f * inv_4piK0 / (float)iter_n;
            }

            float s_re = kw_re * s;
            float s_im = kw_im * s;

            // ---- Convergence check on the SCALED term (match original) ----
            float mag = sqrt(s_re * s_re + s_im * s_im);
            if (mag > threshold) {
                atomicAdd(&smem_n_above, 1);
            }

            // ---- Accumulate to k_series ----
            kseries_re[idx] += s_re;
            kseries_im[idx] += s_im;

            // ---- Store wave for next iteration ----
            // NOTE: store SCALED wave (s_re = kw * c_n) to reproduce the
            // coefficient cascade present in the non-fused path (cvdms.py).
            // In the non-fused path, the buffer swap carries c_n into working,
            // so K(c_n * wave) = c_n * K(wave), and the next scaling gives
            // c_n * c_{n+1} * K^{n+1}(wave). Without this cascade, the outer
            // Taylor terms grow instead of decay for many physical systems
            // (SrTiO3 at 30keV, Si at 100keV, etc.), producing false-positive
            // non-convergence RuntimeWarnings.
            next_re[idx] = s_re;
            next_im[idx] = s_im;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(n_above, smem_n_above);
    }
}
""".replace("__LAPF__", f"{lap_factor}").replace("__INV4PIK0__", f"{inv_4piK0}").replace("__THR__", f"{threshold}")

    kernel = cp.RawKernel(kernel_src, "k_iteration_fused")
    _kernel_cache[cache_key] = kernel
    return kernel


def compute_k_series_fused(
    waves_array: cp.ndarray,
    transmission_function: cp.ndarray,
    wavelength: float,
    convergence_threshold: float = 1e-6,
    max_inner_iter: int = 100,
    check_interval: int = 2,
    prefactor: float | None = None,
    stencil_raw: np.ndarray | None = None,
) -> cp.ndarray:
    """
    Fused inner K-series using per-iteration CUDA kernel.

    Replaces the Python loop in ``_cvdms_inner_k_series`` with per-iteration
    fused CUDA kernel launches. Each kernel combines Laplacian, K-operator,
    scaling, and accumulation into a single launch.

    Compared to the original Python loop:
    - 1 kernel launch vs 5 per iteration (5x fewer launch overhead)
    - No intermediate scratch buffer read/write (eliminates ~3 global writes/iter)
    - Convergence counting via atomic counter avoids D2H sync between iterations
    - Per-pixel convergence detection via shared memory block-level counter

    Parameters
    ----------
    waves_array : cp.ndarray
        Input wave function (complex64), shape (..., H, W) or (H, W).
    transmission_function : cp.ndarray
        Potential * sigma / thickness (float32), shape (H, W).
    wavelength : float
        Electron wavelength in Angstrom.
    convergence_threshold : float
        Per-pixel convergence threshold (default 1e-6).
    max_inner_iter : int
        Maximum K-series iterations (default 100).
    check_interval : int
        Ignored in fused path. Per-iteration checking is optimal here since
        the cost of 1 extra K-iteration far exceeds D2H sync savings.
    prefactor : float
        1 / (dx * dy), the Laplacian prefactor.
    stencil_raw : np.ndarray
        Raw finite-difference coefficients (NOT multiplied by prefactor).

    Returns
    -------
    cp.ndarray
        K-series result (complex64), same shape as ``waves_array``.
    """
    xp = cp

    if prefactor is None or stencil_raw is None:
        raise ValueError("prefactor and stencil_raw are required for fused kernel")

    # ---- Validate and reshape ----
    orig_shape = waves_array.shape
    ndim = waves_array.ndim

    # Flatten all batch dims, keeping last 2 as spatial
    if ndim == 2:
        H, W = orig_shape
        batch = 1
        waves_flat = waves_array.reshape(1, H, W)
    else:
        H, W = orig_shape[-2], orig_shape[-1]
        batch = waves_array.size // (H * W)
        waves_flat = waves_array.reshape(batch, H, W)

    n_pixels = batch * H * W

    # ---- Constants ----
    K0 = 1.0 / wavelength
    inv_4piK0 = 1.0 / (4.0 * math.pi * K0)
    lap_factor = float(prefactor * inv_4piK0)
    threshold = float(convergence_threshold)

    # ---- Device stencil ----
    sc = xp.asarray(stencil_raw, dtype=xp.float32)
    sn = len(stencil_raw) // 2

    # ---- Split input complex wave into contiguous re/im float32 ----
    # .real/.imag on complex give views with stride 2; ascontiguousarray copies
    # to contiguous layout for efficient GPU access.
    buf0_re = xp.ascontiguousarray(waves_flat.real.reshape(-1))
    buf0_im = xp.ascontiguousarray(waves_flat.imag.reshape(-1))
    buf1_re = xp.empty_like(buf0_re)
    buf1_im = xp.empty_like(buf0_im)
    kseries_re = xp.zeros_like(buf0_re)
    kseries_im = xp.zeros_like(buf0_im)

    # ---- V must be contiguous float32 2D ----
    V = xp.ascontiguousarray(transmission_function, dtype=xp.float32)
    if V.ndim == 3:
        V = V[0]  # take first batch slice
    if V.ndim != 2:
        V = V.reshape(H, W)

    # ---- GPU counters ----
    n_above_dev = xp.zeros(1, dtype=xp.int32)
    overflowed_dev = xp.zeros(1, dtype=xp.int32)

    # ---- Select kernel: tiled for large grids, simple for small ----
    # Tiled kernel uses shared memory for stencil reuse, reducing global
    # traffic ~5-10x. Numerically identical (same ops, same coefficients).
    _bench_untiled = _kernel_cache.get('_bench_untiled', False)
    use_tiled = H >= 64 and W >= 64 and not _bench_untiled
    if use_tiled:
        kernel = _get_k_iteration_kernel_tiled(lap_factor, inv_4piK0, threshold, sn)
        block_x = 32
        block_y = 16
        shared_mem_bytes = 2 * (block_y + 2 * sn) * (block_x + 2 * sn) * 4
    else:
        kernel = _get_k_iteration_kernel(lap_factor, inv_4piK0, threshold)
        block_x = 16
        block_y = 16
        shared_mem_bytes = 0

    grid_x = int(math.ceil(W / block_x))
    grid_y = int(math.ceil(H / block_y))
    grid = (grid_x, grid_y, batch)
    block = (block_x, block_y, 1)

    # ---- Ping-pong loop with per-iteration D2H check ----
    # In the fused kernel, each iteration is a single kernel launch + ~28us D2H
    # read. Batching convergence checks saves D2H time but wastes up to
    # (check_interval-1) full K-iterations, which costs more than the sync
    # savings. So we always check every iteration regardless of check_interval.
    cur_re, cur_im = buf0_re, buf0_im
    nxt_re, nxt_im = buf1_re, buf1_im
    prev_n_above = None

    for n in range(1, max_inner_iter + 1):
        n_above_dev.fill(0)
        overflowed_dev.fill(0)

        kernel(
            grid, block,
            (
                cur_re,           # cur_re
                cur_im,           # cur_im
                nxt_re,           # next_re
                nxt_im,           # next_im
                kseries_re,       # kseries_re
                kseries_im,       # kseries_im
                V,                # V (2D)
                n_above_dev,      # n_above counter
                overflowed_dev,   # overflowed flag
                H, W,             # spatial dims
                np.int64(H * W),  # stride
                sc,               # stencil coefficients
                sn,               # stencil half-width
                n,                # iteration number (1-based)
            ),
            shared_mem=shared_mem_bytes,
        )

        # ---- Overflow ----
        if overflowed_dev.item() > 0:
            break

        # ---- Convergence (single int D2H sync) ----
        n_above = int(n_above_dev.item())

        if prev_n_above is not None and n_above >= prev_n_above:
            break  # stagnation

        prev_n_above = n_above

        if n_above == 0:
            break  # fully converged

        # ---- Swap ping-pong ----
        cur_re, cur_im, nxt_re, nxt_im = nxt_re, nxt_im, cur_re, cur_im

    # ---- Reconstruct complex array from re/im ----
    # Interleave kseries_re and kseries_im into complex64
    result = xp.empty(n_pixels, dtype=xp.complex64)
    # Use CuPy's structured assignment or elementwise
    result.real[:] = kseries_re
    result.imag[:] = kseries_im

    return result.reshape(orig_shape)
