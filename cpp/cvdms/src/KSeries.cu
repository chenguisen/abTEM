#include "cvdms/KSeries.h"
#include "cvdms/Convergence.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <algorithm>  // std::swap

namespace cvdms {

// ======================================================================
// Fused K-series iteration kernel
//
// Replaces 4 kernels (laplacian + k_operator_apply + kseries_accumulate
// + convergence_check) with a single kernel per inner iteration.
//
// For each pixel:
//   1. Compute separable Laplacian from cur (neighbor reads)
//   2. Apply V multiply: K(cur) = V * cur + laplacian(cur) / (4πK₀)
//   3. Scale by coeff: result = coeff * K(cur)
//   4. Accumulate: kseries += result
//   5. Store scaled cascade: next = result
//   6. Convergence check: |result| > threshold → atomic counter
//
// Accuracy is a compile-time template for constant-folded stencil coeffs.
// ======================================================================
template <int ACC>
__global__ void kseries_iteration_kernel(
    const float *cur_re, const float *cur_im,
    float *next_re, float *next_im,
    float *kseries_re, float *kseries_im,
    const float *V,
    int nx, int ny, float pref, float inv_4piK0,
    float coeff, float threshold,
    ConvergenceResult *d_result) {

    constexpr int R = ACC / 2;  // stencil radius
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= ny || j >= nx)
        return;

    int idx = i * nx + j;

    float wre = cur_re[idx];
    float wim = cur_im[idx];

    // ---- Separable Laplacian (compile-time coefficients) ----
    float lap_re = 0.0f, lap_im = 0.0f;

    #pragma unroll
    for (int k = -R; k <= R; ++k) {
        float ck;
        if constexpr (ACC == 2) {
            ck = (k == 0) ? -2.0f : 1.0f;
        } else if constexpr (ACC == 4) {
            if (k == 0) ck = -2.5f;
            else if (k == 1 || k == -1) ck = 4.0f / 3.0f;
            else ck = -1.0f / 12.0f;
        } else if constexpr (ACC == 6) {
            if (k == 0) ck = -49.0f / 18.0f;
            else if (k == 1 || k == -1) ck = 1.5f;
            else if (k == 2 || k == -2) ck = -0.15f;
            else ck = 1.0f / 90.0f;
        } else if constexpr (ACC == 8) {
            if (k == 0) ck = -205.0f / 72.0f;
            else if (k == 1 || k == -1) ck = 8.0f / 5.0f;
            else if (k == 2 || k == -2) ck = -0.2f;
            else if (k == 3 || k == -3) ck = 8.0f / 315.0f;
            else ck = -1.0f / 560.0f;
        }

        if (k == 0) {
            lap_re += 2.0f * ck * wre;
            lap_im += 2.0f * ck * wim;
        } else {
            int jk = j + k;
            if (jk < 0) jk += nx;
            else if (jk >= nx) jk -= nx;
            int ik = i + k;
            if (ik < 0) ik += ny;
            else if (ik >= ny) ik -= ny;

            lap_re += ck * (cur_re[i * nx + jk] + cur_re[ik * nx + j]);
            lap_im += ck * (cur_im[i * nx + jk] + cur_im[ik * nx + j]);
        }
    }

    lap_re *= pref;
    lap_im *= pref;

    // ---- K-operator: K(w) = V * w + laplacian / (4πK₀) ----
    float v = V[idx];
    float kw_re = v * wre + lap_re * inv_4piK0;
    float kw_im = v * wim + lap_im * inv_4piK0;

    // ---- NaN/Inf check ----
    if (isnan(kw_re) || isinf(kw_re) || isnan(kw_im) || isinf(kw_im)) {
        atomicAdd(&d_result->n_nan, 1);
        next_re[idx] = 0.0f;
        next_im[idx] = 0.0f;
        return;
    }

    // ---- Scale by coefficient ----
    float s_re = kw_re * coeff;
    float s_im = kw_im * coeff;

    // ---- Accumulate to kseries ----
    kseries_re[idx] += s_re;
    kseries_im[idx] += s_im;

    // ---- Scaled cascade output ----
    next_re[idx] = s_re;
    next_im[idx] = s_im;

    // ---- Convergence: |scaled term| > threshold ----
    float mag2 = s_re * s_re + s_im * s_im;
    if (mag2 > threshold * threshold) {
        atomicAdd(&d_result->n_above, 1);
    }
}

// ======================================================================
// Host function: launch the fused iteration kernel with accuracy dispatch
// ======================================================================
void launch_kseries_iteration(const float *cur_re, const float *cur_im,
                              float *next_re, float *next_im,
                              float *kseries_re, float *kseries_im,
                              const float *V, std::size_t nx, std::size_t ny,
                              float inv_dx, float inv_dy,
                              float inv_4piK0, float coeff,
                              float threshold,
                              ConvergenceResult *d_result,
                              cudaStream_t stream, int accuracy) {

    dim3 block(16, 16);
    dim3 grid((nx + 15) / 16, (ny + 15) / 16);
    float pref = inv_dx * inv_dy;
    int inx = static_cast<int>(nx);
    int iny = static_cast<int>(ny);

    switch (accuracy) {
        case 2:
            kseries_iteration_kernel<2><<<grid, block, 0, stream>>>(
                cur_re, cur_im, next_re, next_im, kseries_re, kseries_im,
                V, inx, iny, pref, inv_4piK0, coeff, threshold,
                d_result);
            break;
        case 4:
            kseries_iteration_kernel<4><<<grid, block, 0, stream>>>(
                cur_re, cur_im, next_re, next_im, kseries_re, kseries_im,
                V, inx, iny, pref, inv_4piK0, coeff, threshold,
                d_result);
            break;
        case 6:
            kseries_iteration_kernel<6><<<grid, block, 0, stream>>>(
                cur_re, cur_im, next_re, next_im, kseries_re, kseries_im,
                V, inx, iny, pref, inv_4piK0, coeff, threshold,
                d_result);
            break;
        case 8:
        default:
            kseries_iteration_kernel<8><<<grid, block, 0, stream>>>(
                cur_re, cur_im, next_re, next_im, kseries_re, kseries_im,
                V, inx, iny, pref, inv_4piK0, coeff, threshold,
                d_result);
            break;
    }
}

// ======================================================================
// compute_k_series — now uses the fused iteration kernel
//
// Inner loop: 1 kernel launch per iteration (was 4), single D2H sync.
// ======================================================================
void compute_k_series(const float *psi_re, const float *psi_im,
                      float *kseries_re, float *kseries_im,
                      const float *V, std::size_t nx, std::size_t ny,
                      float wavelength, float dz,
                      float convergence_threshold, int max_inner,
                      float inv_4piK0, float inv_dx, float inv_dy,
                      DeviceArray<float> &cur_re, DeviceArray<float> &cur_im,
                      DeviceArray<float> &buf_re, DeviceArray<float> &buf_im,
                      ConvergenceResult *d_result, cudaStream_t stream,
                      int accuracy) {

    (void)dz; // not used in inner K-series

    int count = static_cast<int>(nx * ny);

    // Zero out kseries accumulator
    cudaMemsetAsync(kseries_re, 0, count * sizeof(float), stream);
    cudaMemsetAsync(kseries_im, 0, count * sizeof(float), stream);

    // cur = input psi
    cudaMemcpyAsync(cur_re.data(), psi_re, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(cur_im.data(), psi_im, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // Raw pointers for ping-pong swap (DeviceArray has no swap method)
    float *cur_re_ptr = cur_re.data();
    float *cur_im_ptr = cur_im.data();
    float *buf_re_ptr = buf_re.data();
    float *buf_im_ptr = buf_im.data();

    for (int n = 1; n <= max_inner; ++n) {
        // Coefficient c_n matching Python's _cvdms_inner_k_series:
        //   c₁ = 1
        //   cₙ = (0.5 - n + 1) * λ / (π * n)  for n > 1
        float coeff = 1.0f;
        if (n > 1) {
            coeff = (0.5f - static_cast<float>(n) + 1.0f) * wavelength /
                    (static_cast<float>(M_PI) * n);
        }

        // Reset convergence counters (single struct memset)
        cudaMemsetAsync(d_result, 0, sizeof(ConvergenceResult), stream);

        // Single fused kernel: laplacian + K-operator + scale + accumulate
        // + convergence check. Replaces 4 separate kernel launches.
        launch_kseries_iteration(
            cur_re_ptr, cur_im_ptr,
            buf_re_ptr, buf_im_ptr,
            kseries_re, kseries_im,
            V, nx, ny, inv_dx, inv_dy, inv_4piK0, coeff,
            convergence_threshold,
            d_result,
            stream, accuracy);

        // D2H sync for convergence (single struct copy)
        auto result = read_convergence(d_result, stream);
        if (result.n_nan > 0)
            break;
        if (result.n_above == 0)
            break;

        // Swap: cur = coeff * Kⁿ(ψ) → input for next K-operator iteration.
        // K is linear: K(coeff * X) = coeff * K(X), so the scaled cascade
        // propagates the coefficient to all higher-order terms.
        std::swap(cur_re_ptr, buf_re_ptr);
        std::swap(cur_im_ptr, buf_im_ptr);
    }
}

} // namespace cvdms
