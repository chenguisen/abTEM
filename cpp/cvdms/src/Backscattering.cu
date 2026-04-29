#include "cvdms/Backscattering.h"
#include "cvdms/KOperator.h"
#include "cvdms/KSeries.h"
#include "cvdms/Convergence.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <complex>
#include <vector>

namespace cvdms {

// ======================================================================
// Kernel: series += coeff * in  (complex multiply-and-accumulate)
// ======================================================================
__global__ void fs_accumulate_kernel(const float *in_re, const float *in_im,
                                     float *series_re, float *series_im,
                                     int count, float coeff_re,
                                     float coeff_im) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    // complex multiply: in * coeff
    float ir = in_re[idx];
    float ii = in_im[idx];
    float pr = ir * coeff_re - ii * coeff_im;
    float pi = ir * coeff_im + ii * coeff_re;

    series_re[idx] += pr;
    series_im[idx] += pi;
}

// ======================================================================
// Kernel: series *= (0 + 1j) * dz   i.e.  series_re,im = (-im, re) * dz
// ======================================================================
__global__ void fs_finalize_kernel(float *series_re, float *series_im,
                                    int count, float dz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    // multiply by i: (r + i*i) * i = -i + i*r
    float r = series_re[idx];
    float i = series_im[idx];
    series_re[idx] = -i * dz;
    series_im[idx] = r * dz;
}

// ======================================================================
// Kernel: wave = K0 * (psi + kseries)   in-place on kseries buffer
// ======================================================================
__global__ void bsc_wave_kernel(const float *kseries_re,
                                 const float *kseries_im,
                                 const float *psi_re, const float *psi_im,
                                 float *wave_re, float *wave_im, int count,
                                 float K0) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    float wr = psi_re[idx] + kseries_re[idx];
    float wi = psi_im[idx] + kseries_im[idx];
    wave_re[idx] = wr * K0;
    wave_im[idx] = wi * K0;
}

// ======================================================================
// Kernel: wave2 -= wave1  (in-place on wave2)
// ======================================================================
__global__ void bsc_diff_kernel(const float *wave1_re, const float *wave1_im,
                                 float *wave2_re, float *wave2_im,
                                 int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    wave2_re[idx] -= wave1_re[idx];
    wave2_im[idx] -= wave1_im[idx];
}

// ======================================================================
// Kernel: bs = (bs + correction) * inv_2K0   (proper operator addition)
//
// Matches calOneDevideK_forward_back: backscatter += series; /= 2K0
// ======================================================================
__global__ void bsc_add_correct_kernel(float *bs_re, float *bs_im,
                                        const float *corr_re,
                                        const float *corr_im, int count,
                                        float inv_2K0) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    float br = bs_re[idx] + corr_re[idx];
    float bi = bs_im[idx] + corr_im[idx];
    bs_re[idx] = br * inv_2K0;
    bs_im[idx] = bi * inv_2K0;
}

// ======================================================================
// Fused kernel for full_series: Laplacian + K-operator + accumulate
//
// For each pixel:
//   1. Compute separable Laplacian from cur
//   2. K(cur) = V * cur + laplacian / (4πK₀)
//   3. series += prefactor * K(cur)         (complex multiply-accumulate)
//   4. Store K(cur) → next (unscaled cascade)
//
// Accuracy is a compile-time template for constant-folded stencil coeffs.
// ======================================================================
template <int ACC>
__global__ void fs_fused_kernel(const float *cur_re, const float *cur_im,
                                 float *next_re, float *next_im,
                                 float *series_re, float *series_im,
                                 const float *V,
                                 int nx, int ny, float pref, float inv_4piK0,
                                 float coeff_re, float coeff_im) {
    constexpr int R = ACC / 2;
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

    // ---- K-operator: K(cur) = V * cur + laplacian / (4πK₀) ----
    float v = V[idx];
    float kw_re = v * wre + lap_re * inv_4piK0;
    float kw_im = v * wim + lap_im * inv_4piK0;

    // ---- Store K(cur) for cascade (unscaled) ----
    next_re[idx] = kw_re;
    next_im[idx] = kw_im;

    // ---- Accumulate: series += coeff * K(cur) ----
    float pr = kw_re * coeff_re - kw_im * coeff_im;
    float pi = kw_re * coeff_im + kw_im * coeff_re;
    series_re[idx] += pr;
    series_im[idx] += pi;
}

// ======================================================================
// Host dispatch: launch fs_fused_kernel with accuracy dispatch
// ======================================================================
static void launch_fs_fused(const float *cur_re, const float *cur_im,
                             float *next_re, float *next_im,
                             float *series_re, float *series_im,
                             const float *V, std::size_t nx, std::size_t ny,
                             float inv_dx, float inv_dy,
                             float inv_4piK0,
                             float coeff_re, float coeff_im,
                             cudaStream_t stream, int accuracy) {
    dim3 block(16, 16);
    dim3 grid((nx + 15) / 16, (ny + 15) / 16);
    float pref = inv_dx * inv_dy;
    int inx = static_cast<int>(nx);
    int iny = static_cast<int>(ny);

    switch (accuracy) {
        case 2:
            fs_fused_kernel<2><<<grid, block, 0, stream>>>(
                cur_re, cur_im, next_re, next_im, series_re, series_im,
                V, inx, iny, pref, inv_4piK0, coeff_re, coeff_im);
            break;
        case 4:
            fs_fused_kernel<4><<<grid, block, 0, stream>>>(
                cur_re, cur_im, next_re, next_im, series_re, series_im,
                V, inx, iny, pref, inv_4piK0, coeff_re, coeff_im);
            break;
        case 6:
            fs_fused_kernel<6><<<grid, block, 0, stream>>>(
                cur_re, cur_im, next_re, next_im, series_re, series_im,
                V, inx, iny, pref, inv_4piK0, coeff_re, coeff_im);
            break;
        case 8:
        default:
            fs_fused_kernel<8><<<grid, block, 0, stream>>>(
                cur_re, cur_im, next_re, next_im, series_re, series_im,
                V, inx, iny, pref, inv_4piK0, coeff_re, coeff_im);
            break;
    }
}

// ======================================================================
// compute_full_series: K-operator polynomial with override prefactors
//
// Matches Python full_series() from finite_difference.py.
// Uses fused kernel: 1 launch/power instead of 3 (laplacian + apply + acc).
// ======================================================================
void compute_full_series(const float *psi_re, const float *psi_im,
                          float *series_re, float *series_im,
                          const float *V, std::size_t nx, std::size_t ny,
                          float inv_4piK0, float inv_dx, float inv_dy,
                          int order,
                          const std::complex<float> *prefactors, float dz,
                          DeviceArray<float> &temp_re,
                          DeviceArray<float> &temp_im,
                          DeviceArray<float> &buf_re,
                          DeviceArray<float> &buf_im,
                          cudaStream_t stream,
                          int accuracy) {
    if (order < 1)
        return;

    // Zero the series accumulator
    std::size_t count = nx * ny;
    cudaMemsetAsync(series_re, 0, count * sizeof(float), stream);
    cudaMemsetAsync(series_im, 0, count * sizeof(float), stream);

    // --- Term 1: K(psi) → temp, series += prefactors[0] * K(psi) ---
    launch_fs_fused(psi_re, psi_im, temp_re.data(), temp_im.data(),
                    series_re, series_im, V, nx, ny, inv_dx, inv_dy,
                    inv_4piK0, prefactors[0].real(), prefactors[0].imag(),
                    stream, accuracy);

    // --- Higher-order terms: K^i(psi) for i = 2..order ---
    for (int i = 1; i < order; ++i) {
        // buf = K(temp); series += prefactors[i] * buf → single fused kernel
        launch_fs_fused(temp_re.data(), temp_im.data(),
                        buf_re.data(), buf_im.data(),
                        series_re, series_im, V, nx, ny, inv_dx, inv_dy,
                        inv_4piK0, prefactors[i].real(), prefactors[i].imag(),
                        stream, accuracy);

        // swap temp ↔ buf for next cascade iteration
        std::swap(temp_re, buf_re);
        std::swap(temp_im, buf_im);
    }

    // --- series *= 1j * dz ---
    int block_size = 256;
    int grid_size = (static_cast<int>(count) + block_size - 1) / block_size;
    fs_finalize_kernel<<<grid_size, block_size, 0, stream>>>(
        series_re, series_im, static_cast<int>(count), dz);
}

// ======================================================================
// compute_one_over_k_series: 1/k operator polynomial series
//
// Computes  Σ_{i=1}^{order}  binom(-1/2, i) · K^i(psi) / (π·K₀)^i
//
// Unlike compute_full_series, there is NO final multiply by 1j*dz.
// The prefactors[i] = binom(-1/2, i+1) / (π·K₀)^{i+1}  already include
// the correct scaling for the 1/k operator.
//
// Input:  psi_re/psi_im  — wavefunction to apply 1/k correction to
// Output: series_re/series_im  — the 1/k correction series
//
// Corresponds to calOneDevideK_forward_back in ImageSimulation_CGS.
// ======================================================================
void compute_one_over_k_series(const float *psi_re, const float *psi_im,
                                float *series_re, float *series_im,
                                const float *V, std::size_t nx, std::size_t ny,
                                float inv_4piK0, float inv_dx, float inv_dy,
                                int order,
                                const std::complex<float> *prefactors,
                                DeviceArray<float> &temp_re,
                                DeviceArray<float> &temp_im,
                                DeviceArray<float> &buf_re,
                                DeviceArray<float> &buf_im,
                                cudaStream_t stream,
                                int accuracy) {
    if (order < 1)
        return;

    std::size_t count = nx * ny;

    // Zero the series accumulator
    cudaMemsetAsync(series_re, 0, count * sizeof(float), stream);
    cudaMemsetAsync(series_im, 0, count * sizeof(float), stream);

    // --- Term 1: temp = K(psi), series += prefactors[0] * K(psi) ---
    launch_fs_fused(psi_re, psi_im,
                    temp_re.data(), temp_im.data(),
                    series_re, series_im,
                    V, nx, ny, inv_dx, inv_dy,
                    inv_4piK0,
                    prefactors[0].real(), prefactors[0].imag(),
                    stream, accuracy);

    // --- Higher-order terms: K^i(psi) for i = 2..order ---
    for (int i = 1; i < order; ++i) {
        launch_fs_fused(temp_re.data(), temp_im.data(),
                        buf_re.data(), buf_im.data(),
                        series_re, series_im,
                        V, nx, ny, inv_dx, inv_dy,
                        inv_4piK0,
                        prefactors[i].real(), prefactors[i].imag(),
                        stream, accuracy);

        std::swap(temp_re, buf_re);
        std::swap(temp_im, buf_im);
    }
    // NOTE: No final multiply by 1j*dz — prefactors already have correct scaling.
}

// ======================================================================
// apply_backscattering: dual-stream BSC correction
//
// Matches Python _cvdms_backscattering_correction().
// ======================================================================
void apply_backscattering(const float *psi_re, const float *psi_im,
                           float *backscatter_re, float *backscatter_im,
                           const float *V_current, const float *V_next,
                           std::size_t nx, std::size_t ny, float wavelength,
                           float dz, int order, float convergence_threshold,
                           int max_terms, float inv_4piK0, float inv_dx,
                           float inv_dy,
                           DeviceArray<float> &s1_cur_re,
                           DeviceArray<float> &s1_cur_im,
                           DeviceArray<float> &s1_buf_re,
                           DeviceArray<float> &s1_buf_im,
                           DeviceArray<float> &s1_kseries_re,
                           DeviceArray<float> &s1_kseries_im,
                           DeviceArray<float> &s2_cur_re,
                           DeviceArray<float> &s2_cur_im,
                           DeviceArray<float> &s2_buf_re,
                           DeviceArray<float> &s2_buf_im,
                           DeviceArray<float> &s2_kseries_re,
                           DeviceArray<float> &s2_kseries_im,
                           DeviceArray<float> &fs_temp_re,
                           DeviceArray<float> &fs_temp_im,
                           DeviceArray<float> &fs_buf_re,
                           DeviceArray<float> &fs_buf_im,
                           ConvergenceResult *d_result, cudaStream_t stream1,
                           cudaStream_t stream2,
                           int accuracy) {
    int count = static_cast<int>(nx * ny);
    float K0 = 1.0f / wavelength;
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;

    // ================================================================
    // Step 1: Dual-stream K_series computations
    // ================================================================

    // Stream 1: K_series(psi, V_current) → s1_kseries
    compute_k_series(psi_re, psi_im, s1_kseries_re.data(),
                     s1_kseries_im.data(), V_current, nx, ny, wavelength, dz,
                     convergence_threshold, max_terms, inv_4piK0, inv_dx,
                     inv_dy, s1_cur_re, s1_cur_im, s1_buf_re, s1_buf_im,
                     d_result, stream1, accuracy);

    // Stream 2: K_series(psi, V_next) → s2_kseries
    compute_k_series(psi_re, psi_im, s2_kseries_re.data(),
                     s2_kseries_im.data(), V_next, nx, ny, wavelength, dz,
                     convergence_threshold, max_terms, inv_4piK0, inv_dx,
                     inv_dy, s2_cur_re, s2_cur_im, s2_buf_re, s2_buf_im,
                     d_result, stream2, accuracy);

    // Synchronize both streams before combining results
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // ================================================================
    // Step 2: Compute wave_1 and wave_2
    // ================================================================

    // wave_1 = K0 * (psi + K_series(psi, V_current))  — in-place on s1_kseries
    bsc_wave_kernel<<<grid_size, block_size, 0, stream1>>>(
        s1_kseries_re.data(), s1_kseries_im.data(), psi_re, psi_im,
        s1_kseries_re.data(), s1_kseries_im.data(), count, K0);

    // wave_2 = K0 * (psi + K_series(psi, V_next))  — in-place on s2_kseries
    bsc_wave_kernel<<<grid_size, block_size, 0, stream2>>>(
        s2_kseries_re.data(), s2_kseries_im.data(), psi_re, psi_im,
        s2_kseries_re.data(), s2_kseries_im.data(), count, K0);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // ================================================================
    // Step 3: backscatter = wave_2 - wave_1  (in-place on s2_kseries)
    // ================================================================
    bsc_diff_kernel<<<grid_size, block_size, 0, stream1>>>(
        s1_kseries_re.data(), s1_kseries_im.data(),
        s2_kseries_re.data(), s2_kseries_im.data(), count);

    // ================================================================
    // Step 4: 1/k correction series applied to backscatter
    //
    // 对应 calOneDevideK_forward_back in ImageSimulation_CGS
    //
    // corr_prefac[i] = binom(-1/2, i+1) / (π·K₀)^{i+1}  for i=0..order-1
    //
    // The series is applied DIRECTLY to backscatter (s2_kseries),
    // not to psi — this is the physically correct operator application.
    // ================================================================

    // Compute binom(-1/2, n) for n = 0..order
    std::vector<std::complex<float>> binom(static_cast<std::size_t>(order) + 1);
    binom[0] = std::complex<float>(1.0f, 0.0f);
    for (int i = 1; i <= order; ++i) {
        float factor = static_cast<float>(1 - 2 * i) /
                       static_cast<float>(2 * i);
        binom[i] = binom[i - 1] * factor;
    }

    // Compute corr_prefac[i] = binom(-1/2, i+1) / (π·K₀)^{i+1}
    // No 1j*dz factor — compute_one_over_k_series has no final multiply.
    std::vector<std::complex<float>> corr_prefac(
        static_cast<std::size_t>(order));
    for (int i = 0; i < order; ++i) {
        float scale = std::pow(static_cast<float>(M_PI) * K0,
                                static_cast<float>(i + 1));
        corr_prefac[i] = binom[static_cast<std::size_t>(i) + 1] / scale;
    }

    // Compute correction on backscatter (s2_kseries), store in s1_kseries
    // (which held wave_1, no longer needed after wave_2 - wave_1 above)
    // Reuse s1_cur/s1_buf as temp/buf for the series computation
    compute_one_over_k_series(
        s2_kseries_re.data(), s2_kseries_im.data(),  // input = backscatter
        s1_kseries_re.data(), s1_kseries_im.data(),  // output = correction
        V_next, nx, ny, inv_4piK0, inv_dx, inv_dy,
        order, corr_prefac.data(),
        s1_cur_re, s1_cur_im, s1_buf_re, s1_buf_im,
        stream1, accuracy);

    // ================================================================
    // Step 5: backscatter = (backscatter + 1k_correction) / (2*K0)
    // ================================================================
    float inv_2K0 = 1.0f / (2.0f * K0);
    bsc_add_correct_kernel<<<grid_size, block_size, 0, stream1>>>(
        s2_kseries_re.data(), s2_kseries_im.data(),  // bs buffer (in/out)
        s1_kseries_re.data(), s1_kseries_im.data(),  // correction buffer
        count, inv_2K0);

    // ================================================================
    // Step 6: Copy result to output
    // ================================================================
    cudaMemcpyAsync(backscatter_re, s2_kseries_re.data(),
                    count * sizeof(float), cudaMemcpyDeviceToDevice, stream1);
    cudaMemcpyAsync(backscatter_im, s2_kseries_im.data(),
                    count * sizeof(float), cudaMemcpyDeviceToDevice, stream1);

    cudaStreamSynchronize(stream1);
}

} // namespace cvdms
