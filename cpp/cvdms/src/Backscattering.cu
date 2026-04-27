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
// Kernel: bs *= (1 + correction) * inv_2K0   (complex multiply, in-place)
// ======================================================================
__global__ void bsc_correct_kernel(float *bs_re, float *bs_im,
                                    const float *corr_re,
                                    const float *corr_im, int count,
                                    float inv_2K0) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    // factor = (1 + correction) * inv_2K0
    float fr = (1.0f + corr_re[idx]) * inv_2K0;
    float fi = corr_im[idx] * inv_2K0;

    // bs *= factor  (complex multiply)
    float br = bs_re[idx];
    float bi = bs_im[idx];
    bs_re[idx] = br * fr - bi * fi;
    bs_im[idx] = br * fi + bi * fr;
}

// ======================================================================
// compute_full_series: K-operator polynomial with override prefactors
//
// Matches Python full_series() from finite_difference.py.
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
                          cudaStream_t stream) {
    if (order < 1)
        return;

    int count = static_cast<int>(nx * ny);
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;

    // Zero the series accumulator
    cudaMemsetAsync(series_re, 0, count * sizeof(float), stream);
    cudaMemsetAsync(series_im, 0, count * sizeof(float), stream);

    // --- K^1(psi) term: series += prefactors[0] * K(psi) ---
    launch_k_operator(psi_re, psi_im, temp_re.data(), temp_im.data(), V, nx,
                      ny, inv_4piK0, inv_dx, inv_dy, stream);

    fs_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
        temp_re.data(), temp_im.data(), series_re, series_im, count,
        prefactors[0].real(), prefactors[0].imag());

    // --- Higher-order terms: K^i(psi) for i = 2..order ---
    for (int i = 1; i < order; ++i) {
        // buf = K(temp) — unscaled cascade (matches Python full_series)
        launch_k_operator(temp_re.data(), temp_im.data(), buf_re.data(),
                          buf_im.data(), V, nx, ny, inv_4piK0, inv_dx, inv_dy,
                          stream);

        // series += prefactors[i] * buf
        fs_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
            buf_re.data(), buf_im.data(), series_re, series_im, count,
            prefactors[i].real(), prefactors[i].imag());

        // swap temp <-> buf for next iteration
        std::swap(temp_re, buf_re);
        std::swap(temp_im, buf_im);
    }

    // --- series *= 1j * dz ---
    fs_finalize_kernel<<<grid_size, block_size, 0, stream>>>(series_re,
                                                               series_im, count,
                                                               dz);
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
                           int *d_count_above, int *d_count_nan,
                           int *d_count_diverging, cudaStream_t stream1,
                           cudaStream_t stream2) {
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
                     d_count_above, d_count_nan, d_count_diverging, stream1);

    // Stream 2: K_series(psi, V_next) → s2_kseries
    compute_k_series(psi_re, psi_im, s2_kseries_re.data(),
                     s2_kseries_im.data(), V_next, nx, ny, wavelength, dz,
                     convergence_threshold, max_terms, inv_4piK0, inv_dx,
                     inv_dy, s2_cur_re, s2_cur_im, s2_buf_re, s2_buf_im,
                     d_count_above, d_count_nan, d_count_diverging, stream2);

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
    // Step 4: 1/k correction via full_series(psi, V_next)
    // ================================================================

    // Compute prefactors matching Python:
    //   prefactors = [1.0]
    //   for i in range(1, order + 1):
    //       prefactors.append(prefactors[-1] * (1 - 2*i) / (2*i))
    //   for i in range(len(prefactors)):
    //       prefactors[i] /= (1j * dz) * (pi * K0)^i
    std::vector<std::complex<float>> prefac(static_cast<std::size_t>(order) +
                                            1);
    prefac[0] = std::complex<float>(1.0f, 0.0f);
    for (int i = 1; i <= order; ++i) {
        float factor = static_cast<float>(1 - 2 * i) /
                       static_cast<float>(2 * i);
        prefac[i] = prefac[i - 1] * factor;
    }
    for (int i = 0; i <= order; ++i) {
        float denom_real = -dz * std::pow(static_cast<float>(M_PI) * K0, i);
        // 1 / (1j * dz * (pi*K0)^i) = -1j / (dz * (pi*K0)^i)
        // = complex(0, -1) / (dz * (pi*K0)^i)
        // Actually: 1/(1j * x) where x = dz * (pi*K0)^i
        // 1/(1j) = -j, so 1/(1j*x) = -j/x = complex(0, -1/x)
        float scale = std::pow(static_cast<float>(M_PI) * K0,
                                static_cast<float>(i));
        prefac[i] /= std::complex<float>(0.0f, dz * scale);
    }

    // full_series output goes into s1_kseries (which held wave_1,
    // no longer needed after wave_2 - wave_1 above)
    // Reuse s1_cur/s1_buf as fs_temp/fs_buf
    compute_full_series(
        psi_re, psi_im, s1_kseries_re.data(), s1_kseries_im.data(), V_next,
        nx, ny, inv_4piK0, inv_dx, inv_dy, order, prefac.data(), dz,
        s1_cur_re, s1_cur_im, s1_buf_re, s1_buf_im, stream1);

    // ================================================================
    // Step 5: backscatter *= 1/(2*K0) * (1 + 1k_correction)
    // ================================================================
    float inv_2K0 = 1.0f / (2.0f * K0);
    bsc_correct_kernel<<<grid_size, block_size, 0, stream1>>>(
        s2_kseries_re.data(), s2_kseries_im.data(),  // bs buffer
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
