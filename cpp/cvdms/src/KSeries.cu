#include "cvdms/KSeries.h"
#include "cvdms/Convergence.h"
#include "cvdms/KOperator.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>

namespace cvdms {

/// Element-wise:
///   kseries += coeff * buf;
///   cur = coeff * buf  (coefficient cascade: carries product of all c_n)
///
/// buf and cur must NOT alias (each thread reads buf[idx], writes cur[idx]).
__global__ void kseries_accumulate_kernel(const float *buf_re, const float *buf_im,
                                          float *kseries_re, float *kseries_im,
                                          float *cur_re, float *cur_im,
                                          int count, float coeff) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    float cr = buf_re[idx];
    float ci = buf_im[idx];

    // kseries += coeff * K^n(psi)
    kseries_re[idx] += cr * coeff;
    kseries_im[idx] += ci * coeff;

    // cur = coeff * K^n(psi) — carries running product of coefficients
    cur_re[idx] = cr * coeff;
    cur_im[idx] = ci * coeff;
}

void compute_k_series(const float *psi_re, const float *psi_im,
                      float *kseries_re, float *kseries_im,
                      const float *V, std::size_t nx, std::size_t ny,
                      float wavelength, float dz,
                      float convergence_threshold, int max_inner,
                      float inv_4piK0, float inv_dx, float inv_dy,
                      DeviceArray<float> &cur_re, DeviceArray<float> &cur_im,
                      DeviceArray<float> &buf_re, DeviceArray<float> &buf_im,
                      int *d_count_above, int *d_count_nan,
                      int *d_count_diverging, cudaStream_t stream) {

    (void)dz; // not used in inner K-series

    int count = static_cast<int>(nx * ny);
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;

    // Zero out kseries accumulator
    cudaMemsetAsync(kseries_re, 0, count * sizeof(float), stream);
    cudaMemsetAsync(kseries_im, 0, count * sizeof(float), stream);

    // cur = input psi  (K-operator INPUT buffer — will hold cascaded product)
    cudaMemcpyAsync(cur_re.data(), psi_re, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(cur_im.data(), psi_im, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    for (int n = 1; n <= max_inner; ++n) {
        // K_operator(cur) → buf  (SAFE: cur and buf are different buffers)
        launch_k_operator(cur_re.data(), cur_im.data(),
                          buf_re.data(), buf_im.data(),
                          V, nx, ny, inv_4piK0, inv_dx, inv_dy, stream);

        // Coefficient c_n matching Python's _cvdms_inner_k_series:
        //   c₁ = 1  (unscaled first order)
        //   cₙ = (0.5 - n + 1) * λ / (π * n)  for n > 1
        float coeff = 1.0f;
        if (n > 1) {
            coeff = (0.5f - static_cast<float>(n) + 1.0f) * wavelength /
                    (static_cast<float>(M_PI) * n);
        }

        // Accumulate with scaling and cascade:
        //   kseries += coeff * buf
        //   cur = coeff * buf  (cascade: curv = coeff * K^n(psi) × prev_cascade)
        kseries_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
            buf_re.data(), buf_im.data(),
            kseries_re, kseries_im,
            cur_re.data(), cur_im.data(),
            count, coeff);

        // Convergence check (on the latest K^n(psi) term = buf)
        reset_counters(d_count_above, d_count_nan, d_count_diverging, stream);
        launch_convergence_check(buf_re.data(), buf_im.data(), nullptr, nullptr,
                                 count, convergence_threshold, d_count_above,
                                 d_count_nan, d_count_diverging, stream);

        auto result = read_convergence(d_count_above, d_count_nan,
                                       d_count_diverging, stream);
        if (result.n_nan > 0)
            break;
        if (result.n_above == 0)
            break;
    }
}

} // namespace cvdms
