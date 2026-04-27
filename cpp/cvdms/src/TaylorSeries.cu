#include "cvdms/TaylorSeries.h"
#include "cvdms/Convergence.h"
#include "cvdms/KSeries.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cvdms {

/// Element-wise:
///   work = kseries * (i * dz / n)
///   exit += work
__global__ void taylor_scale_accumulate_kernel(const float *kseries_re,
                                               const float *kseries_im,
                                               float *work_re, float *work_im,
                                               float *exit_re, float *exit_im,
                                               int count, float dz, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    // Taylor term: kseries * i * dz / n
    // Complex multiply: (kr + i*ki) * i * dz/n = (-ki + i*kr) * dz/n
    float kr = kseries_re[idx];
    float ki = kseries_im[idx];
    float scale = dz / static_cast<float>(n);

    float wr = -ki * scale;
    float wi = kr * scale;

    work_re[idx] = wr;
    work_im[idx] = wi;
    exit_re[idx] += wr;
    exit_im[idx] += wi;
}

void compute_taylor_series(const float *psi_in_re, const float *psi_in_im,
                           float *psi_out_re, float *psi_out_im,
                           const float *V, std::size_t nx, std::size_t ny,
                           float wavelength, float dz,
                           float convergence_threshold, int max_terms,
                           float inv_4piK0, float inv_dx, float inv_dy,
                           int *d_count_above, int *d_count_nan,
                           int *d_count_diverging, bool &converged,
                           bool &overflow, DeviceArray<float> &work_re,
                           DeviceArray<float> &work_im,
                           DeviceArray<float> &kseries_re,
                           DeviceArray<float> &kseries_im,
                           DeviceArray<float> &kcur_re,
                           DeviceArray<float> &kcur_im,
                           DeviceArray<float> &kwork_re,
                           DeviceArray<float> &kwork_im,
                           cudaStream_t stream,
                           int accuracy) {

    int count = static_cast<int>(nx * ny);
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;

    converged = false;
    overflow = false;

    // exit_wave = psi_in (0th-order term)
    cudaMemcpyAsync(psi_out_re, psi_in_re, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(psi_out_im, psi_in_im, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // work = psi_in (working buffer for Taylor cascade)
    cudaMemcpyAsync(work_re.data(), psi_in_re, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(work_im.data(), psi_in_im, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // Outer Taylor loop matching Python's _cvdms_forward_scattering:
    //   for n_exp_order in range(1, max_terms + 1):
    //       k_series = inner_k_series(working, ...)
    //       working = k_series
    //       working *= i * dz / n
    //       exit_wave += working
    for (int n = 1; n <= max_terms; ++n) {
        // Step 1: Compute K_series(work) → kseries
        // work holds the cascaded result from the previous Taylor iteration
        // (or psi for n=1)
        compute_k_series(work_re.data(), work_im.data(),
                         kseries_re.data(), kseries_im.data(),
                         V, nx, ny, wavelength, dz,
                         convergence_threshold, 100, inv_4piK0, inv_dx, inv_dy,
                         kcur_re, kcur_im,    // K-operator input
                         kwork_re, kwork_im,  // K-operator output
                         d_count_above, d_count_nan, d_count_diverging,
                         stream, accuracy);

        // Step 2: work = kseries;  work *= i * dz / n;  exit += work
        // Fused kernel to minimize memory traffic
        taylor_scale_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
            kseries_re.data(), kseries_im.data(),
            work_re.data(), work_im.data(),
            psi_out_re, psi_out_im,
            count, dz, n);

        // Convergence check
        reset_counters(d_count_above, d_count_nan, d_count_diverging, stream);
        launch_convergence_check(work_re.data(), work_im.data(), nullptr,
                                 nullptr, count, convergence_threshold,
                                 d_count_above, d_count_nan, d_count_diverging,
                                 stream);

        auto result = read_convergence(d_count_above, d_count_nan,
                                       d_count_diverging, stream);
        if (result.n_nan > 0) {
            overflow = true;
            break;
        }
        if (result.n_above == 0) {
            converged = true;
            break;
        }
    }
}

} // namespace cvdms
