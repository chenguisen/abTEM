#include "cvdms/Convergence.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cvdms {

__global__ void convergence_check_kernel(const float *re, const float *im,
                                         const float *prev_re,
                                         const float *prev_im, int count,
                                         float threshold,
                                         ConvergenceResult *d_result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    float r = re[idx];
    float i = im[idx];
    float mag2 = r * r + i * i;

    // NaN/Inf detection
    bool is_nan = isnan(mag2) || isinf(mag2);
    if (is_nan) {
        atomicAdd(&d_result->n_nan, 1);
        return;
    }

    // Above threshold
    if (mag2 > threshold * threshold) {
        atomicAdd(&d_result->n_above, 1);
    }

    // Divergence: compare with previous iteration
    if (prev_re && prev_im) {
        float pr = prev_re[idx];
        float pi = prev_im[idx];
        float prev_mag2 = pr * pr + pi * pi;
        if (prev_mag2 > 1e-30f && mag2 > prev_mag2 * 1.5f) {
            atomicAdd(&d_result->n_diverging, 1);
        }
    }
}

void launch_convergence_check(const float *re, const float *im,
                              const float *prev_re, const float *prev_im,
                              std::size_t count, float threshold,
                              ConvergenceResult *d_result,
                              cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (static_cast<int>(count) + block_size - 1) / block_size;

    convergence_check_kernel<<<grid_size, block_size, 0, stream>>>(
        re, im, prev_re, prev_im, static_cast<int>(count), threshold,
        d_result);
}

ConvergenceResult read_convergence(ConvergenceResult *d_result,
                                   cudaStream_t stream) {
    ConvergenceResult result = {0, 0, 0};

    auto err = cudaMemcpyAsync(&result, d_result, sizeof(ConvergenceResult),
                               cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return {0, 0, 0};
    }
    cudaStreamSynchronize(stream);

    return result;
}

void reset_counters(ConvergenceResult *d_result,
                    cudaStream_t stream) {
    cudaMemsetAsync(d_result, 0, sizeof(ConvergenceResult), stream);
}

} // namespace cvdms
