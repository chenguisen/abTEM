#include "cvdms/Convergence.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cvdms {

__global__ void convergence_check_kernel(const float *re, const float *im,
                                         const float *prev_re,
                                         const float *prev_im, int count,
                                         float threshold, int *count_above,
                                         int *count_nan,
                                         int *count_diverging) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    float r = re[idx];
    float i = im[idx];
    float mag2 = r * r + i * i;

    // NaN/Inf detection
    bool is_nan = isnan(mag2) || isinf(mag2);
    if (is_nan) {
        atomicAdd(count_nan, 1);
        return;
    }

    // Above threshold
    if (mag2 > threshold * threshold) {
        atomicAdd(count_above, 1);
    }

    // Divergence: compare with previous iteration
    if (prev_re && prev_im) {
        float pr = prev_re[idx];
        float pi = prev_im[idx];
        float prev_mag2 = pr * pr + pi * pi;
        if (prev_mag2 > 1e-30f && mag2 > prev_mag2 * 1.5f) {
            atomicAdd(count_diverging, 1);
        }
    }
}

void launch_convergence_check(const float *re, const float *im,
                              const float *prev_re, const float *prev_im,
                              std::size_t count, float threshold,
                              int *d_count_above, int *d_count_nan,
                              int *d_count_diverging, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (static_cast<int>(count) + block_size - 1) / block_size;

    convergence_check_kernel<<<grid_size, block_size, 0, stream>>>(
        re, im, prev_re, prev_im, static_cast<int>(count), threshold,
        d_count_above, d_count_nan, d_count_diverging);
}

ConvergenceResult read_convergence(int *d_count_above, int *d_count_nan,
                                   int *d_count_diverging,
                                   cudaStream_t stream) {
    ConvergenceResult result;
    int h_count_above[1] = {0};
    int h_count_nan[1] = {0};
    int h_count_diverging[1] = {0};

    auto err = cudaMemcpyAsync(h_count_above, d_count_above, sizeof(int),
                               cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return {0, 0, 0};
    }
    err = cudaMemcpyAsync(h_count_nan, d_count_nan, sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return {0, 0, 0};
    }
    err = cudaMemcpyAsync(h_count_diverging, d_count_diverging, sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return {0, 0, 0};
    }
    cudaStreamSynchronize(stream);

    result.n_above = h_count_above[0];
    result.n_nan = h_count_nan[0];
    result.n_diverging = h_count_diverging[0];
    return result;
}

void reset_counters(int *d_count_above, int *d_count_nan,
                    int *d_count_diverging, cudaStream_t stream) {
    cudaMemsetAsync(d_count_above, 0, sizeof(int), stream);
    cudaMemsetAsync(d_count_nan, 0, sizeof(int), stream);
    cudaMemsetAsync(d_count_diverging, 0, sizeof(int), stream);
}

} // namespace cvdms
