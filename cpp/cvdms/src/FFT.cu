#include "cvdms/FFT.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>

namespace cvdms {

// ======================================================================
// Kernels for FFT-based Laplacian
// ======================================================================

/// Pack separate re/im arrays into interleaved cuComplex buffer.
__global__ void pack_complex_kernel(const float *re, const float *im,
                                     cufftComplex *buf, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;
    buf[idx] = make_cuComplex(re[idx], im[idx]);
}

/// Unpack interleaved cuComplex buffer into separate re/im arrays.
__global__ void unpack_complex_kernel(const cufftComplex *buf,
                                       float *re, float *im, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;
    re[idx] = buf[idx].x;
    im[idx] = buf[idx].y;
}

/// Multiply FFT data by k² factor and scale: buf *= factor * inv_N.
///
/// Combined with the scaling 1/N (cuFFT C2C is unnormalized) so that
/// after IFFT the result is the correctly normalized Laplacian.
__global__ void fft_multiply_factor_kernel(cufftComplex *buf,
                                            const float *factor,
                                            int count, float inv_N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;
    float f = factor[idx] * inv_N;
    buf[idx].x *= f;
    buf[idx].y *= f;
}

// ======================================================================
// FFTLaplacian implementation
// ======================================================================

FFTLaplacian::FFTLaplacian()
    : nx_(0), ny_(0), initialized_(false), plan_(0),
      d_buffer_(nullptr), d_factor_(nullptr) {}

FFTLaplacian::~FFTLaplacian() {
    if (plan_) {
        cufftDestroy(plan_);
    }
    if (d_buffer_) {
        cudaFree(d_buffer_);
    }
    if (d_factor_) {
        cudaFree(d_factor_);
    }
}

void FFTLaplacian::initialize(std::size_t nx, std::size_t ny,
                               float sampling_x, float sampling_y) {
    if (initialized_ && nx == nx_ && ny == ny_)
        return;

    // Destroy old resources
    if (plan_) {
        cufftDestroy(plan_);
        plan_ = 0;
    }
    if (d_buffer_) {
        cudaFree(d_buffer_);
        d_buffer_ = nullptr;
    }
    if (d_factor_) {
        cudaFree(d_factor_);
        d_factor_ = nullptr;
    }

    nx_ = nx;
    ny_ = ny;
    std::size_t count = nx * ny;

    // Create C2C FFT plan
    auto err = cufftPlan2d(&plan_, static_cast<int>(nx),
                           static_cast<int>(ny), CUFFT_C2C);
    if (err != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftPlan2d failed: " +
                                 std::to_string(err));
    }

    // Allocate interleaved buffer
    if (cudaMalloc(&d_buffer_, count * sizeof(cufftComplex)) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for FFT buffer");
    }

    // Compute k² factor on host and upload
    // kx = fftfreq(j, d=sampling_x), ky = fftfreq(i, d=sampling_y)
    // factor = -4*pi^2 * (kx^2 + ky^2)
    std::vector<float> h_factor(count);
    for (std::size_t i = 0; i < ny; ++i) {
        float ky = static_cast<float>(i);
        if (ky > ny / 2)
            ky -= static_cast<float>(ny);
        ky /= static_cast<float>(ny) * sampling_y;

        for (std::size_t j = 0; j < nx; ++j) {
            float kx = static_cast<float>(j);
            if (kx > nx / 2)
                kx -= static_cast<float>(nx);
            kx /= static_cast<float>(nx) * sampling_x;

            float k2 = kx * kx + ky * ky;
            h_factor[i * nx + j] = -4.0f * static_cast<float>(M_PI * M_PI) * k2;
        }
    }

    if (cudaMalloc(&d_factor_, count * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for k² factor");
    }
    cudaMemcpy(d_factor_, h_factor.data(), count * sizeof(float),
               cudaMemcpyHostToDevice);

    initialized_ = true;
}

void FFTLaplacian::compute(const float *psi_re, const float *psi_im,
                            float *out_re, float *out_im,
                            cudaStream_t stream) {
    if (!initialized_)
        return;

    std::size_t count = nx_ * ny_;
    int block_size = 256;
    int grid_size = (static_cast<int>(count) + block_size - 1) / block_size;

    float inv_N = 1.0f / static_cast<float>(nx_ * ny_);

    // Step 1: Pack separate re/im → interleaved cuComplex
    pack_complex_kernel<<<grid_size, block_size, 0, stream>>>(
        psi_re, psi_im, d_buffer_, static_cast<int>(count));

    // Step 2: Forward FFT
    cufftExecC2C(plan_, d_buffer_, d_buffer_, CUFFT_FORWARD);

    // Step 3: Multiply by k² factor & scale by 1/N (fused)
    fft_multiply_factor_kernel<<<grid_size, block_size, 0, stream>>>(
        d_buffer_, d_factor_, static_cast<int>(count), inv_N);

    // Step 4: Inverse FFT
    cufftExecC2C(plan_, d_buffer_, d_buffer_, CUFFT_INVERSE);

    // Step 5: Unpack interleaved → separate re/im
    unpack_complex_kernel<<<grid_size, block_size, 0, stream>>>(
        d_buffer_, out_re, out_im, static_cast<int>(count));
}

} // namespace cvdms
