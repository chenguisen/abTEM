#include "cvdms/KOperator.h"
#include "cvdms/Laplacian.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cvdms {

__global__ void k_operator_apply_kernel(const float *lap_re, const float *lap_im,
                                        const float *psi_re, const float *psi_im,
                                        float *Kpsi_re, float *Kpsi_im,
                                        const float *V, int count,
                                        float inv_4piK0) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count)
        return;

    // K(ψ) = V·ψ + ∇²ψ / (4πK₀)
    float Vpsi_re = V[idx] * psi_re[idx];
    float Vpsi_im = V[idx] * psi_im[idx];

    float lap_term_re = lap_re[idx] * inv_4piK0;
    float lap_term_im = lap_im[idx] * inv_4piK0;

    Kpsi_re[idx] = Vpsi_re + lap_term_re;
    Kpsi_im[idx] = Vpsi_im + lap_term_im;
}

void launch_k_operator(const float *psi_re, const float *psi_im,
                       float *Kpsi_re, float *Kpsi_im,
                       const float *V, std::size_t nx, std::size_t ny,
                       float inv_4piK0, float inv_dx, float inv_dy,
                       cudaStream_t stream, int accuracy) {
    int count = static_cast<int>(nx * ny);

    // First compute Laplacian into temporary buffers
    // We reuse Kpsi as temp since it will be overwritten anyway
    launch_laplacian(psi_re, psi_im, Kpsi_re, Kpsi_im, nx, ny, inv_dx,
                      inv_dy, stream, accuracy);

    // Then apply K operator
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;

    k_operator_apply_kernel<<<grid_size, block_size, 0, stream>>>(
        Kpsi_re, Kpsi_im, psi_re, psi_im, Kpsi_re, Kpsi_im, V, count,
        inv_4piK0);
}

} // namespace cvdms
