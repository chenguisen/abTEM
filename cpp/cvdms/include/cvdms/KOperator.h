#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cvdms {

/// Compute K(ψ) = V·ψ + ∇²ψ / (4πK₀) as a fused kernel.
///
/// psi and Kpsi are interleaved complex64 arrays.
/// V is a real float array (potential).
void launch_k_operator(const float *psi_re, const float *psi_im,
                       float *Kpsi_re, float *Kpsi_im,
                       const float *V, std::size_t nx, std::size_t ny,
                       float inv_4piK0, float inv_dx, float inv_dy,
                       cudaStream_t stream = nullptr);

} // namespace cvdms
