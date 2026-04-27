#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cvdms {

/// Compute K(ψ) = V·ψ + ∇²ψ / (4πK₀) as a fused kernel.
///
/// psi and Kpsi are separate re/im arrays.
/// V is a real float array (potential).
///
/// accuracy: Laplacian finite-difference accuracy (2, 4, 6, or 8).
///   Default 8 matches Python CVDMSMultislice(derivative_accuracy=8).
void launch_k_operator(const float *psi_re, const float *psi_im,
                       float *Kpsi_re, float *Kpsi_im,
                       const float *V, std::size_t nx, std::size_t ny,
                       float inv_4piK0, float inv_dx, float inv_dy,
                       cudaStream_t stream = nullptr,
                       int accuracy = 8);

/// Element-wise K(ψ) = V·ψ + ∇²ψ / (4πK₀) using pre-computed Laplacian.
///
/// Unlike launch_k_operator, this does NOT compute the Laplacian internally.
/// Use this when the Laplacian was already computed via FFT.
void launch_k_operator_from_laplacian(const float *psi_re, const float *psi_im,
                                       const float *lap_re, const float *lap_im,
                                       float *Kpsi_re, float *Kpsi_im,
                                       const float *V, std::size_t count,
                                       float inv_4piK0,
                                       cudaStream_t stream = nullptr);

} // namespace cvdms
