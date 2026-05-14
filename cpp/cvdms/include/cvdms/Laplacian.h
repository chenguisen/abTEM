#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cvdms {

/// Compute the 2D Laplacian using a separable finite-difference stencil.
///
/// Matches Python _laplace_operator_stencil(accuracy=acc, ...):
///
///   ∇²ψ = pref · Σ c[k] · (ψ[i+k,j] + ψ[i,j+k])
///
/// where c = np.roll(finite_difference_coefficients(2, acc), -(acc/2)).
///
/// The `accuracy` parameter controls the stencil radius:
///   2 → 3-point 1D stencil (standard 5-pt 2D Laplacian)
///   4 → 5-point 1D stencil
///   6 → 7-point 1D stencil
///   8 → 9-point 1D stencil (Python default)
///
/// Boundary condition: periodic wrap (matching Python mode="wrap").
void launch_laplacian(const float *re_in, const float *im_in,
                      float *re_out, float *im_out,
                      std::size_t nx, std::size_t ny,
                      float inv_dx, float inv_dy,
                      cudaStream_t stream = nullptr,
                      int accuracy = 8);

} // namespace cvdms
