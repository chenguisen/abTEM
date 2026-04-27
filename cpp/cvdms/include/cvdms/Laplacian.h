#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cvdms {

/// Compute the 2D Laplacian using a compact 9-point stencil.
///
/// Stencil (×1/(6·dx·dy)):
///       1    4    1
///       4  -20    4
///       1    4    1
///
/// ∇²ψ ≈ 1/(6h²) · [4·sum(4-neighbors) + sum(diagonals) - 20·center]
///
/// Accuracy: O(h⁴) — cancelles O(h²) errors in both coordinate and mixed
/// partial derivatives.  More isotropic than the 5-point stencil.
///
/// Boundary condition: periodic wrap (matching the Python fused kernel).
void launch_laplacian(const float *re_in, const float *im_in,
                      float *re_out, float *im_out,
                      std::size_t nx, std::size_t ny,
                      float inv_dx, float inv_dy,
                      cudaStream_t stream = nullptr);

} // namespace cvdms
