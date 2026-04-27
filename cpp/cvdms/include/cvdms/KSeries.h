#pragma once

#include "Array.h"

#include <cstddef>

namespace cvdms {

/// Compute one fused K-series iteration for the inner loop.
///
/// Single kernel replaces: laplacian + k_operator_apply + kseries_accumulate
/// + convergence_check.
///
/// cur_re/cur_im: K-operator input (cur = coeff_prev * K^{n-1}(ψ))
/// next_re/next_im: scaled cascade output (next = coeff * K(cur))
/// kseries_re/kseries_im: result accumulator (kseries += coeff * K(cur))
/// V: potential array
/// coeff: K-series coefficient c_n for this iteration
/// threshold: convergence threshold (compared against |coeff * K(cur)|)
void launch_kseries_iteration(const float *cur_re, const float *cur_im,
                              float *next_re, float *next_im,
                              float *kseries_re, float *kseries_im,
                              const float *V, std::size_t nx, std::size_t ny,
                              float inv_dx, float inv_dy,
                              float inv_4piK0, float coeff,
                              float threshold,
                              int *d_count_above, int *d_count_nan,
                              cudaStream_t stream, int accuracy);

/// Compute the inner K-series: Σ cₙ·Kⁿ(ψ) using ping-pong buffers.
///
/// cur_re/cur_im: K-operator INPUT buffer (must NOT alias buf)
/// buf_re/buf_im: K-operator OUTPUT / accumulate buffer
/// kseries: result accumulator (must NOT alias cur or buf)
///
/// Returns the K-series result in kseries.
///
/// accuracy: Laplacian finite-difference accuracy (default 8 = Python default).
void compute_k_series(const float *psi_re, const float *psi_im,
                      float *kseries_re, float *kseries_im,
                      const float *V, std::size_t nx, std::size_t ny,
                      float wavelength, float dz,
                      float convergence_threshold, int max_inner,
                      float inv_4piK0, float inv_dx, float inv_dy,
                      DeviceArray<float> &cur_re, DeviceArray<float> &cur_im,
                      DeviceArray<float> &buf_re, DeviceArray<float> &buf_im,
                      int *d_count_above, int *d_count_nan,
                      int *d_count_diverging,
                      cudaStream_t stream = nullptr,
                      int accuracy = 8);

} // namespace cvdms
