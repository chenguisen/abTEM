#pragma once

#include "Array.h"

#include <cstddef>

namespace cvdms {

/// Compute the inner K-series: Σ cₙ·Kⁿ(ψ) using ping-pong buffers.
///
/// cur_re/cur_im: K-operator INPUT buffer (must NOT alias buf)
/// buf_re/buf_im: K-operator OUTPUT / accumulate buffer
/// kseries: result accumulator (must NOT alias cur or buf)
///
/// Returns the K-series result in kseries.
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
                      cudaStream_t stream = nullptr);

} // namespace cvdms
