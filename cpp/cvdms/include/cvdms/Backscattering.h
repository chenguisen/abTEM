#pragma once

#include "Array.h"

#include <complex>
#include <cstddef>
#include <vector>

namespace cvdms {

/// Compute the K-operator polynomial series with override prefactors.
///
/// Implements Python full_series() from finite_difference.py:
///   series = K(psi)                           (coeff = 1.0)
///   for i = 2..order:
///       temp = K(temp)                        (unscaled cascade)
///       series += temp * prefactors[i-1]      (scaled accumulation)
///   return series * 1j * dz
///
/// prefactors must have at least `order` elements.
/// prefactors[0] is used for K(psi), prefactors[i-1] for K^i(psi).
///
/// Result is written to series_re/series_im.
void compute_full_series(const float *psi_re, const float *psi_im,
                          float *series_re, float *series_im,
                          const float *V, std::size_t nx, std::size_t ny,
                          float inv_4piK0, float inv_dx, float inv_dy,
                          int order,
                          const std::complex<float> *prefactors,
                          float dz,
                          DeviceArray<float> &temp_re,
                          DeviceArray<float> &temp_im,
                          DeviceArray<float> &buf_re,
                          DeviceArray<float> &buf_im,
                          cudaStream_t stream);

/// Apply backscattering (BSC) correction using dual CUDA streams.
///
/// Computes:
///   wave_1 = K0 * (psi + K_series(psi, V_current))       [stream 1]
///   wave_2 = K0 * (psi + K_series(psi, V_next))          [stream 2]
///   backscatter = wave_2 - wave_1
///   1k_correction = full_series(psi, V_next, prefactors)
///   backscatter *= 1/(2*K0) * (1 + 1k_correction)
///
/// Output is written to backscatter_re/backscatter_im.
/// All stream working buffers must be pre-allocated to nx*ny.
void apply_backscattering(const float *psi_re, const float *psi_im,
                           float *backscatter_re, float *backscatter_im,
                           const float *V_current, const float *V_next,
                           std::size_t nx, std::size_t ny,
                           float wavelength, float dz, int order,
                           float convergence_threshold, int max_terms,
                           float inv_4piK0, float inv_dx, float inv_dy,
                           // Stream 1 buffers (K_series for V_current)
                           DeviceArray<float> &s1_cur_re,
                           DeviceArray<float> &s1_cur_im,
                           DeviceArray<float> &s1_buf_re,
                           DeviceArray<float> &s1_buf_im,
                           DeviceArray<float> &s1_kseries_re,
                           DeviceArray<float> &s1_kseries_im,
                           // Stream 2 buffers (K_series for V_next)
                           DeviceArray<float> &s2_cur_re,
                           DeviceArray<float> &s2_cur_im,
                           DeviceArray<float> &s2_buf_re,
                           DeviceArray<float> &s2_buf_im,
                           DeviceArray<float> &s2_kseries_re,
                           DeviceArray<float> &s2_kseries_im,
                           // Full series working buffers (reuse s1 after sync)
                           DeviceArray<float> &fs_temp_re,
                           DeviceArray<float> &fs_temp_im,
                           DeviceArray<float> &fs_buf_re,
                           DeviceArray<float> &fs_buf_im,
                           // Convergence counters
                           int *d_count_above, int *d_count_nan,
                           int *d_count_diverging,
                           cudaStream_t stream1, cudaStream_t stream2);

} // namespace cvdms
