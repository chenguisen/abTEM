#pragma once

#include "Array.h"
#include "Convergence.h"

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
                          cudaStream_t stream,
                          int accuracy = 8);

/// Compute the 1/k operator polynomial series applied to a wavefunction.
///
/// Computes Σ_{i=1}^{order} binom(-1/2, i) · K^i(psi) / (π·K₀)^i.
///
/// Unlike compute_full_series, there is NO final multiply by 1j*dz.
/// The prefactors already include correct 1/(π·K₀)^i scaling.
///
/// Corresponds to calOneDevideK_forward_back in ImageSimulation_CGS.
///
/// prefactors must have at least `order` elements.
/// prefactors[i] is used for K^{i+1}(psi) term.
///
/// Result is written to series_re/series_im.
void compute_one_over_k_series(const float *psi_re, const float *psi_im,
                                float *series_re, float *series_im,
                                const float *V, std::size_t nx, std::size_t ny,
                                float inv_4piK0, float inv_dx, float inv_dy,
                                int order,
                                const std::complex<float> *prefactors,
                                DeviceArray<float> &temp_re,
                                DeviceArray<float> &temp_im,
                                DeviceArray<float> &buf_re,
                                DeviceArray<float> &buf_im,
                                cudaStream_t stream,
                                int accuracy = 8);

/// Apply backscattering (BSC) correction using dual CUDA streams.
///
/// Computes:
///   wave_1 = K0 * (psi + K_series(psi, V_current))       [stream 1]
///   wave_2 = K0 * (psi + K_series(psi, V_next))          [stream 2]
///   backscatter = wave_2 - wave_1                         (raw_diff)
///   correction = 1/k_series(backscatter, V_next)          (operator acts on bs)
///   backscatter = (backscatter + correction) / (2*K0)
///
/// where 1/k_series computes (1 + K/(π·K₀))^{-1/2} - 1 applied to backscatter.
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
                           // Convergence counters (single struct, one D2H copy)
                           ConvergenceResult *d_result,
                           cudaStream_t stream1, cudaStream_t stream2,
                           int accuracy = 8);

} // namespace cvdms
