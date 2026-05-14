#pragma once

#include "Array.h"
#include "Convergence.h"
#include "FFT.h"

#include <cstddef>

namespace cvdms {

/// FFT-based Taylor series: same as compute_taylor_series but uses cuFFT
/// for the Laplacian (∇²ψ = IFFT[-4π²k²·FFT(ψ)]) instead of finite-difference.
///
/// fft_laplacian: initialized FFTLaplacian object.
void compute_taylor_series_fft(const float *psi_in_re, const float *psi_in_im,
                                float *psi_out_re, float *psi_out_im,
                                const float *V, std::size_t nx, std::size_t ny,
                                float wavelength, float dz,
                                float convergence_threshold,
                                int max_terms, int max_inner,
                                float inv_4piK0,
                                ConvergenceResult *d_result,
                                bool &converged, bool &overflow,
                                DeviceArray<float> &work_re,
                                DeviceArray<float> &work_im,
                                DeviceArray<float> &kseries_re,
                                DeviceArray<float> &kseries_im,
                                DeviceArray<float> &kcur_re,
                                DeviceArray<float> &kcur_im,
                                DeviceArray<float> &kwork_re,
                                DeviceArray<float> &kwork_im,
                                FFTLaplacian &fft_laplacian,
                                DeviceArray<float> &lap_re,
                                DeviceArray<float> &lap_im,
                                int *outer_iters = nullptr,
                                cudaStream_t stream = nullptr,
                                AntialiasFilter *antialias_filter = nullptr,
                                float divergence_ratio = 0.0f,
                                float *d_sum_work = nullptr,
                                float *d_sum_exit = nullptr);

/// Compute the outer Taylor series: exp(i·dz·K) ≈ Σ (i·dz·K)ⁿ/n!
///
/// Takes an input wavefunction and computes the multislice exit wave
/// using the Taylor expansion of the scattering operator.
///
/// Buffer layout:
///   work_re/work_im: Taylor working buffer (holds cascaded wave between iterations)
///   kseries_re/kseries_im: K-series result buffer
///   kcur_re/kcur_im: K-series K-operator INPUT (separate from buf)
///   kwork_re/kwork_im: K-series K-operator OUTPUT / scratch
///
/// d_result: ConvergenceResult struct (single D2H copy for all counters).
///
/// Returns the exit wave (in-place on psi_out).
void compute_taylor_series(const float *psi_in_re, const float *psi_in_im,
                           float *psi_out_re, float *psi_out_im,
                           const float *V, std::size_t nx, std::size_t ny,
                           float wavelength, float dz,
                           float convergence_threshold,
                           int max_terms, int max_inner,
                           float inv_4piK0, float inv_dx, float inv_dy,
                           ConvergenceResult *d_result,
                           bool &converged, bool &overflow,
                           DeviceArray<float> &work_re,
                           DeviceArray<float> &work_im,
                           DeviceArray<float> &kseries_re,
                           DeviceArray<float> &kseries_im,
                           DeviceArray<float> &kcur_re,
                           DeviceArray<float> &kcur_im,
                           DeviceArray<float> &kwork_re,
                           DeviceArray<float> &kwork_im,
                           int *outer_iters = nullptr,
                           cudaStream_t stream = nullptr,
                           int accuracy = 8,
                           AntialiasFilter *antialias_filter = nullptr,
                           float divergence_ratio = 0.0f,
                           float *d_sum_work = nullptr,
                           float *d_sum_exit = nullptr);

} // namespace cvdms
