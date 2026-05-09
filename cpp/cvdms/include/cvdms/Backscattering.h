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

/// Back-propagate BSC waves through per-original-slice stepping.
///
/// For each exit plane block (bottom to top), back-propagates through each
/// ORIGINAL potential slice individually (not an aggregated effective slice)
/// to keep dz small, preventing float32 overflow in compute_taylor_series.
///
/// For each block ep (from num_exit_planes-2 down to 0):
///   1. Copy bsc_waves[ep+1] → work
///   2. For each original slice sl in the block (reverse order):
///        a. conj(work) → work                      (time reversal, if use_conj)
///        b. compute_taylor_series(work, V[sl])      (forward propagate)
///        c. copy Taylor output → work
///        d. conj(work) → work                      (time reversal undo, if use_conj)
///   3. bsc_waves[ep] += work                       (accumulate)
///
/// bsc_waves_re/bsc_waves_im: device pointer arrays, size num_exit_planes.
///   bsc_waves[i] is BSC at exit plane i (0 = entrance/top surface).
/// V_slices: device pointer arrays, size num_total_slices.
///   All original (non-aggregated) transmission functions.
/// exit_plane_indices: host int array, size num_exit_planes.
///   Block ep spans V_slices[exit_plane_indices[ep] : exit_plane_indices[ep+1]].
///   Blocks may be empty when consecutive exit planes share the same index
///   (e.g. entrance plane with exit_plane_indices[0] = -1 + 1 = 0).
/// dz: slice thickness (Å), uniform across all slices.
///
/// exit_re/exit_im: separate buffer for compute_taylor_series output
///   (must NOT alias work_re/work_im).
/// All other DeviceArray working buffers must be pre-allocated to nx*ny.
/// Running accumulation back-propagation of BSC through ALL slices.
///
/// Uses the running accumulation algorithm to back-propagate EVERY slice's
/// BSC through all overlying slices to the entrance surface:
///   work = 0
///   for sl = num_slices-1 down to 0:
///       work += bsc_slices[sl]
///       if use_conj: work = conj(forward(conj(work), V_slices[sl]))
///       else:        work = forward(work, V_slices[sl])
///   ep_bsc[0] = work  (total BSC at entrance surface)
///
/// bsc_slices_re/bsc_slices_im: device pointer arrays, size num_slices.
///   bsc_slices[sl] is the raw BSC computed at the exit of slice sl.
///   On output, bsc_slices[0] is modified in-place to the total accumulated
///   BSC at the entrance surface. Other entries are unchanged.
/// V_slices: device pointer arrays, size num_slices.
/// dz: slice thickness (A), uniform across all slices.
///
/// All DeviceArray working buffers must be pre-allocated to nx*ny.
void running_accumulate_bsc(
    float *const *bsc_slices_re, float *const *bsc_slices_im,
    int num_slices,
    const float *const *V_slices,
    std::size_t nx, std::size_t ny,
    float wavelength, float dz,
    float convergence_threshold, int max_terms, int max_inner,
    float inv_4piK0, float inv_dx, float inv_dy,
    // Exit plane output: arrays filled with accumulated BSC at each EP
    float *const *ep_bsc_re, float *const *ep_bsc_im,
    int num_exit_planes, const int *exit_plane_indices,
    DeviceArray<float> &work_re, DeviceArray<float> &work_im,
    DeviceArray<float> &exit_re, DeviceArray<float> &exit_im,
    DeviceArray<float> &kseries_re, DeviceArray<float> &kseries_im,
    DeviceArray<float> &kcur_re, DeviceArray<float> &kcur_im,
    DeviceArray<float> &kwork_re, DeviceArray<float> &kwork_im,
    ConvergenceResult *d_result,
    cudaStream_t stream,
    int accuracy = 8,
    bool use_conj = true);

void back_propagate_bsc_series(
    float *const *bsc_waves_re, float *const *bsc_waves_im,
    int num_exit_planes,
    const float *const *V_slices,
    int num_total_slices,
    const int *exit_plane_indices,
    std::size_t nx, std::size_t ny,
    float wavelength, float dz,
    float convergence_threshold, int max_terms, int max_inner,
    float inv_4piK0, float inv_dx, float inv_dy,
    DeviceArray<float> &work_re, DeviceArray<float> &work_im,
    DeviceArray<float> &exit_re, DeviceArray<float> &exit_im,
    DeviceArray<float> &kseries_re, DeviceArray<float> &kseries_im,
    DeviceArray<float> &kcur_re, DeviceArray<float> &kcur_im,
    DeviceArray<float> &kwork_re, DeviceArray<float> &kwork_im,
    ConvergenceResult *d_result,
    cudaStream_t stream,
    int accuracy = 8,
    bool use_conj = true);

} // namespace cvdms
