#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cvdms {

/// Convergence detection results from a GPU kernel.
struct ConvergenceResult {
    int n_above;     ///< elements above convergence threshold
    int n_nan;       ///< NaN elements
    int n_diverging; ///< elements diverging (growing compared to previous)
};

/// Launch the convergence-check kernel.
///
/// Compares |working| against convergence_threshold and overflow limit.
/// Writes atomic counters to d_result (single struct = 1 D2H copy).
void launch_convergence_check(const float *working_re, const float *working_im,
                              const float *prev_re, const float *prev_im,
                              std::size_t count, float threshold,
                              ConvergenceResult *d_result,
                              cudaStream_t stream = nullptr);

/// Copy convergence counters from device to host (single struct copy).
ConvergenceResult read_convergence(ConvergenceResult *d_result,
                                   cudaStream_t stream = nullptr);

/// Reset device counters to zero (single struct memset).
void reset_counters(ConvergenceResult *d_result,
                    cudaStream_t stream = nullptr);

} // namespace cvdms
