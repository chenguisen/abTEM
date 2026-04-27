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
/// Writes atomic counters to device memory.
void launch_convergence_check(const float *working_re, const float *working_im,
                              const float *prev_re, const float *prev_im,
                              std::size_t count, float threshold,
                              int *d_count_above, int *d_count_nan,
                              int *d_count_diverging,
                              cudaStream_t stream = nullptr);

/// Copy convergence counters from device to host.
ConvergenceResult read_convergence(int *d_count_above, int *d_count_nan,
                                   int *d_count_diverging,
                                   cudaStream_t stream = nullptr);

/// Reset device counters to zero.
void reset_counters(int *d_count_above, int *d_count_nan,
                    int *d_count_diverging, cudaStream_t stream = nullptr);

} // namespace cvdms
