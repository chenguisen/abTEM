#include "cvdms/TaylorSeries.h"
#include "cvdms/Convergence.h"
#include "cvdms/KOperator.h"
#include "cvdms/KSeries.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>

namespace cvdms {

/// Fused kernel: work = kseries * i*dz/n, exit += work, convergence check,
/// and sum reduction for divergence ratio check.
///
/// Replaces taylor_scale_accumulate_kernel + convergence_check_kernel
/// (2 kernel launches → 1 per outer iteration).
///
/// Shared memory layout: [blockDim.x] floats for |work| + [blockDim.x] for |exit|.
__global__ void taylor_fused_kernel(const float *kseries_re,
                                     const float *kseries_im,
                                     float *work_re, float *work_im,
                                     float *exit_re, float *exit_im,
                                     int count, float dz, int n,
                                     float threshold,
                                     ConvergenceResult *d_result,
                                     float *d_sum_work,
                                     float *d_sum_exit) {
    extern __shared__ float sdata[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;

    float abs_work = 0.0f;
    float abs_exit = 0.0f;

    if (idx < count) {
        // Taylor term: kseries * i * dz / n
        // Complex multiply: (kr + i*ki) * i * dz/n = (-ki + i*kr) * dz/n
        float kr = kseries_re[idx];
        float ki = kseries_im[idx];
        float scale = dz / static_cast<float>(n);

        float wr = -ki * scale;
        float wi = kr * scale;

        // Compute |exit_before| for divergence ratio BEFORE modifying exit.
        // The ratio sum(|work|)/sum(|exit_before|) detects when a new term
        // dwarfs the accumulated exit, which sum(|work|)/sum(|exit_after|)
        // cannot (exit_after ≈ work when work dominates, forcing ratio ~1).
        float ebr = exit_re[idx];
        float ebi = exit_im[idx];
        abs_exit = sqrtf(fmaxf(ebr * ebr + ebi * ebi, 0.0f));

        // Update exit and work
        work_re[idx] = wr;
        work_im[idx] = wi;
        exit_re[idx] = ebr + wr;
        exit_im[idx] = ebi + wi;

        // NaN/Inf check on current term
        float mag2 = wr * wr + wi * wi;
        bool term_nan = isnan(mag2) || isinf(mag2);

        // NaN/Inf check on exit after accumulation
        float er = exit_re[idx];
        float ei = exit_im[idx];
        bool exit_nan = isnan(er) || isinf(er) || isnan(ei) || isinf(ei);

        if (term_nan || exit_nan) {
            atomicAdd(&d_result->n_nan, 1);
            // Still compute sums with whatever finite values we have
        } else {
            if (mag2 > threshold * threshold) {
                atomicAdd(&d_result->n_above, 1);
            }
        }

        abs_work = sqrtf(fmaxf(mag2, 0.0f));
    }

    // Block-level reduction for sum(|work|) and sum(|exit|)
    sdata[tid] = abs_work;
    sdata[bd + tid] = abs_exit;
    __syncthreads();

    for (int s = bd / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[bd + tid] += sdata[bd + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_sum_work, sdata[0]);
        atomicAdd(d_sum_exit, sdata[bd]);
    }
}

/// Undo last Taylor step: exit -= work.
/// Called when divergence ratio check triggers truncation.
__global__ void undo_taylor_step_kernel(float *exit_re, float *exit_im,
                                        const float *work_re, const float *work_im,
                                        int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;
    exit_re[idx] -= work_re[idx];
    exit_im[idx] -= work_im[idx];
}

void compute_taylor_series(const float *psi_in_re, const float *psi_in_im,
                           float *psi_out_re, float *psi_out_im,
                           const float *V, std::size_t nx, std::size_t ny,
                           float wavelength, float dz,
                           float convergence_threshold,
                           int max_terms, int max_inner,
                           float inv_4piK0, float inv_dx, float inv_dy,
                           ConvergenceResult *d_result, bool &converged,
                           bool &overflow, DeviceArray<float> &work_re,
                           DeviceArray<float> &work_im,
                           DeviceArray<float> &kseries_re,
                           DeviceArray<float> &kseries_im,
                           DeviceArray<float> &kcur_re,
                           DeviceArray<float> &kcur_im,
                           DeviceArray<float> &kwork_re,
                           DeviceArray<float> &kwork_im,
                           int *outer_iters,
                           cudaStream_t stream,
                           int accuracy,
                           AntialiasFilter *antialias_filter,
                           float divergence_ratio,
                           float *d_sum_work,
                           float *d_sum_exit) {

    int count = static_cast<int>(nx * ny);
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    size_t shmem = 2 * block_size * sizeof(float);

    converged = false;
    overflow = false;

    // exit_wave = psi_in (0th-order term)
    cudaMemcpyAsync(psi_out_re, psi_in_re, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(psi_out_im, psi_in_im, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // work = psi_in (working buffer for Taylor cascade)
    cudaMemcpyAsync(work_re.data(), psi_in_re, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(work_im.data(), psi_in_im, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // Outer Taylor loop matching Python's _cvdms_forward_scattering:
    //   for n_exp_order in range(1, max_terms + 1):
    //       k_series = inner_k_series(working, ...)
    //       working = k_series
    //       working *= i * dz / n
    //       exit_wave += working
    int iter = 0;
    for (int n = 1; n <= max_terms; ++n) {
        // Step 1: Compute K_series(work) → kseries
        // work holds the cascaded result from the previous Taylor iteration
        // (or psi for n=1)
        compute_k_series(work_re.data(), work_im.data(),
                         kseries_re.data(), kseries_im.data(),
                         V, nx, ny, wavelength, dz,
                         convergence_threshold, max_inner, inv_4piK0, inv_dx, inv_dy,
                         kcur_re, kcur_im,    // K-operator input
                         kwork_re, kwork_im,  // K-operator output
                         d_result,
                         stream, accuracy,
                         antialias_filter);

        // Step 2: work = kseries * i*dz/n, exit += work, convergence check,
        // and sum reduction for divergence ratio.
        cudaMemsetAsync(d_result, 0, sizeof(ConvergenceResult), stream);
        cudaMemsetAsync(d_sum_work, 0, sizeof(float), stream);
        cudaMemsetAsync(d_sum_exit, 0, sizeof(float), stream);
        taylor_fused_kernel<<<grid_size, block_size, shmem, stream>>>(
            kseries_re.data(), kseries_im.data(),
            work_re.data(), work_im.data(),
            psi_out_re, psi_out_im,
            count, dz, n, convergence_threshold,
            d_result,
            d_sum_work, d_sum_exit);

        auto result = read_convergence(d_result, stream);
        if (result.n_nan > 0) {
            overflow = true;
            ++iter;
            break;
        }
        if (result.n_above == 0) {
            converged = true;
            ++iter;
            break;
        }

        // Divergence ratio check (matching Python lines 510-525)
        if (n > 1 && divergence_ratio > 0.0f) {
            float h_sum_work = 0.0f, h_sum_exit = 0.0f;
            cudaMemcpyAsync(&h_sum_work, d_sum_work, sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&h_sum_exit, d_sum_exit, sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            float ratio = h_sum_work / fmaxf(h_sum_exit, 1e-30f);
            if (ratio > divergence_ratio) {
                // Undo last term: exit -= work
                undo_taylor_step_kernel<<<grid_size, block_size, 0, stream>>>(
                    psi_out_re, psi_out_im,
                    work_re.data(), work_im.data(),
                    count);
                ++iter;
                break;
            }
        }

        ++iter;
    }

    if (outer_iters)
        *outer_iters = iter;
}

// ======================================================================
// FFT-based K-series step kernel
//
// For each pixel:
//   1. kw *= coeff (scale)
//   2. kseries += scaled_kw
//   3. store scaled_kw → cur (cascade)
//   4. convergence check
// ======================================================================
__global__ void fft_kseries_step_kernel(const float *kw_re, const float *kw_im,
                                         float *kseries_re, float *kseries_im,
                                         float *cur_re, float *cur_im,
                                         int count, float coeff,
                                         float threshold,
                                         ConvergenceResult *d_result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    float wr = kw_re[idx] * coeff;
    float wi = kw_im[idx] * coeff;

    // Accumulate to kseries
    kseries_re[idx] += wr;
    kseries_im[idx] += wi;

    // Store scaled result as cascade input for next iteration
    cur_re[idx] = wr;
    cur_im[idx] = wi;

    // Convergence check
    float mag2 = wr * wr + wi * wi;
    bool is_nan = isnan(mag2) || isinf(mag2);
    if (is_nan) {
        atomicAdd(&d_result->n_nan, 1);
        return;
    }
    if (mag2 > threshold * threshold) {
        atomicAdd(&d_result->n_above, 1);
    }
}

// ======================================================================
// FFT-based Taylor series
//
// Uses cuFFT for the Laplacian (∇²ψ = IFFT[-4π²k²·FFT(ψ)]) instead of
// the finite-difference stencil. Each inner K-series iteration does:
//   1. FFT Laplacian of cur → lap
//   2. K-operator: kw = V*cur + lap/(4πK₀)
//   3. Scale + accumulate + convergence (fused kernel)
//
// This is ~5 launches/iter vs 1 for the fused stencil, but cuFFT gives
// the exact band-limited Laplacian and can be faster for large grids.
// ======================================================================
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
                                int *outer_iters,
                                cudaStream_t stream,
                                AntialiasFilter *antialias_filter,
                                float divergence_ratio,
                                float *d_sum_work,
                                float *d_sum_exit) {

    int count = static_cast<int>(nx * ny);
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    size_t shmem = 2 * block_size * sizeof(float);

    converged = false;
    overflow = false;

    // Zero kseries accumulator
    cudaMemsetAsync(kseries_re.data(), 0, count * sizeof(float), stream);
    cudaMemsetAsync(kseries_im.data(), 0, count * sizeof(float), stream);

    // exit_wave = psi_in (0th-order term)
    cudaMemcpyAsync(psi_out_re, psi_in_re, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(psi_out_im, psi_in_im, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // cur = psi_in (K-series cascade input for inner loop)
    cudaMemcpyAsync(kcur_re.data(), psi_in_re, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(kcur_im.data(), psi_in_im, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    int prev_ks_n_above = -1;
    for (int n = 1; n <= max_inner; ++n) {
        float coeff = 1.0f;
        if (n > 1) {
            coeff = (0.5f - static_cast<float>(n) + 1.0f) * wavelength /
                    (static_cast<float>(M_PI) * n);
        }

        // ---- Inner K-series iteration using FFT Laplacian ----

        // Step 1: FFT Laplacian of cur → lap
        fft_laplacian.compute(kcur_re.data(), kcur_im.data(),
                               lap_re.data(), lap_im.data(), stream);

        // Step 2: K-operator: kw = V*cur + lap/(4πK₀)
        launch_k_operator_from_laplacian(
            kcur_re.data(), kcur_im.data(),
            lap_re.data(), lap_im.data(),
            kwork_re.data(), kwork_im.data(),
            V, count, inv_4piK0, stream);

        // Step 3: Scale, accumulate, convergence check
        cudaMemsetAsync(d_result, 0, sizeof(ConvergenceResult), stream);
        fft_kseries_step_kernel<<<grid_size, block_size, 0, stream>>>(
            kwork_re.data(), kwork_im.data(),
            kseries_re.data(), kseries_im.data(),
            kcur_re.data(), kcur_im.data(),
            count, coeff, convergence_threshold, d_result);

        // ---- Internal antialias: prevent bandwidth explosion ----
        if (antialias_filter && antialias_filter->initialized()) {
            antialias_filter->apply(kcur_re.data(), kcur_im.data(),
                                     kwork_re.data(), kwork_im.data(), stream);
            cudaMemcpyAsync(kcur_re.data(), kwork_re.data(),
                            count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(kcur_im.data(), kwork_im.data(),
                            count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }

        auto result = read_convergence(d_result, stream);
        if (result.n_nan > 0) {
            overflow = true;  // K-series overflow
            break;
        }
        if (result.n_above == 0) {
            break;  // K-series converged
        }
        // Stagnation detection
        if (prev_ks_n_above >= 0 && result.n_above >= prev_ks_n_above)
            break;
        prev_ks_n_above = result.n_above;
    }

    int iter = 0;

    // ---- Outer Taylor iteration ----
    // work holds the cascaded Taylor value (starts as psi_in for n=1)
    cudaMemcpyAsync(work_re.data(), psi_in_re, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(work_im.data(), psi_in_im, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    for (int n = 1; n <= max_terms; ++n) {
        // Compute K_series(work) → kseries using FFT-based inner K-series
        // Reuse the same inner loop as above but with work as input
        // Instead of duplicating the inner loop, we recompute it here

        // Zero kseries for this outer iteration
        cudaMemsetAsync(kseries_re.data(), 0, count * sizeof(float), stream);
        cudaMemsetAsync(kseries_im.data(), 0, count * sizeof(float), stream);

        // cur = work (cascade input for inner K-series)
        cudaMemcpyAsync(kcur_re.data(), work_re.data(), count * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(kcur_im.data(), work_im.data(), count * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        bool inner_overflow = false;
        int prev_inner_above = -1;
        for (int inner_n = 1; inner_n <= max_inner; ++inner_n) {
            float inner_coeff = 1.0f;
            if (inner_n > 1) {
                inner_coeff = (0.5f - static_cast<float>(inner_n) + 1.0f) *
                              wavelength / (static_cast<float>(M_PI) * inner_n);
            }

            // FFT Laplacian
            fft_laplacian.compute(kcur_re.data(), kcur_im.data(),
                                   lap_re.data(), lap_im.data(), stream);

            // K-operator
            launch_k_operator_from_laplacian(
                kcur_re.data(), kcur_im.data(),
                lap_re.data(), lap_im.data(),
                kwork_re.data(), kwork_im.data(),
                V, count, inv_4piK0, stream);

            // Scale + accumulate + convergence
            cudaMemsetAsync(d_result, 0, sizeof(ConvergenceResult), stream);
            fft_kseries_step_kernel<<<grid_size, block_size, 0, stream>>>(
                kwork_re.data(), kwork_im.data(),
                kseries_re.data(), kseries_im.data(),
                kcur_re.data(), kcur_im.data(),
                count, inner_coeff, convergence_threshold, d_result);

            // ---- Internal antialias: prevent bandwidth explosion ----
            if (antialias_filter && antialias_filter->initialized()) {
                antialias_filter->apply(kcur_re.data(), kcur_im.data(),
                                         kwork_re.data(), kwork_im.data(), stream);
                cudaMemcpyAsync(kcur_re.data(), kwork_re.data(),
                                count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(kcur_im.data(), kwork_im.data(),
                                count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }

            auto ir = read_convergence(d_result, stream);
            if (ir.n_nan > 0) { inner_overflow = true; break; }
            if (ir.n_above == 0) break;
            // Stagnation detection: stop if unconverged pixel count stops decreasing
            if (prev_inner_above >= 0 && ir.n_above >= prev_inner_above)
                break;
            prev_inner_above = ir.n_above;
        }

        // Taylor step: work = kseries * i*dz/n, exit += work
        cudaMemsetAsync(d_result, 0, sizeof(ConvergenceResult), stream);
        if (d_sum_work) cudaMemsetAsync(d_sum_work, 0, sizeof(float), stream);
        if (d_sum_exit) cudaMemsetAsync(d_sum_exit, 0, sizeof(float), stream);
        taylor_fused_kernel<<<grid_size, block_size, shmem, stream>>>(
            kseries_re.data(), kseries_im.data(),
            work_re.data(), work_im.data(),
            psi_out_re, psi_out_im,
            count, dz, n, convergence_threshold,
            d_result,
            d_sum_work ? d_sum_work : psi_out_re,  // fallback (will be zero)
            d_sum_exit ? d_sum_exit : psi_out_im);

        auto result = read_convergence(d_result, stream);
        if (result.n_nan > 0 || inner_overflow) {
            overflow = true;
            ++iter;
            break;
        }
        if (result.n_above == 0) {
            converged = true;
            ++iter;
            break;
        }

        // Divergence ratio check (matching Python lines 510-525)
        if (n > 1 && divergence_ratio > 0.0f && d_sum_work && d_sum_exit) {
            float h_sum_work = 0.0f, h_sum_exit = 0.0f;
            cudaMemcpyAsync(&h_sum_work, d_sum_work, sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&h_sum_exit, d_sum_exit, sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            float ratio = h_sum_work / fmaxf(h_sum_exit, 1e-30f);
            if (ratio > divergence_ratio) {
                undo_taylor_step_kernel<<<grid_size, block_size, 0, stream>>>(
                    psi_out_re, psi_out_im,
                    work_re.data(), work_im.data(),
                    count);
                ++iter;
                break;
            }
        }

        ++iter;
    }

    if (outer_iters)
        *outer_iters = iter;
}

} // namespace cvdms
