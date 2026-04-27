#include "cvdms/Laplacian.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cvdms {

/// Separable finite-difference Laplacian matching Python's
/// _laplace_operator_stencil(accuracy=acc, prefactor=inv_dx*inv_dy, ...).
///
/// The accuracy parameter determines the stencil radius:
///   accuracy  |  stencil size  |  radius
///   ----------|----------------|---------
///   2         |  3             |  1       (standard 5-pt Laplacian)
///   4         |  5             |  2
///   6         |  7             |  3
///   8         |  9             |  4
///
/// Python's default derivative_accuracy=8 gives accuracy=8, corresponding
/// to a separable 9-point 1D stencil (radius 4):
///
///   ∇²ψ[i,j] = pref * Σ c[k] * (ψ[i+k,j] + ψ[i,j+k])   k=-4..4
///
/// where c = finite_difference_coefficients(2, 8), then rolled so that
/// c[0] corresponds to the center coefficient.
///
/// This is a SEPARABLE (dimension-by-dimension) stencil — no diagonal
/// coupling — unlike the 9-point compact stencil which uses a 3×3
/// neighborhood.  Both are "9-point" in name but structurally different.
///
/// Boundary condition: periodic wrap (matching Python mode="wrap").
template <int ACC>
__global__ void laplacian_kernel_separable(const float *re_in, const float *im_in,
                                            float *re_out, float *im_out,
                                            int nx, int ny, float pref) {
    constexpr int R = ACC / 2;  // stencil radius
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= ny || j >= nx)
        return;

    int idx = i * nx + j;

    // accumulate over x and y directions with periodic wrapping
    float sum_re = 0.0f;
    float sum_im = 0.0f;

    // k=0 (center): coefficient c0 appears twice (once for x, once for y direction)
    //                so effective center weight = 2 * c[0]
    #pragma unroll
    for (int k = -R; k <= R; ++k) {
        // These coefficients match finite_difference_coefficients(2, ACC)
        // after np.roll(c, -(len(c)//2)).
        //   c[-R], ..., c[-1], c[0], c[1], ..., c[R]
        float coeff;
        if constexpr (ACC == 2) {
            // c = [1, -2, 1] → roll(-1) → [-2, 1, 1]
            coeff = (k == 0) ? -2.0f : 1.0f;
        } else if constexpr (ACC == 4) {
            // c = [-1/12, 4/3, -5/2, 4/3, -1/12] → roll(-2)
            //   → [-5/2, 4/3, -1/12, -1/12, 4/3]
            if (k == 0) coeff = -2.5f;
            else if (k == 1 || k == -1) coeff = 4.0f / 3.0f;
            else coeff = -1.0f / 12.0f;
        } else if constexpr (ACC == 6) {
            // c = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] → roll(-3)
            if (k == 0) coeff = -49.0f / 18.0f;
            else if (k == 1 || k == -1) coeff = 1.5f;
            else if (k == 2 || k == -2) coeff = -0.15f;
            else coeff = 1.0f / 90.0f;
        } else if constexpr (ACC == 8) {
            // c = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
            //   → roll(-4)
            if (k == 0) coeff = -205.0f / 72.0f;
            else if (k == 1 || k == -1) coeff = 8.0f / 5.0f;
            else if (k == 2 || k == -2) coeff = -0.2f;
            else if (k == 3 || k == -3) coeff = 8.0f / 315.0f;
            else coeff = -1.0f / 560.0f;
        }

        if (k == 0) {
            // center: counted twice (x + y directions)
            sum_re += 2.0f * coeff * re_in[idx];
            sum_im += 2.0f * coeff * im_in[idx];
        } else {
            // Periodic wrap for offset k
            int jk = j + k;
            int ik = i + k;
            if (jk < 0) jk += nx;
            else if (jk >= nx) jk -= nx;
            if (ik < 0) ik += ny;
            else if (ik >= ny) ik -= ny;

            // x-direction: ψ[i, j+k]
            sum_re += coeff * re_in[i * nx + jk];
            sum_im += coeff * im_in[i * nx + jk];

            // y-direction: ψ[i+k, j]
            sum_re += coeff * re_in[ik * nx + j];
            sum_im += coeff * im_in[ik * nx + j];
        }
    }

    re_out[idx] = pref * sum_re;
    im_out[idx] = pref * sum_im;
}

void launch_laplacian(const float *re_in, const float *im_in,
                      float *re_out, float *im_out,
                      std::size_t nx, std::size_t ny,
                      float inv_dx, float inv_dy, cudaStream_t stream,
                      int accuracy) {
    dim3 block(16, 16);
    dim3 grid((nx + 15) / 16, (ny + 15) / 16);

    float pref = inv_dx * inv_dy;  // = 1/(dx*dy)

    switch (accuracy) {
        case 2:
            laplacian_kernel_separable<2><<<grid, block, 0, stream>>>(
                re_in, im_in, re_out, im_out, static_cast<int>(nx),
                static_cast<int>(ny), pref);
            break;
        case 4:
            laplacian_kernel_separable<4><<<grid, block, 0, stream>>>(
                re_in, im_in, re_out, im_out, static_cast<int>(nx),
                static_cast<int>(ny), pref);
            break;
        case 6:
            laplacian_kernel_separable<6><<<grid, block, 0, stream>>>(
                re_in, im_in, re_out, im_out, static_cast<int>(nx),
                static_cast<int>(ny), pref);
            break;
        case 8:
        default:
            laplacian_kernel_separable<8><<<grid, block, 0, stream>>>(
                re_in, im_in, re_out, im_out, static_cast<int>(nx),
                static_cast<int>(ny), pref);
            break;
    }
}

} // namespace cvdms
