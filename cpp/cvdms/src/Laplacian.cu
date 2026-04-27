#include "cvdms/Laplacian.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cvdms {

/// Compact 9-point 2D Laplacian with periodic boundary conditions.
///
/// Stencil (multiplied by 1/(6·dx·dy)):
///       1    4    1
///       4  -20    4
///       1    4    1
///
/// ∇²ψ[i,j] ≈ 1/(6·h²) · (4·sum_cardinal + sum_diagonal - 20·center)
///
/// where:
///   sum_cardinal = ψ[i-1,j] + ψ[i+1,j] + ψ[i,j-1] + ψ[i,j+1]
///   sum_diagonal = ψ[i-1,j-1] + ψ[i-1,j+1] + ψ[i+1,j-1] + ψ[i+1,j+1]
///
/// This is O(h⁴) accurate and more isotropic than the 5-point stencil.
/// Derived from the Taylor expansion: the coefficients are chosen to cancel
/// the O(h²) error terms in both coordinate and mixed partial directions.
__global__ void laplacian_kernel_9pt(const float *re_in, const float *im_in,
                                     float *re_out, float *im_out,
                                     int nx, int ny, float inv_area) {
    int j = blockDim.x * blockIdx.x + threadIdx.x; // col (x)
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row (y)
    if (i >= ny || j >= nx)
        return;

    int idx = i * nx + j;

    // Periodic wrap indices for cardinal neighbors (radius 1)
    int j_m1 = (j >= 1)    ? j - 1   : j - 1 + nx;
    int j_p1 = (j < nx - 1) ? j + 1   : j + 1 - nx;
    int i_m1 = (i >= 1)    ? i - 1   : i - 1 + ny;
    int i_p1 = (i < ny - 1) ? i + 1   : i + 1 - ny;

    // Periodic wrap for diagonal neighbors (corner of the 3x3 stencil)
    // Computed directly from the periodic indices above
    int i_m1_nx = i_m1 * nx;
    int i_p1_nx = i_p1 * nx;

    // Cardinal neighbors: NSEW
    float card_re = re_in[i_m1_nx + j] + re_in[i_p1_nx + j] +
                    re_in[idx + (j_m1 - j)] + re_in[idx + (j_p1 - j)];
    float card_im = im_in[i_m1_nx + j] + im_in[i_p1_nx + j] +
                    im_in[idx + (j_m1 - j)] + im_in[idx + (j_p1 - j)];

    // Diagonal neighbors: NW, NE, SW, SE
    float diag_re = re_in[i_m1_nx + j_m1] + re_in[i_m1_nx + j_p1] +
                    re_in[i_p1_nx + j_m1] + re_in[i_p1_nx + j_p1];
    float diag_im = im_in[i_m1_nx + j_m1] + im_in[i_m1_nx + j_p1] +
                    im_in[i_p1_nx + j_m1] + im_in[i_p1_nx + j_p1];

    // 9-point compact Laplacian:
    //   (4 * card + diag - 20 * center) / (6 * h²)
    float scale = inv_area / 6.0f;

    float lap_re = scale * (4.0f * card_re + diag_re - 20.0f * re_in[idx]);
    float lap_im = scale * (4.0f * card_im + diag_im - 20.0f * im_in[idx]);

    re_out[idx] = lap_re;
    im_out[idx] = lap_im;
}

void launch_laplacian(const float *re_in, const float *im_in,
                      float *re_out, float *im_out,
                      std::size_t nx, std::size_t ny,
                      float inv_dx, float inv_dy, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((nx + 15) / 16, (ny + 15) / 16);

    float inv_area = inv_dx * inv_dy;

    laplacian_kernel_9pt<<<grid, block, 0, stream>>>(
        re_in, im_in, re_out, im_out, static_cast<int>(nx),
        static_cast<int>(ny), inv_area);
}

} // namespace cvdms
