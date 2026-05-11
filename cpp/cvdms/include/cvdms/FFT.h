#pragma once

#include <cufft.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cvdms {

/// RAII wrapper for a cuFFT plan (2D).
class FFTPlan2D {
  public:
    FFTPlan2D() : plan_(0), nx_(0), ny_(0) {}

    FFTPlan2D(std::size_t nx, std::size_t ny, cufftType type, int batch = 1)
        : plan_(0), nx_(nx), ny_(ny) {
        auto err = cufftPlan2d(&plan_, nx, ny, type);
        if (err != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftPlan2d failed: " +
                                     std::to_string(err));
        }
        if (batch > 1) {
            err = cufftSetAutoAllocation(plan_, 0);
            if (err != CUFFT_SUCCESS) {
                throw std::runtime_error("cufftSetAutoAllocation failed");
            }
        }
    }

    ~FFTPlan2D() {
        if (plan_) {
            cufftDestroy(plan_);
        }
    }

    // Move
    FFTPlan2D(FFTPlan2D &&other) noexcept
        : plan_(other.plan_), nx_(other.nx_), ny_(other.ny_) {
        other.plan_ = 0;
        other.nx_ = 0;
        other.ny_ = 0;
    }

    FFTPlan2D &operator=(FFTPlan2D &&other) noexcept {
        if (this != &other) {
            if (plan_)
                cufftDestroy(plan_);
            plan_ = other.plan_;
            nx_ = other.nx_;
            ny_ = other.ny_;
            other.plan_ = 0;
            other.nx_ = 0;
            other.ny_ = 0;
        }
        return *this;
    }

    // No copy
    FFTPlan2D(const FFTPlan2D &) = delete;
    FFTPlan2D &operator=(const FFTPlan2D &) = delete;

    cufftHandle handle() const { return plan_; }
    std::size_t nx() const { return nx_; }
    std::size_t ny() const { return ny_; }

  private:
    cufftHandle plan_;
    std::size_t nx_, ny_;
};

/// FFT-based Laplacian operator using cuFFT.
///
/// ∇²ψ = IFFT[-4π²·k²·FFT(ψ)]
///
/// This is the exact band-limited Laplacian, matching Python's
/// _laplace_operator_fft() in finite_difference.py.
///
/// Internally uses float2 interleaved buffer for cuFFT.
class FFTLaplacian {
  public:
    FFTLaplacian();

    /// Initialize with grid dimensions and sampling.
    /// Must be called at least once before compute().
    /// Re-initializes if dimensions changed.
    void initialize(std::size_t nx, std::size_t ny,
                    float sampling_x, float sampling_y);

    /// Compute ∇²ψ = IFFT[-4π²k²·FFT(ψ)].
    /// psi_re/psi_im: input wavefunction (nx*ny each, device)
    /// out_re/out_im: output Laplacian (nx*ny each, device, may alias input)
    void compute(const float *psi_re, const float *psi_im,
                 float *out_re, float *out_im,
                 cudaStream_t stream = nullptr);

    ~FFTLaplacian();

    std::size_t nx() const { return nx_; }
    std::size_t ny() const { return ny_; }

  private:
    std::size_t nx_, ny_;
    bool initialized_;
    cufftHandle plan_;
    cufftComplex *d_buffer_;  // workspace (nx*ny cuComplex)
    float *d_factor_;         // -4π²k² factor (nx*ny float)
};

} // namespace cvdms
