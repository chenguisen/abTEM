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
    FFTPlan2D() : plan_(nullptr), nx_(0), ny_(0) {}

    FFTPlan2D(std::size_t nx, std::size_t ny, cufftType type, int batch = 1)
        : plan_(nullptr), nx_(nx), ny_(ny) {
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
        other.plan_ = nullptr;
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
            other.plan_ = nullptr;
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

} // namespace cvdms
