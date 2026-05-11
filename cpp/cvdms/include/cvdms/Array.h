#pragma once

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace cvdms {

namespace py = pybind11;

/// RAII wrapper for a typed device-side array.
template <typename T> class DeviceArray {
  public:
    DeviceArray() : ptr_(nullptr), size_(0) {}

    explicit DeviceArray(std::size_t n) : ptr_(nullptr), size_(n) {
        if (n > 0) {
            auto err = cudaMalloc(&ptr_, n * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed: " +
                                         std::string(cudaGetErrorString(err)));
            }
        }
    }

    ~DeviceArray() { free(); }

    // Move
    DeviceArray(DeviceArray &&other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceArray &operator=(DeviceArray &&other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // No copy
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    void resize(std::size_t n) {
        if (n != size_) {
            free();
            size_ = n;
            if (n > 0) {
                auto err = cudaMalloc(&ptr_, n * sizeof(T));
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaMalloc failed: " +
                                             std::string(cudaGetErrorString(err)));
                }
            }
        }
    }

    void zero() {
        if (ptr_ && size_ > 0) {
            auto err = cudaMemset(ptr_, 0, size_ * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemset failed");
            }
        }
    }

    void copy_from_host(const T *host, std::size_t n) {
        if (n > size_)
            n = size_;
        if (n > 0) {
            auto err = cudaMemcpy(ptr_, host, n * sizeof(T),
                                  cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy H2D failed");
            }
        }
    }

    void copy_to_host(T *host, std::size_t n) const {
        if (n > size_)
            n = size_;
        if (n > 0) {
            auto err = cudaMemcpy(host, ptr_, n * sizeof(T),
                                  cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy D2H failed");
            }
        }
    }

  private:
    T *ptr_;
    std::size_t size_;

    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }
};

/// View a pybind11 array_t as device pointer (does NOT own).
/// Only valid for arrays that are already on the device (CuPy etc.).
template <typename T> class DeviceView {
  public:
    DeviceView(py::array_t<T> arr) : arr_(arr) {
        ptr_ = arr_.mutable_data();
    }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }
    py::ssize_t size() const { return arr_.size(); }

    auto shape() const { return arr_.shape(); }
    auto strides() const { return arr_.strides(); }
    int ndim() const { return arr_.ndim(); }

  private:
    py::array_t<T> arr_;
    T *ptr_;
};

// Common complex types
using complex64 = std::complex<float>;

} // namespace cvdms
