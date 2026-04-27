#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cvdms/Array.h"
#include "cvdms/Backscattering.h"
#include "cvdms/Convergence.h"
#include "cvdms/KSeries.h"
#include "cvdms/TaylorSeries.h"

namespace py = pybind11;
namespace cvdms {

// ──────────────────────────────────────────────
// Helper: extract device pointer via __cuda_array_interface__
// ──────────────────────────────────────────────
static float *get_device_ptr(py::handle obj) {
    auto cai = obj.attr("__cuda_array_interface__").cast<py::dict>();
    auto data = cai["data"].cast<py::tuple>();
    uintptr_t ptr = data[0].cast<uintptr_t>();
    return reinterpret_cast<float *>(ptr);
}

// ──────────────────────────────────────────────
// Python-facing wrapper for full Taylor series
// ──────────────────────────────────────────────
class PyTaylorEngine {
  public:
    PyTaylorEngine()
        : d_count_above_(nullptr), d_count_nan_(nullptr),
          d_count_diverging_(nullptr), initialized_(false) {}

    void initialize(std::size_t nx, std::size_t ny) {
        if (initialized_ && nx == nx_ && ny == ny_)
            return;

        nx_ = nx;
        ny_ = ny;
        std::size_t count = nx * ny;

        // Free old buffers
        work_re_ = DeviceArray<float>();
        work_im_ = DeviceArray<float>();
        kseries_re_ = DeviceArray<float>();
        kseries_im_ = DeviceArray<float>();
        kcur_re_ = DeviceArray<float>();
        kcur_im_ = DeviceArray<float>();
        kwork_re_ = DeviceArray<float>();
        kwork_im_ = DeviceArray<float>();

        // Allocate new buffers
        work_re_ = DeviceArray<float>(count);
        work_im_ = DeviceArray<float>(count);
        kseries_re_ = DeviceArray<float>(count);
        kseries_im_ = DeviceArray<float>(count);
        kcur_re_ = DeviceArray<float>(count);
        kcur_im_ = DeviceArray<float>(count);
        kwork_re_ = DeviceArray<float>(count);
        kwork_im_ = DeviceArray<float>(count);

        // Convergence counters
        if (!d_count_above_)
            cudaMalloc(&d_count_above_, sizeof(int));
        if (!d_count_nan_)
            cudaMalloc(&d_count_nan_, sizeof(int));
        if (!d_count_diverging_)
            cudaMalloc(&d_count_diverging_, sizeof(int));

        initialized_ = true;
    }

    ~PyTaylorEngine() {
        if (d_count_above_)
            cudaFree(d_count_above_);
        if (d_count_nan_)
            cudaFree(d_count_nan_);
        if (d_count_diverging_)
            cudaFree(d_count_diverging_);
    }

    /// Compute full Taylor-series forward scattering.
    ///
    /// All arrays must be CuPy float32 device arrays on the same GPU.
    /// psi_re, psi_im: contiguous float32 arrays — MODIFIED IN-PLACE.
    ///   Shape must be (..., nx, ny) with total elements divisible by nx*ny.
    ///   When batched (total > nx*ny), each batch item is processed sequentially.
    /// V: contiguous float32 array (nx*ny, potential) — shared across batch.
    /// laplace_prefactor: 1.0 / (dx * dy) for Laplacian stencil
    ///
    /// Returns: (converged: bool, overflow: bool)
    py::tuple compute(py::object psi_re, py::object psi_im, py::object V,
                      std::size_t nx, std::size_t ny, float wavelength,
                      float dz, float convergence_threshold, int max_terms,
                      float laplace_prefactor,
                      int accuracy = 8) {

        // Extract device pointers
        float *re_ptr = get_device_ptr(psi_re);
        float *im_ptr = get_device_ptr(psi_im);
        float *V_ptr = get_device_ptr(V);

        // Detect batch size from array shape
        auto cai = psi_re.attr("__cuda_array_interface__").cast<py::dict>();
        auto shape = cai["shape"].cast<py::tuple>();
        std::size_t total_elems = 1;
        for (auto &dim : shape)
            total_elems *= dim.cast<std::size_t>();
        std::size_t batch = total_elems / (nx * ny);
        if (batch == 0) batch = 1;  // safety

        // Initialize/resize buffers (single nx*ny batch size)
        initialize(nx, ny);

        // Physical constants
        float K0 = 1.0f / wavelength;
        float inv_4piK0 = 1.0f / (4.0f * static_cast<float>(M_PI) * K0);

        // Decompose prefactor into inv_dx * inv_dy for the Laplacian stencil
        float inv_dx = std::sqrt(laplace_prefactor);
        float inv_dy = inv_dx;

        bool all_converged = true;
        bool any_overflow = false;

        for (std::size_t b = 0; b < batch; ++b) {
            float *batch_re = re_ptr + b * nx * ny;
            float *batch_im = im_ptr + b * nx * ny;

            bool item_converged = false;
            bool item_overflow = false;

            compute_taylor_series(
                batch_re, batch_im,        // input wave
                batch_re, batch_im,        // output wave (in-place)
                V_ptr,                     // potential (shared)
                nx, ny,
                wavelength, dz,
                convergence_threshold, max_terms,
                inv_4piK0, inv_dx, inv_dy,
                d_count_above_, d_count_nan_, d_count_diverging_,
                item_converged, item_overflow,
                work_re_, work_im_,
                kseries_re_, kseries_im_,
                kcur_re_, kcur_im_,
                kwork_re_, kwork_im_,
                nullptr,                  // default CUDA stream
                accuracy);

            all_converged &= item_converged;
            any_overflow |= item_overflow;
        }

        return py::make_tuple(all_converged, any_overflow);
    }

  private:
    int *d_count_above_;
    int *d_count_nan_;
    int *d_count_diverging_;
    bool initialized_;
    std::size_t nx_, ny_;

    // Taylor working buffers (outer loop)
    DeviceArray<float> work_re_;
    DeviceArray<float> work_im_;

    // K-series result (inner loop accumulator)
    DeviceArray<float> kseries_re_;
    DeviceArray<float> kseries_im_;

    // K-operator input (K-series cascade buffer)
    DeviceArray<float> kcur_re_;
    DeviceArray<float> kcur_im_;

    // K-operator output (Laplacian scratch + K-result)
    DeviceArray<float> kwork_re_;
    DeviceArray<float> kwork_im_;
};

// ──────────────────────────────────────────────
// Python-facing wrapper for backscattering correction
// ──────────────────────────────────────────────
class PyBSCEngine {
  public:
    PyBSCEngine()
        : d_count_above_(nullptr), d_count_nan_(nullptr),
          d_count_diverging_(nullptr), initialized_(false) {
        cudaStreamCreate(&stream1_);
        cudaStreamCreate(&stream2_);
    }

    ~PyBSCEngine() {
        cudaStreamDestroy(stream1_);
        cudaStreamDestroy(stream2_);
        if (d_count_above_)
            cudaFree(d_count_above_);
        if (d_count_nan_)
            cudaFree(d_count_nan_);
        if (d_count_diverging_)
            cudaFree(d_count_diverging_);
    }

    void initialize(std::size_t nx, std::size_t ny) {
        if (initialized_ && nx == nx_ && ny == ny_)
            return;
        nx_ = nx;
        ny_ = ny;
        std::size_t count = nx * ny;

        // Free old buffers
        auto reset = [](DeviceArray<float> &arr) {
            arr = DeviceArray<float>();
        };
        reset(s1_cur_re_); reset(s1_cur_im_);
        reset(s1_buf_re_); reset(s1_buf_im_);
        reset(s1_kseries_re_); reset(s1_kseries_im_);
        reset(s2_cur_re_); reset(s2_cur_im_);
        reset(s2_buf_re_); reset(s2_buf_im_);
        reset(s2_kseries_re_); reset(s2_kseries_im_);

        // Allocate new buffers
        auto alloc = [count](DeviceArray<float> &arr) {
            arr = DeviceArray<float>(count);
        };
        alloc(s1_cur_re_); alloc(s1_cur_im_);
        alloc(s1_buf_re_); alloc(s1_buf_im_);
        alloc(s1_kseries_re_); alloc(s1_kseries_im_);
        alloc(s2_cur_re_); alloc(s2_cur_im_);
        alloc(s2_buf_re_); alloc(s2_buf_im_);
        alloc(s2_kseries_re_); alloc(s2_kseries_im_);

        // Convergence counters
        if (!d_count_above_)
            cudaMalloc(&d_count_above_, sizeof(int));
        if (!d_count_nan_)
            cudaMalloc(&d_count_nan_, sizeof(int));
        if (!d_count_diverging_)
            cudaMalloc(&d_count_diverging_, sizeof(int));

        initialized_ = true;
    }

    /// Compute backscattering correction.
    ///
    /// psi_re, psi_im: input wave (float32 device arrays).
    /// V_current, V_next: potential arrays for current/next slice.
    /// bs_re, bs_im: output backscatter arrays (pre-allocated, same shape).
    ///
    /// Returns: (success: bool)
    py::tuple compute(py::object psi_re, py::object psi_im,
                      py::object V_current, py::object V_next,
                      py::object bs_re, py::object bs_im,
                      std::size_t nx, std::size_t ny, float wavelength,
                      float dz, int order, float convergence_threshold,
                      int max_terms, float laplace_prefactor,
                      int accuracy = 8) {

        float *re_ptr = get_device_ptr(psi_re);
        float *im_ptr = get_device_ptr(psi_im);
        float *V_cur_ptr = get_device_ptr(V_current);
        float *V_next_ptr = get_device_ptr(V_next);
        float *bs_re_ptr = get_device_ptr(bs_re);
        float *bs_im_ptr = get_device_ptr(bs_im);

        // Detect batch size
        auto cai = psi_re.attr("__cuda_array_interface__").cast<py::dict>();
        auto shape = cai["shape"].cast<py::tuple>();
        std::size_t total_elems = 1;
        for (auto &d : shape)
            total_elems *= d.cast<std::size_t>();
        std::size_t batch = total_elems / (nx * ny);
        if (batch == 0)
            batch = 1;

        initialize(nx, ny);

        float K0 = 1.0f / wavelength;
        float inv_4piK0 = 1.0f / (4.0f * static_cast<float>(M_PI) * K0);
        float inv_dx = std::sqrt(laplace_prefactor);
        float inv_dy = inv_dx;

        for (std::size_t b = 0; b < batch; ++b) {
            float *batch_re = re_ptr + b * nx * ny;
            float *batch_im = im_ptr + b * nx * ny;
            float *batch_bs_re = bs_re_ptr + b * nx * ny;
            float *batch_bs_im = bs_im_ptr + b * nx * ny;

            apply_backscattering(
                batch_re, batch_im, batch_bs_re, batch_bs_im, V_cur_ptr,
                V_next_ptr, nx, ny, wavelength, dz, order,
                convergence_threshold, max_terms, inv_4piK0, inv_dx, inv_dy,
                s1_cur_re_, s1_cur_im_, s1_buf_re_, s1_buf_im_,
                s1_kseries_re_, s1_kseries_im_, s2_cur_re_, s2_cur_im_,
                s2_buf_re_, s2_buf_im_, s2_kseries_re_, s2_kseries_im_,
                s1_cur_re_, s1_cur_im_,   // fs_temp aliases s1_cur
                s1_buf_re_, s1_buf_im_,   // fs_buf aliases s1_buf
                d_count_above_, d_count_nan_, d_count_diverging_, stream1_,
                stream2_, accuracy);
        }

        return py::make_tuple(true);
    }

  private:
    DeviceArray<float> s1_cur_re_, s1_cur_im_;
    DeviceArray<float> s1_buf_re_, s1_buf_im_;
    DeviceArray<float> s1_kseries_re_, s1_kseries_im_;
    DeviceArray<float> s2_cur_re_, s2_cur_im_;
    DeviceArray<float> s2_buf_re_, s2_buf_im_;
    DeviceArray<float> s2_kseries_re_, s2_kseries_im_;
    int *d_count_above_, *d_count_nan_, *d_count_diverging_;
    cudaStream_t stream1_, stream2_;
    std::size_t nx_, ny_;
    bool initialized_;
};

} // namespace cvdms

// ──────────────────────────────────────────────
// Standalone K-series function (for Python integration)
// ──────────────────────────────────────────────
static py::tuple py_compute_k_series(py::object psi_re, py::object psi_im,
                                     py::object V, std::size_t nx,
                                     std::size_t ny, float wavelength,
                                     float dz, float convergence_threshold,
                                     int max_terms, float laplace_prefactor) {

    float *re_ptr = cvdms::get_device_ptr(psi_re);
    float *im_ptr = cvdms::get_device_ptr(psi_im);
    float *V_ptr = cvdms::get_device_ptr(V);

    std::size_t count = nx * ny;
    float K0 = 1.0f / wavelength;
    float inv_4piK0 = 1.0f / (4.0f * static_cast<float>(M_PI) * K0);
    float inv_dx = std::sqrt(laplace_prefactor);
    float inv_dy = inv_dx;

    // Allocate buffers
    cvdms::DeviceArray<float> cur_re(count);
    cvdms::DeviceArray<float> cur_im(count);
    cvdms::DeviceArray<float> buf_re(count);
    cvdms::DeviceArray<float> buf_im(count);
    cvdms::DeviceArray<float> kseries_re(count);
    cvdms::DeviceArray<float> kseries_im(count);

    int *d_above, *d_nan, *d_div;
    cudaMalloc(&d_above, sizeof(int));
    cudaMalloc(&d_nan, sizeof(int));
    cudaMalloc(&d_div, sizeof(int));

    cvdms::compute_k_series(re_ptr, im_ptr,
                            kseries_re.data(), kseries_im.data(),
                            V_ptr, nx, ny, wavelength, dz,
                            convergence_threshold, max_terms,
                            inv_4piK0, inv_dx, inv_dy,
                            cur_re, cur_im, buf_re, buf_im,
                            d_above, d_nan, d_div, nullptr, 8);

    // Allocate CuPy-like output on device
    // We can't create CuPy arrays from C++, so we write back to input arrays
    // Or: we can create pybind11 arrays with device pointers
    // For now, just return success status
    cudaFree(d_above);
    cudaFree(d_nan);
    cudaFree(d_div);

    return py::make_tuple(true);
}

PYBIND11_MODULE(_cvdms_backend, m) {
    m.doc() = "CVDMS C++/CUDA backend for abTEM";

    py::class_<cvdms::PyTaylorEngine>(m, "TaylorEngine")
        .def(py::init<>())
        .def("compute", &cvdms::PyTaylorEngine::compute,
             py::arg("psi_re"), py::arg("psi_im"), py::arg("V"),
             py::arg("nx"), py::arg("ny"),
             py::arg("wavelength"), py::arg("dz"),
             py::arg("convergence_threshold"), py::arg("max_terms"),
             py::arg("laplace_prefactor"),
             py::arg("accuracy") = 8);

    py::class_<cvdms::PyBSCEngine>(m, "BSCEngine")
        .def(py::init<>())
        .def("compute", &cvdms::PyBSCEngine::compute,
             py::arg("psi_re"), py::arg("psi_im"),
             py::arg("V_current"), py::arg("V_next"),
             py::arg("bs_re"), py::arg("bs_im"),
             py::arg("nx"), py::arg("ny"),
             py::arg("wavelength"), py::arg("dz"),
             py::arg("order"),
             py::arg("convergence_threshold"), py::arg("max_terms"),
             py::arg("laplace_prefactor"),
             py::arg("accuracy") = 8);
}
