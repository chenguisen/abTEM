#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cvdms/Array.h"
#include "cvdms/Backscattering.h"
#include "cvdms/Convergence.h"
#include "cvdms/FFT.h"
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
        : initialized_(false), nx_(0), ny_(0) {}

    ~PyTaylorEngine() {
        for (auto &ctx : contexts_) {
            if (ctx.stream)
                cudaStreamDestroy(ctx.stream);
            if (ctx.d_result)
                cudaFree(ctx.d_result);
        }
    }

    /// Compute full Taylor-series forward scattering.
    ///
    /// All arrays must be CuPy float32 device arrays on the same GPU.
    /// psi_re, psi_im: contiguous float32 arrays — MODIFIED IN-PLACE.
    ///   Shape must be (..., nx, ny) with total elements divisible by nx*ny.
    ///   When batched (total > nx*ny), batch items are distributed across
    ///   concurrent CUDA streams (up to kMaxStreams) for GPU-level parallelism.
    /// V: contiguous float32 array (nx*ny, potential) — shared across batch.
    /// laplace_prefactor: 1.0 / (dx * dy) for Laplacian stencil.
    /// laplace_method: "finite-difference" (default) or "fft".
    ///   When "fft", sampling_x and sampling_y must be > 0.
    ///
    /// Returns: (converged: bool, overflow: bool)
    py::tuple compute(py::object psi_re, py::object psi_im, py::object V,
                      std::size_t nx, std::size_t ny, float wavelength,
                      float dz, float convergence_threshold, int max_terms,
                      int max_inner,
                      float laplace_prefactor,
                      int accuracy = 8,
                      const std::string &laplace_method = "finite-difference",
                      float sampling_x = 0.0f, float sampling_y = 0.0f) {

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

        bool use_fft = (laplace_method == "fft");

        // Physical constants
        float K0 = 1.0f / wavelength;
        float inv_4piK0 = 1.0f / (4.0f * static_cast<float>(M_PI) * K0);

        // Decompose prefactor into inv_dx * inv_dy for the Laplacian stencil
        float inv_dx = std::sqrt(laplace_prefactor);
        float inv_dy = inv_dx;

        // Re-initialize if dimensions changed
        if (!initialized_ || nx != nx_ || ny != ny_) {
            nx_ = nx;
            ny_ = ny;
            for (auto &ctx : contexts_)
                ctx.buffers_valid = false;
            initialized_ = true;
        }

        // Number of concurrent streams (capped by batch size and max streams)
        int num_streams = std::min<int>(batch, kMaxStreams);
        ensure_contexts(num_streams, nx, ny);

        // Initialize per-stream FFT Laplacian if needed
        if (use_fft) {
            if (sampling_x <= 0.0f) sampling_x = sampling_y;
            if (sampling_y <= 0.0f) sampling_y = sampling_x;
            for (int s = 0; s < num_streams; ++s) {
                auto &ctx = contexts_[s];
                ctx.fft_laplacian.initialize(nx, ny, sampling_x, sampling_y);
                if (!ctx.lap_initialized) {
                    ctx.lap_re = DeviceArray<float>(nx * ny);
                    ctx.lap_im = DeviceArray<float>(nx * ny);
                    ctx.lap_initialized = true;
                }
            }
        }

        // Collected results from all batch items
        std::vector<bool> item_converged(batch, true);
        std::vector<bool> item_overflow(batch, false);

        // Dispatch batch items round-robin across streams using host threads.
        // Each thread processes items assigned to one stream sequentially;
        // GPU work from different streams can execute concurrently.
        std::vector<std::thread> threads;
        for (int s = 0; s < num_streams; ++s) {
            threads.emplace_back([&, s]() {
                auto &ctx = contexts_[s];
                for (std::size_t b = s; b < batch; b += num_streams) {
                    float *batch_re = re_ptr + b * nx * ny;
                    float *batch_im = im_ptr + b * nx * ny;

                    int iters = 0;
                    bool conv = false, ovf = false;
                    if (use_fft) {
                        compute_taylor_series_fft(
                            batch_re, batch_im, batch_re, batch_im,
                            V_ptr, nx, ny, wavelength, dz,
                            convergence_threshold, max_terms, max_inner,
                            inv_4piK0,
                            ctx.d_result,
                            conv, ovf,
                            ctx.work_re, ctx.work_im,
                            ctx.kseries_re, ctx.kseries_im,
                            ctx.kcur_re, ctx.kcur_im,
                            ctx.kwork_re, ctx.kwork_im,
                            ctx.fft_laplacian,
                            ctx.lap_re, ctx.lap_im,
                            &iters,
                            ctx.stream);
                    } else {
                        compute_taylor_series(
                            batch_re, batch_im, batch_re, batch_im,
                            V_ptr, nx, ny, wavelength, dz,
                            convergence_threshold, max_terms, max_inner,
                            inv_4piK0, inv_dx, inv_dy,
                            ctx.d_result,
                            conv, ovf,
                            ctx.work_re, ctx.work_im,
                            ctx.kseries_re, ctx.kseries_im,
                            ctx.kcur_re, ctx.kcur_im,
                            ctx.kwork_re, ctx.kwork_im,
                            &iters,
                            ctx.stream,
                            accuracy);
                    }
                    item_converged[b] = conv;
                    item_overflow[b] = ovf;
                }
            });
        }

        for (auto &t : threads)
            t.join();

        // Aggregate results
        bool all_converged = true;
        bool any_overflow = false;
        for (std::size_t b = 0; b < batch; ++b) {
            all_converged &= item_converged[b];
            any_overflow |= item_overflow[b];
        }

        return py::make_tuple(all_converged, any_overflow);
    }

  private:
    static constexpr int kMaxStreams = 4;

    struct StreamCtx {
        cudaStream_t stream = nullptr;
        DeviceArray<float> work_re, work_im;
        DeviceArray<float> kseries_re, kseries_im;
        DeviceArray<float> kcur_re, kcur_im;
        DeviceArray<float> kwork_re, kwork_im;
        DeviceArray<float> lap_re, lap_im;
        FFTLaplacian fft_laplacian;
        ConvergenceResult *d_result = nullptr;
        bool lap_initialized = false;
        bool buffers_valid = false;
    };

    std::vector<StreamCtx> contexts_;
    bool initialized_;
    std::size_t nx_, ny_;

    void ensure_contexts(int n, std::size_t nx, std::size_t ny) {
        std::size_t count = nx * ny;

        // Create streams and allocate buffers for first n contexts
        while (static_cast<int>(contexts_.size()) < n) {
            StreamCtx ctx;
            cudaStreamCreate(&ctx.stream);
            cudaMalloc(&ctx.d_result, sizeof(ConvergenceResult));
            contexts_.push_back(std::move(ctx));
        }

        // (Re)allocate buffers for active contexts if needed (dims changed)
        for (int s = 0; s < n; ++s) {
            auto &ctx = contexts_[s];
            if (!ctx.buffers_valid) {
                auto reset = [](DeviceArray<float> &arr) {
                    arr = DeviceArray<float>();
                };
                reset(ctx.work_re); reset(ctx.work_im);
                reset(ctx.kseries_re); reset(ctx.kseries_im);
                reset(ctx.kcur_re); reset(ctx.kcur_im);
                reset(ctx.kwork_re); reset(ctx.kwork_im);
                reset(ctx.lap_re); reset(ctx.lap_im);
                ctx.lap_initialized = false;

                ctx.work_re = DeviceArray<float>(count);
                ctx.work_im = DeviceArray<float>(count);
                ctx.kseries_re = DeviceArray<float>(count);
                ctx.kseries_im = DeviceArray<float>(count);
                ctx.kcur_re = DeviceArray<float>(count);
                ctx.kcur_im = DeviceArray<float>(count);
                ctx.kwork_re = DeviceArray<float>(count);
                ctx.kwork_im = DeviceArray<float>(count);

                ctx.buffers_valid = true;
            }
        }
    }
};

// ──────────────────────────────────────────────
// Python-facing wrapper for backscattering correction
// ──────────────────────────────────────────────
class PyBSCEngine {
  public:
    PyBSCEngine()
        : d_result_(nullptr), initialized_(false) {
        cudaStreamCreate(&stream1_);
        cudaStreamCreate(&stream2_);
    }

    ~PyBSCEngine() {
        cudaStreamDestroy(stream1_);
        cudaStreamDestroy(stream2_);
        if (d_result_)
            cudaFree(d_result_);
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

        // Convergence counter struct (single D2H copy)
        if (!d_result_)
            cudaMalloc(&d_result_, sizeof(ConvergenceResult));

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
                d_result_, stream1_,
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
    ConvergenceResult *d_result_;
    cudaStream_t stream1_, stream2_;
    std::size_t nx_, ny_;
    bool initialized_;
};

// ──────────────────────────────────────────────
// Helper: extract device pointers from a Python list of CuPy arrays
// ──────────────────────────────────────────────
static std::vector<float*> extract_ptr_list(py::list lst) {
    std::vector<float*> result;
    for (py::handle item : lst) {
        py::object obj = py::reinterpret_borrow<py::object>(item);
        result.push_back(get_device_ptr(obj));
    }
    return result;
}

// ──────────────────────────────────────────────
// Python-facing wrapper for BSC backward propagation
// ──────────────────────────────────────────────
class PyBSCBackPropEngine {
  public:
    PyBSCBackPropEngine()
        : d_result_(nullptr), initialized_(false) {
        cudaStreamCreate(&stream_);
    }

    ~PyBSCBackPropEngine() {
        cudaStreamDestroy(stream_);
        if (d_result_)
            cudaFree(d_result_);
    }

    void initialize(std::size_t nx, std::size_t ny) {
        if (initialized_ && nx == nx_ && ny == ny_)
            return;
        nx_ = nx;
        ny_ = ny;
        std::size_t count = nx * ny;

        auto reset = [](DeviceArray<float> &arr) { arr = DeviceArray<float>(); };
        reset(work_re_); reset(work_im_);
        reset(exit_re_); reset(exit_im_);
        reset(kseries_re_); reset(kseries_im_);
        reset(kcur_re_); reset(kcur_im_);
        reset(kwork_re_); reset(kwork_im_);

        auto alloc = [count](DeviceArray<float> &arr) { arr = DeviceArray<float>(count); };
        alloc(work_re_); alloc(work_im_);
        alloc(exit_re_); alloc(exit_im_);
        alloc(kseries_re_); alloc(kseries_im_);
        alloc(kcur_re_); alloc(kcur_im_);
        alloc(kwork_re_); alloc(kwork_im_);

        if (!d_result_)
            cudaMalloc(&d_result_, sizeof(ConvergenceResult));

        initialized_ = true;
    }

    /// Back-propagate BSC waves through per-original-slice stepping.
    ///
    /// bsc_waves_re, bsc_waves_im: Python lists of CuPy float32 arrays.
    ///   Length = num_exit_planes. Modified in-place to accumulated BSC.
    /// V_slices: Python list of CuPy float32 arrays.
    ///   ALL original (non-aggregated) transmission functions.
    /// exit_plane_indices: Python list of ints, length = num_exit_planes.
    ///   Block ep spans V_slices[exit_plane_indices[ep] :
    ///                         exit_plane_indices[ep+1]].
    /// dz: slice thickness (Å), uniform across all slices.
    ///
    /// All CuPy arrays must be on the same GPU device.
    py::tuple compute(py::list bsc_waves_re, py::list bsc_waves_im,
                      py::list V_slices,
                      py::list exit_plane_indices,
                      std::size_t nx, std::size_t ny,
                      float wavelength, float dz,
                      float convergence_threshold, int max_terms,
                      int max_inner, float laplace_prefactor,
                      int accuracy = 8,
                      bool use_conj = true) {

        int num_exit_planes = static_cast<int>(py::len(bsc_waves_re));
        int num_total_slices = static_cast<int>(py::len(V_slices));

        if (static_cast<int>(py::len(exit_plane_indices)) != num_exit_planes) {
            throw std::runtime_error(
                "len(exit_plane_indices) must equal len(bsc_waves)");
        }

        initialize(nx, ny);

        float K0 = 1.0f / wavelength;
        float inv_4piK0 = 1.0f / (4.0f * static_cast<float>(M_PI) * K0);
        float inv_dx = std::sqrt(laplace_prefactor);
        float inv_dy = inv_dx;

        // Extract device pointer arrays from Python lists
        std::vector<float*> bsc_re_ptrs = extract_ptr_list(bsc_waves_re);
        std::vector<float*> bsc_im_ptrs = extract_ptr_list(bsc_waves_im);
        std::vector<float*> V_ptrs = extract_ptr_list(V_slices);

        // Copy exit_plane_indices to host array
        std::vector<int> ep_indices(num_exit_planes);
        for (int i = 0; i < num_exit_planes; ++i) {
            ep_indices[i] = py::cast<int>(exit_plane_indices[i]);
        }

        back_propagate_bsc_series(
            bsc_re_ptrs.data(), bsc_im_ptrs.data(), num_exit_planes,
            const_cast<const float**>(V_ptrs.data()), num_total_slices,
            ep_indices.data(),
            nx, ny, wavelength, dz,
            convergence_threshold, max_terms, max_inner,
            inv_4piK0, inv_dx, inv_dy,
            work_re_, work_im_,
            exit_re_, exit_im_,
            kseries_re_, kseries_im_,
            kcur_re_, kcur_im_,
            kwork_re_, kwork_im_,
            d_result_, stream_, accuracy,
            use_conj);

        return py::make_tuple(true);
    }

    /// Running accumulation: back-propagate EVERY slice's BSC through all
    /// overlying slices to the entrance surface.
    ///
    /// bsc_slices_re, bsc_slices_im: Python lists of CuPy float32 arrays.
    ///   Length = num_slices. bsc_slices_re[0]/bsc_slices_im[0] are modified
    ///   in-place to the total accumulated BSC at the entrance surface.
    /// V_slices: Python list of CuPy float32 arrays, length = num_slices.
    /// dz: slice thickness (A), uniform across all slices.
    ///
    /// All CuPy arrays must be on the same GPU device.
    py::tuple compute_accumulate(py::list bsc_slices_re, py::list bsc_slices_im,
                                  py::list V_slices,
                                  py::list ep_re, py::list ep_im,
                                  py::list exit_plane_indices_list,
                                  std::size_t nx, std::size_t ny,
                                  float wavelength, float dz,
                                  float convergence_threshold, int max_terms,
                                  int max_inner, float laplace_prefactor,
                                  int accuracy = 8,
                                  bool use_conj = true) {

        int num_slices = static_cast<int>(py::len(bsc_slices_re));
        int num_exit_planes = static_cast<int>(py::len(ep_re));

        if (static_cast<int>(py::len(bsc_slices_im)) != num_slices) {
            throw std::runtime_error(
                "len(bsc_slices_re) must equal len(bsc_slices_im)");
        }
        if (static_cast<int>(py::len(V_slices)) != num_slices) {
            throw std::runtime_error(
                "len(V_slices) must equal len(bsc_slices)");
        }
        if (static_cast<int>(py::len(ep_im)) != num_exit_planes) {
            throw std::runtime_error(
                "len(ep_re) must equal len(ep_im)");
        }
        if (static_cast<int>(py::len(exit_plane_indices_list)) != num_exit_planes) {
            throw std::runtime_error(
                "len(exit_plane_indices) must equal num_exit_planes");
        }

        initialize(nx, ny);

        float K0 = 1.0f / wavelength;
        float inv_4piK0 = 1.0f / (4.0f * static_cast<float>(M_PI) * K0);
        float inv_dx = std::sqrt(laplace_prefactor);
        float inv_dy = inv_dx;

        // Extract device pointer arrays from Python lists
        std::vector<float*> bsc_re_ptrs = extract_ptr_list(bsc_slices_re);
        std::vector<float*> bsc_im_ptrs = extract_ptr_list(bsc_slices_im);
        std::vector<float*> V_ptrs = extract_ptr_list(V_slices);
        std::vector<float*> ep_re_ptrs = extract_ptr_list(ep_re);
        std::vector<float*> ep_im_ptrs = extract_ptr_list(ep_im);

        // Extract exit plane indices (host ints)
        std::vector<int> ep_indices(num_exit_planes);
        for (int i = 0; i < num_exit_planes; ++i) {
            ep_indices[i] = py::cast<int>(exit_plane_indices_list[i]);
        }

        running_accumulate_bsc(
            bsc_re_ptrs.data(), bsc_im_ptrs.data(), num_slices,
            const_cast<const float**>(V_ptrs.data()),
            nx, ny, wavelength, dz,
            convergence_threshold, max_terms, max_inner,
            inv_4piK0, inv_dx, inv_dy,
            ep_re_ptrs.data(), ep_im_ptrs.data(),
            num_exit_planes, ep_indices.data(),
            work_re_, work_im_,
            exit_re_, exit_im_,
            kseries_re_, kseries_im_,
            kcur_re_, kcur_im_,
            kwork_re_, kwork_im_,
            d_result_, stream_, accuracy,
            use_conj);

        return py::make_tuple(true);
    }

  private:
    DeviceArray<float> work_re_, work_im_;
    DeviceArray<float> exit_re_, exit_im_;
    DeviceArray<float> kseries_re_, kseries_im_;
    DeviceArray<float> kcur_re_, kcur_im_;
    DeviceArray<float> kwork_re_, kwork_im_;
    ConvergenceResult *d_result_;
    cudaStream_t stream_;
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

    cvdms::ConvergenceResult *d_result;
    cudaMalloc(&d_result, sizeof(cvdms::ConvergenceResult));

    cvdms::compute_k_series(re_ptr, im_ptr,
                            kseries_re.data(), kseries_im.data(),
                            V_ptr, nx, ny, wavelength, dz,
                            convergence_threshold, max_terms,
                            inv_4piK0, inv_dx, inv_dy,
                            cur_re, cur_im, buf_re, buf_im,
                            d_result, nullptr, 8);

    cudaFree(d_result);

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
             py::arg("max_inner") = 100,
             py::arg("laplace_prefactor"),
             py::arg("accuracy") = 8,
             py::arg("laplace_method") = "finite-difference",
             py::arg("sampling_x") = 0.0f,
             py::arg("sampling_y") = 0.0f);

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

    py::class_<cvdms::PyBSCBackPropEngine>(m, "BSCBackPropEngine")
        .def(py::init<>())
        .def("compute", &cvdms::PyBSCBackPropEngine::compute,
             py::arg("bsc_waves_re"), py::arg("bsc_waves_im"),
             py::arg("V_slices"),
             py::arg("exit_plane_indices"),
             py::arg("nx"), py::arg("ny"),
             py::arg("wavelength"), py::arg("dz"),
             py::arg("convergence_threshold"), py::arg("max_terms"),
             py::arg("max_inner") = 100,
             py::arg("laplace_prefactor"),
             py::arg("accuracy") = 8,
             py::arg("use_conj") = true)
        .def("compute_accumulate",
             &cvdms::PyBSCBackPropEngine::compute_accumulate,
             py::arg("bsc_slices_re"), py::arg("bsc_slices_im"),
             py::arg("V_slices"),
             py::arg("ep_re"), py::arg("ep_im"),
             py::arg("exit_plane_indices"),
             py::arg("nx"), py::arg("ny"),
             py::arg("wavelength"), py::arg("dz"),
             py::arg("convergence_threshold"), py::arg("max_terms"),
             py::arg("max_inner") = 100,
             py::arg("laplace_prefactor"),
             py::arg("accuracy") = 8,
             py::arg("use_conj") = true);
}
