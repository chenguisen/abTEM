"""Chen-Van Dyck Multislice (CVDMS) algorithm module.

This module implements the CVDMS algorithm based on:
J.H. Chen, D. Van Dyck, "Accurate multislice theory for elastic electron
scattering in transmission electron microscopy".

The algorithm is ported from the ImageSimulation_CGS project's C++/CUDA
implementation in main_diffraction_cbed.cu and wave_kernels.cu, fully
aligned with the original pixel-by-pixel convergence control.

Note: _backend_reported is a module-level flag to print the backend
selection message only once per session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import warnings

import numpy as np

from abtem.core.backend import get_array_module
from abtem.core.energy import energy2sigma, energy2wavelength

_backend_reported = False  # print backend selection only once
_bsc_engine = None         # cached BSCEngine to avoid per-slice cudaMalloc thrashing
_taylor_engine = None      # cached TaylorEngine, same reason

from abtem.finite_difference import LaplaceOperator, DivergedError

if TYPE_CHECKING:
    from abtem.potentials.iam import PotentialArray
    from abtem.waves import Waves


def cvdms_multislice_step(
    waves: Waves,
    potential_slice: PotentialArray,
    next_slice: PotentialArray | None,
    laplace: LaplaceOperator,
    *,
    max_terms: int = 50,
    max_inner: int = 100,
    convergence_threshold: float = 1e-7,
    backscattering: bool = False,
    calculate_backscattered: bool = False,
    fully_corrected: bool = False,
    divergence_ratio: float = 5.0,
    check_interval: int = 2,
    antialias: bool = True,
    use_fused_kernel: bool = True,
    backend: str = "auto",
    antialias_inner: bool = True,
) -> Waves | Sequence[Waves]:
    """
    Performs a single CVDMS (Chen-Van Dyck Multislice) step.

    This implements the CVDMS algorithm ported from ImageSimulation_CGS,
    fully aligned with the original pixel-by-pixel convergence logic.

    Parameters
    ----------
    waves : Waves
        Input wave function.
    potential_slice : PotentialArray
        Potential array for the current slice.
    next_slice : PotentialArray, optional
        Potential array for the next slice (used for backscattering).
    laplace : LaplaceOperator
        Finite-difference Laplacian operator.
    max_terms : int, optional
        Maximum Taylor series terms (default 50).
    convergence_threshold : float, optional
        Pixel-wise convergence threshold (default 1e-7).
        Applied to both the outer Taylor series and the 1/k BSC correction series.
    backscattering : bool, optional
        If True, enable inter-slice backscattering coupling. The BSC operator
        is computed and subtracted from the forward wave:
        ψ_corrected = ψ_forward - BSC(ψ_forward). (default False).
    calculate_backscattered : bool, optional
        Whether to compute the backscattered wave (default False).
    fully_corrected : bool, optional
        Internal parameter. When True, the function always returns a
        (Waves, Waves) tuple to satisfy the caller's unconditional tuple
        unpacking. Derived from backscattering in the CVDMSMultislice path,
        or from expansion_scope in the RealSpaceMultislice path.
    divergence_ratio : float, optional
        When |term|_sum > divergence_ratio * |accumulated|_sum, the Taylor
        series is truncated (with a warning) instead of raising DivergedError.
        Default 5.0.
    antialias : bool, optional
        If True, apply antialias low-pass filter to the potential (default True).
        When enabled, the projected potential is bandlimited to 2/3 Nyquist
        with a cosine taper, matching the Fourier multislice antialias aperture
        treatment. This ensures both algorithms operate on the same bandlimited
        input, enabling fair comparison. Set to False to use the full-frequency
        potential (may introduce aliasing in FFT Laplacian).
    use_fused_kernel : bool, optional
        If True, use fused CUDA kernel for inner K-series (default True).
    backend : str, optional
        Backend selection for the K-operator computation (default "auto").
        "auto": try C++ CUDA backend first if available, fall through to CuPy/Python.
        "c++": force C++ CUDA backend; raises RuntimeError if unavailable.
        "cupy": skip C++ CUDA backend, use CuPy fused kernel or Python loops.
    antialias_inner : bool, optional
        If True, apply antialias filter after each K-operator application within
        the inner K-series (default True). Prevents bandwidth explosion from
        V * psi multiplication, which creates above-Nyquist frequencies that
        cause overflow at fine sampling. This adds 2 FFTs per K-series iteration.

    Returns
    -------
    Waves or list of Waves
        The exit wave, and optionally the backscattered wave.
    """
    if max_terms < 1:
        raise ValueError("max_terms must be >= 1")

    # Ensure devices match
    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)
        if next_slice is not None:
            next_slice = next_slice.copy_to_device(device=waves.device)

    thickness = potential_slice.thickness
    wavelength = energy2wavelength(waves._valid_energy)
    sigma = energy2sigma(waves._valid_energy)
    K0 = 1.0 / wavelength

    # Transmission function: sigma * V (per angstrom)
    transmission_function = potential_slice.array[0] * sigma / thickness

    # Next slice transmission function for backscattering
    if next_slice is not None and backscattering:
        transmission_function_next = (
            next_slice.array[0] * sigma / thickness
        )
    else:
        transmission_function_next = None

    # Apply antialias bandlimit to the potential, matching Fourier multislice.
    # This removes spatial frequencies above 2/3 Nyquist from the potential
    # before the K-operator expansion, ensuring a fair comparison between
    # CVDMS and Fourier.
    aa_kernel = None
    if antialias:
        from abtem.antialias import antialias_aperture

        xp_aa = get_array_module(waves._array)
        aa_kernel = antialias_aperture(
            waves._valid_gpts, waves._valid_sampling, xp=xp_aa
        )
        # Use xp.fft directly (not abTEM's dispatch) to handle both
        # real-valued potentials and pyfftw limitations with real arrays.
        tf_f = xp_aa.fft.fft2(transmission_function)
        transmission_function = xp_aa.fft.ifft2(tf_f * aa_kernel).real
        if transmission_function_next is not None:
            tf_next_f = xp_aa.fft.fft2(transmission_function_next)
            transmission_function_next = xp_aa.fft.ifft2(
                tf_next_f * aa_kernel
            ).real

    laplace_stencil = laplace.get_stencil(waves, device=waves.device)

    # Prefactor for Laplacian stencil: 1/(dx*dy)
    prefactor = 1.0 / (waves.sampling[0] * waves.sampling[1])

    # Extract raw stencil coefficients for fused kernel
    from abtem.finite_difference import finite_difference_coefficients
    stencil_raw = finite_difference_coefficients(2, laplace._accuracy).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Step 1: Pure forward scattering
    #  对应 calPureForwardScatter + calK_PureForward
    # ------------------------------------------------------------------ #
    pure_forward = _cvdms_forward_scattering(
        waves._array,
        transmission_function,
        laplace_stencil,
        wavelength,
        thickness,
        max_terms,
        max_inner,
        convergence_threshold,
        divergence_ratio=divergence_ratio,
        check_interval=check_interval,
        use_fused_kernel=use_fused_kernel,
        prefactor=prefactor,
        stencil_raw=stencil_raw,
        backend=backend,
        laplace_method=getattr(laplace, '_method', 'finite-difference'),
        sampling=waves.sampling,
        antialias_inner=antialias_inner,
        aa_kernel=aa_kernel,
    )

    # ------------------------------------------------------------------ #
    # Step 2: Backscattering correction
    #  对应 calBSC
    # ------------------------------------------------------------------ #
    if backscattering and next_slice is not None:
        # Per-slice backscattering correction (NOT the accumulated backscattered
        # wave). This is the electron flux scattered backward at this interface,
        # subtracted from the forward wave:
        #   exit_wave = pure_forward - backscatter
        #   B · ψ = (k_{j+1} - k_j) / (2·k_{j+1}) · ψ
        #
        # The accumulated backscattered wave (bsc_wave_conj) is computed later
        # by back-propagating these per-slice corrections through all overlying
        # slices via _back_propagate_bsc_impl().
        #
        # Guard: skip BSC when the current slice has negligible potential.
        # The SBA formula assumes k_{j+1} ≈ k_j, which fails when the current
        # slice is vacuum — K_series(ψ, 0) = 0 while K_series(ψ, V_next) is
        # enormous, blowing up I/I₀ to 10^16+.  Physically, vacuum has no
        # backscattering; Fresnel propagation already handles free space.
        xp_bs = get_array_module(transmission_function)
        tf_max = float(xp_bs.max(xp_bs.abs(transmission_function)))
        if tf_max < 1e-10:
            exit_wave = pure_forward
            backscatter = xp_bs.zeros_like(pure_forward)
        else:
            backscatter = _cvdms_backscattering_correction(
                pure_forward,
                transmission_function,
                transmission_function_next,
                laplace_stencil,
                wavelength,
                thickness,
                convergence_threshold=convergence_threshold,
                max_inner_iter=max_inner,
                check_interval=check_interval,
                use_fused_kernel=use_fused_kernel,
                prefactor=prefactor,
                stencil_raw=stencil_raw,
                backend=backend,
                antialias_inner=antialias_inner,
                aa_kernel=aa_kernel,
            )

            # Corrected forward wave = pure forward - backscattering
            # 对应: phi_j = (1 - B_{j+1,j}) · ψ_j
            exit_wave = pure_forward - backscatter

        # Bandlimit the exit wave and backscatter (match Fourier antialias)
        if antialias:
            xp_f = get_array_module(pure_forward)
            exit_wave = xp_f.fft.ifft2(xp_f.fft.fft2(exit_wave) * aa_kernel)
            backscatter = xp_f.fft.ifft2(xp_f.fft.fft2(backscatter) * aa_kernel)

        kwargs = waves._copy_kwargs(exclude=("array",))
        exit_waves_obj = waves.__class__(exit_wave, **kwargs)

        # NOTE: backscattered_waves_obj contains per-slice correction fields,
        # NOT the physical backscattered wave. The caller back-propagates
        # these to produce the accumulated backscattered wave (bsc_wave_conj).
        backscattered_waves_obj = waves.__class__(backscatter, **kwargs)

        if calculate_backscattered:
            return exit_waves_obj, backscattered_waves_obj

        return exit_waves_obj, backscattered_waves_obj

    # Bandlimit the exit wave (match Fourier antialias)
    if antialias:
        xp_f = get_array_module(pure_forward)
        pure_forward = xp_f.fft.ifft2(xp_f.fft.fft2(pure_forward) * aa_kernel)

    kwargs = waves._copy_kwargs(exclude=("array",))

    if calculate_backscattered or fully_corrected:
        # When fully_corrected=True, the caller always unpacks (Waves, Waves).
        # For the last slice where BSC is not computed, return zero
        # backscattered wave. When calculate_backscattered=True, same tuple
        # return is needed.
        xp = get_array_module(pure_forward)
        zero_back = xp.zeros_like(pure_forward)
        backscattered_waves_obj = waves.__class__(zero_back, **kwargs)
        return waves.__class__(pure_forward, **kwargs), backscattered_waves_obj

    return waves.__class__(pure_forward, **kwargs)


# ======================================================================
# Forward scattering: outer loop (指数展开)
#  对应 calPureForwardScatter
# ======================================================================
def _cvdms_forward_scattering(
    waves_array: np.ndarray,
    transmission_function: np.ndarray,
    laplace: callable,
    wavelength: float,
    thickness: float,
    max_terms: int,
    max_inner: int = 100,
    convergence_threshold: float = 1e-6,
    divergence_ratio: float = 5.0,
    return_diagnostics: bool = False,
    check_interval: int = 2,
    use_fused_kernel: bool = True,
    prefactor: float | None = None,
    stencil_raw: np.ndarray | None = None,
    backend: str = "auto",
    laplace_method: str = "finite-difference",
    sampling: tuple[float, float] | None = None,
    antialias_inner: bool = False,
    aa_kernel: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Pure forward scattering with double Taylor series expansion.

    Outer loop (指数展开):
        phi = Σ (i·dz)ⁿ/n! · K_inner_seriesⁿ(psi_0)

    Inner loop (平方根展开, inside _cvdms_inner_k_series):
        K_inner_series(psi) = Σ cₙ · Kⁿ(psi)

    Both loops use pixel-by-pixel convergence:
    `applyThread` equivalent — count pixels where |term| > cutoff.
    When ALL pixels are below cutoff, the series has converged.

    GPU utilization optimization:
    Convergence checks force D2H synchronization, stalling the GPU pipeline.
    This is the primary reason CVDMS GPU utilization is lower than Fourier.
    The `check_interval` parameter controls how often convergence is checked:
    - check_interval=1 (original): check every iteration → max ~9,600 syncs/slice
    - check_interval=2 (default): check every 2 iterations → halved syncs
    - check_interval=3: check every 3 iterations → further reduced
    The cost is at most `check_interval - 1` extra terms, which is negligible
    for the convergent Taylor series.

    对应: calPureForwardScatter + calK_PureForward in wave_kernels.cu
    """
    xp = get_array_module(waves_array)
    dz = thickness

    # Diagnostic tracking (only populated when return_diagnostics=True)
    diag_ratios = []
    diag_n_above = []
    overflow_detected = False
    divergence_truncated = False
    global _backend_reported

    # ---- Backend selection ----
    # Determine if C++ CUDA path is eligible
    cpp_eligible = (use_fused_kernel
                    and xp.__name__ == "cupy"
                    and prefactor is not None
                    and waves_array.dtype == np.complex64
                    and waves_array.ndim >= 2)

    # ---- C++ CUDA backend path ----
    # Replaces the entire outer Taylor + inner K-series loop with a single
    # pybind11 call to _cvdms_backend.TaylorEngine.
    use_cpp = False
    if backend == "c++":
        if not cpp_eligible:
            raise RuntimeError(
                "C++ CUDA backend requested but not available. "
                "Requirements: CuPy, complex64 dtype, ndim >= 2, "
                "use_fused_kernel=True, and a valid prefactor."
            )
        use_cpp = True
    elif backend == "auto":
        use_cpp = cpp_eligible
    # backend == "cupy": use_cpp stays False

    if use_cpp:
        try:
            global _taylor_engine
            if _taylor_engine is None:
                from _cvdms_backend import TaylorEngine
                _taylor_engine = TaylorEngine()
            if not _backend_reported:
                print("[cvdms] Using C++ CUDA backend")
                _backend_reported = True

            psi_re = xp.ascontiguousarray(
                xp.real(waves_array).astype(xp.float32))
            psi_im = xp.ascontiguousarray(
                xp.imag(waves_array).astype(xp.float32))
            V = xp.ascontiguousarray(
                transmission_function.astype(xp.float32))

            nx, ny = waves_array.shape[-2:]
            engine = _taylor_engine

            converged = True
            overflow = False
            if laplace_method == "fft" and sampling is not None:
                sx, sy = sampling
                converged, overflow = engine.compute(
                    psi_re, psi_im, V,
                    nx, ny, wavelength, dz,
                    convergence_threshold, max_terms, max_inner,
                    prefactor, 8,
                    "fft", sx, sy,
                    aa_kernel,
                    divergence_ratio=divergence_ratio,
                )
            else:
                converged, overflow = engine.compute(
                    psi_re, psi_im, V,
                    nx, ny, wavelength, dz,
                    convergence_threshold, max_terms, max_inner,
                    prefactor,
                    aa_kernel=aa_kernel,
                    divergence_ratio=divergence_ratio,
                )

            exit_wave = xp.empty_like(waves_array)
            exit_wave.real = psi_re
            exit_wave.imag = psi_im

            if overflow:
                warnings.warn(
                    f"CVDMS numerical overflow detected. "
                    f"The accumulated wave function exceeds complex64 range. "
                    f"Use a coarser sampling, higher voltage, or thinner sample, "
                    f"or switch to complex128 precision.",
                    RuntimeWarning, stacklevel=2,
                )
                overflow_detected = True

            if return_diagnostics:
                diag = {
                    "n_terms_used": -1 if converged else max_terms,
                    "ratios_per_order": [],
                    "n_above_per_order": [],
                    "overflow_detected": overflow_detected,
                    "divergence_truncated": False,
                    "max_amplitude": float(xp.max(xp.abs(exit_wave))),
                }
                return exit_wave, diag
            return exit_wave
        except ImportError:
            if backend == "c++":
                raise RuntimeError(
                    "C++ CUDA backend requested but _cvdms_backend module "
                    "not found. Build the C++ backend first."
                ) from None
            pass  # Fall through to Python path

    # ---- Python backend path ----
    if not _backend_reported:
        _backend_name = "CuPy fused kernel" if use_fused_kernel and xp.__name__ == "cupy" else "Python (CuPy/NumPy)"
        print(f"[cvdms] Using {_backend_name} backend")
        _backend_reported = True

    exit_wave = waves_array.copy()
    working = None  # first allocation comes from inner_k_series

    # Outer Taylor series: exp(i·K·dz) = Σ (i·dz·K)ⁿ/n!
    for n_exp_order in range(1, max_terms + 1):
        # ---- Inner series: compute K_series(working) ----
        k_series = _cvdms_inner_k_series(
            working if working is not None else waves_array,
            transmission_function,
            laplace,
            wavelength,
            convergence_threshold,
            max_inner_iter=max_inner,
            check_interval=check_interval,
            use_fused_kernel=use_fused_kernel,
            prefactor=prefactor,
            stencil_raw=stencil_raw,
            antialias_inner=antialias_inner,
            aa_kernel=aa_kernel,
        )

        # Reuse k_series memory as working buffer for next iteration
        working = k_series

        # multiplyComplex_i_CGS:  working *= i * dz / n_exp_order
        scale = complex(0, dz / float(n_exp_order))
        working *= scale  # in-place

        # Compute sum_exit BEFORE adding working to exit_wave.
        # Using sum(|work|) / sum(|exit_before|) correctly detects when
        # a new term dwarfs the previous accumulated exit.  The old
        # denominator sum(|exit_after|) was always >= sum(|work|), so the
        # ratio was bounded near 1 and the check was ineffective.
        sum_exit_before = 0.0
        if (n_exp_order % check_interval == 0 or n_exp_order == max_terms):
            if n_exp_order > 1 and divergence_ratio > 0:
                sum_exit_before = float(xp.abs(exit_wave).sum())

        # addArray_1dthread: exit_wave += working
        exit_wave += working

        # ---- Batched convergence + stability check ----
        if n_exp_order % check_interval == 0 or n_exp_order == max_terms:
            # Overflow check (inf/nan)
            if xp.any(xp.isinf(exit_wave) | xp.isnan(exit_wave)):
                exit_wave -= working
                warnings.warn(
                    f"CVDMS numerical overflow at order {n_exp_order}. "
                    f"The accumulated wave function exceeds complex64 range. "
                    f"Use a coarser sampling, higher voltage, or thinner "
                    f"sample, or switch to complex128 precision.",
                    RuntimeWarning, stacklevel=2,
                )
                overflow_detected = True
                break

            n_above = int(xp.sum(xp.abs(working) > convergence_threshold))

            if n_exp_order > 1 and divergence_ratio > 0:
                sum_working = float(xp.abs(working).sum())
            else:
                sum_working = 0.0

            # Record diagnostics
            if return_diagnostics:
                diag_n_above.append((n_exp_order, n_above))

            if n_above == 0:
                break

            # Divergence check: ratio = |new term| / |exit before adding it|
            if n_exp_order > 1 and divergence_ratio > 0:
                ratio = sum_working / max(sum_exit_before, 1e-30)
                if return_diagnostics:
                    diag_ratios.append((n_exp_order, ratio))
                if ratio > divergence_ratio:
                    exit_wave -= working
                    divergence_truncated = True
                    warnings.warn(
                        f"CVDMS series truncated at order {n_exp_order - 1} "
                        f"(term/accum ratio={ratio:.4f} > "
                        f"divergence_ratio={divergence_ratio}). "
                        f"Partial sum may have reduced accuracy.",
                        RuntimeWarning, stacklevel=2,
                    )
                    break
    else:
        n_remaining = int(xp.sum(xp.abs(working) > convergence_threshold))
        warnings.warn(
            f"CVDMS forward scattering did not fully converge in "
            f"{max_terms} terms. {n_remaining} pixels above threshold "
            f"({convergence_threshold}). "
            "Try increasing max_terms or convergence_threshold.",
            RuntimeWarning, stacklevel=2,
        )


    if return_diagnostics:
        diag = {
            "n_terms_used": n_exp_order,
            "ratios_per_order": diag_ratios,
            "n_above_per_order": diag_n_above,
            "overflow_detected": overflow_detected,
            "divergence_truncated": divergence_truncated,
            "max_amplitude": float(xp.max(xp.abs(exit_wave))),
        }
        return exit_wave, diag
    return exit_wave


# ======================================================================
# Inner K-operator series (平方根展开)
#  对应 calK_PureForward
# ======================================================================
def _cvdms_inner_k_series(
    waves_array: np.ndarray,
    transmission_function: np.ndarray,
    laplace: callable,
    wavelength: float,
    convergence_threshold: float,
    max_inner_iter: int = 100,
    check_interval: int = 2,
    use_fused_kernel: bool = True,
    prefactor: float | None = None,
    stencil_raw: np.ndarray | None = None,
    antialias_inner: bool = False,
    aa_kernel: np.ndarray | None = None,
) -> np.ndarray:
    """
    Inner K-operator Taylor series with pixel-by-pixel convergence.

    Computes:
        K_series(psi) = Σ_(n=1..∞) cₙ · Kⁿ(psi)

    where K is the multislice operator:
        K(psi) = V · psi + ∇²(psi) / (4π·K₀)

    and cₙ are the binomial scaling coefficients:
        c₁ = 1
        cₙ = (0.5 - n + 1) · λ / (π · n)   for n > 1

    The convergence check is pixel-by-pixel (同 applyThread):
        count pixels where |latest_term| > cutoff
        if count == 0 → converged

    GPU utilization optimization:
    Convergence checks force D2H synchronization. The `check_interval`
    parameter controls how often we sync. At check_interval=2 (default),
    sync frequency is halved with at most 1 extra iteration of work.

    Optimizations:
    - Pre-allocated scratch buffer avoids allocating new arrays each iteration
    - In-place operations where safe

    对应: calK_PureForward in wave_kernels.cu
    """
    xp = get_array_module(waves_array)

    # ---- Fused kernel path ----
    if use_fused_kernel and xp.__name__ == "cupy" and prefactor is not None and stencil_raw is not None:
        from .cvdms_kernels import compute_k_series_fused

        return compute_k_series_fused(
            waves_array,
            transmission_function,
            wavelength,
            convergence_threshold,
            max_inner_iter,
            check_interval,
            prefactor=prefactor,
            stencil_raw=stencil_raw,
            antialias_inner=antialias_inner,
            aa_kernel=aa_kernel,
        )

    # ---- Original Python loop path ----
    K0 = 1.0 / wavelength
    inv_4piK0 = 1.0 / (4.0 * np.pi * K0)

    # Pre-allocate scratch buffer (reused each iteration)
    scratch = xp.empty_like(waves_array)

    # k_series = 0 (initialize series result to zero)
    k_series = xp.zeros_like(waves_array)

    # working = input wave (gets overwritten)
    working = waves_array.copy()

    n_sqrt_order = 1
    prev_n_above = None

    while True:
        # ---- K operator: V * working + laplace(working) / (4πK₀) ----
        # scratch = laplace(working)
        scratch[:] = laplace(working)
        scratch *= inv_4piK0                     # in-place: ∇²/(4πK₀)
        working *= transmission_function          # in-place: V * working
        scratch += working                        # in-place: K(working)

        # ---- Internal antialias: prevent bandwidth explosion ----
        # V * cur doubles the function bandwidth, creating above-Nyquist
        # components that the Laplacian amplifies ~k². Re-bandlimit to the
        # same 2/3 Nyquist aperture used for the potential.
        if antialias_inner and aa_kernel is not None:
            scratch_f = xp.fft.fft2(scratch)
            scratch_f *= aa_kernel
            scratch[:] = xp.fft.ifft2(scratch_f)

        # ---- Numerical stability check ----
        # Deferred to check_interval boundaries to avoid D2H sync.
        # Combined isnan/isinf into a single sync point.
        if n_sqrt_order % check_interval == 0:
            if xp.any(xp.isinf(scratch) | xp.isnan(scratch)):
                break

        # ---- Scaling for higher orders ----
        if n_sqrt_order == 1:
            k_series += scratch  # first order: no scaling
        else:
            scale = (
                (0.5 - n_sqrt_order + 1.0) * wavelength / (np.pi * n_sqrt_order)
            )
            # In-place accumulate scaled result
            scratch *= scale
            k_series += scratch

        # ---- Batched convergence/stagnation check ----
        # Check every `check_interval` iterations to reduce D2H syncs.
        # Stagnation detection (prev_n_above) still uses gpu-side values
        # from the last sync point; the lag is bounded by check_interval.
        if n_sqrt_order % check_interval == 0:
            n_above = int(xp.sum(xp.abs(scratch) > convergence_threshold))

            if prev_n_above is not None and n_above >= prev_n_above:
                break

            prev_n_above = n_above
            n_sqrt_order += 1

            if n_above == 0:
                break  # fully converged
        else:
            # Non-sync iteration: just increment counter, no convergence check
            n_sqrt_order += 1

        if n_sqrt_order > max_inner_iter:
            break  # safety limit

        # ---- Prepare for next iteration: working = K(working) ----
        working, scratch = scratch, working

    return k_series


# ======================================================================
# Backscattering correction
#  对应 calBSC
# ======================================================================
def _cvdms_backscattering_correction(
    waves_array: np.ndarray,
    transmission_function: np.ndarray,
    transmission_function_next: np.ndarray,
    laplace: callable,
    wavelength: float,
    thickness: float,
    convergence_threshold: float = 1e-16,
    max_inner_iter: int = 100,
    check_interval: int = 2,
    use_fused_kernel: bool = True,
    prefactor: float | None = None,
    stencil_raw: np.ndarray | None = None,
    backend: str = "auto",
    antialias_inner: bool = False,
    aa_kernel: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate per-slice backscattering correction field.

    Returns the correction field subtracted from the forward wave at each
    slice interface: exit_wave = pure_forward - backscatter. This is NOT
    the accumulated backscattered wave (bsc_wave_conj) — that is computed
    by back-propagating these per-slice corrections through all overlying slices.

    对应: calBSC in wave_kernels.cu

    BSC operator (flux-conserving Fresnel reflection):
        wave_1 = k_j * phi       (current slice potential)
        wave_2 = k_{j+1} * phi   (next slice potential)
        R = (wave_1 - wave_2) / (wave_1 + wave_2)   (Fresnel reflection amplitude)
        T = sqrt(1 - |R|²)                           (forward transmission)
        backscatter = phi · (1 - T)
        |ψ_out|² = T² · |ψ|² ≤ |ψ|² always.

    Reference: Eq.(7-10) in Micron 190 (2025) 103778.
    """
    xp = get_array_module(waves_array)
    K0 = 1.0 / wavelength
    dz = thickness
    global _backend_reported

    # ---- Backend selection ----
    cpp_eligible = (use_fused_kernel
                    and xp.__name__ == "cupy"
                    and waves_array.dtype == np.complex64
                    and prefactor is not None
                    and transmission_function_next is not None)

    use_cpp = False
    if backend == "c++":
        if not cpp_eligible:
            raise RuntimeError(
                "C++ CUDA backend requested but not available. "
                "Requirements: CuPy, complex64 dtype, use_fused_kernel=True, "
                "and a valid prefactor."
            )
        use_cpp = True
    elif backend == "auto":
        use_cpp = cpp_eligible
    # backend == "cupy": use_cpp stays False

    if use_cpp:
        try:
            global _bsc_engine
            if _bsc_engine is None:
                from _cvdms_backend import BSCEngine
                _bsc_engine = BSCEngine()

            if not _backend_reported:
                print("[cvdms] Using C++ CUDA backend")
                _backend_reported = True

            psi_re = xp.ascontiguousarray(
                xp.real(waves_array).astype(xp.float32))
            psi_im = xp.ascontiguousarray(
                xp.imag(waves_array).astype(xp.float32))
            V_cur = xp.ascontiguousarray(
                transmission_function.astype(xp.float32))
            V_next = xp.ascontiguousarray(
                transmission_function_next.astype(xp.float32))
            bs_re = xp.empty_like(psi_re)
            bs_im = xp.empty_like(psi_im)

            nx, ny = waves_array.shape[-2:]

            _bsc_engine.compute(
                psi_re, psi_im, V_cur, V_next, bs_re, bs_im,
                nx, ny, wavelength, dz, max_inner_iter,
                convergence_threshold=convergence_threshold,
                max_terms=100,
                laplace_prefactor=prefactor,
            )

            result = xp.empty_like(waves_array)
            result.real = bs_re
            result.imag = bs_im
            return result
        except ImportError:
            if backend == "c++":
                raise RuntimeError(
                    "C++ CUDA backend requested but _cvdms_backend module "
                    "not found. Build the C++ backend first."
                ) from None
            pass  # Fall through to Python path

    # ---- Python backend path ----
    from abtem.finite_difference import conventional_operator

    # wave_1 = K_0 · psi + 1/(2π) · K_series(psi, V_current)
    #  对应 calK_forward_back with current slice potential
    #
    # NOTE: _cvdms_inner_k_series uses c₁=1 (forward scattering convention).
    # For BSC, CGS calK_forward_back uses c₁=λ/(2π), which propagates through
    # the cascade to all higher-order terms. We correct post-hoc by scaling
    # the k_series output by λ/(2π) (= 1/(2πK₀)), so that:
    #   K₀·ψ + K₀·(λ/(2π))·k_series = K₀·ψ + k_series/(2π)
    wave_1 = _cvdms_inner_k_series(
        waves_array,
        transmission_function,
        laplace,
        wavelength,
        convergence_threshold=1e-16,  # strict for backscattering
        use_fused_kernel=use_fused_kernel,
        prefactor=prefactor,
        stencil_raw=stencil_raw,
        antialias_inner=antialias_inner,
        aa_kernel=aa_kernel,
    )
    wave_1 = wave_1 / (2.0 * np.pi) + waves_array * K0

    # wave_2 = K_0 · psi + 1/(2π) · K_series(psi, V_next)
    #  对应 calK_forward_back with next slice potential
    wave_2 = _cvdms_inner_k_series(
        waves_array,
        transmission_function_next,
        laplace,
        wavelength,
        convergence_threshold=1e-16,
        use_fused_kernel=use_fused_kernel,
        prefactor=prefactor,
        stencil_raw=stencil_raw,
        antialias_inner=antialias_inner,
        aa_kernel=aa_kernel,
    )
    wave_2 = wave_2 / (2.0 * np.pi) + waves_array * K0

    # Flux-conserving Fresnel reflection:
    #   R = (k_cur - k_next) / (k_cur + k_next)   (Fresnel reflection amplitude)
    #     = (wave_1 - wave_2) / (wave_1 + wave_2)  (pixel-wise, since wave≈k·ψ)
    #
    # The reflected flux fraction is |R|².  The forward transmission
    # amplitude that conserves flux is T = sqrt(1 - |R|²), so that:
    #   |ψ_out|² = T² · |ψ_fwd|² = (1 - |R|²) · |ψ_fwd|² ≤ |ψ_fwd|²
    #
    # This replaces the SBA formula (B = Δk/(2·k_next)) which gives
    # non-unitary |1-B|² > 1 when k_cur > k_next (potential decreasing),
    # and also avoids the 1/k correction expansion used for SBA.
    sum_waves = wave_1 + wave_2
    diff_waves = wave_1 - wave_2
    with np.errstate(divide='ignore', invalid='ignore'):
        R_sq = xp.abs(diff_waves) ** 2 / xp.abs(sum_waves) ** 2
        R_sq = xp.clip(R_sq, 0.0, 1.0)  # numerical safety
        T = xp.sqrt(1.0 - R_sq)
        backscatter = waves_array * (1.0 - T)
    # Pixels where sum_waves ≈ 0 (both potentials negligible): no backscattering
    zero_mask = xp.abs(sum_waves) < xp.finfo(sum_waves.dtype).eps * 10
    if xp.any(zero_mask):
        if xp is np:
            backscatter[zero_mask] = 0.0 + 0.0j
        else:
            backscatter[zero_mask] = xp.zeros(1, dtype=backscatter.dtype)[0]

    return backscatter
