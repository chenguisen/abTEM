"""Coupled-Wave Dynamical Multislice (CVDMS) algorithm module.

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
    convergence_threshold: float = 1e-6,
    order: int = 1,
    backscattering: bool = False,
    calculate_backscattered: bool = False,
    fully_corrected: bool = False,
    divergence_ratio: float = 5.0,
    check_interval: int = 2,
    antialias: bool = True,
    use_fused_kernel: bool = True,
    backend: str = "auto",
) -> Waves | Sequence[Waves]:
    """
    Performs a single CVDMS (Coupled-Wave Dynamical Multislice) step.

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
        Pixel-wise convergence threshold (default 1e-6).
    order : int, optional
        Operator expansion order (default 1).
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
        convergence_threshold,
        divergence_ratio=divergence_ratio,
        check_interval=check_interval,
        use_fused_kernel=use_fused_kernel,
        prefactor=prefactor,
        stencil_raw=stencil_raw,
        backend=backend,
    )

    # ------------------------------------------------------------------ #
    # Step 2: Backscattering correction
    #  对应 calBSC
    # ------------------------------------------------------------------ #
    if backscattering and next_slice is not None:
        # BSC operator applied to the forward-propagated wave ψ = e^{i·K·dz}·φ,
        # giving the physical backscattered wave at the interface:
        #   B · ψ = (k_{j+1} - k_j) / (2·k_{j+1}) · ψ
        backscatter = _cvdms_backscattering_correction(
            pure_forward,
            transmission_function,
            transmission_function_next,
            laplace_stencil,
            wavelength,
            thickness,
            order,
            use_fused_kernel=use_fused_kernel,
            prefactor=prefactor,
            stencil_raw=stencil_raw,
            backend=backend,
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

        backscattered_waves_obj = waves.__class__(backscatter, **kwargs)

        if calculate_backscattered:
            # Return raw BSC term (no forward propagation). The caller
            # (multislice_and_detect) accumulates these and performs full
            # backward propagation via _back_propagate_backscattered_waves.
            return exit_waves_obj, backscattered_waves_obj

        # Always return tuple in BSC branch: the full-convention path
        # in multislice_and_detect unconditionally unpacks (waves, backscatter).
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
    convergence_threshold: float,
    divergence_ratio: float = 5.0,
    return_diagnostics: bool = False,
    check_interval: int = 2,
    use_fused_kernel: bool = True,
    prefactor: float | None = None,
    stencil_raw: np.ndarray | None = None,
    backend: str = "auto",
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
            from _cvdms_backend import TaylorEngine
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
            engine = TaylorEngine()
            converged, overflow = engine.compute(
                psi_re, psi_im, V,
                nx, ny, wavelength, dz,
                convergence_threshold, max_terms,
                prefactor,
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

    # Pre-allocate: exit_wave starts as copy of input (first series term)
    # working buffer reused across outer iterations
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
            check_interval=check_interval,
            use_fused_kernel=use_fused_kernel,
            prefactor=prefactor,
            stencil_raw=stencil_raw,
        )

        # Reuse k_series memory as working buffer for next iteration
        working = k_series

        # multiplyComplex_i_CGS:  working *= i * dz / n_exp_order
        scale = complex(0, dz / float(n_exp_order))
        working *= scale  # in-place

        # addArray_1dthread: exit_wave += working
        exit_wave += working

        # ---- Batched convergence + stability check ----
        # Check every `check_interval` iterations to reduce D2H syncs.
        # The convergence check is the main source of GPU underutilization
        # (each `int(xp.sum(...))` stalls the GPU pipeline).
        if n_exp_order % check_interval == 0 or n_exp_order == max_terms:
            # Overflow check (inf/nan)
            if xp.any(xp.isinf(exit_wave) | xp.isnan(exit_wave)):
                exit_wave -= working
                warnings.warn(
                    f"CVDMS numerical overflow at order {n_exp_order}. "
                    f"The accumulated wave function exceeds complex64 range. "
                    f"Use a coarser sampling, higher voltage, or thinner sample, "
                    f"or switch to complex128 precision.",
                    RuntimeWarning, stacklevel=2,
                )
                overflow_detected = True
                break

            n_above = int(xp.sum(xp.abs(working) > convergence_threshold))

            if n_exp_order > 1 and divergence_ratio > 0:
                sum_working = float(xp.abs(working).sum())
                sum_exit = float(xp.abs(exit_wave).sum())
            else:
                sum_working = sum_exit = 0.0

            # Record diagnostics
            if return_diagnostics:
                diag_n_above.append((n_exp_order, n_above))

            if n_above == 0:
                break

            # Divergence check
            if n_exp_order > 1 and divergence_ratio > 0:
                ratio = sum_working / max(sum_exit, 1e-30)
                if return_diagnostics:
                    diag_ratios.append((n_exp_order, ratio))
                if ratio > divergence_ratio:
                    exit_wave -= working
                    divergence_truncated = True
                    warnings.warn(
                        f"CVDMS series truncated at order {n_exp_order - 1} "
                        f"(term/accum ratio={ratio:.4f} > divergence_ratio={divergence_ratio}). "
                        f"Partial sum may have reduced accuracy.",
                        RuntimeWarning, stacklevel=2,
                    )
                    break
    else:
        n_remaining = int(xp.sum(xp.abs(working) > convergence_threshold))
        warnings.warn(
            f"CVDMS forward scattering did not fully converge in {max_terms} terms. "
            f"{n_remaining} pixels above threshold ({convergence_threshold}). "
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
    order: int,
    use_fused_kernel: bool = True,
    prefactor: float | None = None,
    stencil_raw: np.ndarray | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """
    Calculate backscattering correction.

    对应: calBSC in wave_kernels.cu

    BSC operator:
        wave_1 = k_{j-1} * phi   (current slice potential)
        wave_2 = k_j * phi       (next slice potential)
        backscatter = (1 / (2*K₀)) * (wave_2 - wave_1) · (1 + 1/k_correction)

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
            from _cvdms_backend import BSCEngine

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
            engine = BSCEngine()
            engine.compute(
                psi_re, psi_im, V_cur, V_next, bs_re, bs_im,
                nx, ny, wavelength, dz, order,
                convergence_threshold=1e-16,
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
    from abtem.finite_difference import full_series

    # wave_1 = K_0 · (phi + K_series(phi, V_current))
    #  对应 calK_forward_back with current slice potential
    wave_1 = _cvdms_inner_k_series(
        waves_array,
        transmission_function,
        laplace,
        wavelength,
        convergence_threshold=1e-16,  # strict for backscattering
        use_fused_kernel=use_fused_kernel,
        prefactor=prefactor,
        stencil_raw=stencil_raw,
    )
    wave_1 = (waves_array + wave_1) * K0

    # wave_2 = K_0 · (phi + K_series(phi, V_next))
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
    )
    wave_2 = (waves_array + wave_2) * K0

    # backscatter = wave_2 - wave_1 (reuse wave_2's memory;
    # wave_1 and wave_2 are not needed after this point)
    #  对应 substractArray(incidentWave, exitwave_2_d, exitwave_1_d)
    backscatter = wave_2
    backscatter -= wave_1

    # 1/k correction series
    #  对应 calOneDevideK_forward_back
    #  Use full_series for the 1/k operator (well-tested in finite_difference)
    prefactors = [1.0]
    for i in range(1, order + 1):
        prefactors.append(prefactors[-1] * (1 - 2 * i) / (2 * i))
    for i in range(len(prefactors)):
        prefactors[i] = prefactors[i] / (1.0j * dz) / (np.pi * K0) ** i

    backscatter *= (
        1.0
        / (2.0 * K0)
        * (
            1.0
            + full_series(
                waves_array,
                laplace,
                transmission_function_next,
                order,
                wavelength,
                dz,
                override_prefactor=prefactors,
            )
        )
    )

    return backscatter
