"""Coupled-Wave Dynamical Multislice (CVDMS) algorithm module.

This module implements the CVDMS algorithm based on:
J.H. Chen, D. Van Dyck, "Accurate multislice theory for elastic electron
scattering in transmission electron microscopy".

The algorithm is ported from the ImageSimulation_CGS project's C++/CUDA
implementation in main_diffraction_cbed.cu and wave_kernels.cu, fully
aligned with the original pixel-by-pixel convergence control.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import warnings

import numpy as np

from abtem.core.backend import get_array_module
from abtem.core.energy import energy2sigma, energy2wavelength
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

    laplace_stencil = laplace.get_stencil(waves, device=waves.device)

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
        )

        # Corrected forward wave = pure forward - backscattering
        # 对应: phi_j = (1 - B_{j+1,j}) · ψ_j
        exit_wave = pure_forward - backscatter

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
) -> np.ndarray:
    """
    Pure forward scattering with double Taylor series expansion.

    Outer loop (指数展开):
        phi = Σ (i·dz)ⁿ/n! · K_inner_seriesⁿ(psi_0)

    Inner loop (平方根展开, inside _cvdms_inner_k_series):
        K_inner_series(psi) = Σ cₙ · Kⁿ(psi)

    Both loops use pixel-by-pixel convergence:
    `applyThread` equivalent — count pixels where |term| > cutoff.
    When ALL pixels are below cutoff, the series has converged.

    对应: calPureForwardScatter + calK_PureForward in wave_kernels.cu
    """
    xp = get_array_module(waves_array)
    dz = thickness

    # incidentWave_d = initial wave (first term of the series)
    exit_wave = waves_array.copy()

    # ctemp2D0_d = initial working copy
    working = waves_array.copy()

    # Outer Taylor series: exp(i·K·dz) = Σ (i·dz·K)ⁿ/n!
    for n_exp_order in range(1, max_terms + 1):
        # ---- Inner series: compute K_series(working) ----
        #  对应 calK_PureForward
        #  input:  working  (= ctemp2D0_d)
        #  output: k_series (= ctemp2D1_d)
        k_series = _cvdms_inner_k_series(
            working,
            transmission_function,
            laplace,
            wavelength,
            convergence_threshold,
        )

        # cudaMemcpy(ctemp2D0_d, ctemp2D1_d) — swap reference instead of copy
        # _cvdms_inner_k_series returns a freshly allocated array, so we can
        # reuse its memory by swapping the reference, avoiding an unnecessary copy.
        working = k_series

        # multiplyComplex_i_CGS:  working *= i * dz / n_exp_order
        scale = complex(0, dz / float(n_exp_order))
        working *= scale  # i * dz / n, in-place

        # addArray_1dthread: exit_wave += working
        exit_wave += working

        # ---- Numerical overflow detection ----
        # complex64 (float32) overflows above ~3.4e38. At low voltage + fine
        # sampling, accumulated amplitude across many slices can exceed this.
        if xp.any(xp.isnan(exit_wave)) or xp.any(xp.isinf(exit_wave)):
            # Undo the term that caused overflow
            exit_wave -= working
            n_overflow = int(xp.sum(xp.isinf(exit_wave)) + xp.sum(xp.isnan(exit_wave)))
            warnings.warn(
                f"CVDMS numerical overflow at order {n_exp_order} "
                f"({n_overflow} pixels inf/nan). "
                "The accumulated wave function exceeds complex64 range. "
                "Use a coarser sampling, higher voltage, or thinner sample, "
                "or switch to complex128 precision.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # ---- Pixel-by-pixel convergence check ----
        #  applyThread: count pixels where |working| > cutoff
        n_above = float(xp.sum(xp.abs(working) > convergence_threshold))

        if n_above == 0:
            break

        # Divergence check: truncate series when term grows too large.
        #  对应 fcms_taylor_max_iter() — the original C++ code accepts
        #  partial convergence rather than raising a hard error.
        if n_exp_order > 1 and divergence_ratio > 0:
            ratio = float(xp.abs(working).sum()) / max(float(xp.abs(exit_wave).sum()), 1e-30)
            if ratio > divergence_ratio:
                # Undo the latest term and accept partial sum as best approximation.
                # Earlier terms are valid; only the latest exceeded the stability bound.
                exit_wave -= working
                warnings.warn(
                    f"CVDMS series truncated at order {n_exp_order - 1} "
                    f"(term/accum ratio={ratio:.4f} > divergence_ratio={divergence_ratio}). "
                    "Partial sum may have reduced accuracy. Consider using a smaller "
                    "slice thickness or tighter convergence_threshold.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
    else:
        # Series did not fully converge within max_terms, but return the best
        # approximation with a warning (matches C++ fcms_taylor_max_iter()
        # behavior of accepting partial convergence).
        n_remaining = float(xp.sum(xp.abs(working) > convergence_threshold))
        warnings.warn(
            f"CVDMS forward scattering did not fully converge in {max_terms} terms. "
            f"{int(n_remaining)} pixels above threshold ({convergence_threshold}). "
            "Try increasing max_terms or convergence_threshold.",
            RuntimeWarning,
            stacklevel=2,
        )

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

    对应: calK_PureForward in wave_kernels.cu
    """
    xp = get_array_module(waves_array)
    K0 = 1.0 / wavelength

    # ctemp2D1_d = 0 (initialize series result to zero)
    k_series = xp.zeros_like(waves_array)

    # ctemp2D0_d = input wave (working copy that gets overwritten)
    working = waves_array.copy()

    n_sqrt_order = 1
    prev_n_above = None

    while True:
        # ---- K operator: V * working + laplace(working) / (4πK₀) ----
        # Single scratch buffer replaces potential_term + laplace_term temporaries.
        # laplace(working) allocates a new array → scratch holds ∇²(working).
        scratch = laplace(working)
        scratch /= (4.0 * np.pi * K0)            # in-place: ∇²/(4πK₀)
        working *= transmission_function          # in-place: V * working
        scratch += working                        # in-place: K(working)

        # ---- Numerical stability check ----
        if xp.any(xp.isnan(scratch)) or xp.any(xp.isinf(scratch)):
            # NaN/Inf at high order means accumulated numerical error:
            # truncate the series and return the partial sum (the NaN term is
            # NOT added to k_series). The outer loop's own divergence check
            # catches genuinely unstable parameter regimes.
            # This matches fcms_taylor_max_iter() in the original C++ code.
            break

        # ---- Scaling for higher orders ----
        #  if nSqrtOrder != 1 in calK_PureForward
        if n_sqrt_order == 1:
            k_series += scratch  # first order: no scaling
        else:
            scale = (
                (0.5 - n_sqrt_order + 1.0) * wavelength / (np.pi * n_sqrt_order)
            )
            k_series += scratch * scale

        # ---- Pixel-by-pixel convergence check ----
        #  applyThread: count pixels where |K(working)| > cutoff
        n_above = float(xp.sum(xp.abs(scratch) > convergence_threshold))

        # ---- Divergence / stagnation detection ----
        # If the number of unconverged pixels increases or stagnates, the series
        # has reached its optimal truncation point. Further iterations would not
        # meaningfully improve the sum (oscillating limit cycle).
        # 对应 fcms_taylor_max_iter() in original C++ code.
        if prev_n_above is not None and n_above >= prev_n_above:
            break

        prev_n_above = n_above
        n_sqrt_order += 1

        if n_above == 0:
            break  # fully converged

        if n_sqrt_order > max_inner_iter:
            break  # safety limit

        # ---- Prepare for next iteration: working = K(working) ----
        # Swap references instead of copying. scratch is reallocated by
        # laplace() next iteration, so no aliasing issues.
        working = scratch

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

    from abtem.finite_difference import full_series

    # wave_1 = K_0 · (phi + K_series(phi, V_current))
    #  对应 calK_forward_back with current slice potential
    wave_1 = _cvdms_inner_k_series(
        waves_array,
        transmission_function,
        laplace,
        wavelength,
        convergence_threshold=1e-16,  # strict for backscattering
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
