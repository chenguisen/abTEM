"""Coupled-Wave Dynamical Multislice (CVDMS) algorithm module.

This module implements the CVDMS algorithm based on:
J.H. Chen, D. Van Dyck, "Accurate multislice theory for elastic electron
scattering in transmission electron microscopy".

The algorithm is ported from the ImageSimulation_CGS project's C++/CUDA
implementation in main_diffraction_cbed.cu and wave_kernels.cu.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from abtem.core.backend import get_array_module
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.finite_difference import LaplaceOperator, DivergedError, NotConvergedError

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
    include_backscattering: bool = True,
    calculate_backscattered: bool = False,
    fully_corrected: bool = False,
) -> Waves | Sequence[Waves]:
    """
    Performs a single CVDMS (Coupled-Wave Dynamical Multislice) step.

    This implements the CVDMS algorithm ported from ImageSimulation_CGS.

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
        Convergence threshold for Taylor series (default 1e-6).
    order : int, optional
        Operator expansion order (default 1).
    include_backscattering : bool, optional
        Whether to include backscattering coupling (default True).
    calculate_backscattered : bool, optional
        Whether to compute the backscattered wave (default False).
    fully_corrected : bool, optional
        If True, both transmission and propagator are expanded to order.

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

    # Transmission function: sigma * V (per thickness)
    transmission_function = potential_slice.array[0] * sigma / thickness

    # Next slice transmission function for backscattering
    if next_slice is not None and include_backscattering:
        transmission_function_next = (
            next_slice.array[0] * sigma / thickness
        )
    else:
        transmission_function_next = None

    laplace_stencil = laplace.get_stencil(waves, device=waves.device)

    # ---- Step 1: Pure forward scattering ----
    # Reference: calPureForwardScatter in wave_kernels.cu
    pure_forward = _cvdms_forward_scattering(
        waves._array,
        transmission_function,
        laplace_stencil,
        wavelength,
        thickness,
        max_terms,
        convergence_threshold,
        order,
        fully_corrected,
    )

    # ---- Step 2: Backscattering correction ----
    # Reference: transmitSmallProbe_propCVDMS_CGS_BSC and calBSC
    if include_backscattering and next_slice is not None:
        backscatter = _cvdms_backscattering_correction(
            waves._array,
            transmission_function,
            transmission_function_next,
            laplace_stencil,
            wavelength,
            thickness,
            order,
        )

        # Corrected forward wave = pure forward - backscattering
        exit_wave = pure_forward - backscatter

        # ---- Step 3: Backscattered wave propagation ----
        backscattered_wave = None
        if calculate_backscattered:
            backscattered_wave = _cvdms_forward_scattering(
                backscatter,
                transmission_function,
                laplace_stencil,
                wavelength,
                thickness,
                max_terms,
                convergence_threshold,
                order,
                fully_corrected,
            )

        kwargs = waves._copy_kwargs(exclude=("array",))
        exit_waves_obj = waves.__class__(exit_wave, **kwargs)

        if calculate_backscattered:
            backscattered_waves_obj = waves.__class__(
                backscattered_wave, **kwargs
            )
            return exit_waves_obj, backscattered_waves_obj

        return exit_waves_obj

    kwargs = waves._copy_kwargs(exclude=("array",))
    return waves.__class__(pure_forward, **kwargs)


def _cvdms_forward_scattering(
    waves_array: np.ndarray,
    transmission_function: np.ndarray,
    laplace: callable,
    wavelength: float,
    thickness: float,
    max_terms: int,
    convergence_threshold: float,
    order: int,
    fully_corrected: bool,
) -> np.ndarray:
    """
    Pure forward scattering calculation.

    Implements the CVDMS forward scattering operator:
        phi_j = exp(i * K * dz) * phi_{j-1}

    Following the structure of calPureForwardScatter / calK_PureForward
    in wave_kernels.cu.

    The key difference from the standard multislice is that the Taylor
    series for the exponential is computed with explicit convergence
    control over the entire wavefront.
    """
    xp = get_array_module(waves_array)
    waves_out = waves_array.copy()
    accumulated = waves_array.copy()

    K0 = 1.0 / wavelength
    scal_ = 1.0 / waves_array.shape[-1]  # 1/N for FFT scaling
    dz = thickness

    initial_amplitude = xp.abs(waves_out).sum()

    for n_exp_order in range(2, max_terms + 1):
        # Apply the K operator (inner series)
        # Reference: calK_PureForward
        k_result = _apply_k_operator(
            waves_out if n_exp_order == 2 else temp_wave,
            transmission_function,
            laplace,
            scal_,
            wavelength,
            order,
            fully_corrected,
        )

        temp_wave = k_result

        # Scale: multiply by i * dz / n_exp_order
        # Reference: multiplyComplex_i_CGS(ctemp2D0_d, dz/nExpOrder, waveSize_)
        scale = dz / float(n_exp_order)
        temp_wave = temp_wave * 1.0j * scale

        accumulated += temp_wave
        temp_amplitude = xp.abs(temp_wave).sum()

        if temp_amplitude / initial_amplitude <= convergence_threshold:
            break

        if temp_amplitude > initial_amplitude:
            raise DivergedError(
                "CVDMS forward scattering series diverged"
            )
    else:
        raise NotConvergedError(
            f"CVDMS forward scattering series did not converge to "
            f"{convergence_threshold} in {max_terms} terms"
        )

    return accumulated


def _apply_k_operator(
    waves_array: np.ndarray,
    transmission_function: np.ndarray,
    laplace: callable,
    scal_: float,
    wavelength: float,
    order: int,
    fully_corrected: bool,
) -> np.ndarray:
    """
    Apply the K operator for forward scattering.

    Reference: calK_PureForward in wave_kernels.cu

    The K operator consists of:
    1. Multiply wave by potential (scattering)
    2. Apply Laplacian to wave (propagation)
    3. Sum: K(wave) = laplace(wave) / (4*pi*K0) + V * wave
    """
    xp = get_array_module(waves_array)
    K0 = 1.0 / wavelength

    # Initialize result to zero
    result = xp.zeros_like(waves_array)

    # Start with waves_out as the working copy
    working = waves_array.copy()
    n_sqrt_order = 1

    while True:
        # Step 1: Multiply by potential (scattering)
        # multiplyElementwise(ctemp2D_d, ctemp2D0_d, temp_pot2d_d)
        potential_term = working * transmission_function

        # Step 2: Apply Laplacian (propagation)
        # For order 1: standard laplace
        # Reference: the Fourier/real-space laplace in calK_PureForward
        laplace_term = laplace(working) / (4.0 * np.pi * K0)

        # Step 3: Sum both contributions
        # addArray(ctemp2D0_d, ctemp_wave, ctemp2D_d)
        k_working = laplace_term + potential_term

        # Apply scaling for n_sqrt_order > 1
        # Reference: scaleSqrt = (0.5 - nSqrtOrder + 1)*wavelength/(pi*nSqrtOrder)
        if n_sqrt_order == 1:
            # First order: no scaling
            result += k_working
        else:
            scale_sqrt = (
                (0.5 - n_sqrt_order + 1.0) * wavelength / (np.pi * n_sqrt_order)
            )
            result += k_working * scale_sqrt

        # Prepare for next iteration
        # For higher-order corrections, apply the operator again
        if n_sqrt_order >= order and not fully_corrected:
            break

        working = k_working
        n_sqrt_order += 1

        # Safety limit
        if n_sqrt_order > 100:
            break

    return result


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

    Reference: calBSC in wave_kernels.cu

    Implements the BSC (Back-Scattering-Coefficient) operator:
        BSC = (k_j - k_{j-1}) / (2 * k_j)

    where k_j and k_{j-1} are the multislice operators for the
    current and next slices respectively.
    """
    xp = get_array_module(waves_array)
    K0 = 1.0 / wavelength
    dz = thickness

    # Use the existing full_series from finite_difference for backscattering
    # as it's already well-tested and matches the physics
    from abtem.finite_difference import full_series

    # Wave 1: k_{j-1} * phi (using current slice potential)
    wave_1 = full_series(
        waves_array, laplace, transmission_function, order,
        wavelength, dz,
    )

    # Wave 2: k_j * phi (using next slice potential)
    wave_2 = full_series(
        waves_array, laplace, transmission_function_next, order,
        wavelength, dz,
    )

    # BSC: (wave_2 - wave_1) / (2 * K0)
    # Reference: calBSC lines 6681-6697
    backscatter = (wave_2 - wave_1) / (2.0 * K0)

    # Apply 1/k operator series
    # Reference: calOneDevideK_forward_back and Eq.10 in Micron 190 (2025) 103778
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
