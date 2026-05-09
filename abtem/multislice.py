"""Module for running the multislice algorithm."""

from __future__ import annotations

import copy
from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypeGuard, cast

import numpy as np
from ase import Atoms

from abtem.antialias import AntialiasAperture, antialias_aperture
from abtem.core import config
from abtem.core.axes import AxisMetadata
from abtem.core.backend import get_array_module
from abtem.core.chunks import Chunks, ValidatedChunks, validate_chunks
from abtem.core.complex import complex_exponential
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import energy2wavelength
from abtem.core.ensemble import _wrap_with_array, unpack_blockwise_args
from abtem.core.fft import CachedFFTWConvolution, fft2_convolve
from abtem.core.grid import spatial_frequencies
from abtem.core.utils import expand_dims_to_broadcast
from abtem.detectors import BaseDetector, WavesDetector, validate_detectors
from abtem.finite_difference import LaplaceOperator
from abtem.finite_difference import multislice_step as realspace_multislice_step
from abtem.inelastic.core_loss import TransitionPotential, TransitionPotentialArray
from abtem.inelastic.plasmons import _update_plasmon_axes
from abtem.measurements import BaseMeasurements
from abtem.potentials.iam import (
    BasePotential,
    PotentialArray,
    TransmissionFunction,
    validate_potential,
)
from abtem.slicing import SliceIndexedAtoms
from abtem.tilt import _get_tilt_axes
from abtem.transform import WavesTransform

if TYPE_CHECKING:
    from abtem.waves import Waves


def _fresnel_propagator_array(
    thickness: float,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    energy: float,
    device: str,
    order: int = 1,
):
    if order > 2:
        raise ValueError(
            """
            Only orders 1 and 2 are supported in Fourier space.
            For higher orders, use the realspace multislice instead.
            """
        )

    xp = get_array_module(device)
    wavelength = energy2wavelength(energy)
    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)
    kx, ky = kx[:, None], ky[None]

    f = complex_exponential(
        -(kx**2) * np.pi * thickness * wavelength
    ) * complex_exponential(-(ky**2) * np.pi * thickness * wavelength)

    # Propagator corrected in Fourier-space, only valid for order=2
    # Eq. (4) from Microscopy and Microanalysis (2020), 26, 1147-1157
    if order == 2:
        f = f * complex_exponential(
            (-np.pi * thickness * wavelength**3) / 4.0 * (kx**4 + ky**4)
        )
    return f


def _apply_tilt_to_fresnel_propagator_array(
    array: np.ndarray,
    sampling: tuple[float, float],
    thickness: float,
    tilt: tuple[float, float] | tuple[tuple[float, float], ...] | np.ndarray,
):
    xp = get_array_module(array)
    tilt = cast(np.ndarray, xp.array(tilt))

    squeeze = False
    if tilt.shape == (2,):
        squeeze = True
        tilt = tilt[None]

    kx, ky = spatial_frequencies(array.shape[-2:], sampling, xp=xp)
    kx, ky = kx[None, :, None], ky[None, None]

    tilt = complex_exponential(
        -kx * xp.tan(tilt[:, 0, None, None] / 1e3) * thickness * 2 * np.pi
    ) * complex_exponential(
        -ky * xp.tan(tilt[:, 1, None, None] / 1e3) * thickness * 2 * np.pi
    )

    tilt, array = expand_dims_to_broadcast(tilt, array, match_dims=((-2, -1), (-2, -1)))

    array = tilt * array

    if squeeze:
        array = array[0]

    return array


class FresnelPropagator:
    """
    The Fresnel propagator is used for propagating wave functions using the near-field
    approximation (Fresnel diffraction).
    """

    def __init__(self):
        self._array = None
        self._key = None
        self._cached_fftw_convolution = CachedFFTWConvolution()

    def get_array(self, waves: Waves, thickness: float, order: int = 1) -> np.ndarray:
        """
        Get the Fresnel propagator as an array for the given wave functions and
        thickness.

        Parameters
        ----------
        waves : Waves
            The wave functions to propagate.
        thickness : float
            Distance in free space to propagate [Å].

        Returns
        -------
        array : np.ndarray
            The Fresnel propagator as an array.
        """
        key: tuple[Any, ...] = (
            waves._valid_gpts,
            waves._valid_sampling,
            thickness,
            waves.base_tilt,
            waves._valid_energy,
            waves.device,
        )

        tilt_axes_metadata = _get_tilt_axes(waves)
        if len(tilt_axes_metadata) > 0:
            key = key + copy.deepcopy(tilt_axes_metadata)

        if key == self._key:
            return self._array

        self._array = self._calculate_array(waves, thickness, order=order)
        self._key = key

        return self._array

    @staticmethod
    def _calculate_array(waves: Waves, thickness: float, order: int = 1) -> np.ndarray:
        array = _fresnel_propagator_array(
            thickness=thickness,
            gpts=waves._valid_gpts,
            sampling=waves._valid_sampling,
            energy=waves._valid_energy,
            device=waves.device,
            order=order,
        )

        array *= antialias_aperture(
            waves._valid_gpts,
            waves._valid_sampling,
            get_array_module(waves.device),
        )

        if waves.base_tilt != (0.0, 0.0):
            array = _apply_tilt_to_fresnel_propagator_array(
                array,
                sampling=waves._valid_sampling,
                thickness=thickness,
                tilt=waves.base_tilt,
            )

        xp = get_array_module(waves.device)

        tilt_axes = _get_tilt_axes(waves)
        if not tilt_axes:
            return array

        for axis in reversed(waves.ensemble_axes_metadata):
            if hasattr(axis, "tilt"):
                tilt = xp.asarray(axis.tilt)
                array = _apply_tilt_to_fresnel_propagator_array(
                    array,
                    sampling=waves._valid_sampling,
                    tilt=tilt,
                    thickness=thickness,
                )

            else:
                array = array[..., None, :, :]

        return array

    def propagate(
        self, waves: Waves, thickness: float, in_place: bool = False, order: int = 1
    ) -> Waves:
        """
        Propagate wave functions through free space.

        Parameters
        ----------
        waves : Waves
            The wave functions to propagate.
        thickness : float
            Distance in free space to propagate.
        in_place : bool
            If True, the waves are overwritten.

        Returns
        -------
        propagated_wave_functions : Waves
            Propagated wave functions.
        """
        kernel = self.get_array(waves, thickness, order=order)
        if (config.get("fft") == "fftw") and isinstance(waves._array, np.ndarray):
            array = self._cached_fftw_convolution(
                waves._array, kernel, overwrite_x=in_place
            )
        else:
            array = fft2_convolve(waves._array, kernel, overwrite_x=in_place)

        if in_place:
            waves._array = array
        else:
            kwargs = waves._copy_kwargs(exclude=("array",))
            waves = waves.__class__(array, **kwargs)

        return waves


def allocate_measurement(
    waves: Waves,
    detector: BaseDetector,
    extra_ensemble_axes_shape: tuple[int, ...],
    extra_ensemble_axes_metadata: list[AxisMetadata],
) -> BaseMeasurements | Waves:
    """
    Allocate a measurement matching the given wave functions and detector.

    Parameters
    ----------
    waves : BaseWaves
        The wave functions to derive the allocated measurement from.
    detector : BaseDetector
        The detector to derive the allocated measurement from.
    extra_ensemble_axes_shape : tuple of int, optional
        The shape of additional ensemble axes not in the waves.
    extra_ensemble_axes_metadata : list of AxisMetadata
        The axes metadata of additional ensemble axes not in the waves.

    Returns
    -------
    allocated_measurement : BaseMeasurements or Waves
        The allocated measurement
    """
    xp = get_array_module(detector._out_meta(waves)[0])

    measurement_type = detector._out_type(waves)[0]

    axes_metadata = detector._out_axes_metadata(waves)[0]

    shape = detector._out_shape(waves)[0]
    #
    if extra_ensemble_axes_shape is not None:
        assert len(extra_ensemble_axes_shape) == len(extra_ensemble_axes_shape)
        shape = extra_ensemble_axes_shape + shape
        axes_metadata = extra_ensemble_axes_metadata + axes_metadata

    metadata = detector._out_metadata(waves)[0]

    array = xp.zeros(shape, dtype=detector._out_dtype(waves)[0])

    out_measurement = measurement_type.from_array_and_metadata(
        array=array, axes_metadata=axes_metadata, metadata=metadata
    )

    return out_measurement


def _potential_ensemble_shape_and_metadata(
    potential: BasePotential,
) -> tuple[tuple[int, ...], list[AxisMetadata]]:
    extra_ensemble_axes_shape = potential.ensemble_shape
    extra_ensemble_axes_metadata = potential.ensemble_axes_metadata

    if len(potential.exit_planes) > 1:
        extra_ensemble_axes_shape = (
            *extra_ensemble_axes_shape,
            len(potential.exit_planes),
        )
        extra_ensemble_axes_metadata = [
            *extra_ensemble_axes_metadata,
            potential._get_exit_planes_axes_metadata(),
        ]

    return extra_ensemble_axes_shape, extra_ensemble_axes_metadata


def allocate_multislice_measurements(
    waves: Waves,
    detectors: list[BaseDetector],
    extra_ensemble_axes_shape: tuple[int, ...],
    extra_ensemble_axes_metadata: list[AxisMetadata],
) -> list[BaseMeasurements | Waves]:
    """
    Allocate the multislice measurements that would be produced by detecting the given
    set of wave functions with the given set of detectors.

    Parameters
    ----------
    waves : Waves
        The waves to derive the allocated measurement from.
    detectors : list of BaseDetector
        The detectors to derive the allocated measurement from.
    extra_ensemble_axes_shape : tuple of int, optional
        The shape of additional ensemble axes not in the waves.
    extra_ensemble_axes_metadata : list of AxisMetadata
        The axes metadata of additional ensemble axes not in the waves.

    Returns
    -------
    allocated_measurements : list
        List of allocated to measurements.
    """

    measurements = []
    for detector in detectors:
        measurements.append(
            allocate_measurement(
                waves, detector, extra_ensemble_axes_shape, extra_ensemble_axes_metadata
            )
        )

    return measurements


def conventional_multislice_step(
    waves: Waves,
    potential_slice: PotentialArray | TransmissionFunction,
    propagator: FresnelPropagator,
    antialias_aperture: AntialiasAperture,
    conjugate: bool = False,
    transpose: bool = False,
    order: int = 1,
) -> Waves:
    """
    Calculate one step of the multislice algorithm for the given batch of wave functions
    through a given potential slice.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a :class:`.Waves` object.
    potential_slice : PotentialArray or TransmissionFunction
        A potential slice as a :class:`.PotentialArray` or
        :class:`.TransmissionFunction`.
    propagator : FresnelPropagator, optional
        A Fresnel propagator type matching the wave functions. The main reason for using
        this argument is to reuse a previously calculated propagator. If not provided a
        new propagator is created.
    antialias_aperture : AntialiasAperture, optional
        An antialias aperture type matching the wave functions. The main reason for
        using this argument is to reuse a previously calculated antialias aperture.
        If not provided a new antialias aperture is created.
    conjugate : bool, optional
        If True, use the conjugate of the transmission function (default is False).
    transpose : bool, optional
        If True, reverse the order of propagation and transmission (default is False).

    Returns
    -------
    forward_stepped_waves : Waves
        Wave functions propagated and transmitted through the potential slice.
    """
    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)

    if isinstance(potential_slice, TransmissionFunction):
        transmission_function = potential_slice

    else:
        transmission_function = potential_slice.transmission_function(
            energy=waves._valid_energy
        )
        transmission_function = antialias_aperture.bandlimit(
            transmission_function, in_place=False
        )

    thickness = transmission_function.slice_thickness[0]

    if conjugate:
        thickness = -thickness

    if transpose:
        waves = propagator.propagate(
            waves, thickness=thickness, in_place=True, order=order
        )
        waves = transmission_function.transmit(waves, conjugate=conjugate)
    else:
        waves = transmission_function.transmit(waves, conjugate=conjugate)
        waves = propagator.propagate(
            waves, thickness=thickness, in_place=True, order=order
        )

    return waves


def _update_measurements(
    waves: Waves,
    detectors: list[BaseDetector],
    measurements: list[BaseMeasurements | Waves],
    measurement_index: tuple[int, ...] = (0,),
    additive: bool = False,
) -> None:
    assert len(detectors) == len(measurements)

    for i, detector in enumerate(detectors):
        new_measurement = detector.detect(waves)

        if additive:
            measurements[i].array[measurement_index] += new_measurement.array
        else:
            measurements[i].array[measurement_index] = new_measurement.array
    return


def _validate_potential_ensemble_indices(
    potential_index: int | tuple[int, ...],
    exit_plane_index: int | tuple[int, ...],
    potential: BasePotential,
) -> tuple[int, ...]:
    if not potential.ensemble_shape:
        potential_index = ()
    elif not isinstance(potential_index, tuple):
        potential_index = (potential_index,)

    if len(potential.exit_planes) == 1:
        exit_plane_index = ()
    elif not isinstance(exit_plane_index, tuple):
        exit_plane_index = (exit_plane_index,)

    measurement_indices = potential_index + exit_plane_index

    return measurement_indices


def _generate_potential_configurations(potential):
    for potential_index, _, potential_configuration in potential.generate_blocks():
        potential_configuration = potential_configuration.item()

        if len(potential.ensemble_shape):
            potential_index = np.unravel_index(
                potential_index, potential.ensemble_shape
            )

        yield potential_index, potential_configuration


def lookahead(iterable):
    """
    Generator that yields (current, next) items from an iterable.
    The last item is yielded as (last, None).
    """
    it = iter(iterable)
    try:
        current_item = next(it)
    except StopIteration:
        return

    for next_item in it:
        yield current_item, next_item
        current_item = next_item

    yield current_item, None


@dataclass(frozen=True)
class FourierMultislice:
    """
    Multislice algorithm computed fast in Fourier space.

    Parameters
    ----------
    order : int, optional
        Propagator order, one of 1 or 2 (default 1)
    expansion_scope: str
        Specified for compatibility. Must be "propagator" (default "propagator")
    conjugate : bool, optional
        If True, use the conjugate of the transmission function (default is False)
    transpose : bool, optional
        If True, reverse the order of propagation and transmission (default is False)
    """

    order: Literal[1, 2] = 1
    expansion_scope: Literal["propagator"] = "propagator"
    conjugate: bool = False
    transpose: bool = False


@dataclass(frozen=True)
class RealSpaceMultislice:
    """
    Multislice algorithm computed in real-space.

    Parameters
    ----------
    order : int, optional
        Propagator and/or transmission operator order (default 1)
    expansion_scope: str
        If "propagator" (default) only the propagator operator is expanded to order
        If "full" both the propagator and transmission operators are expanded to order
    derivative_accuracy : int, optional
        Finite-difference stencil accuracy for Laplace operator (default 8,
        corresponding to a 9-point stencil, matching the C++ "9点法").
        Must be a positive even integer.
    laplace_method : str, optional
        Laplacian computation method: "finite-difference" or "fft" (default "finite-difference").
        "fft" uses FFT in reciprocal space, matching the approach in ImageSimulation_CGS.
    max_terms: int, optional
        Max terms in exponent Taylor series expansion (default 80)
    """

    order: int = 1
    expansion_scope: Literal["propagator", "full"] = "propagator"
    derivative_accuracy: int = 8
    laplace_method: str = "finite-difference"
    max_terms: int = 80


@dataclass(frozen=True)
class CVDMSMultislice:
    """
    Coupled-Wave Dynamical Multislice (CVDMS) algorithm.

    Based on the CVDMS theory by Jiang Hua Chen and Dirk Van Dyck
    (Accurate multislice theory for elastic electron scattering in
    transmission electron microscopy). Accounts for backscattering
    coupling between slices, yielding higher accuracy than conventional
    multislice methods.

    Parameters
    ----------
    order : int, optional
        Taylor expansion order (default 1).
    max_terms : int, optional
        Maximum terms in the Taylor series expansion (default 50).
    convergence_threshold : float, optional
        Threshold for Taylor series convergence (default 1e-6).
    backscattering : bool, optional
        If True, enable inter-slice backscattering coupling (default False).
        Controls both the calling convention (passing next_slice, tuple return)
        and the physical BSC operator. The forward wave is corrected for
        backscattering loss at each slice interface:
        ψ_corrected = ψ_forward - BSC(ψ_forward).
        If False, only forward propagation is performed; no backscattering
        coupling between slices.
    calculate_backscattered : bool, optional
        If True, separately track the backscattered wave at each interface and
        perform full backward propagation to the sample surface (default False).
        Requires ``backscattering=True``.
    back_prop_mode : str, optional
        Back-propagation direction for the backscattered wave (default "conj").
        "conj": use conj-trick conj(forward(conj(ψ))) = exp(-i·K·dz)·ψ,
                physically correct backward propagation (time-reversal).
        "forward": use forward propagator forward(ψ) = exp(+i·K·dz)·ψ,
                matching ImageSimulation_CGS code behavior.
        Set to "forward" to cross-validate against CGS results.
        Only used when ``calculate_backscattered=True``.
    derivative_accuracy : int, optional
        If True, separately track the backscattered wave at each interface and
        perform full backward propagation to the sample surface (default False).
        Requires ``backscattering=True``.
    derivative_accuracy : int, optional
        Accuracy for the Laplacian operator (default 8, corresponding to a
        9-point stencil, matching the C++ "9点法").
        For method="finite-difference": centered finite-difference stencil accuracy,
        must be a positive even integer.
        For method="fft": ignored (using exact FFT-based Laplacian).
    laplace_method : str, optional
        Laplacian computation method: "finite-difference" or "fft" (default "finite-difference").
        "fft" uses FFT in reciprocal space, corresponding to
        ImageSimulation_CGS wave_kernels.cu:6002 (MultiCoefInReciprocalSpace).
    divergence_ratio : float, optional
        Ratio threshold for Taylor series divergence detection (default 5.0).
        When the latest term's total amplitude exceeds this multiple of the
        accumulated sum, the series is truncated and a warning is issued.
        Set to 0 to disable soft truncation (raises DivergedError instead).
    check_interval : int, optional
        Convergence check interval for GPU utilization optimization (default 2).
        Controls how often the GPU is synchronized to check Taylor series
        convergence. Higher values reduce D2H synchronization overhead at the
        cost of at most (check_interval - 1) extra iterations.
        - check_interval=1: check every iteration (original behavior)
        - check_interval=2 (default): halved sync frequency
        - check_interval=3: further reduced, but may overshoot convergence
    antialias : bool, optional
        If True, apply antialias low-pass filter to the potential (default True).
        Enables a fair comparison with Fourier multislice by bandlimiting the
        projected potential to 2/3 Nyquist before the K-operator expansion,
        matching the Fourier antialias aperture treatment.
    use_fused_kernel : bool, optional
        If True, use fused CUDA kernel for inner K-series (default True).
        Replaces the Python loop with a single kernel launch, eliminating
        Python overhead and intermediate global memory traffic.
        Only available when device="gpu" and cupy is installed.
    backend : str, optional
        Backend selection for the K-operator computation (default "auto").
        "auto": try C++ CUDA backend first if available, fall through to
                CuPy fused kernel or Python loops.
        "c++":  force C++ CUDA backend; raises RuntimeError if unavailable.
        "cupy": skip C++ CUDA backend, use CuPy fused kernel or Python loops.
    """

    order: int = 1
    max_terms: int = 50
    max_inner: int = 100
    convergence_threshold: float = 1e-7
    backscattering: bool = False
    calculate_backscattered: bool = False
    back_prop_mode: str = "conj"
    derivative_accuracy: int = 8
    laplace_method: str = "finite-difference"
    divergence_ratio: float = 5.0
    check_interval: int = 2
    antialias: bool = True
    use_fused_kernel: bool = True
    backend: str = "auto"


def multislice_and_detect(
    waves: Waves,
    potential: BasePotential,
    detectors: Optional[list[BaseDetector]] = None,
    algorithm: FourierMultislice | RealSpaceMultislice | CVDMSMultislice = FourierMultislice(),
    return_backscattered: bool = False,
    pbar: bool = False,
) -> BaseMeasurements | Waves | list[BaseMeasurements | Waves]:
    """
    Calculate the full multislice algorithm for the given batch of wave functions
    through a given potential, detecting at each of the exit planes specified in the
    potential.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a :class:`.Waves` object.
    potential : BasePotential
        A potential as :class:`.BasePotential` object.
    detectors : (list of) BaseDetector, optional
        A detector or a list of detectors defining how the wave functions should be
        converted to measurements after running the multislice algorithm.
    algorithm: FourierMultislice, RealSpaceMultislice or CVDMSMultislice, optional
        Algorithm used for multislice operator (default is FourierMultislice())
    return_backscattered: bool, optional
        If True and algorithm is CVDMSMultislice with backscattering=True, the
        backscattered components are also returned. Requires potential exit_planes.

    """
    waves = waves.ensure_real_space()
    detectors = validate_detectors(detectors)
    waves = waves.copy()

    def _algorithm_uses_backscattering(alg) -> bool:
        """Check whether the algorithm enables inter-slice backscattering coupling."""
        if isinstance(alg, CVDMSMultislice):
            return alg.backscattering
        return alg.expansion_scope == "full"

    if return_backscattered:
        if not _algorithm_uses_backscattering(algorithm):
            raise ValueError(
                "Backscattering contributions require backscattering=True "
                "(CVDMSMultislice) or expansion_scope='full' (RealSpaceMultislice)."
            )
        if potential.num_exit_planes == 1:
            raise ValueError(
                "Backscattering contributions require potential.exit_planes."
            )

        # Keep BSC at full resolution by using a WavesDetector for the BSC
        # channel, avoiding shape mismatches from detector cropping.
        if len(detectors) == 0:
            # No user detectors: need one for forward exit wave + one for BSC
            detectors = [WavesDetector(), WavesDetector()]
        else:
            detectors = list(detectors) + [WavesDetector()]

    if isinstance(algorithm, FourierMultislice):
        antialias_aperture = AntialiasAperture()
        propagator = FresnelPropagator()

        def multislice_step(waves, potential_slice, next_slice=None):
            return conventional_multislice_step(
                waves,
                potential_slice=potential_slice,
                antialias_aperture=antialias_aperture,
                propagator=propagator,
                conjugate=algorithm.conjugate,
                transpose=algorithm.transpose,
                order=algorithm.order,
            )

    elif isinstance(algorithm, RealSpaceMultislice):
        laplace_operator = LaplaceOperator(
            algorithm.derivative_accuracy, method=algorithm.laplace_method
        )

        def multislice_step(waves, potential_slice, next_slice=None):
            return realspace_multislice_step(
                waves,
                potential_slice=potential_slice,
                next_slice=next_slice,
                laplace=laplace_operator,
                max_terms=algorithm.max_terms,
                order=algorithm.order,
                fully_corrected=algorithm.expansion_scope == "full",
            )

    else:
        from .cvdms import cvdms_multislice_step as cvdms_step

        def multislice_step(waves, potential_slice, next_slice=None):
            return cvdms_step(
                waves,
                potential_slice=potential_slice,
                next_slice=next_slice,
                laplace=LaplaceOperator(
                    algorithm.derivative_accuracy, method=algorithm.laplace_method
                ),
                max_terms=algorithm.max_terms,
                max_inner=algorithm.max_inner,
                convergence_threshold=algorithm.convergence_threshold,
                order=algorithm.order,
                backscattering=algorithm.backscattering,
                calculate_backscattered=algorithm.calculate_backscattered,
                fully_corrected=algorithm.backscattering,
                divergence_ratio=algorithm.divergence_ratio,
                check_interval=algorithm.check_interval,
                antialias=algorithm.antialias,
                use_fused_kernel=algorithm.use_fused_kernel,
                backend=algorithm.backend,
            )

    (
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    ) = _potential_ensemble_shape_and_metadata(potential)

    if sum(extra_ensemble_axes_shape) == 1:
        measurements = None
    else:
        measurements = allocate_multislice_measurements(
            waves,
            detectors,
            extra_ensemble_axes_shape,
            extra_ensemble_axes_metadata,
        )

    n_waves = np.prod(waves.shape[:-2])
    n_slices = n_waves * potential.num_slices * potential.num_configurations

    tqdm_pbar = TqdmWrapper(
        enabled=pbar, total=int(n_slices), leave=False, desc="multislice"
    )

    # Per-slice BSC storage for running accumulation back-propagation.
    # Structure: per_slice_bsc_data[config_idx][slice_idx] = BSC array.
    per_slice_bsc_data = None
    if _algorithm_uses_backscattering(algorithm) and return_backscattered:
        per_slice_bsc_data = []

    # Save the initial waves to reset between frozen phonon configurations
    initial_waves = waves.copy()

    for potential_index, potential_configuration in _generate_potential_configurations(
        potential
    ):
        waves = initial_waves.copy()

        exit_plane_index = 0

        # Handle entrance plane detection (before first slice)
        if potential.exit_planes[0] == -1:
            measurement_index = _validate_potential_ensemble_indices(
                potential_index, exit_plane_index, potential
            )

            if measurements is not None:
                _update_measurements(waves, detectors, measurements, measurement_index)

            exit_plane_index += 1

        depth = 0.0

        # Per-config per-slice BSC list
        config_bsc = []

        for potential_slice, next_slice in lookahead(
            potential_configuration.generate_slices()
        ):
            if _algorithm_uses_backscattering(algorithm):
                waves, backscatter_waves = multislice_step(
                    waves, potential_slice, next_slice=next_slice
                )
                if per_slice_bsc_data is not None:
                    config_bsc.append(backscatter_waves.array.copy())
            else:
                waves = multislice_step(waves, potential_slice, next_slice=None)
            tqdm_pbar.update_if_exists(int(n_waves))

            depth += potential_slice.axes_metadata[0].values[0]

            _update_plasmon_axes(waves, depth)

            if potential_slice.exit_planes:
                measurement_index = _validate_potential_ensemble_indices(
                    potential_index, exit_plane_index, potential
                )

                if measurements is not None:
                    if _algorithm_uses_backscattering(algorithm) and return_backscattered:
                        _update_measurements(
                            waves, detectors[:-1], measurements[:-1], measurement_index
                        )
                        _update_measurements(
                            backscatter_waves,
                            detectors[-1:],
                            measurements[-1:],
                            measurement_index,
                        )
                    else:
                        _update_measurements(
                            waves, detectors, measurements, measurement_index
                        )
                exit_plane_index += 1

        # After all slices for this config: save per-slice BSC list
        if per_slice_bsc_data is not None:
            per_slice_bsc_data.append(config_bsc)

    # Handle final output if not using intermediate measurements
    if measurements is None:
        measurements = [
            detector.detect(waves)[(None,) * len(potential.ensemble_shape)]
            for detector in detectors
        ]

    elif return_backscattered:
        # Back-propagate per-slice BSC through the specimen. The BSC
        # channel uses WavesDetector (added above), so measurements[-1]
        # is a Waves object with raw complex wavefunction data at the
        # native grid resolution.
        _back_propagate_backscattered_waves(
            measurements[-1], potential, multislice_step,
            per_slice_bsc_arrays=per_slice_bsc_data,
            back_prop_mode=getattr(algorithm, 'back_prop_mode', 'conj'),
        )
    tqdm_pbar.close_if_exists()

    return measurements


def _aggregate_slices_by_exit_planes(potential_slices, exit_planes):
    """
    Group potential slices between exit_planes, summing their thicknesses.

    Parameters
    ----------
    potential_slices : list of PotentialSlice
        Original slices along the beam direction.
    exit_planes : list of int
        Indices of exit planes (first can be -1 for entrance plane).

    Returns
    -------
    effective_slices : list of PotentialSlice
        Aggregated slices with summed potential arrays and summed thicknesses.
    """

    effective_slices = []

    for i in range(0, len(exit_planes) - 1):
        idx_start = exit_planes[i] + 1  # slice after previous exit plane
        idx_end = exit_planes[i + 1] + 1  # include this exit plane

        # Aggregate slices in this block
        combined_slice = potential_slices[idx_start].copy()
        thickness = combined_slice.slice_thickness[0]
        # Add remaining slices in the block
        for in_bw_slice in potential_slices[idx_start + 1 : idx_end]:
            combined_slice += in_bw_slice
            thickness += in_bw_slice.slice_thickness[0]
            combined_slice._slice_thickness = (thickness,)
            combined_slice._slice_limits = [(0, thickness)]

        effective_slices.append(combined_slice)

    return effective_slices


def _back_propagate_backscattered_waves(
    backscattered_waves: Waves,
    potential: BasePotential,
    multislice_step: Callable,
    per_slice_bsc_arrays: list | None = None,
    back_prop_mode: str = "conj",
) -> Waves:
    """
    For each slice in the multislice step, a small part of the wave get backscattered.
    This function runs the multislice in reverse for each backscattered wave summing
    them for a final backscattered wave result.

    When per_slice_bsc_arrays is provided, uses running accumulation over ALL
    original slices (not just exit planes) — each slice's BSC is back-propagated
    through all overlying slices to the entrance surface. This ensures physical
    correctness matching ImageSimulation_CGS.

    Supports potential ensemble dimensions (e.g., frozen phonons) by recursively
    processing each ensemble member independently.
    """

    xp = get_array_module(backscattered_waves.device)
    potential_slices = [
        slice
        for _, config in _generate_potential_configurations(potential)
        for slice in config.generate_slices()
    ]

    exit_planes = potential.exit_planes

    return _back_propagate_bsc_impl(
        backscattered_waves, potential_slices, exit_planes, multislice_step,
        per_slice_bsc_arrays=per_slice_bsc_arrays,
        back_prop_mode=back_prop_mode,
    )


def _back_propagate_bsc_impl(
    backscattered_waves: Waves,
    potential_slices: list,
    exit_planes: list,
    multislice_step: Callable,
    per_slice_bsc_arrays: list | None = None,
    back_prop_mode: str = "conj",
) -> Waves:
    """
    Internal BSC back-propagation with per-original-slice stepping.

    Split from _back_propagate_backscattered_waves so that the slice list
    and exit planes are computed once and reused across recursive ensemble
    calls, avoiding repeated GPU memory allocations.

    When per_slice_bsc_arrays is provided, uses running accumulation over
    ALL original slices: each slice's BSC is back-propagated through all
    overlying slices to the entrance surface, matching ImageSimulation_CGS.

    Otherwise uses the exit-plane block approach (original behavior).

    Parameters
    ----------
    back_prop_mode : str
        "conj" (default): conj-trick for time-reversed backward propagation.
        "forward": use forward propagator (CGS-compatible mode).
    """

    num_exit_planes = len(exit_planes)

    # ---- Recursive case: handle potential ensemble dims ----
    # The exit_planes axis is the last ensemble axis. If there are additional
    # leading ensemble dimensions (e.g., frozen phonon configs), iterate over
    # them and process each independently.
    if len(backscattered_waves.ensemble_shape) > 1:
        num_configs = len(backscattered_waves)
        slices_per_config = len(potential_slices) // num_configs if num_configs > 0 else 0
        for i in range(num_configs):
            # Use config-specific potential slices for correct BSC back-propagation
            # (frozen phonon potentials differ per config)
            start_sl = i * slices_per_config
            end_sl = (i + 1) * slices_per_config
            config_slices = potential_slices[start_sl:end_sl]

            _back_propagate_bsc_impl(
                backscattered_waves[i],
                config_slices,
                exit_planes,
                multislice_step,
                per_slice_bsc_arrays=(
                    per_slice_bsc_arrays[i]
                    if per_slice_bsc_arrays is not None
                    else None
                ),
                back_prop_mode=back_prop_mode,
            )
        return backscattered_waves

    # ---- Single ensemble case: exit_planes is the only ensemble axis ----
    if len(backscattered_waves) != num_exit_planes:
        raise ValueError(
            f"Shape mismatch: len(backscattered_waves)={len(backscattered_waves)}, "
            f"exit_planes={num_exit_planes}"
        )

    # zero intensity in incoming wave (entrance / top surface)
    backscattered_waves[0]._array[:] = 0

    xp = get_array_module(backscattered_waves.device)
    num_slices = len(potential_slices)

    # ================================================================== #
    # Running accumulation path (per-slice BSC)
    # ================================================================== #
    # Uses running accumulation from bottom to top: each slice's BSC is
    # added to a running accumulator, then back-propagated through that
    # slice via the conj-trick. This guarantees every intermediate slice's
    # BSC reaches the entrance surface — matching ImageSimulation_CGS
    # physical correctness.
    # ================================================================== #
    if per_slice_bsc_arrays is not None:
        # Unwrap: in the non-recursive single-config case,
        # per_slice_bsc_arrays = [[arr_0, arr_1, ...]] (list of 1 config's list).
        # The running accumulation expects a flat list [arr_0, arr_1, ...].
        if (isinstance(per_slice_bsc_arrays, list) and
            len(per_slice_bsc_arrays) == 1 and
            isinstance(per_slice_bsc_arrays[0], list)):
            per_slice_bsc_arrays = per_slice_bsc_arrays[0]

        # ---- C++ CUDA running accumulation ----
        # One engine call, all per-slice steps on the same CUDA stream.
        # per_slice_bsc_arrays come from GPU forward sim (CuPy), so we try
        # the C++ CUDA path regardless of the measurement object's device.
        is_gpu = (
            len(per_slice_bsc_arrays) > 0
            and hasattr(per_slice_bsc_arrays[0], '__cuda_array_interface__')
        )

        if is_gpu:
            try:
                from _cvdms_backend import BSCBackPropEngine
                from abtem.core.energy import energy2sigma, energy2wavelength

                # Get energy from the backscattered_waves metadata (may be
                # Waves or DiffractionPatterns depending on detector)
                if hasattr(backscattered_waves, '_valid_energy'):
                    _energy = backscattered_waves._valid_energy
                else:
                    _energy = backscattered_waves._get_from_metadata("energy")
                wavelength = energy2wavelength(_energy)
                sigma = energy2sigma(_energy)
                dx, dy = backscattered_waves.sampling
                laplace_prefactor = 1.0 / (dx * dy)
                dz = float(potential_slices[0].thickness)

                # Build per-slice BSC re/im lists (float32 CuPy contiguos)
                import cupy as _cupy
                bsc_re_list = []
                bsc_im_list = []
                for arr in per_slice_bsc_arrays:
                    bsc_re_list.append(_cupy.ascontiguousarray(
                        _cupy.real(arr).astype(_cupy.float32)))
                    bsc_im_list.append(_cupy.ascontiguousarray(
                        _cupy.imag(arr).astype(_cupy.float32)))

                # Build transmission functions for ALL original slices
                V_list = []
                for sl in potential_slices:
                    tf = sl.array[0] * sigma / float(sl.thickness)
                    if not hasattr(tf, '__cuda_array_interface__'):
                        tf = _cupy.asarray(tf)
                    V_list.append(_cupy.ascontiguousarray(
                        tf.astype(_cupy.float32)))

                nx, ny = bsc_re_list[0].shape[-2:]

                # Build exit plane output buffers (re/im lists from the
                # backscattered_waves Waves object's existing array storage).
                ep_re_list = []
                ep_im_list = []
                for ep_idx in range(len(backscattered_waves)):
                    ep_arr = backscattered_waves._array[ep_idx]
                    if not hasattr(ep_arr, '__cuda_array_interface__'):
                        ep_arr = _cupy.asarray(ep_arr)
                    ep_re_list.append(_cupy.ascontiguousarray(
                        _cupy.real(ep_arr).astype(_cupy.float32)))
                    ep_im_list.append(_cupy.ascontiguousarray(
                        _cupy.imag(ep_arr).astype(_cupy.float32)))

                # Build exit plane indices list (host ints)
                ep_indices_list = [int(idx) for idx in exit_planes]

                engine = BSCBackPropEngine()
                engine.compute_accumulate(
                    bsc_re_list, bsc_im_list, V_list,
                    ep_re_list, ep_im_list, ep_indices_list,
                    nx, ny, wavelength, dz,
                    convergence_threshold=1e-7,
                    max_terms=50, max_inner=100,
                    laplace_prefactor=laplace_prefactor,
                    accuracy=8,
                    use_conj=(back_prop_mode == "conj"),
                )

                # Read back all exit plane buffers (C++ CUDA wrote in-place
                # to the ep_re/ep_im device buffers).
                for ep_idx in range(len(backscattered_waves)):
                    ep_result = ep_re_list[ep_idx] + 1.0j * ep_im_list[ep_idx]
                    target = backscattered_waves._array[ep_idx]
                    if hasattr(ep_result, 'get') and not hasattr(target, '__cuda_array_interface__'):
                        ep_result = ep_result.get()
                    backscattered_waves._array[ep_idx] = ep_result

                # Negate all exit planes for forward mode: φ₁ᵇ = -Σ ...
                if back_prop_mode == "forward":
                    for ep_idx in range(len(backscattered_waves)):
                        backscattered_waves._array[ep_idx] = -backscattered_waves._array[ep_idx]

                return backscattered_waves
            except ImportError:
                pass  # Fall through to Python path

        # ---- Python running accumulation path ----
        # Running accumulation: O(N) time, O(1) extra storage.
        # Accumulates BSC from bottom to top, saving the accumulated value
        # at each exit plane to produce a physically correct depth profile
        # (BSC increases with depth).
        #
        # per_slice_bsc_arrays may be CuPy (captured from GPU forward sim)
        # while backscattered_waves may be numpy. We convert BSC arrays
        # to numpy for consistent CPU-side accumulation.
        working = backscattered_waves[0].copy()
        working.array[:] = 0

        # Pre-convert all BSC arrays to numpy
        bsc_np = []
        for arr in per_slice_bsc_arrays:
            if hasattr(arr, 'get'):
                arr = arr.get()
            bsc_np.append(arr)

        # Build slice-to-exit-plane mapping (skip EP 0 = entrance surface)
        sl_to_ep = {}
        for k in range(1, num_exit_planes):
            ep_sl = exit_planes[k]
            if 0 <= ep_sl < num_slices:
                sl_to_ep[ep_sl] = k

        # Process all slices bottom-to-top (including last slice, fixing
        # off-by-one vs the original exit-plane-block approach)
        for sl_idx in range(num_slices - 1, -1, -1):
            # Add this slice's BSC to the running accumulator
            working.array += bsc_np[sl_idx]

            # Save accumulated BSC at this exit plane BEFORE conj-trick.
            # At this point work is at the bottom of slice sl_idx, which is
            # the correct physical position for the exit plane.
            if sl_idx in sl_to_ep:
                backscattered_waves._array[sl_to_ep[sl_idx]] = working.array.copy()

            # conj-trick: conj → forward → conj = backward propagation
            # forward mode: forward only (CGS-compatible)
            if back_prop_mode == "conj":
                working.array = np.conj(working.array)

            result = multislice_step(
                working, potential_slices[sl_idx], next_slice=None
            )
            if isinstance(result, tuple):
                working = result[0]
            else:
                working = result

            if back_prop_mode == "conj":
                working.array = np.conj(working.array)

        # Write total BSC to entrance surface (exit plane 0)
        backscattered_waves._array[0] = working.array

        # Negate all exit planes for forward mode: φ₁ᵇ = -Σ ...
        if back_prop_mode == "forward":
            for ep_idx in range(len(backscattered_waves)):
                backscattered_waves._array[ep_idx] = -backscattered_waves._array[ep_idx]

        return backscattered_waves

    # ================================================================== #
    # Original exit-plane block path (per_slice_bsc_arrays is None)
    # ================================================================== #
    #
    # Only back-propagates BSC at exit planes through their respective
    # blocks. Intermediate (non-exit-plane) slice BSC is NOT back-propagated.
    # Retained for backward compatibility.
    # ================================================================== #

    # ---- C++ CUDA backend path (exit-plane blocks) ----
    if (xp.__name__ == "cupy"
        and backscattered_waves.array.dtype == np.complex64
        and back_prop_mode == "conj"):  # C++ engine only supports conj mode
        try:
            from _cvdms_backend import BSCBackPropEngine
            from abtem.core.energy import energy2sigma, energy2wavelength

            wavelength = energy2wavelength(backscattered_waves._valid_energy)
            sigma = energy2sigma(backscattered_waves._valid_energy)
            dx, dy = backscattered_waves.sampling
            laplace_prefactor = 1.0 / (dx * dy)

            # Get uniform slice thickness
            dz = float(potential_slices[0].thickness)

            # Build real/imag lists for BSC waves (float32 device arrays)
            bsc_re_list = []
            bsc_im_list = []
            for w in backscattered_waves:
                arr = w._array
                bsc_re_list.append(xp.ascontiguousarray(
                    xp.real(arr).astype(xp.float32)))
                bsc_im_list.append(xp.ascontiguousarray(
                    xp.imag(arr).astype(xp.float32)))

            # Build transmission function for ALL original slices
            V_list = []
            for sl in potential_slices:
                tf = sl.array[0] * sigma / float(sl.thickness)
                if not hasattr(tf, '__cuda_array_interface__'):
                    tf = xp.asarray(tf)
                V_list.append(xp.ascontiguousarray(tf.astype(xp.float32)))

            # Build exit_plane_indices: block i spans slices
            # [exit_planes[i] + 1 : exit_planes[i+1] + 1]
            ep_indices = [ep + 1 for ep in exit_planes]
            # Sentinel: ensure last block is bounded
            if ep_indices[-1] > len(V_list):
                ep_indices[-1] = len(V_list)

            nx, ny = bsc_re_list[0].shape[-2:]

            engine = BSCBackPropEngine()
            engine.compute(
                bsc_re_list, bsc_im_list, V_list, ep_indices,
                nx, ny, wavelength, dz,
                convergence_threshold=1e-7,
                max_terms=50, max_inner=100,
                laplace_prefactor=laplace_prefactor,
                accuracy=8,
            )

            # Reconstruct complex waves from modified re/im arrays
            for i in range(len(backscattered_waves)):
                backscattered_waves._array[i] = (
                    bsc_re_list[i] + 1.0j * bsc_im_list[i]
                )

            return backscattered_waves
        except ImportError:
            pass  # Fall through to Python path

    # ---- Python exit-plane block path ----
    for i in range(num_exit_planes - 2, -1, -1):
        start = exit_planes[i] + 1        # first original slice after EP i
        end = exit_planes[i + 1] + 1      # last original slice at EP i+1

        # Copy the accumulated BSC at exit plane i+1 (includes contributions
        # from deeper planes that were already back-propagated in earlier
        # iterations of this loop).
        wave = backscattered_waves[i + 1].copy()

        # Back-propagate through each original slice in this block,
        # going from bottom to top (reverse order).
        for sl_idx in range(end - 1, start - 1, -1):
            # conj-trick: conj → forward → conj = backward
            # forward mode: forward only (CGS-compatible)
            if back_prop_mode == "conj":
                wave.array = xp.conj(wave.array)

            result = multislice_step(
                wave, potential_slices[sl_idx], next_slice=None
            )
            if isinstance(result, tuple):
                wave, _ = result
            else:
                wave = result

            if back_prop_mode == "conj":
                wave.array = xp.conj(wave.array)

        # Accumulate back-propagated contribution into exit plane i
        backscattered_waves[i].array += wave.array

    # Negate all exit planes for forward mode: φ₁ᵇ = -Σ ...
    if back_prop_mode == "forward":
        for ep_idx in range(len(backscattered_waves)):
            backscattered_waves._array[ep_idx] = -backscattered_waves._array[ep_idx]

    return backscattered_waves


def transition_potential_multislice_and_detect(
    waves: Waves,
    potential: BasePotential,
    transition_potential: TransitionPotential | TransitionPotentialArray,
    detectors: Optional[list[BaseDetector]] = None,
    detectors_elastic: Optional[list[BaseDetector]] = None,
    double_channel: bool = True,
    threshold: float = 1.0,
    sites: Optional[SliceIndexedAtoms | Atoms] = None,
    algorithm: FourierMultislice | RealSpaceMultislice = FourierMultislice(),
    pbar: bool = False,
) -> list[BaseMeasurements | Waves] | BaseMeasurements | Waves:
    """
    Calculate the full multislice algorithm for the given batch of wave functions
    through a given potential, detecting at each of the exit planes specified in the
    potential.

    Parameters
    ----------
    waves : Waves
        A batch of wave functions as a :class:`.Waves` object.
    potential : BasePotential
        A potential as :class:`.BasePotential` object.
    detectors : (list of) BaseDetector, optional
        A detector or a list of detectors defining how the wave functions should be
        converted to measurements after running the multislice algorithm.
    algorithm: FourierMultislice or RealSpaceMultislice, optional
        Algorithm used for multislice operator (default is FourierMultislice())

    Returns
    -------
    measurements : Waves or tuple of :class:`.BaseMeasurement`
        Exit waves or detected measurements or lists of measurements.
    """

    def _update_loss_measurements(
        measurements, waves, detectors, potential, slice_index, potential_index
    ):
        if slice_index in potential.exit_planes:
            exit_plane_index = potential.exit_planes.index(slice_index)

            measurement_index = _validate_potential_ensemble_indices(
                potential_index, exit_plane_index, potential
            )

            for i, detector in enumerate(detectors):
                new_measurement = detector.detect(waves)
                new_measurement = new_measurement.sum((0,))
                measurements[i].array[measurement_index] += new_measurement.array

    waves = waves.ensure_real_space()

    if isinstance(algorithm, FourierMultislice):
        antialias_aperture = AntialiasAperture()
        propagator = FresnelPropagator()

        def multislice_step(waves, potential_slice):
            return conventional_multislice_step(
                waves,
                potential_slice=potential_slice,
                antialias_aperture=antialias_aperture,
                propagator=propagator,
                conjugate=algorithm.conjugate,
                transpose=algorithm.transpose,
                order=algorithm.order,
            )

    else:
        laplace_operator = LaplaceOperator(
            algorithm.derivative_accuracy,
            method=getattr(algorithm, "laplace_method", "finite-difference"),
        )

        def multislice_step(waves, potential_slice):
            return realspace_multislice_step(
                waves,
                potential_slice=potential_slice,
                next_slice=None,
                laplace=laplace_operator,
                max_terms=algorithm.max_terms,
                order=algorithm.order,
                fully_corrected=algorithm.expansion_scope == "full",
            )

    if detectors is None:
        detectors = [WavesDetector()]

    (
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    ) = _potential_ensemble_shape_and_metadata(potential)

    measurements = allocate_multislice_measurements(
        waves,
        detectors,
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    )

    transition_potential.grid.match(waves)
    transition_potential.accelerator.match(waves)

    if isinstance(transition_potential, TransitionPotential):
        transition_potential = transition_potential.build()

    transition_potential = transition_potential.copy_to_device(waves.device)

    if sites is None and hasattr(potential, "get_sliced_atoms"):
        sites = potential.get_sliced_atoms()
    elif sites is None and hasattr(potential, "atoms"):
        sites = potential.atoms

    if isinstance(sites, Atoms):
        sites = SliceIndexedAtoms(sites, slice_thickness=potential.slice_thickness)
    elif not isinstance(sites, SliceIndexedAtoms):
        raise ValueError()

    n_sites = np.sum(sites.atoms.numbers == transition_potential.Z)

    if n_sites == 0:
        raise RuntimeError(
            "No scattering sites matching transition potential for element"
            f"{transition_potential.Z}"
        )

    absolute_threshold = transition_potential.absolute_threshold(
        waves, threshold=threshold
    )

    n_waves = np.prod(waves.shape[:-2])
    n_slices = n_waves * potential.num_slices * potential.num_configurations

    tqdm_pbar = TqdmWrapper(
        enabled=pbar, total=int(n_slices), leave=False, desc="multislice"
    )

    for (
        potential_index,
        potential_configuration,
    ) in _generate_potential_configurations(potential):
        if potential.exit_planes[0] == -1:
            measurement_index = _validate_potential_ensemble_indices(
                potential_index, 0, potential
            )
            _update_measurements(waves, detectors, measurements, measurement_index)

        depth = 0.0
        for scatter_index, potential_slice in enumerate(
            potential_configuration.generate_slices()
        ):
            waves = multislice_step(
                waves,
                potential_slice,
            )
            depth += potential_slice.axes_metadata[0].values[0]

            _update_plasmon_axes(waves, depth)

            sites_slice = sites.get_atoms_in_slices(
                scatter_index, atomic_number=transition_potential.Z
            )

            tqdm_pbar.update_if_exists(int(n_waves))

            if len(sites_slice) == 0:
                continue

            for (
                included_sites,
                scattered_waves,
            ) in transition_potential.generate_scattered_waves(
                waves, sites_slice, max_batch=1, threshold=absolute_threshold
            ):
                if len(scattered_waves) == 0:
                    continue

                if double_channel:
                    _update_loss_measurements(
                        measurements,
                        scattered_waves,
                        detectors,
                        potential,
                        scatter_index,
                        potential_index,
                    )

                    # if scatter_index + 1 == len(potential):
                    #    break

                    for inner_slice_index, inner_potential_slice in enumerate(
                        potential_configuration.generate_slices(
                            first_slice=scatter_index + 1
                        )
                    ):
                        scattered_waves = multislice_step(
                            scattered_waves,
                            inner_potential_slice,
                        )

                        _update_plasmon_axes(waves, depth)

                        _update_loss_measurements(
                            measurements,
                            scattered_waves,
                            detectors,
                            potential,
                            inner_slice_index + scatter_index + 1,
                            potential_index,
                        )

                else:
                    exit_plane_index = bisect_left(potential.exit_planes, scatter_index)

                    measurement_plane_indices: tuple[slice] | tuple = ()
                    if len(potential.exit_planes) > 1:
                        exit_planes = slice(
                            exit_plane_index, len(potential.exit_planes)
                        )
                        measurement_plane_indices = (exit_planes,)

                    for i, detector in enumerate(detectors):
                        new_measurement = detector.detect(scattered_waves).sum((0,))
                        measurements[i].array[measurement_plane_indices] += (
                            new_measurement.array[
                                (None,) * len(measurement_plane_indices)
                            ]
                        )

    tqdm_pbar.close_if_exists()

    return measurements


def is_waves_base_measurements_or_list(
    value: Any,
) -> TypeGuard["Waves | BaseMeasurements | list[Waves | BaseMeasurements]"]:
    waves_class_name = "Waves"
    base_measurements_class_name = "BaseMeasurements"
    waves_module_name = "abtem.waves"
    base_measurements_module_name = "abtem.measurements"

    def is_instance_of_waves_or_base_measurements(obj: Any) -> bool:
        return (
            obj.__class__.__name__ == waves_class_name
            and obj.__class__.__module__ == waves_module_name
        ) or (
            obj.__class__.__name__ == base_measurements_class_name
            and obj.__class__.__module__ == base_measurements_module_name
        )

    if is_instance_of_waves_or_base_measurements(value):
        return True
    if isinstance(value, list) and all(
        is_instance_of_waves_or_base_measurements(item) for item in value
    ):
        return True
    return False


class MultisliceTransform(WavesTransform[BaseMeasurements]):
    """
    Transformation applying the multislice algorithm to wave functions, producing new
    wave functions or measurements.

    Parameters
    ----------
    potential : BasePotential
        A potential as :class:`.BasePotential` object.
    detectors : (list of) BaseDetector, optional
        A detector or a list of detectors defining how the wave functions should be
        converted to measurements after running the multislice algorithm.
    multislice_func : callable, optional
        The multislice function defining the multislice algorithm used
        (default is :func:`.multislice_and_detect`).
    **multislice_func_kwargs
        Additional keyword arguments passed to the multislice function.
    """

    def __init__(
        self,
        potential: BasePotential,
        detectors: Optional[BaseDetector | list[BaseDetector]] = None,
        multislice_func: Optional[Callable] = None,
        **multislice_func_kwargs,
    ):
        if multislice_func is None:
            multislice_func = multislice_and_detect

        potential = validate_potential(potential)

        self._potential = potential

        detectors = validate_detectors(detectors)
        self._user_detectors = detectors

        if multislice_func_kwargs.get("return_backscattered", False):
            if len(detectors) == 0:
                detectors = [WavesDetector(), WavesDetector()]
            else:
                detectors = detectors + [WavesDetector()]

        if "pbar" not in multislice_func_kwargs:
            multislice_func_kwargs["pbar"] = config.get(
                "diagnostics.task_progress", False
            )

        self._detectors = detectors
        self._multislice_func = multislice_func
        self._multislice_func_kwargs = multislice_func_kwargs

    @property
    def multislice_func(self) -> Callable:
        """The multislice function defining the multislice algorithm used."""
        return self._multislice_func

    @property
    def _num_outputs(self):
        return len(self._detectors)

    @property
    def potential(self) -> BasePotential:
        """Electrostatic potential for each multislice slice."""
        return self._potential

    @property
    def detectors(self) -> list[BaseDetector]:
        """List of detectors defining how the wave functions should be converted to
        measurements."""
        return self._detectors

    @property
    def ensemble_axes_metadata(self):
        ensemble_axes_metadata = self.potential.ensemble_axes_metadata

        if len(self.potential.exit_planes) > 1:
            exit_planes_metadata = [self.potential._get_exit_planes_axes_metadata()]
        else:
            exit_planes_metadata = []

        ensemble_axes_metadata = [
            *ensemble_axes_metadata,
            *exit_planes_metadata,
        ]
        return ensemble_axes_metadata

    @property
    def ensemble_shape(self):
        ensemble_shape = self._potential.ensemble_shape
        if len(self._potential.exit_planes) > 1:
            ensemble_shape = (*ensemble_shape, len(self._potential.exit_planes))
        return ensemble_shape

    def _out_metadata(self, waves: Waves) -> tuple[dict, ...]:
        return tuple(detector._out_metadata(waves)[0] for detector in self.detectors)

    def _out_dtype(self, waves: Waves) -> tuple[np.dtype, ...]:
        return tuple(detector._out_dtype(waves)[0] for detector in self.detectors)

    def _out_meta(self, waves: Waves) -> tuple[np.ndarray, ...]:
        return tuple(detector._out_meta(waves)[0] for detector in self.detectors)

    def _out_type(self, waves: Waves) -> tuple[type, ...]:
        return tuple(detector._out_type(waves)[0] for detector in self.detectors)

    def _out_ensemble_shape(self, waves: Waves) -> tuple[tuple[int, ...], ...]:
        shape = tuple(
            self.ensemble_shape + detector._out_ensemble_shape(waves)[0]
            for detector in self.detectors
        )
        return shape

    def _out_base_shape(self, waves: Waves) -> tuple[tuple[int, ...], ...]:
        base_shape = tuple(
            detector._out_base_shape(waves)[0] for detector in self.detectors
        )
        return base_shape

    def _out_base_axes_metadata(self, waves: Waves) -> tuple[list[AxisMetadata], ...]:
        return tuple(
            detector._out_base_axes_metadata(waves)[0] for detector in self.detectors
        )

    def _out_ensemble_axes_metadata(
        self, waves: Waves
    ) -> tuple[list[AxisMetadata], ...]:
        if len(self.potential.exit_planes) > 1:
            potential_axes_metadata = self.potential.ensemble_axes_metadata + [
                self.potential._get_exit_planes_axes_metadata()
            ]
        else:
            potential_axes_metadata = self.potential.ensemble_axes_metadata

        ensemble_axes_metadata = tuple(
            potential_axes_metadata + detector._out_ensemble_axes_metadata(waves)[0]
            for detector in self.detectors
        )

        return ensemble_axes_metadata

    @property
    def _default_ensemble_chunks(self) -> Chunks:
        chunks: tuple[int, ...] = ()

        if len(self.potential.ensemble_shape) > 0:
            chunks = chunks + (1,)

        if len(self.potential.exit_planes) > 1:
            chunks = chunks + (len(self.potential.exit_planes),)

        return chunks

    def _validate_ensemble_chunks(
        self, chunks: Optional[Chunks] = None, limit: str | int = "auto"
    ) -> ValidatedChunks:
        if chunks is None:
            chunks = self._default_ensemble_chunks

        if (
            isinstance(chunks, int)
            and len(self.ensemble_shape) > 1
            and self.potential.num_exit_planes > 1
        ):
            chunks = (chunks, self.potential.num_exit_planes)

        chunks = validate_chunks(self.ensemble_shape, chunks, max_elements=limit)

        if self.potential.num_exit_planes > 1:
            chunks = chunks[:-1] + ((self.potential.num_exit_planes,),)

        return chunks

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        chunks = self._validate_ensemble_chunks(chunks)

        if self.potential.num_exit_planes > 1:
            chunks = chunks[:-1]

        args = self._potential._partition_args(chunks=chunks, lazy=lazy)

        if len(self._potential.exit_planes) > 1:
            args = (args[0][..., None],)

        return args

    @staticmethod
    def _multislice_transform_member(*args, potential_partial: Callable, **kwargs):
        args = unpack_blockwise_args(args)

        potential = potential_partial(*args)
        potential = potential.item()
        transform = MultisliceTransform(potential, **kwargs)

        ndims = len(transform.ensemble_shape)
        wrapped_transform = _wrap_with_array(transform, ndims)
        return wrapped_transform

    def _from_partitioned_args(self) -> Callable:
        potential_partial = self._potential._from_partitioned_args()
        return partial(
            self._multislice_transform_member,
            potential_partial=potential_partial,
            detectors=self._user_detectors,
            multislice_func=self.multislice_func,
            **self._multislice_func_kwargs,
        )

    def _calculate_new_array(self, waves: Waves):
        # Pass user detectors (without BSC WavesDetector) because
        # multislice_and_detect adds the necessary WavesDetectors internally
        # when return_backscattered=True. Using self.detectors here would
        # cause double-counting (one from __init__, one from
        # multislice_and_detect).
        input_detectors = self._user_detectors

        measurements = self.multislice_func(
            waves=waves,
            potential=self.potential,
            detectors=input_detectors,
            **self._multislice_func_kwargs,
        )

        if len(measurements) != len(self.detectors):
            raise RuntimeError(
                f"Expected {len(self.detectors)} outputs, got {len(measurements)}"
            )

        arrays = tuple(measurement.array for measurement in measurements)
        if len(arrays) == 1:
            arrays = arrays[0]

        return arrays

    def apply(
        self, waves: Waves, max_batch: int | str = "auto"
    ) -> Waves | BaseMeasurements | list[Waves | BaseMeasurements]:
        """
        Run the multislice algorithm on the given wave functions. An output is returned
        for each detector.

        Parameters
        ----------
        waves : Waves
            The wave functions to run the multislice algorithm on.
        max_batch : int or str, optional
            The maximum batch size to use for the multislice algorithm. If 'auto' the
            batch size is chosen automatically based on the available memory.

        Returns
        -------
        waves : tuple of Waves and BaseMeasurements
            The wave functions after running the multislice algorithm.
        """
        output = waves.apply_transform(self, max_batch=max_batch)
        # assert is_waves_base_measurements_or_list(output)

        return output
