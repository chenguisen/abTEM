"""Tests for the CVDMS multislice algorithm."""

import pytest

import ase
import numpy as np

import abtem
from abtem.multislice import CVDMSMultislice, FourierMultislice


@pytest.fixture
def atoms():
    """Create a simple test structure."""
    return ase.build.bulk("Au", cubic=True)


@pytest.fixture
def potential(atoms):
    """Create a simple potential for testing."""
    return abtem.Potential(
        atoms,
        gpts=(64, 64),
        slice_thickness=0.5,
        sampling=0.2,
    )


@pytest.fixture
def probe(potential):
    """Create a probe matched to the potential."""
    return abtem.Probe(
        semiangle_cutoff=30,
        energy=80e3,
    ).match_grid(potential)


class TestCVDMSBasic:
    """Basic CVDMS algorithm tests."""

    def test_cvdms_import(self):
        """Test that CVDMSMultislice can be imported."""
        from abtem.multislice import CVDMSMultislice
        assert CVDMSMultislice is not None

    def test_cvdms_default_creation(self):
        """Test default CVDMSMultislice creation."""
        algo = CVDMSMultislice()
        assert algo.order == 1
        assert algo.max_terms == 50
        assert algo.convergence_threshold == 1e-6
        assert algo.backscattering is False
        assert algo.calculate_backscattered is False
        assert algo.derivative_accuracy == 8
        assert algo.laplace_method == "finite-difference"

    def test_cvdms_custom_params(self):
        """Test CVDMSMultislice with custom parameters."""
        algo = CVDMSMultislice(
            order=2,
            max_terms=100,
            convergence_threshold=1e-8,
            backscattering=True,
            calculate_backscattered=True,
            derivative_accuracy=10,
            laplace_method="fft",
        )
        assert algo.order == 2
        assert algo.max_terms == 100
        assert algo.convergence_threshold == 1e-8
        assert algo.backscattering is True
        assert algo.calculate_backscattered is True
        assert algo.derivative_accuracy == 10
        assert algo.laplace_method == "fft"

    def test_cvdms_basic_computation(self, probe, potential):
        """Test basic CVDMS computation (forward only)."""
        algorithm = CVDMSMultislice(order=1)
        result = probe.multislice(potential, algorithm=algorithm)
        assert result is not None
        assert hasattr(result, "array")

    def test_cvdms_with_backscattering(self, probe, potential):
        """Test CVDMS with backscattering enabled."""
        algorithm = CVDMSMultislice(
            order=1,
            backscattering=True,
        )
        result = probe.multislice(potential, algorithm=algorithm)
        assert result is not None

    def test_cvdms_compare_with_fourier(self, probe, potential):
        """Compare CVDMS results with Fourier multislice (order=1)."""
        # Both should produce similar results for a thin sample at order=1
        algo_fourier = FourierMultislice(order=1)
        algo_cvdms = CVDMSMultislice(
            order=1,
            convergence_threshold=1e-6,
            max_terms=50,
        )

        result_fourier = probe.multislice(potential, algorithm=algo_fourier)
        result_cvdms = probe.multislice(potential, algorithm=algo_cvdms)

        # For thin sample with no backscattering, results should be similar
        diff = np.abs(result_cvdms.array - result_fourier.array)
        assert np.mean(diff) < 0.1


class TestCVDMSConvergence:
    """CVDMS convergence tests."""

    @pytest.mark.parametrize("threshold", [1e-4, 1e-6])
    def test_convergence_thresholds(self, probe, potential, threshold):
        """Test different convergence thresholds."""
        algorithm = CVDMSMultislice(
            order=1,
            convergence_threshold=threshold,
        )
        result = probe.multislice(potential, algorithm=algorithm)
        assert result is not None

    @pytest.mark.parametrize("max_terms", [10, 50])
    def test_max_terms(self, probe, potential, max_terms):
        """Test different maximum Taylor series terms."""
        algorithm = CVDMSMultislice(
            order=1,
            max_terms=max_terms,
            convergence_threshold=1e-8,
        )
        result = probe.multislice(potential, algorithm=algorithm)
        assert result is not None


class TestCVDMSErrors:
    """CVDMS error handling tests."""

    def test_invalid_max_terms(self):
        """Test that invalid max_terms raises an error."""
        from abtem.cvdms import cvdms_multislice_step
        with pytest.raises(ValueError):
            cvdms_multislice_step(None, None, None, None, max_terms=0)

    def test_with_detectors(self, probe, potential):
        """Test CVDMS with detectors."""
        algorithm = CVDMSMultislice(order=1)
        detector = abtem.WavesDetector()
        result = probe.multislice(
            potential, algorithm=algorithm, detectors=[detector]
        )
        assert result is not None


class TestCVDMSLaplacian:
    """CVDMS Laplacian method tests."""

    def test_fft_laplacian_basic(self, probe, potential):
        """Test CVDMS with FFT-based Laplacian."""
        algorithm = CVDMSMultislice(
            order=1,
            laplace_method="fft",
        )
        result = probe.multislice(potential, algorithm=algorithm)
        assert result is not None
        assert hasattr(result, "array")

    def test_fft_laplacian_numerical(self):
        """Verify FFT Laplacian on a simple analytical function.

        For ψ = sin(kx·x) + sin(kx·y):
          ∇²ψ = ∂²/∂x² sin(kx·x) + ∂²/∂y² sin(kx·y)
               = -kx² · sin(kx·x) - kx² · sin(kx·y)
               = -kx² · ψ
        """
        import abtem.finite_difference as fd

        sampling = (0.1, 0.1)
        N = 64
        extent = N * sampling[0]
        kx = 2.0 * np.pi / extent * 4  # 4 cycles across grid

        x = np.arange(N) * sampling[0]
        psi = np.sin(kx * x)[:, None] + np.sin(kx * x)[None, :]
        psi = psi.astype(np.complex64)

        stencil = fd._laplace_operator_fft(sampling)
        result = stencil(psi)

        expected = -kx**2 * psi
        diff = np.abs(result - expected)
        assert np.mean(diff) / np.mean(np.abs(expected)) < 0.05

    def test_fft_laplacian_works_with_light_atoms(self):
        """FFT Laplacian works for light atoms (Si) where high-freq growth is less severe."""
        atoms_si = ase.build.bulk("Si", cubic=True)
        potential_si = abtem.Potential(
            atoms_si,
            gpts=(64, 64),
            slice_thickness=1.0,
            sampling=0.2,
        )
        probe_si = abtem.Probe(
            semiangle_cutoff=30,
            energy=80e3,
        ).match_grid(potential_si)

        algo_fft = CVDMSMultislice(
            order=1,
            laplace_method="fft",
        )
        result = probe_si.multislice(potential_si, algorithm=algo_fft)
        assert result is not None

    def test_fft_with_backscattering(self, probe, potential):
        """Test FFT Laplacian with backscattering enabled."""
        algorithm = CVDMSMultislice(
            order=1,
            backscattering=True,
            laplace_method="fft",
        )
        result = probe.multislice(potential, algorithm=algorithm)
        assert result is not None


class TestIntensityConservation:
    """Total intensity (electron flux) conservation during multislice.

    For elastic scattering, Parseval's theorem requires sum(|ψ|²) to be
    constant (total electron flux is conserved). Fourier multislice is
    exactly unitary; CVDMS uses a Taylor series approximation so small
    deviations are expected.
    """

    def test_fourier_conservation(self, probe, potential):
        """Fourier multislice: sum(|ψ|²) is exactly conserved (machine precision)."""
        algorithm = FourierMultislice(order=1)

        # Initial probe total intensity
        I0 = float(np.sum(np.abs(np.asarray(probe.array)) ** 2))

        result = probe.multislice(potential, algorithm=algorithm)

        # Total intensity at each exit plane
        arr = np.asarray(result.array)
        I_final = float(np.sum(np.abs(arr) ** 2))

        rel_diff = abs(I_final - I0) / I0

        # Tolerance: abTEM default is complex64 (float32) → ~1e-5 precision.
        # For complex128 (float64) the tolerance could be 1e-14, but using
        # 1e-5 keeps the test robust across precision settings.
        assert rel_diff < 1e-5, (
            f"Fourier multislice intensity not conserved: "
            f"|ΔI|/I₀ = {rel_diff:.2e}"
        )

    def test_cvdms_conservation(self, probe, potential):
        """CVDMS: sum(|ψ|²) is approximately conserved (Taylor series approx)."""
        algorithm = CVDMSMultislice(order=1)

        I0 = float(np.sum(np.abs(np.asarray(probe.array)) ** 2))

        result = probe.multislice(potential, algorithm=algorithm)

        arr = np.asarray(result.array)
        I_final = float(np.sum(np.abs(arr) ** 2))

        rel_diff = abs(I_final - I0) / I0
        assert rel_diff < 0.01, (
            f"CVDMS multislice intensity not conserved: "
            f"|ΔI|/I₀ = {rel_diff:.2e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
