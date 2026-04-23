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
        assert algo.include_backscattering is True
        assert algo.calculate_backscattered is False
        assert algo.expansion_scope == "propagator"
        assert algo.derivative_accuracy == 6

    def test_cvdms_custom_params(self):
        """Test CVDMSMultislice with custom parameters."""
        algo = CVDMSMultislice(
            order=2,
            max_terms=100,
            convergence_threshold=1e-8,
            include_backscattering=False,
            expansion_scope="full",
        )
        assert algo.order == 2
        assert algo.max_terms == 100
        assert algo.convergence_threshold == 1e-8
        assert algo.include_backscattering is False
        assert algo.expansion_scope == "full"

    def test_cvdms_basic_computation(self, probe, potential):
        """Test basic CVDMS computation."""
        algorithm = CVDMSMultislice(order=1, include_backscattering=False)
        result = probe.multislice(potential, algorithm=algorithm)
        assert result is not None
        assert hasattr(result, "array")

    def test_cvdms_with_backscattering(self, probe, potential):
        """Test CVDMS with backscattering enabled."""
        algorithm = CVDMSMultislice(
            order=1,
            include_backscattering=True,
        )
        result = probe.multislice(potential, algorithm=algorithm)
        assert result is not None

    def test_cvdms_compare_with_fourier(self, probe, potential):
        """Compare CVDMS results with Fourier multislice (order=1)."""
        # Both should produce similar results for a thin sample at order=1
        algo_fourier = FourierMultislice(order=1)
        algo_cvdms = CVDMSMultislice(
            order=1,
            include_backscattering=False,
            convergence_threshold=1e-10,
            max_terms=100,
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
            include_backscattering=False,
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
            include_backscattering=False,
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
        algorithm = CVDMSMultislice(
            order=1,
            include_backscattering=False,
        )
        detector = abtem.WavesDetector()
        result = probe.multislice(
            potential, algorithm=algorithm, detectors=[detector]
        )
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
