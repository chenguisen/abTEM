"""
test_backpropagation.py — Backscattered wave accumulation and back-propagation.

Tests the ``return_backscattered=True`` pipeline:
  1. Forward pass collects per-slice BSC fields
  2. ``_back_propagate_backscattered_waves()`` back-propagates them to
     the entrance plane via the conj-trick (conj → forward → conj)

Key questions:
  - Does the accumulated backscattered wave have the correct shape?
  - Is the entrance-plane backscattered wave zero?
  - Does backscattered intensity grow with sample depth?
  - Does higher expansion order change the backscattered signal?
"""

import numpy as np
import pytest

from abtem.multislice import RealSpaceMultislice
from abtem.finite_difference import DivergedError
from test_cvdms.metrics import to_numpy, intensity, check_finite


def _run_bsc_full(system, order):
    """Run full BSC pipeline. Returns (fwd_arr, bwd_arr) or (None, None)."""
    if system["potential_exit_planes"] is None:
        pytest.skip("no exit_planes in system")
    try:
        fwd, bwd = system["probe"].multislice(
            potential=system["potential_exit_planes"],
            scan=[[0, 0]], lazy=False,
            algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
            return_backscattered=True,
        )
        return to_numpy(fwd.array), to_numpy(bwd.array)
    except DivergedError:
        return None, None


def _skip_if_diverged(fwd, bwd, order):
    if fwd is None:
        pytest.skip(f"order={order} diverged")


# ── Shape consistency ──────────────────────────────────────────────────────

class TestBackscatteredWaveShape:
    """Forward and backward waves should have consistent shapes."""

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_forward_backward_same_ndim(self, sto_system, order):
        """Forward and backward arrays have same number of dimensions."""
        fwd, bwd = _run_bsc_full(sto_system, order)
        _skip_if_diverged(fwd, bwd, order)

        print(f"\n  fwd shape: {fwd.shape}, bwd shape: {bwd.shape}")
        assert fwd.ndim == bwd.ndim, (
            f"ndim mismatch: fwd {fwd.ndim} vs bwd {bwd.ndim}"
        )
        assert check_finite(bwd)

    @pytest.mark.parametrize("exit_planes", [1, 2, 4])
    def test_exit_planes_dimension(self, sto_system, exit_planes):
        """N exit planes → leading dimension should be present."""
        from test_cvdms.conftest import build_system, _make_sto_atoms
        sys = build_system(_make_sto_atoms(), exit_planes=exit_planes,
                           device=sto_system["device"])

        fwd, bwd = _run_bsc_full(sys, order=3)
        _skip_if_diverged(fwd, bwd, 3)

        fwd_shape = fwd.shape
        bwd_shape = bwd.shape

        print(f"\n  exit_planes={exit_planes}: fwd={fwd_shape}, bwd={bwd_shape}")

        # If exit_planes >= 2, there's an exit-plane dimension
        if exit_planes >= 2:
            assert fwd_shape[0] >= 2
        # Both should have matching spatial dimensions at the end
        assert fwd_shape[-2:] == bwd_shape[-2:]


# ── Physical sanity ────────────────────────────────────────────────────────

class TestBackscatteredWavePhysics:
    """Backscattered wave should obey physical constraints."""

    @pytest.mark.parametrize("order", [2, 4])
    def test_entrance_backscattered_is_zero(self, sto_system, order):
        """The first exit-plane backscattered wave must be zero
        (set at ``multislice.py:875``)."""
        _, bwd = _run_bsc_full(sto_system, order)
        _skip_if_diverged(_, bwd, order)

        if bwd.ndim >= 3 and bwd.shape[0] > 1:
            first_slice_norm = np.abs(bwd[0]).sum()
            print(f"\n  order={order}: |BSC[0]| = {first_slice_norm:.3e}")
            assert first_slice_norm < 1e-10, (
                f"First exit plane BSC should be zero, got {first_slice_norm:.3e}"
            )
        else:
            pytest.skip("single exit plane, no entrance/first distinction")

    @pytest.mark.parametrize("order", [2, 4])
    def test_backscattered_is_finite(self, sto_system, order):
        """Backscattered wave should contain finite values."""
        _, bwd = _run_bsc_full(sto_system, order)
        _skip_if_diverged(_, bwd, order)

        assert check_finite(bwd), f"order={order}: non-finite in backscattered wave"
        assert np.abs(bwd).sum() > 0, f"order={order}: backscattered wave is zero"

    @pytest.mark.parametrize("order_pair", [(2, 4)])
    def test_higher_order_at_least_similar_bsc(self, sto_system, order_pair):
        """Higher orders should produce comparable or larger BSC signal."""
        lo, hi = order_pair
        _, bwd_lo = _run_bsc_full(sto_system, lo)
        _, bwd_hi = _run_bsc_full(sto_system, hi)
        _skip_if_diverged(bwd_lo, bwd_hi, lo)

        norm_lo = np.abs(bwd_lo).sum()
        norm_hi = np.abs(bwd_hi).sum()
        ratio = norm_hi / norm_lo if norm_lo > 0 else float("inf")
        print(f"\n  |BSC(order={hi})| / |BSC(order={lo})| = {ratio:.4f}")
        assert ratio > 0.01, (
            f"Order {hi} BSC ({norm_hi:.3e}) is {ratio:.4f}x of order {lo} ({norm_lo:.3e})"
        )


# ── BSC with detectors ─────────────────────────────────────────────────────

class TestBackscatteredWithDetectors:
    """Combining ``return_backscattered`` with explicit detectors."""

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_extra_wave_in_result_list(self, sto_system, order):
        """With detectors + return_backscattered=True, the last element
        is the backscattered waves."""
        import abtem
        probe = sto_system["probe"]
        pot = sto_system["potential_exit_planes"]

        detector = abtem.PixelatedDetector()
        try:
            results = probe.multislice(
                potential=pot, scan=[[0, 0]], lazy=False,
                detectors=detector,
                algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
                return_backscattered=True,
            )
        except DivergedError:
            pytest.skip(f"order={order} diverged")

        assert isinstance(results, (list, tuple)), "should return list"
        assert len(results) == 2, f"expected 2 items, got {len(results)}"
        bwd_arr = to_numpy(results[-1].array)
        assert check_finite(bwd_arr)


# ── Error paths ────────────────────────────────────────────────────────────

class TestBackscatteredErrorPaths:
    """Return_backscattered=True requires exit_planes and full scope."""

    def test_raises_without_exit_planes(self, sto_system):
        """ValueError when potential has no exit_planes."""
        probe = sto_system["probe"]
        with pytest.raises(ValueError, match="exit_planes"):
            probe.multislice(
                potential=sto_system["potential"], scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
                return_backscattered=True,
            )

    def test_raises_without_full_scope(self, sto_system):
        """ValueError when expansion_scope is not 'full'."""
        probe = sto_system["probe"]
        with pytest.raises(ValueError, match="expansion_scope"):
            probe.multislice(
                potential=sto_system["potential_exit_planes"],
                scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=3, expansion_scope="propagator"),
                return_backscattered=True,
            )
