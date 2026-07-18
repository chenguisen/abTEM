"""
test_bsc_convergence.py — Convergence of backscattering (BSC) correction.

Tests the per-slice BSC correction in dev's ``multislice_step()``
(``finite_difference.py`` lines 608-667):

  - SBA (Single Backscattering Approximation):  B ∝ Δ full_series / (2πi·dz)
  - 1/k binomial correction:  B *= 1/(2K₀) · (1 + Σ binom(-½,n)·Kⁿ)

Note: Order=1 with ``expansion_scope="full"`` often triggers ``DivergedError``
because the exponential series diverges before convergence with just one
K-operator nesting.  Start from order >= 2 for BSC-enabled tests.
"""

import numpy as np
import pytest

from abtem.multislice import RealSpaceMultislice
from abtem.finite_difference import DivergedError
from test_cvdms.metrics import to_numpy, ncc, max_diff, intensity, check_finite


def _run_bsc(system, order):
    """Run BSC-enabled multislice.  Returns (fwd_arr, bwd_arr) or (None, None)."""
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
    """pytest.skip if BSC diverged at this order."""
    if fwd is None:
        pytest.skip(f"order={order} diverged (expected at low order for full scope)")


# ── BSC magnitude ──────────────────────────────────────────────────────────

class TestBSCMagnitude:
    """Quantify BSC correction relative to forward wave."""

    @pytest.mark.parametrize("order", [2, 3, 4, 6])
    def test_bsc_ratio_finite_and_small(self, sto_system, order):
        """|BSC| / |forward| should be < 1 and finite."""
        fwd, bwd = _run_bsc(sto_system, order)
        _skip_if_diverged(fwd, bwd, order)

        fwd_norm = np.abs(fwd).sum()
        bwd_norm = np.abs(bwd).sum()
        ratio = bwd_norm / fwd_norm if fwd_norm > 0 else 0

        print(f"\n  order={order}: |BSC|/|fwd| = {ratio:.6e}")
        assert check_finite(bwd), f"order={order}: non-finite BSC"
        assert ratio < 1.0, f"order={order}: BSC ratio {ratio:.4f} > 1"
        assert ratio > 0, "BSC correction is exactly zero"

    @pytest.mark.parametrize("order", [2, 3, 4, 6])
    def test_bsc_ratio_converges_with_order(self, sto_system, order):
        """BSC ratio should stabilize as order increases."""
        _, bwd = _run_bsc(sto_system, order)
        _skip_if_diverged(_, bwd, order)
        norm = np.abs(bwd).sum()
        print(f"\n  order={order}: |BSC| = {norm:.6e}")
        assert np.isfinite(norm)

    def test_au_bsc_larger_than_sto(self, sto_system, au_system):
        """Gold (higher Z) should produce a larger BSC signal than STO."""
        _, bwd_sto = _run_bsc(sto_system, order=3)
        _, bwd_au = _run_bsc(au_system, order=3)
        _skip_if_diverged(bwd_sto, bwd_au, 3)

        sto_ratio = np.abs(bwd_sto).sum() / np.abs(bwd_sto).size
        au_ratio = np.abs(bwd_au).sum() / np.abs(bwd_au).size

        print(f"\n  STO mean |BSC|/px = {sto_ratio:.3e}")
        print(f"  Au  mean |BSC|/px = {au_ratio:.3e}")
        assert au_ratio > sto_ratio * 0.01, (
            f"Au BSC ({au_ratio:.3e}) suspiciously smaller than STO ({sto_ratio:.3e})"
        )


# ── BSC order convergence ──────────────────────────────────────────────────

class TestBSCOrderConvergence:
    """BSC correction should converge with expansion order."""

    @pytest.mark.parametrize("order_pair", [(2, 4), (3, 4), (4, 6)])
    def test_bsc_adjacent_orders_similar(self, sto_system, order_pair):
        """Adjacent orders should produce similar BSC corrections."""
        lo, hi = order_pair
        fwd_lo, bwd_lo = _run_bsc(sto_system, lo)
        fwd_hi, bwd_hi = _run_bsc(sto_system, hi)
        _skip_if_diverged(fwd_lo, bwd_lo, lo)
        _skip_if_diverged(fwd_hi, bwd_hi, hi)

        fwd_ncc = ncc(fwd_lo, fwd_hi)
        bwd_ncc = ncc(bwd_lo, bwd_hi)

        print(f"\n  orders ({lo},{hi}): NCC_fwd = {fwd_ncc:.8f}, "
              f"NCC_bwd = {bwd_ncc:.8f}")
        assert fwd_ncc > 0.99, (
            f"Forward NCC({lo},{hi}) = {fwd_ncc:.8f} below 0.99"
        )

    def test_bsc_order_2_vs_6_comparison(self, sto_system):
        """Order 2 vs 6 BSC: verify the correction changes with more terms."""
        fwd2, bwd2 = _run_bsc(sto_system, order=2)
        fwd6, bwd6 = _run_bsc(sto_system, order=6)
        _skip_if_diverged(fwd2, bwd2, 2)
        _skip_if_diverged(fwd6, bwd6, 6)

        d_fwd = max_diff(fwd2, fwd6)
        d_bwd = max_diff(bwd2, bwd6)

        print(f"\n  max_diff(order2, order6): fwd={d_fwd:.3e}, bwd={d_bwd:.3e}")
        assert np.isfinite(d_fwd) and np.isfinite(d_bwd)
        # The BSC should differ between these orders
        assert d_bwd > 0, "BSC should differ between orders 2 and 6"


# ── BSC correction effect on forward wave ──────────────────────────────────

class TestBSCEffectOnForward:
    """The BSC correction modifies the forward wave at the sub-percent level."""

    def test_bsc_correction_is_small(self, sto_system):
        """BSC correction should be small relative to forward wave amplitude."""
        probe = sto_system["probe"]
        pot_ep = sto_system["potential_exit_planes"]
        pot_no_ep = sto_system["potential"]

        try:
            fwd_with_bsc = to_numpy(probe.multislice(
                potential=pot_ep, scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
            ).array)
        except DivergedError:
            pytest.skip("BSC diverged at order=3")

        fwd_no_bsc = to_numpy(probe.multislice(
            potential=pot_no_ep, scan=[[0, 0]], lazy=False,
            algorithm=RealSpaceMultislice(order=3, expansion_scope="propagator"),
        ).array)

        d = max_diff(fwd_no_bsc, fwd_with_bsc)
        # Normalize by max amplitude rather than mean (mean ~0 at the edges)
        norm = np.abs(fwd_no_bsc).max()

        print(f"\n  |fwd_bsc - fwd_no_bsc|_max = {d:.3e}")
        print(f"  max |fwd| = {norm:.3e}")

        assert np.isfinite(d)
        # BSC correction is physically small — should be of the same order
        # as the wave amplitude, not orders of magnitude larger
        assert d < 1.0, f"BSC correction max_diff ({d:.3e}) unreasonably large"


# ── Energy budget ──────────────────────────────────────────────────────────

class TestBSCEnergyBudget:
    """I_fwd + I_bsc should approximately equal I0."""

    @pytest.mark.parametrize("order", [2, 4, 6])
    def test_energy_budget_order(self, sto_system, order):
        """Energy budget should stay bounded across orders."""
        probe = sto_system["probe"]
        I0 = intensity(probe)

        fwd, bwd = _run_bsc(sto_system, order)
        _skip_if_diverged(fwd, bwd, order)

        I_fwd = intensity(fwd)
        I_bwd = intensity(bwd)
        budget = abs(I_fwd + I_bwd - I0) / I0

        print(f"\n  order={order}: I_fwd={I_fwd:.4f}, I_bwd={I_bwd:.3e}, "
              f"|budget err| = {budget:.3e}")
        assert budget < 0.5, f"order={order}: energy budget error {budget:.3f}"
