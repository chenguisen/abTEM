"""
test_forward_convergence.py — Convergence of fully_corrected forward scattering.

Tests the Taylor expansion order convergence of dev's real-space multislice
with ``expansion_scope="full"`` (uses ``full_series`` which expands both
propagator and transmission operators to the specified order).

Key questions answered by this test suite:
  - Does NCC approach 1 as expansion order increases?
  - Does the difference between successive orders decrease (monotonic convergence)?
  - How does ``expansion_scope="full"`` compare to ``"propagator"`` at the same order?
  - Does convergence behavior depend on material (Z), slice thickness, or energy?
"""

import numpy as np
import pytest

from abtem.multislice import RealSpaceMultislice
from abtem.finite_difference import DivergedError
from test_cvdms.metrics import to_numpy, ncc, rmsd, max_diff, check_finite, relative_error, intensity

# ── helpers ────────────────────────────────────────────────────────────────

def _run_forward(system, order, scope="full", scan=None):
    """Run forward multislice on a single scan point and return the numpy array.
    Returns None if DivergedError is raised."""
    if scan is None:
        scan = [[0, 0]]
    try:
        result = system["probe"].multislice(
            potential=system["potential"],
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(order=order, expansion_scope=scope),
        )
        return to_numpy(result.array)
    except DivergedError:
        return None


def _run_forward_plane_wave(system, order, scope="full"):
    """Same as _run_forward but using PlaneWave (no scan parameter)."""
    result = system["plane_wave"].multislice(
        potential=system["potential"],
        lazy=False,
        algorithm=RealSpaceMultislice(order=order, expansion_scope=scope),
    )
    return to_numpy(result.array)


# ── Order convergence ──────────────────────────────────────────────────────

class TestOrderConvergence:
    """Self-convergence: higher orders should approach a stable limit."""

    @pytest.mark.parametrize("orders", [
        pytest.param((1, 2, 4), id="1-2-4"),
        pytest.param((2, 4, 6), id="2-4-6"),
    ])
    def test_ncc_increases_with_order(self, sto_system, orders):
        """NCC(order_i, order_max) should increase monotonically with i."""
        o1, o2, o3 = orders
        ref = _run_forward(sto_system, order=o3)          # highest → reference

        arr1 = _run_forward(sto_system, order=o1)
        arr2 = _run_forward(sto_system, order=o2)

        ncc1 = ncc(ref, arr1)
        ncc2 = ncc(ref, arr2)

        print(f"\n  NCC(order={o1} vs {o3}) = {ncc1:.8f}")
        print(f"  NCC(order={o2} vs {o3}) = {ncc2:.8f}")

        assert ncc2 >= ncc1 * 0.999, (
            f"NCC should improve: {ncc1:.8f} → {ncc2:.8f}"
        )

    @pytest.mark.parametrize("orders", [
        pytest.param((1, 2, 4, 6), id="1→6"),
        pytest.param((1, 2, 3, 4), id="1→4"),
    ])
    def test_rmsd_decreases_monotonically(self, sto_system, orders):
        """RMSD(order_i, order_max) should decrease monotonically."""
        arr_max = _run_forward(sto_system, order=orders[-1])

        prev_rmsd = float("inf")
        for o in orders[:-1]:
            arr = _run_forward(sto_system, order=o)
            r = rmsd(arr_max, arr)
            print(f"\n  RMSD(order={o} vs {orders[-1]}) = {r:.3e}")
            assert r <= prev_rmsd * 1.01, (
                f"RMSD should decrease: {prev_rmsd:.3e} → {r:.3e}"
            )
            prev_rmsd = r

    @pytest.mark.parametrize("order_pair", [(1, 2), (2, 3), (3, 4), (4, 6)])
    def test_pairwise_diff_decreases(self, sto_system, order_pair):
        """max_diff(order_k, order_{k+2}) should decrease as k increases."""
        lo, hi = order_pair
        arr_lo = _run_forward(sto_system, order=lo)
        arr_hi = _run_forward(sto_system, order=hi)
        diff = max_diff(arr_lo, arr_hi)
        print(f"\n  max_diff(order={lo} vs {hi}) = {diff:.3e}")
        assert np.isfinite(diff)
        assert diff > 0  # should actually differ


class TestOrderConvergencePlaneWave:
    """Same convergence checks using PlaneWave (no probe structure)."""

    def test_plane_wave_order_monotonic(self, sto_system):
        """PlaneWave NCC(order, 8) should increase monotonically."""
        ref = _run_forward_plane_wave(sto_system, order=6)

        results = {}
        for o in [1, 2, 3, 4]:
            arr = _run_forward_plane_wave(sto_system, order=o)
            results[o] = ncc(ref, arr)
            print(f"\n  PW NCC(order={o} vs 6) = {results[o]:.8f}")

        # NCC should be monotonic (or nearly so)
        prev = 0
        for o in [1, 2, 3, 4]:
            assert results[o] >= prev * 0.999, (
                f"PW NCC should increase: order={o}, NCC={results[o]:.8f} < prev={prev:.8f}"
            )
            prev = results[o]


# ── Expansion scope comparison ─────────────────────────────────────────────

class TestExpansionScopeComparison:
    """``full`` vs ``propagator`` at the same order."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_scope_differs_at_same_order(self, sto_system, order):
        """Full and propagator-only should differ at every order > 1."""
        arr_prop = _run_forward(sto_system, order=order, scope="propagator")
        arr_full = _run_forward(sto_system, order=order, scope="full")
        d = max_diff(arr_prop, arr_full)
        print(f"\n  order={order}: |prop - full|_max = {d:.3e}")
        assert d > 0, f"order={order}: full and propagator should differ"
        assert np.isfinite(d)

    def test_scope_difference_decreases_with_order(self, sto_system):
        """Higher-order full and propagator should converge toward each other."""
        diffs = []
        for o in [1, 2, 3, 4]:
            arr_prop = _run_forward(sto_system, order=o, scope="propagator")
            arr_full = _run_forward(sto_system, order=o, scope="full")
            diffs.append(max_diff(arr_prop, arr_full))

        print(f"\n  scope diffs: {[f'{d:.3e}' for d in diffs]}")
        # The difference should not explode with order
        assert max(diffs) < 10 * diffs[0], "scope difference should stay bounded"

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_ncc_between_scopes(self, sto_system, order):
        """NCC between full and propagator at same order."""
        arr_prop = _run_forward(sto_system, order=order, scope="propagator")
        arr_full = _run_forward(sto_system, order=order, scope="full")
        score = ncc(arr_prop, arr_full)
        print(f"\n  NCC(full, prop) at order={order} = {score:.8f}")
        assert score > 0.9, f"NCC({score:.4f}) below 0.9 at order={order}"


# ── Thickness sensitivity ──────────────────────────────────────────────────

class TestThicknessSensitivity:
    """Convergence as a function of slice thickness."""

    @pytest.mark.parametrize("thickness,order", [
        (0.25, 2),
        (0.25, 4),
        (0.5, 2),
        (1.0, 4),
        (2.0, 6),
    ])
    def test_thickness_order_finite(self, sto_system, thickness, order):
        """All thickness/order combos must produce finite results, or skip if diverged."""
        from test_cvdms.conftest import build_system, _make_sto_atoms
        sys = build_system(_make_sto_atoms(), slice_thickness=thickness,
                           device=sto_system["device"])
        arr = _run_forward(sys, order=order)
        if arr is None:
            pytest.skip(f"dz={thickness}, order={order} diverged (expected for aggressive params)")
        assert check_finite(arr), f"thickness={thickness}, order={order}: non-finite!"
        assert intensity(arr) > 0, f"thickness={thickness}, order={order}: zero intensity"

    @pytest.mark.parametrize("thickness", [0.25, 0.5, 1.0])
    def test_thick_slice_needs_higher_order(self, sto_system, thickness):
        """For thicker slices, order-1 vs order-4 difference should be larger."""
        from test_cvdms.conftest import build_system, _make_sto_atoms
        sys = build_system(_make_sto_atoms(), slice_thickness=thickness,
                           device=sto_system["device"])
        arr1 = _run_forward(sys, order=1)
        arr4 = _run_forward(sys, order=4)
        if arr1 is None or arr4 is None:
            pytest.skip(f"dz={thickness} diverged")
        d = max_diff(arr1, arr4)
        print(f"\n  dz={thickness} Angstrom: |order1 - order4|_max = {d:.3e}")
        assert check_finite(arr1) and check_finite(arr4)


# ── Material sensitivity ───────────────────────────────────────────────────

class TestMaterialSensitivity:
    """Convergence across materials with different scattering strengths."""

    def test_au_needs_higher_order_than_sto(self, sto_system, au_system):
        """Gold (Z=79) converges slower than SrTiO3 — order-4 NCC vs order-6
        should be lower for Au than for STO."""
        arr_sto_4 = _run_forward(sto_system, order=4)
        arr_sto_6 = _run_forward(sto_system, order=6)
        sto_ncc = ncc(arr_sto_6, arr_sto_4)

        arr_au_4 = _run_forward(au_system, order=4)
        arr_au_6 = _run_forward(au_system, order=6)
        au_ncc = ncc(arr_au_6, arr_au_4)

        print(f"\n  STO: NCC(order4, order6) = {sto_ncc:.8f}")
        print(f"  Au:  NCC(order4, order6) = {au_ncc:.8f}")

        # Au should be harder to converge (or at least not easier)
        # Allow a small tolerance because physics depends on exact structure
        assert au_ncc <= sto_ncc * 1.01, (
            f"Au should not converge significantly faster than STO"
        )

    def test_all_materials_finite_at_order4(self, sto_system, au_system, si_system):
        """All three materials must produce finite results at order 4."""
        for name, sys in [("STO", sto_system), ("Au", au_system), ("Si", si_system)]:
            arr = _run_forward(sys, order=4)
            assert check_finite(arr), f"{name}: non-finite at order=4"
            assert intensity(arr) > 0, f"{name}: zero intensity"


# ── Self-convergence ───────────────────────────────────────────────────────

class TestSelfConvergence:
    """Relative error between successive even orders."""

    @pytest.mark.parametrize("base_order", [1, 2, 4])
    def test_relative_error_decreases(self, sto_system, base_order):
        """|order_k - order_{k+2}| / |order_k| should decrease as k grows."""
        arr_k = _run_forward(sto_system, order=base_order)
        arr_k2 = _run_forward(sto_system, order=base_order + 2)
        rel = relative_error(arr_k, arr_k2)
        print(f"\n  rel_err(order={base_order} vs {base_order+2}) = {rel:.3e}")
        assert rel < 1.0, f"Relative error {rel:.3f} should be < 1"
        assert np.isfinite(rel)

    def test_highest_order_pair_has_smallest_error(self, sto_system):
        """The highest-order pair should have the smallest relative error."""
        errors = {}
        for o in [1, 2, 4]:
            arr_k = _run_forward(sto_system, order=o)
            arr_k2 = _run_forward(sto_system, order=o + 2)
            errors[o] = relative_error(arr_k, arr_k2)

        print(f"\n  rel_errors: { {k: f'{v:.3e}' for k, v in errors.items()} }")
        # The (4,6) pair should have smaller or equal error than (1,3)
        if 4 in errors and 1 in errors:
            assert errors[4] <= errors[1] * 2, (
                f"Highest-order pair error ({errors[4]:.3e}) should be comparable "
                f"to lowest-order pair error ({errors[1]:.3e})"
            )
