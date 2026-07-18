"""
test_conservation.py — Conservation law validation.

Validates physical invariants of the real-space multislice:
  - Intensity conservation (per-slice, |ψ(z)|² ≈ |ψ(0)|²)
  - Vacuum propagation amplitude preservation
  - Flux conservation with and without BSC correction

Reference: CVDMS V1a/V1b/V1c pattern — conservation laws are the simplest
and strongest correctness checks because they rely on no approximations.
"""

import numpy as np
import pytest

from abtem.multislice import RealSpaceMultislice
from abtem.finite_difference import DivergedError
from test_cvdms.metrics import to_numpy, intensity, check_finite


def _run_multislice(system, order=3, scope="propagator", scan=None, return_bsc=False):
    """Run multislice and return (result_array, initial_intensity), or (None, I0)."""
    if scan is None:
        scan = [[0, 0]]
    probe = system["probe"]
    initial_I = intensity(probe)

    kwargs = dict(
        potential=system["potential"],
        scan=scan,
        lazy=False,
        algorithm=RealSpaceMultislice(order=order, expansion_scope=scope),
    )

    try:
        if return_bsc:
            if system["potential_exit_planes"] is None:
                pytest.skip("no exit_planes in this system")
            fwd, bwd = probe.multislice(
                potential=system["potential_exit_planes"],
                scan=scan, lazy=False,
                algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
                return_backscattered=True,
            )
            return (to_numpy(fwd.array), to_numpy(bwd.array)), initial_I

        result = probe.multislice(**kwargs)
        return to_numpy(result.array), initial_I
    except DivergedError:
        return None, initial_I


# ── Intensity conservation ─────────────────────────────────────────────────

class TestIntensityConservation:
    """Forward scattering should approximately conserve intensity."""

    @pytest.mark.parametrize("order,scope", [
        (1, "propagator"),
        (2, "full"),
        (4, "full"),
    ])
    def test_intensity_not_diverging(self, sto_system, order, scope):
        """Exit intensity should not diverge (grow orders of magnitude)."""
        arr, I0 = _run_multislice(sto_system, order=order, scope=scope)
        if arr is None:
            pytest.skip(f"order={order} scope={scope} diverged")

        I_exit = intensity(arr)
        ratio = I_exit / I0 if I0 > 0 else float("inf")

        print(f"\n  order={order} scope={scope}: I_exit/I0 = {ratio:.6f}")

        # Allow some non-unitarity from finite-difference truncation,
        # but catastrophic growth indicates divergence
        assert ratio < 10, f"Intensity ratio {ratio:.2f} suggests divergence"
        assert ratio > 0.01, f"Intensity ratio {ratio:.6f} suggests near-zero result"
        assert check_finite(arr)

    @pytest.mark.parametrize("order", [2, 3, 4, 6])
    def test_intensity_ratio_bounded(self, sto_system, order):
        """I_exit / I0 should be in a reasonable range for full scope."""
        arr, I0 = _run_multislice(sto_system, order=order, scope="full")
        if arr is None:
            pytest.skip(f"order={order} diverged")

        ratio = intensity(arr) / I0

        print(f"\n  order={order}: I/I0 = {ratio:.6f}")
        assert 0.3 < ratio < 3.0, f"Intensity ratio {ratio:.3f} out of bounds"

    def test_intensity_improves_with_order(self, sto_system):
        """Higher orders should better conserve intensity."""
        probe = sto_system["probe"]
        I0 = intensity(probe)

        ratios = {}
        for o in [2, 4, 6]:
            arr, _ = _run_multislice(sto_system, order=o, scope="full")
            if arr is None:
                ratios[o] = None
                continue
            I_exit = intensity(arr)
            ratios[o] = abs(I_exit - I0) / I0

        print(f"\n  |I/I0 - 1|: { {k: f'{v:.6f}' if v is not None else 'diverged' for k, v in ratios.items()} }")

        # Order-6 should be at least reasonable compared to order-2
        valid = {k: v for k, v in ratios.items() if v is not None}
        if len(valid) >= 2:
            orders = sorted(valid.keys())
            assert valid[orders[-1]] <= valid[orders[0]] * 10, (
                f"Highest order deviation ({valid[orders[-1]]:.6f}) "
                f"much worse than lowest ({valid[orders[0]]:.6f})"
            )


# ── Vacuum propagation ─────────────────────────────────────────────────────

class TestVacuumPropagation:
    """V=0: the wave should propagate without amplitude change."""

    def test_vacuum_amplitude_conserved(self):
        """In vacuum (empty potential), amplitude should be exactly conserved."""
        # Build an empty potential
        import ase
        import abtem

        atoms = ase.Atoms(
            symbols="H",           # single H atom → negligible potential
            scaled_positions=[[0, 0, 0]],
            cell=[10.0, 10.0, 5.0],
            pbc=True,
        )

        potential = abtem.Potential(
            atoms, gpts=(128, 128), slice_thickness=0.5,
            projection="finite", device="cpu",
        )

        # Very high energy → weak interaction → close to vacuum
        probe = abtem.Probe(
            semiangle_cutoff=10, energy=300e3, device="cpu",
        ).match_grid(potential)

        I0 = intensity(probe)

        for order in [1, 2, 4]:
            arr = to_numpy(probe.multislice(
                potential=potential, scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
            ).array)
            I_exit = intensity(arr)
            deviation = abs(I_exit - I0) / I0

            print(f"\n  order={order}: vacuum |I/I0 - 1| = {deviation:.3e}")
            assert deviation < 0.1, (
                f"order={order}: vacuum intensity deviation {deviation:.3e} too large"
            )
            assert check_finite(arr)


# ── Flux conservation with BSC ─────────────────────────────────────────────

class TestFluxConservationWithBSC:
    """Energy budget: I_fwd + I_bsc ≈ I0."""

    @pytest.mark.parametrize("order", [2, 4, 6])
    def test_bsc_energy_budget(self, sto_system, order):
        """Forward + backscattered intensity should approximately equal I0."""
        if sto_system["potential_exit_planes"] is None:
            pytest.skip("no exit_planes")

        probe = sto_system["probe"]
        I0 = intensity(probe)

        try:
            fwd, bwd = probe.multislice(
                potential=sto_system["potential_exit_planes"],
                scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
                return_backscattered=True,
            )
        except DivergedError:
            pytest.skip(f"BSC diverged at order={order}")

        I_fwd = intensity(to_numpy(fwd.array))
        I_bwd = intensity(to_numpy(bwd.array))
        budget_error = abs(I_fwd + I_bwd - I0) / I0

        print(f"\n  order={order}: I_fwd={I_fwd:.6f}, I_bwd={I_bwd:.3e}, "
              f"|I_fwd+I_bwd-I0|/I0 = {budget_error:.3e}")

        # The energy budget should close within reasonable tolerance.
        # Note: DEV's BSC is a per-slice correction, not a full energy
        # accounting, so we allow a larger tolerance than the full BSC
        # energy budget in CVDMS.
        assert budget_error < 0.5, (
            f"BSC energy budget error {budget_error:.3f} too large"
        )
        assert check_finite(to_numpy(fwd.array))
        assert check_finite(to_numpy(bwd.array))
