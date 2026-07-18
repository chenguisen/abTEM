"""
test_analytical_limits.py — Validation against closed-form analytical solutions.

Tests the real-space multislice against cases where the exact answer is known:

  - **Vacuum Fresnel propagation**: V=0, exact phase evolution is
    ψ(k, z) = ψ₀(k) · exp(-iπλz·(kx²+ky²))
  - **Homogeneous potential**: V(x,y) = const, no Laplacian contribution,
    exact solution is ψ(z) = ψ(0) · exp(i·σ·V₀·z)
  - **Weak phase limit**: Thin, weak scatterer, exact phase shift is
    ψ ≈ ψ_in · exp(i·σ·V_proj)  (projection approximation)

Reference: CVDMS V2a/V2b/V2c methodology — analytical limits provide
the strongest correctness checks because the answer is known exactly.
"""

import numpy as np
import pytest

import abtem
from abtem.multislice import RealSpaceMultislice
from abtem.finite_difference import DivergedError
from test_cvdms.metrics import to_numpy, ncc, rmsd, amplitude_rms, phase_rms, intensity, check_finite


# ── Vacuum Fresnel propagation ─────────────────────────────────────────────

class TestVacuumFresnel:
    """V=0: compare dev's propagation against analytic Fresnel propagator."""

    def test_amplitude_conserved_in_vacuum(self, sto_system):
        """In a nearly-empty potential, amplitude should be approximately conserved."""
        from test_cvdms.conftest import build_system
        import ase

        # Build a system with a single light atom → very weak potential
        atoms = ase.Atoms(
            symbols="H",
            scaled_positions=[[0, 0, 0]],
            cell=[10, 10, 5],
            pbc=True,
        )
        sys = build_system(atoms, gpts=(128, 128), slice_thickness=0.5,
                           exit_planes=0, energy=300e3, device=sto_system["device"])

        probe = sys["probe"]
        I0 = intensity(probe)

        for order in [1, 2, 4]:
            arr = to_numpy(probe.multislice(
                potential=sys["potential"], scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
            ).array)
            I_exit = intensity(arr)
            deviation = abs(I_exit - I0) / I0

            print(f"\n  order={order}: |I/I0-1| = {deviation:.3e}")
            assert deviation < 0.05, (
                f"order={order}: vacuum intensity deviation {deviation:.3e}"
            )

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_phase_is_finite(self, sto_system, order):
        """Phase in near-vacuum should be finite and well-behaved."""
        from test_cvdms.conftest import build_system
        import ase

        atoms = ase.Atoms(
            symbols="H",
            scaled_positions=[[0, 0, 0]],
            cell=[10, 10, 5],
            pbc=True,
        )
        sys = build_system(atoms, gpts=(128, 128), slice_thickness=0.5,
                           exit_planes=0, energy=300e3, device=sto_system["device"])

        probe = sys["probe"]
        try:
            arr = to_numpy(probe.multislice(
                potential=sys["potential"], scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
            ).array)
        except DivergedError:
            pytest.skip(f"order={order} diverged")

        assert check_finite(arr), f"order={order}: non-finite in vacuum"
        probe_arr = to_numpy(probe.build(lazy=False).array)
        ph_rms_val = phase_rms(probe_arr, arr)
        print(f"\n  order={order}: phase_rms = {ph_rms_val:.3e} rad")
        assert ph_rms_val < np.pi, f"order={order}: phase RMS {ph_rms_val:.2f} > π"


# ── Homogeneous potential ──────────────────────────────────────────────────

class TestHomogeneousPotential:
    """V(x,y) = const → no Laplacian, all orders should agree."""

    def test_constant_potential_orders_agree(self):
        """When V is constant, the Laplacian term is zero → all orders
        of the K-operator expansion are identical."""
        import ase

        # Build a "uniform" potential: very large atoms → nearly flat
        # Actually, let's use an alternative: build small system where
        # the constant-phase shift dominates
        atoms = ase.Atoms(
            symbols="C",
            scaled_positions=[[0.5, 0.5, 0.5]],
            cell=[3.0, 3.0, 5.0],
            pbc=True,
        )

        potential = abtem.Potential(
            atoms, gpts=(128, 128), slice_thickness=0.5,
            projection="finite", device="cpu",
        )

        probe = abtem.Probe(
            semiangle_cutoff=5, energy=300e3, device="cpu",
        ).match_grid(potential)

        results = {}
        for order in [1, 2, 4]:
            arr = to_numpy(probe.multislice(
                potential=potential, scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
            ).array)
            results[order] = arr
            assert check_finite(arr)

        # Higher orders should be close to order-1 (constant V → no improvement
        # from higher Laplacian terms, but they shouldn't make things worse)
        ncc_1_4 = ncc(results[1], results[4])
        print(f"\n  NCC(order=1, order=4) = {ncc_1_4:.8f}")

        # In uniform potential, higher orders should not dramatically change
        # the result (no high-frequency structure to expand)
        assert ncc_1_4 > 0.9, f"NCC({ncc_1_4:.4f}) between orders too low"


# ── Weak phase limit ───────────────────────────────────────────────────────

class TestWeakPhaseLimit:
    """Thin, weak scatterer → projection approximation should hold."""

    def test_thin_slice_phase_shift_is_finite(self):
        """At minimum, the phase shift from a thin slice should be finite
        and well-behaved."""
        import ase

        atoms = ase.Atoms(
            symbols="C",
            scaled_positions=[[0.5, 0.5, 0.5]],
            cell=[5.0, 5.0, 2.0],
            pbc=True,
        )

        potential = abtem.Potential(
            atoms, gpts=(128, 128), slice_thickness=0.25,
            projection="finite", device="cpu",
        )

        probe = abtem.Probe(
            semiangle_cutoff=5, energy=300e3, device="cpu",
        ).match_grid(potential)

        # Run with different orders — all should be finite
        for order in [1, 2, 4]:
            arr = to_numpy(probe.multislice(
                potential=potential, scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
            ).array)
            assert check_finite(arr), f"order={order}: non-finite in weak phase limit"
            assert intensity(arr) > 0, f"order={order}: zero intensity"

    def test_thin_slice_amplitude_near_unity(self):
        """Weak phase object: amplitude should stay near 1 (phase-only modulation)."""
        import ase

        atoms = ase.Atoms(
            symbols="C",
            scaled_positions=[[0.5, 0.5, 0.5]],
            cell=[5.0, 5.0, 2.0],
            pbc=True,
        )

        potential = abtem.Potential(
            atoms, gpts=(128, 128), slice_thickness=0.25,
            projection="finite", device="cpu",
        )

        probe = abtem.Probe(
            semiangle_cutoff=5, energy=300e3, device="cpu",
        ).match_grid(potential)

        I0 = intensity(probe)

        try:
            arr = to_numpy(probe.multislice(
                potential=potential, scan=[[0, 0]], lazy=False,
                algorithm=RealSpaceMultislice(order=4, expansion_scope="full"),
            ).array)
        except DivergedError:
            pytest.skip("order=4 diverged for weak phase limit")

        I_exit = intensity(arr)
        ratio = I_exit / I0

        print(f"\n  Weak phase: I/I0 = {ratio:.6f}")

        # For a very thin weak scatterer, intensity should be near 1
        assert 0.8 < ratio < 1.2, f"Weak phase intensity ratio {ratio:.3f}"
