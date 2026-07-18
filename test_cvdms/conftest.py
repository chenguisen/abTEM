"""
Shared test fixtures for the cvdms convergence test suite.

Provides multi-material test systems (STO, Au, Si) with Stage A / Stage B
grid sizes, GPU marker, and pytest configuration.

All fixtures use ``projection="finite"`` and a default slice thickness of
0.5 Angstrom (finer than the 0.75 Angstrom default in test/).
"""

import warnings

import ase
import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, settings

import abtem
from abtem import config

config.set({"diagnostics.progress_bar": False})

settings.register_profile(
    "dev",
    max_examples=20,
    print_blob=True,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    phases=[Phase.generate],
)
settings.load_profile("dev")


def pytest_configure(config):
    warnings.filterwarnings("ignore", category=UserWarning)
    config.addinivalue_line(
        "markers",
        "multigpu: requires >=2 GPUs and dask-cuda; skipped otherwise",
    )
    config.addinivalue_line(
        "markers",
        "stage_b: larger grid (512x512), longer runtime; use -k 'not stage_b' for CI",
    )


# ── GPU marker (from test/utils.py pattern) ────────────────────────────────

try:
    from abtem.core.backend import cp
except ImportError:
    cp = None

gpu = pytest.param("gpu", marks=pytest.mark.skipif(cp is None, reason="no gpu"))


# ── Stage grid sizes ───────────────────────────────────────────────────────

STAGE_A_GPTS = (128, 128)
STAGE_B_GPTS = (512, 512)
DEFAULT_SLICE_THICKNESS = 0.5  # finer than test/ default of 0.75


# ── Atom builders ──────────────────────────────────────────────────────────

def _make_sto_atoms():
    """SrTiO3 [001], spacegroup 221, a = 3.9127 Angstrom."""
    unit_cell = ase.Atoms(
        symbols="SrTiO3",
        scaled_positions=[
            [0.0, 0.0, 0.0],       # Sr
            [0.5, 0.5, 0.5],       # Ti
            [0.5, 0.0, 0.5],       # O
            [0.5, 0.5, 0.0],       # O
            [0.0, 0.5, 0.5],       # O
        ],
        cell=[3.9127, 3.9127, 3.9127],
        pbc=True,
    )
    return unit_cell * (2, 2, 8)  # ~31 Angstrom thick, plenty of slices


def _make_au_atoms():
    """Gold [001], FCC, a = 4.08 Angstrom.  Strong scatterer (Z=79)."""
    unit_cell = ase.Atoms(
        symbols="Au",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=[4.08, 4.08, 4.08],
        pbc=True,
    )
    return unit_cell * (2, 2, 8)


def _make_si_atoms():
    """Silicon [001], diamond cubic, a = 5.43 Angstrom.  Medium scatterer."""
    unit_cell = ase.Atoms(
        symbols="Si2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    return unit_cell * (2, 2, 8)


# ── System builders (reusable by both fixtures and standalone scripts) ─────

def build_system(atoms, gpts=STAGE_A_GPTS, slice_thickness=DEFAULT_SLICE_THICKNESS,
                 exit_planes=1, energy=30e3, semiangle_cutoff=20, device="cpu"):
    """Build a complete test system dict from atoms.

    Parameters
    ----------
    atoms : ase.Atoms
    gpts : tuple[int, int]
    slice_thickness : float
        Slice thickness in Angstrom.
    exit_planes : int
        Number of exit planes (equally spaced).  If 0 or None the potential
        is built *without* exit_planes (forward-only use).
    energy : float
        Electron energy in eV.  30 keV is the primary test energy.
    semiangle_cutoff : float
        Probe semiangle in mrad.
    device : str
        "cpu" or "gpu".
    """
    kwargs = dict(
        gpts=gpts,
        slice_thickness=slice_thickness,
        projection="finite",
        device=device,
    )

    if exit_planes:
        potential_ep = abtem.Potential(atoms, exit_planes=exit_planes, **kwargs)
    else:
        potential_ep = None

    potential = abtem.Potential(atoms, **kwargs)

    probe = abtem.Probe(
        semiangle_cutoff=semiangle_cutoff,
        energy=energy,
        device=device,
    ).match_grid(potential)

    plane_wave = abtem.PlaneWave(energy=energy, device=device).match_grid(potential)

    return {
        "atoms": atoms,
        "potential": potential,
        "potential_exit_planes": potential_ep,
        "probe": probe,
        "plane_wave": plane_wave,
        "device": device,
        "gpts": gpts,
        "energy": energy,
        "slice_thickness": slice_thickness,
    }


# ── Pytest fixtures ────────────────────────────────────────────────────────

def _parse_fixture_params(request):
    """Parse request.param which may be a string (legacy CPU/GPU), a dict, or None."""
    if not hasattr(request, "param") or request.param is None:
        return {"device": "cpu", "stage": "A", "energy": 30e3, "exit_planes": 1}

    param = request.param
    if isinstance(param, str):
        # Legacy: "cpu" or gpu marker
        return {"device": param, "stage": "A", "energy": 30e3, "exit_planes": 1}

    if isinstance(param, dict):
        return {
            "device": param.get("device", "cpu"),
            "stage": param.get("stage", "A"),
            "energy": param.get("energy", 30e3),
            "exit_planes": param.get("exit_planes", 1),
        }

    # Fallback for pytest Param objects
    return {"device": str(param), "stage": "A", "energy": 30e3, "exit_planes": 1}


@pytest.fixture
def sto_system(request):
    """SrTiO3 test system.

    Can be parametrized via ``@pytest.mark.parametrize`` with ``indirect=True``::

        @pytest.mark.parametrize("sto_system", ["cpu", gpu], indirect=True)
        def test_on_both_devices(sto_system): ...

        @pytest.mark.parametrize("sto_system", [
            pytest.param({"device": "cpu", "stage": "B"}, id="cpu-B"),
        ], indirect=True)
        def test_stage_b(sto_system): ...
    """
    p = _parse_fixture_params(request)
    gpts = STAGE_A_GPTS if p["stage"] == "A" else STAGE_B_GPTS
    return build_system(_make_sto_atoms(), gpts=gpts, exit_planes=p["exit_planes"],
                        energy=p["energy"], device=p["device"])


@pytest.fixture
def au_system(request):
    """Gold test system (strong scatterer, Z=79).  Same parametrization as ``sto_system``."""
    p = _parse_fixture_params(request)
    gpts = STAGE_A_GPTS if p["stage"] == "A" else STAGE_B_GPTS
    return build_system(_make_au_atoms(), gpts=gpts, exit_planes=p["exit_planes"],
                        energy=p["energy"], device=p["device"])


@pytest.fixture
def si_system(request):
    """Silicon test system (medium scatterer, Z=14).  Same parametrization as ``sto_system``."""
    p = _parse_fixture_params(request)
    gpts = STAGE_A_GPTS if p["stage"] == "A" else STAGE_B_GPTS
    return build_system(_make_si_atoms(), gpts=gpts, exit_planes=p["exit_planes"],
                        energy=p["energy"], device=p["device"])
