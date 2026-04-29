"""
Baseline parameters and sweep definitions for the CVDMS benchmark suite.

All values extracted from cbed_quickstart.ipynb (SrTiO3, 30 keV).
"""
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Baseline (from cbed_quickstart.ipynb)
# ---------------------------------------------------------------------------
@dataclass
class Baseline:
    """Fixed baseline parameters inherited by all sweeps."""
    # Material
    material: str = "SrTiO3"
    spacegroup: int = 221  # Pm-3m
    lattice_constant: float = 3.905  # Angstrom
    supercell_xy: tuple = (8, 8)
    supercell_z: int = 50  # → ~195 Angstrom total thickness

    # Grid / sampling
    sampling: float = 0.05   # Angstrom
    gpts: tuple = (627, 627)  # full resolution

    # Multislice
    slice_thickness: float = 0.4  # Angstrom
    exit_planes: int = 60

    # Probe
    energy: float = 30e3       # eV
    semiangle_cutoff: float = 35.0  # mrad

    # Frozen phonon
    frozen_phonons: int = 32

    # Algorithm (common)
    convergence_threshold: float = 1e-7
    max_terms: int = 50
    order: int = 1
    backend: str = "auto"

    @property
    def total_thickness(self) -> float:
        return self.supercell_z * self.lattice_constant


# ---------------------------------------------------------------------------
# Fast-mode grid resolution
# ---------------------------------------------------------------------------
FAST_GPTS = 256  # fixed small grid for fast mode
FAST_FROZEN_PHONONS = 4  # reduced FP count for fast mode (except fp sweep)

def fast_gpts(sampling: float) -> tuple:
    """Return (gx, gy) for fast mode — capped at FAST_GPTS."""
    return (FAST_GPTS, FAST_GPTS)


def sampling_gpts(sampling: float, supercell_xy=(8, 8),
                  lattice_constant=3.905, max_gpts=640):
    """Compute gpts from sampling for the sampling sweep (capped at max_gpts).

    The sampling sweep MUST use variable grid sizes to probe
    reciprocal-space resolution. A cap prevents numerical instability
    (CVDMS NaN on very large grids like 781²).
    """
    cell_size = supercell_xy[0] * lattice_constant
    gpts = int(round(cell_size / sampling))
    gpts = min(gpts, max_gpts)
    gpts = max(gpts, 128)
    return (gpts, gpts)


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------
@dataclass
class SweepDef:
    """A single parameter sweep."""
    name: str                      # short key: "voltage", "fp", ...
    parameter: str                 # kwarg name passed to simulation
    label: str                     # human-readable, e.g. "Accelerating voltage (keV)"
    values: list                   # parameter values
    unit: str = ""
    full_resolution: bool = False  # run at 627x627 instead of fast grid

    # Fixed overrides (relative to baseline)
    fixed: dict = field(default_factory=dict)


SWEEPS: list[SweepDef] = [
    SweepDef(
        name="voltage",
        parameter="energy",
        label="Accelerating voltage",
        values=[30e3, 80e3, 100e3, 200e3, 300e3],
        unit="eV",
        full_resolution=False,
        fixed={},
    ),
    SweepDef(
        name="fp",
        parameter="frozen_phonons",
        label="Frozen phonon configurations",
        values=[1, 4, 8, 16, 32],
        full_resolution=False,
        fixed={},
    ),
    SweepDef(
        name="sampling",
        parameter="sampling",
        label="Real-space sampling",
        values=[0.04, 0.05, 0.07, 0.10],
        unit="Å",
        full_resolution=False,
        fixed={},
    ),
    SweepDef(
        name="thickness",
        parameter="total_thickness",
        label="Total specimen thickness",
        values=[5, 10, 15, 20, 25],
        unit="nm",
        full_resolution=False,
        fixed={},
    ),
    SweepDef(
        name="slice_thickness",
        parameter="slice_thickness",
        label="Slice thickness",
        values=[0.2, 0.4, 0.6, 0.8, 1.0],
        unit="Å",
        full_resolution=False,
        fixed={},
    ),
]

ALGORITHMS = [
    "fourier",
    "cvdms_fd",
    "cvdms_bsc",
]

ALGORITHM_LABELS = {
    "fourier": "Fourier multislice",
    "cvdms_fd": "CVDMS (forward only)",
    "cvdms_bsc": "CVDMS (with backscattering)",
}

ALGORITHM_COLORS = {
    "fourier": "#2166AC",
    "cvdms_fd": "#1B7837",
    "cvdms_bsc": "#B2182B",
}


def resolve_sweep_params(baseline: Baseline, sweep: SweepDef, value):
    """Return a dict of parameters for a specific sweep point."""
    params = {
        "material": baseline.material,
        "spacegroup": baseline.spacegroup,
        "lattice_constant": baseline.lattice_constant,
        "supercell_xy": baseline.supercell_xy,
        "sampling": baseline.sampling,
        "slice_thickness": baseline.slice_thickness,
        "supercell_z": baseline.supercell_z,
        "energy": baseline.energy,
        "semiangle_cutoff": baseline.semiangle_cutoff,
        "frozen_phonons": baseline.frozen_phonons,
        "convergence_threshold": baseline.convergence_threshold,
        "max_terms": baseline.max_terms,
        "order": baseline.order,
        "backend": baseline.backend,
        "exit_planes": baseline.exit_planes,
        "total_thickness": baseline.total_thickness,
    }
    # Apply sweep value
    params[sweep.parameter] = value

    # Apply fixed overrides
    params.update(sweep.fixed)

    # Special: thickness sweep changes supercell_z
    if sweep.parameter == "total_thickness":
        thickness_angstrom = value * 10  # nm → Å
        z_repeat = max(3, int(round(thickness_angstrom / baseline.lattice_constant)))
        params["supercell_z"] = z_repeat
        params["total_thickness"] = z_repeat * baseline.lattice_constant / 10  # nm

    return params


def format_value(sweep: SweepDef, value) -> str:
    """Human-readable label for a sweep value."""
    if sweep.parameter == "energy":
        return f"{value/1e3:.0f} keV"
    if sweep.parameter == "total_thickness":
        return f"{value} nm"
    if sweep.unit:
        return f"{value} {sweep.unit}"
    return str(value)
