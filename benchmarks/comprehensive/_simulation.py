"""
Simulation runner — builds structures, potentials, probes, and runs algorithms.
"""
import time
import numpy as np
from ase.spacegroup import crystal
from abtem import Potential, Probe, FrozenPhonons
from abtem.multislice import FourierMultislice, CVDMSMultislice


class SimulationRunner:
    """Build and run multislice simulations for benchmark sweeps."""

    def __init__(self, device="gpu", verbose=True):
        self.device = device
        self.verbose = verbose

        # Configure abTEM for GPU
        from abtem.core import config as _cfg
        _cfg.set({"device": "gpu", "fft": "cupy"})

    def log(self, msg):
        if self.verbose:
            print(f"  [runner] {msg}")

    def build_structure(self, supercell_xy=(8, 8), supercell_z=50,
                        material="SrTiO3", spacegroup=221,
                        lattice_constant=3.905):
        """Build SrTiO3 supercell."""
        self.log(f"Building SrTiO3 ({supercell_xy[0]}x{supercell_xy[1]}x{supercell_z})...")
        # SrTiO3 perovskite: Pm-3m (#221), a=3.905
        atoms = crystal(
            ["Sr", "Ti", "O"],
            basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
            spacegroup=221,
            cellpar=[lattice_constant, lattice_constant, lattice_constant,
                     90, 90, 90],
        )
        atoms *= (supercell_xy[0], supercell_xy[1], supercell_z)
        return atoms

    def build_potential(self, atoms, sampling=0.05, slice_thickness=0.4,
                        gpts=None, frozen_phonons=32, exit_planes=60):
        """Build abTEM Potential with frozen phonons."""
        self.log(f"Building potential: sampling={sampling}, dz={slice_thickness}, "
                 f"gpts={gpts}, FP={frozen_phonons}...")

        # Apply frozen phonons if requested
        if frozen_phonons > 1:
            # SrTiO3 isotropic RMSD values (from notebook)
            fp_sigmas = {
                'Sr': 0.164356,
                'Ti': 0.116584,
                'O': 0.148198,
            }
            # Build sigmas list matching atom order in atoms
            sigmas = [fp_sigmas.get(atoms[i].symbol, 0.1)
                      for i in range(len(atoms))]
            atoms = FrozenPhonons(atoms, num_configs=frozen_phonons,
                                  sigmas=sigmas, seed=42)
            self.log(f"  FP wrapper created: {frozen_phonons} configs")

        # Pass either gpts or sampling, not both (to avoid overspecification)
        pot_kwargs = dict(
            slice_thickness=slice_thickness,
            projection="finite",
            exit_planes=exit_planes,
        )
        if gpts is not None:
            pot_kwargs["gpts"] = gpts
        else:
            pot_kwargs["sampling"] = sampling

        pot = Potential(atoms, **pot_kwargs)
        return pot

    def build_probe(self, potential, energy=30e3, semiangle_cutoff=35.0):
        """Build abTEM Probe matched to the potential grid."""
        self.log(f"Building probe: energy={energy/1e3:.0f} keV, "
                 f"semiangle={semiangle_cutoff} mrad...")
        probe = Probe(energy=energy, semiangle_cutoff=semiangle_cutoff)
        probe.match_grid(potential)
        return probe

    def run_algorithm(self, potential, probe, algorithm="fourier",
                      convergence_threshold=1e-6, max_terms=50,
                      order=1, backend="auto"):
        """Run one algorithm and return exit waves + CBED pattern.

        Returns dict with keys:
            exit_wave: complex ndarray (batch, nx, ny) or (nx, ny)
            cbed: ndarray (diffraction pattern)
            time: wall-clock seconds
            diagnostics: dict with extra info
        """
        self.log(f"Running {algorithm}...")

        if algorithm == "fourier":
            algo = FourierMultislice(order=1)
        elif algorithm == "cvdms_fd":
            algo = CVDMSMultislice(
                order=order,
                max_terms=max_terms,
                convergence_threshold=convergence_threshold,
                backscattering=False,
                use_fused_kernel=True,
                backend=backend,
                laplace_method="finite-difference",
            )
        elif algorithm == "cvdms_bsc":
            algo = CVDMSMultislice(
                order=order,
                max_terms=max_terms,
                convergence_threshold=convergence_threshold,
                backscattering=True,
                use_fused_kernel=True,
                backend=backend,
                laplace_method="finite-difference",
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        t0 = time.time()
        exit_waves = probe.multislice(potential, algorithm=algo)
        t1 = time.time()

        # Diffraction pattern: CBED via Fourier transform of exit wave
        # For the full exit wave, compute the diffraction pattern
        if hasattr(exit_waves, 'array'):
            ew_array = exit_waves.array
        else:
            ew_array = exit_waves

        # Compute CBED: |FFT(exit_wave)|^2, summed over batch
        ew = self._to_numpy(ew_array)
        cbed = self._compute_cbed(ew)

        # Track I0 (initial probe intensity)
        if hasattr(probe, 'array'):
            probe_array = self._to_numpy(probe.array)
            I0 = float(np.sum(np.abs(probe_array) ** 2))
        else:
            I0 = 1.0  # fallback

        # Intensity map at exit
        intensity_map = np.abs(ew) ** 2

        result = {
            "exit_wave": ew,
            "cbed": cbed,
            "intensity_map": intensity_map,
            "I0": I0,
            "time": t1 - t0,
            "diagnostics": {
                "algorithm": algorithm,
                "converged": True,  # would need algorithm-specific diagnostics
            },
        }
        self.log(f"  → done in {t1-t0:.1f}s")
        return result

    @staticmethod
    def _to_numpy(arr):
        """Convert cupy/dask/device array to numpy."""
        import cupy as cp
        # Convert dask -> cupy first
        if hasattr(arr, 'compute'):
            arr = arr.compute()
        # Convert cupy -> numpy
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    @staticmethod
    def _compute_cbed(exit_wave):
        """Compute CBED pattern from exit wave (numpy array)."""
        ew = np.asarray(exit_wave)
        if ew.ndim == 3:
            f = np.fft.fft2(ew, axes=(-2, -1))
            cbed = np.mean(np.abs(f) ** 2, axis=0)
        else:
            f = np.fft.fft2(ew)
            cbed = np.abs(f) ** 2
        return np.fft.fftshift(cbed)

    def cleanup(self):
        """Free GPU memory."""
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass
