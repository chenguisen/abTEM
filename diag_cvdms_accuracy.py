"""
CVDMS accuracy diagnostic: systematically map valid parameter regimes.

Tests combinations of (voltage, sampling, thickness) and compares CVDMS
against Fourier multislice reference. Reports:
  - Valid/invalid (inf/nan overflow)
  - Intensity conservation error ΔI/I₀
  - Mean absolute difference from Fourier

Usage:
    python diag_cvdms_accuracy.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import ase
import abtem
from abtem.multislice import CVDMSMultislice, FourierMultislice
# Switch FFT backend from pyfftw (not installed) to numpy
from abtem.core import config as abtem_config
abtem_config.config["fft"] = "numpy"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Test grid ──────────────────────────────────────────────────────
VOLTAGES   = [10e3, 30e3, 80e3, 200e3, 300e3]
SAMPLINGS  = [0.05, 0.10, 0.20]
SLICE_THICKNESS = 0.97
N_SLICES_LIST   = [3, 15, 30]

BULK = ase.build.bulk("Si", cubic=True)
SUPERCELL = BULK * (4, 4, 4)

# ── Print header ───────────────────────────────────────────────────
hdr = f"{'keV':>6} | {'samp':>6} | {'n_slc':>5} | {'thick':>7} | {'CVDMS':>7} | {'Four.':>6} | {'ΔI/I₀':>9} | {'MAE':>9} | {'note'}"
sep = "-" * len(hdr)
print(hdr)
print(sep)

# ── Run tests ──────────────────────────────────────────────────────
for voltage in VOLTAGES:
    for sampling in SAMPLINGS:
        for n_slices in N_SLICES_LIST:
            try:
                # Build potential + probe
                potential = abtem.Potential(
                    SUPERCELL,
                    gpts=(128, 128),
                    slice_thickness=SLICE_THICKNESS,
                    sampling=sampling,
                )
                n_slices_actual = len(potential)
                n_use = min(n_slices, n_slices_actual)
                thickness = n_use * SLICE_THICKNESS

                probe = abtem.Probe(
                    semiangle_cutoff=9.4,
                    energy=voltage,
                ).match_grid(potential)
                probe_arr = probe.build().array
                I0 = float(np.sum(np.abs(probe_arr) ** 2))

                # Fourier reference
                algo_f = FourierMultislice(order=1)
                result_f = probe.multislice(potential[:n_use], algorithm=algo_f)
                arr_f = np.asarray(result_f.array)
                fourier_ok = not (np.any(np.isnan(arr_f)) or np.any(np.isinf(arr_f)))

                # CVDMS
                algo_c = CVDMSMultislice(
                    order=1,
                    convergence_threshold=1e-6,
                    max_terms=50,
                    divergence_ratio=5.0,
                )
                result_c = probe.multislice(potential[:n_use], algorithm=algo_c)
                arr_c = np.asarray(result_c.array)
                cvdms_ok = not (np.any(np.isnan(arr_c)) or np.any(np.isinf(arr_c)))

                # Metrics
                if cvdms_ok and I0 > 0:
                    dI = abs(float(np.sum(np.abs(arr_c)**2)) - I0) / I0
                else:
                    dI = np.nan

                if cvdms_ok and fourier_ok:
                    mae = float(np.mean(np.abs(arr_c - arr_f)))
                else:
                    mae = np.nan

                # Note
                note = ""
                if not cvdms_ok:
                    note = "INF/NAN"
                elif dI > 0.05:
                    note = f"high ΔI ({dI:.1%})"
                elif dI > 0.01:
                    note = f"mod ΔI ({dI:.1%})"

                cvdms_str = "OK" if cvdms_ok else "INF"
                f_str = "OK" if fourier_ok else "INF"
                dI_str = f"{dI:.2e}" if not np.isnan(dI) else "N/A"
                mae_str = f"{mae:.2e}" if not np.isnan(mae) else "N/A"

                print(f"{voltage/1e3:>6.0f} | {sampling:>6.2f} | {n_use:>5d} | "
                      f"{thickness:>7.2f} | {cvdms_str:>7} | {f_str:>6} | "
                      f"{dI_str:>9} | {mae_str:>9} | {note}")

            except Exception as e:
                print(f"{voltage/1e3:>6.0f} | {sampling:>6.2f} | {n_slices:>5d} | "
                      f"  ERR   | {type(e).__name__}: {e}")

print("\nDone.")
