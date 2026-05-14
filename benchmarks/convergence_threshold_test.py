"""
Self-convergence test for CVDMS convergence_threshold.

Tests whether convergence_threshold=1e-6 is sufficient by sweeping over
thresholds and comparing each result against the tightest (1e-8).

When NCC between a given threshold and the tightest reference → 1.0,
the series is numerically converged at that threshold.

Tests both cvdms_fd (forward-only) and cvdms_bsc (backscattering).
"""
import sys, os, time, json, warnings
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABTEM_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if ABTEM_ROOT not in sys.path:
    sys.path.insert(0, ABTEM_ROOT)

from abtem import Potential, Probe, FrozenPhonons
from abtem.multislice import CVDMSMultislice
from benchmarks.comprehensive._simulation import SimulationRunner
from benchmarks.comprehensive._metrics import ncc, rmsd, compute_all_metrics

# ── Parameters (fast mode) ──────────────────────────────────────
SAMPLING = 0.05
GPT = 256
FP = 1          # single FP; convergence is per-config
ENERGY = 30e3
SEMIANGLE = 35.0
SLICE_DZ = 0.4
SUPERCELL_XY = (8, 8)
SUPERCELL_Z = 50
EXIT_PLANES = 60
ORDER = 1
MAX_TERMS = 50
MAX_INNER = 100
FP_SIGMAS = {"Sr": 0.164356, "Ti": 0.116584, "O": 0.148198}

THRESHOLDS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
REF_THRESH = 1e-8   # tightest = convergence reference

# ── Build shared structure + potential ──────────────────────────
runner = SimulationRunner(device="gpu", verbose=False)
atoms = runner.build_structure(SUPERCELL_XY, SUPERCELL_Z)
fp_sigmas_list = [FP_SIGMAS.get(atoms[i].symbol, 0.1) for i in range(len(atoms))]
fp_atoms = FrozenPhonons(atoms, num_configs=FP, sigmas=fp_sigmas_list, seed=42)
potential = Potential(
    fp_atoms, sampling=SAMPLING, gpts=GPT,
    slice_thickness=SLICE_DZ, projection="finite",
    exit_planes=EXIT_PLANES,
)
probe = Probe(energy=ENERGY, semiangle_cutoff=SEMIANGLE)
probe.match_grid(potential)
total_z = atoms.cell[2, 2]
n_slices = int(np.ceil(total_z))
probe_wave = probe.build(lazy=False)
probe_array = runner._to_numpy(probe_wave.array)
I0 = float(np.sum(np.abs(probe_array) ** 2))

print("=" * 70)
print("CVDMS Convergence Threshold Self-Test")
print("=" * 70)
print(f"Grid: {GPT}×{GPT}, Sampling: {SAMPLING} Å, FP={FP}")
print(f"Energy: {ENERGY/1e3:.0f} keV, Semiangle: {SEMIANGLE} mrad")
print(f"Slice: {SLICE_DZ} Å, Slices: {n_slices}, dz={SLICE_DZ} Å")
print(f"Reference: threshold = {REF_THRESH:.0e}")
print()

# ── Run CVDMS: FD + BSC at every threshold ──────────────────────
results = {}   # (alg, thresh) -> {cbed, ew, time, converged, warnings}

for bsc, alg_name in [(False, "cvdms_fd"), (True, "cvdms_bsc")]:
    print(f"── {alg_name} ──")
    for thresh in THRESHOLDS:
        algo = CVDMSMultislice(
            order=ORDER, max_terms=MAX_TERMS, max_inner=MAX_INNER,
            convergence_threshold=thresh,
            backscattering=bsc,
            use_fused_kernel=True, backend="c++",
            laplace_method="finite-difference",
        )

        warn_list = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t0 = time.time()
            cw = probe.multislice(potential, algorithm=algo)
            if hasattr(cw, 'compute'):
                cw.compute()
            ct = time.time() - t0
            warn_list = w

        ew_np = runner._to_numpy(cw.array if hasattr(cw, 'array') else cw)
        cbed = runner._compute_cbed(ew_np)

        converged = True
        for warning in warn_list:
            if "did not fully converge" in str(warning.message):
                converged = False
                break

        results[(alg_name, thresh)] = {
            "cbed": cbed,
            "ew": ew_np,
            "time": ct,
            "converged": converged,
            "warnings": [str(w.message) for w in warn_list],
        }
        print(f"  {thresh:.0e}  {ct:.1f}s  converged={converged}")
        if not converged:
            for ww in warn_list:
                print(f"    ⚠ {ww.message}")
    print()

runner.cleanup()

# ── Self-convergence: each threshold vs REF_THRESH (1e-8) ───────
print("=" * 70)
print(f"Self-Convergence: NCC vs threshold = {REF_THRESH:.0e} (tightest)")
print("=" * 70)

for alg_name in ["cvdms_fd", "cvdms_bsc"]:
    print(f"\n── {alg_name} ──")
    ref_cbed = results[(alg_name, REF_THRESH)]["cbed"]
    ref_ew = results[(alg_name, REF_THRESH)]["ew"]
    print(f"{'Threshold':>10} | {'NCC':>10} | {'1-NCC':>10} | {'RMSD':>10} | {'IC':>12} | {'Time':>8} | {'Conv':>8}")
    print("-" * 80)

    for th in THRESHOLDS:
        r = results[(alg_name, th)]
        ncc_val = ncc(ref_cbed, r["cbed"])
        rmsd_val = rmsd(ref_cbed, r["cbed"])
        ic_val = compute_all_metrics(r["ew"], r["cbed"], I0=I0).get("intensity_conservation", -1)

        print(f"{th:>10.0e} | {ncc_val:>10.6f} | {1-ncc_val:>10.2e} | {rmsd_val:>10.2e} | {ic_val:>12.4e} | {r['time']:>8.1f} | {str(r['converged']):>8}")

# ── Successive-difference: adjacent thresholds ──────────────────
print(f"\n{'='*70}")
print("Successive Difference: NCC between adjacent thresholds")
print("(measures incremental change from tightening)")
print(f"{'='*70}")

for alg_name in ["cvdms_fd", "cvdms_bsc"]:
    print(f"\n── {alg_name} ──")
    print(f"{'Loose':>10} → {'Tight':>10} | {'NCC':>10} | {'1-NCC':>10}")
    print("-" * 50)
    for i in range(1, len(THRESHOLDS)):
        t_lo = THRESHOLDS[i-1]
        t_hi = THRESHOLDS[i]
        ncc_val = ncc(results[(alg_name, t_hi)]["cbed"], results[(alg_name, t_lo)]["cbed"])
        print(f"{t_lo:>10.0e} → {t_hi:>10.0e} | {ncc_val:>10.6f} | {1-ncc_val:>10.2e}")

# ── Conclusion ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Conclusion")
print("=" * 70)

for alg_name in ["cvdms_fd", "cvdms_bsc"]:
    print(f"\n── {alg_name} ──")
    # Find the loosest threshold where 1-NCC vs reference < 1e-6
    for th in THRESHOLDS:
        ncc_val = ncc(results[(alg_name, REF_THRESH)]["cbed"], results[(alg_name, th)]["cbed"])
        delta = 1 - ncc_val
        if delta < 1e-6:
            print(f"  ✓ NCC={ncc_val:.6f} (Δ={delta:.2e}) at threshold ≤ {th:.0e}  ← sufficient")
            break
        else:
            print(f"  Δ={delta:.2e} at {th:.0e}")

# ── Save ────────────────────────────────────────────────────────
out = {
    "parameters": {
        "gpts": GPT, "sampling": SAMPLING, "frozen_phonons": FP,
        "energy": ENERGY, "semiangle_cutoff": SEMIANGLE,
        "slice_thickness": SLICE_DZ, "order": ORDER,
        "max_terms": MAX_TERMS, "max_inner": MAX_INNER,
    },
    "thresholds": THRESHOLDS,
    "reference_threshold": REF_THRESH,
    "results": {
        alg: [
            {
                "threshold": th,
                "time": results[(alg, th)]["time"],
                "converged": results[(alg, th)]["converged"],
                "n_warnings": len(results[(alg, th)]["warnings"]),
            }
            for th in THRESHOLDS
        ]
        for alg in ["cvdms_fd", "cvdms_bsc"]
    },
}
out_path = os.path.join(SCRIPT_DIR, "convergence_threshold_test.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to {out_path}")
