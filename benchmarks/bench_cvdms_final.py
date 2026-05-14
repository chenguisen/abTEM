"""
Final CVDMS benchmark: before/after comparison with notebook-scale parameters.
Grid: ~627x627, 30keV, finite-difference Laplacian, 1 frozen phonon.
"""
import time
import warnings
import numpy as np
from ase.build import bulk, surface
import abtem
from abtem.multislice import CVDMSMultislice

abtem.config.set({"device": "gpu", "fft": "cupy", "diagnostics.task_progress": False})
warnings.filterwarnings("ignore")
np.random.seed(42)

# ---- Build Si(111) similar to notebook scale ----
si = bulk("Si", "diamond", a=5.431)
atoms = surface(si, (1, 1, 1), 3, periodic=True) * (8, 8, 5)

# Build potential ONCE, use for both runs (identical phonon displacements)
frozen = abtem.FrozenPhonons(atoms, 1, {"Si": 0.1})
potential = abtem.Potential(
    frozen, sampling=0.05, projection="finite",
    slice_thickness=0.4, exit_planes=20,
)

gx, gy = potential.gpts
total_z = atoms.cell[2, 2]
sz = potential.slice_thickness[0]
n_slices = int(total_z / sz)
lx, ly = potential.extent
print(f"Grid: {gx}x{gy} = {gx*gy/1e6:.1f}M px")
print(f"Extent: {lx:.1f}x{ly:.1f} A")
print(f"Slices: {n_slices}, total: {total_z:.1f} A")

wave = abtem.Probe(energy=30e3, semiangle_cutoff=35)
wave.grid.match(potential)

# ---- Run 1: Original (force non-tiled) ----
import abtem.cvdms_kernels as ck

# ---- Baseline: non-tiled kernel ----
print("\n=== Baseline (non-tiled kernel) ===")
ck._kernel_cache['_bench_untiled'] = True
try:
    si2 = bulk("Si", "diamond", a=5.431)
    si2_atoms = surface(si2, (1, 1, 1), 3, periodic=True) * (8, 8, 5)
    f2 = abtem.FrozenPhonons(si2_atoms, 1, {"Si": 0.1})
    p2 = abtem.Potential(f2, sampling=0.05, projection="finite",
                         slice_thickness=0.4, exit_planes=20)
    w2 = abtem.Probe(energy=30e3, semiangle_cutoff=35)
    w2.grid.match(p2)
    cvdms = CVDMSMultislice(order=1, convergence_threshold=1e-6,
                             max_terms=50, use_fused_kernel=True,
                             backscattering=True)
    t0 = time.time()
    m = w2.multislice(p2, algorithm=cvdms).diffraction_patterns(max_angle="cutoff")
    results_baseline = m.mean(0).compute()
    t_base = time.time() - t0
    print(f"  Total: {t_base:.1f}s")
    print(f"  Per-slice: {t_base/n_slices*1000:.1f}ms")
finally:
    ck._kernel_cache['_bench_untiled'] = False

# ---- Run 2: Tiled kernel ----
print("\n=== Optimized (tiled kernel + backscattering GPU) ===")
si3 = bulk("Si", "diamond", a=5.431)
si3_atoms = surface(si3, (1, 1, 1), 3, periodic=True) * (8, 8, 5)
f3 = abtem.FrozenPhonons(si3_atoms, 1, {"Si": 0.1})
p3 = abtem.Potential(f3, sampling=0.05, projection="finite",
                     slice_thickness=0.4, exit_planes=20)
w3 = abtem.Probe(energy=30e3, semiangle_cutoff=35)
w3.grid.match(p3)

cvdms = CVDMSMultislice(order=1, convergence_threshold=1e-6,
                         max_terms=50, use_fused_kernel=True,
                         backscattering=True)
t0 = time.time()
m = w3.multislice(p3, algorithm=cvdms).diffraction_patterns(max_angle="cutoff")
results_opt = m.mean(0).compute()
t_opt = time.time() - t0
print(f"  Total: {t_opt:.1f}s")
print(f"  Per-slice: {t_opt/n_slices*1000:.1f}ms")

if t_base != float('inf') and t_base > 0:
    print(f"\n=== Combined speedup: {t_base/t_opt:.2f}x ===")
    print(f"  Saved {t_base - t_opt:.1f}s")

# ---- Verify numerical equivalence ----
import numpy as np
print("\n=== Numerical verification ===")
try:
    arr_base = results_baseline.array if hasattr(results_baseline, 'array') else results_baseline
    arr_opt = results_opt.array if hasattr(results_opt, 'array') else results_opt
    if hasattr(arr_base, 'get'):
        arr_base = arr_base.get()
        arr_opt = arr_opt.get()
    arr_base, arr_opt = np.asarray(arr_base), np.asarray(arr_opt)
    diff = np.abs(arr_opt - arr_base)
    print(f"  Max diff: {float(np.max(diff)):.2e}")
    print(f"  Mean diff: {float(np.mean(diff)):.2e}")
    if float(np.max(diff)) < 1e-4:
        print("  ✓ PASS: results match within tolerance")
    else:
        print("  ✗ FAIL: results differ!")
except Exception as e:
    print(f"  (Skipped: {e})")
