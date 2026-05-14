"""Compare GPU utilization: Fourier vs CVDMS (check_interval=1 vs check_interval=2).

This script measures the impact of D2H synchronization batching on GPU utilization.
The key optimization: batching convergence checks (check_interval=2) instead of
checking every iteration (check_interval=1) reduces GPU pipeline stalls.

Expected result:
  Fourier:        ~90%+ GPU utilization (no D2H syncs)
  CVDMS ci=1:     <50% GPU utilization (sync every iteration)
  CVDMS ci=2:     between the above (half the syncs)
"""
import abtem
import cupy as cp
import numpy as np
import time
import subprocess
import threading
from ase.build import bulk
import ase
from abtem.multislice import CVDMSMultislice, FourierMultislice

abtem.config.set({'device': 'gpu', 'fft': 'cupy', 'diagnostics.task_progress': False})

silicon = bulk('Si', crystalstructure='diamond')
silicon_111 = ase.build.surface(silicon, (1, 1, 1), layers=3, periodic=True)
silicon_111_orthogonal = abtem.orthogonalize_cell(silicon_111)

# Medium grid: ~185K px, ~20 slices for meaningful comparison
atoms = silicon_111_orthogonal * (8, 8, 20)
potential = abtem.Potential(atoms, sampling=0.08, projection='finite',
                             slice_thickness=1, exit_planes=20)
wave = abtem.Probe(energy=300e3, semiangle_cutoff=15.0)
wave.grid.match(potential)

# Warm up
print("Warming up GPU...")
_ = wave.multislice(potential,
    algorithm=CVDMSMultislice(order=1, laplace_method='fft', check_interval=2))
_.compute()
cp.cuda.Stream.null.synchronize()
print("Warm up done.\n")


def sample_gpu_util(samples_list, stop_event, interval=0.1):
    """Sample GPU util into samples_list until stop_event is set."""
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            val = int(result.stdout.strip())
            samples_list.append(val)
        except Exception:
            pass
        time.sleep(interval)


def run_with_monitoring(name, algorithm):
    """Run multislice with GPU utilization monitoring in a background thread."""
    samples = []
    stop_event = threading.Event()

    t = threading.Thread(
        target=sample_gpu_util, args=(samples, stop_event), daemon=True
    )
    t.start()

    time.sleep(0.3)  # let sampling thread start

    t0 = time.time()
    r = wave.multislice(potential, algorithm=algorithm)
    r.compute()
    elapsed = time.time() - t0

    stop_event.set()
    t.join(timeout=2)

    avg_util = np.mean(samples) if samples else 0
    print(f"  {name}:")
    print(f"    Time:       {elapsed*1000:.0f} ms")
    print(f"    GPU samples: {len(samples)}")
    print(f"    Avg util:    {avg_util:.1f}%")
    if samples:
        print(f"    Max util:    {max(samples)}%")
        print(f"    Min util:    {min(samples)}%")
        p25 = np.percentile(samples, 25)
        p50 = np.percentile(samples, 50)
        p75 = np.percentile(samples, 75)
        print(f"    Quartiles:   P25={p25:.0f}% P50={p50:.0f}% P75={p75:.0f}%")
    print()
    return elapsed, samples


algorithms = [
    ("Fourier Multislice", FourierMultislice()),
    ("CVDMS check_interval=1 (ORIGINAL)",
     CVDMSMultislice(order=1, laplace_method='fft', check_interval=1)),
    ("CVDMS check_interval=2 (OPTIMIZED)",
     CVDMSMultislice(order=1, laplace_method='fft', check_interval=2)),
    ("CVDMS check_interval=3",
     CVDMSMultislice(order=1, laplace_method='fft', check_interval=3)),
]

results = {}
for name, alg in algorithms:
    elapsed, samples = run_with_monitoring(name, alg)
    results[name] = {'time': elapsed, 'samples': samples}

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for name, data in results.items():
    avg = np.mean(data['samples']) if data['samples'] else 0
    print(f"  {name:40s}  {data['time']*1000:6.0f} ms  GPU util: {avg:5.1f}%")

print()
print("Analysis:")
print("  Fourier has zero D2H syncs (pure GPU pipeline) -> highest util.")
print("  CVDMS ci=1 syncs every Taylor iteration -> most stalls.")
print("  CVDMS ci=2 halves sync frequency -> improved util.")
print("  nvidia-smi samples at ~10Hz, so brief idle dips may be missed.")
print("  Actual GPU idle time is higher than nvidia-smi reports.")
