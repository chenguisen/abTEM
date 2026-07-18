#!/usr/bin/env python3
"""
run_convergence.py — Standalone convergence diagnostics for DEV's
fully_corrected + BSC implementations.

Usage::

    conda activate abtem-env
    cd F:/abTEM

    # Quick scan (Stage A, STO, CPU)
    python test_cvdms/run_convergence.py --mode all

    # Detailed scan with plots
    python test_cvdms/run_convergence.py --mode all --materials STO,Au,Si \\
        --orders 1,2,3,4,6,8 --stage B --plot

    # Forward scattering only, GPU
    python test_cvdms/run_convergence.py --mode forward --gpu --orders 1,2,4,6

Modes
-----
  forward     Forward scattering convergence (order sweep)
  bsc         BSC correction convergence (forward + backscattered wave)
  backprop    Backscattered wave accumulation vs exit_planes
  stress      Divergence stress test (thick slices, low energy)
  all         Run all modes
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make test_cvdms importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_cvdms.conftest import build_system, _make_sto_atoms, _make_au_atoms, _make_si_atoms
from test_cvdms.metrics import to_numpy, ncc, rmsd, max_diff, intensity, check_finite, relative_error

import abtem
from abtem.multislice import RealSpaceMultislice

MATERIALS = {"STO": _make_sto_atoms, "Au": _make_au_atoms, "Si": _make_si_atoms}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════════════
# Mode: forward
# ═══════════════════════════════════════════════════════════════════════════

def run_forward(system, orders):
    """Sweep expansion order. Report NCC, RMSD, max_diff vs highest order."""
    print(f"\n{'─'*65}")
    print(f"Forward scattering convergence  ({system['device']}, gpts={system['gpts']})")
    print(f"{'─'*65}")

    ref_order = max(orders)
    print(f"Computing reference (order={ref_order})...")
    t0 = time.perf_counter()
    ref_arr = _forward(system, ref_order)
    ref_time = time.perf_counter() - t0
    print(f"  reference done in {ref_time:.1f}s")

    results = {}
    prev_arr = None
    print(f"\n{'Order':>6} {'time_s':>8} {'NCC_vs_ref':>12} {'RMSD':>10} "
          f"{'max_diff':>10} {'rel_err':>10}")
    print(f"{'─'*65}")

    for o in orders:
        t0 = time.perf_counter()
        arr = _forward(system, o)
        dt = time.perf_counter() - t0

        n = ncc(ref_arr, arr)
        r = rmsd(ref_arr, arr)
        d = max_diff(ref_arr, arr)
        rel = relative_error(ref_arr, arr)

        print(f"{o:>6} {dt:>8.1f} {n:>12.8f} {r:>10.2e} {d:>10.2e} {rel:>10.2e}")

        results[o] = {"time_s": dt, "ncc": n, "rmsd": r,
                       "max_diff": d, "rel_err": rel}
        prev_arr = arr

    return results


def _forward(system, order):
    probe = system["probe"]
    pot = system["potential"]
    return to_numpy(probe.multislice(
        potential=pot, scan=[[0, 0]], lazy=False,
        algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
    ).array)


# ═══════════════════════════════════════════════════════════════════════════
# Mode: bsc
# ═══════════════════════════════════════════════════════════════════════════

def run_bsc(system, orders):
    """Sweep expansion order with BSC enabled. Report forward + backward metrics."""
    if system["potential_exit_planes"] is None:
        system = build_system(system["atoms"], gpts=system["gpts"],
                              exit_planes=1, energy=system["energy"],
                              device=system["device"])

    print(f"\n{'─'*75}")
    print(f"BSC correction convergence  ({system['device']}, gpts={system['gpts']})")
    print(f"{'─'*75}")

    probe = system["probe"]
    pot = system["potential_exit_planes"]
    I0 = intensity(probe)

    results = {}
    print(f"\n{'Order':>6} {'time_s':>8} {'|fwd|':>10} {'|bwd|':>12} "
          f"{'|bwd|/|fwd|':>12} {'I_fwd+I_bwd-I0':>14}")
    print(f"{'─'*75}")

    for o in orders:
        t0 = time.perf_counter()
        fwd, bwd = probe.multislice(
            potential=pot, scan=[[0, 0]], lazy=False,
            algorithm=RealSpaceMultislice(order=o, expansion_scope="full"),
            return_backscattered=True,
        )
        dt = time.perf_counter() - t0

        fwd_arr = to_numpy(fwd.array)
        bwd_arr = to_numpy(bwd.array)

        fwd_norm = np.abs(fwd_arr).sum()
        bwd_norm = np.abs(bwd_arr).sum()
        ratio = bwd_norm / fwd_norm if fwd_norm > 0 else 0
        I_fwd = intensity(fwd_arr)
        I_bwd = intensity(bwd_arr)
        budget = I_fwd + I_bwd - I0

        print(f"{o:>6} {dt:>8.1f} {fwd_norm:>10.4f} {bwd_norm:>12.3e} "
              f"{ratio:>12.6f} {budget:>14.3e}")

        results[o] = {"time_s": dt, "fwd_norm": fwd_norm, "bwd_norm": bwd_norm,
                       "ratio": ratio, "budget_error": budget}

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Mode: backprop
# ═══════════════════════════════════════════════════════════════════════════

def run_backprop(system, orders, exit_planes_list=(1, 2, 4, 8)):
    """Check backscattered wave accumulation vs number of exit planes."""
    print(f"\n{'─'*75}")
    print(f"Backscattered wave accumulation vs exit_planes  "
          f"({system['device']}, gpts={system['gpts']})")
    print(f"{'─'*75}")

    order = min(4, max(orders))  # use a reasonable fixed order
    print(f"Using order={order}")

    results = {}
    print(f"\n{'EP':>4} {'order':>6} {'time_s':>8} {'|bwd|':>12} "
          f"{'bwd[0]==0':>10} {'depth_growth':>14}")
    print(f"{'─'*75}")

    for ep in exit_planes_list:
        sys = build_system(system["atoms"], gpts=system["gpts"],
                           exit_planes=ep, energy=system["energy"],
                           device=system["device"])

        t0 = time.perf_counter()
        _, bwd = sys["probe"].multislice(
            potential=sys["potential_exit_planes"], scan=[[0, 0]], lazy=False,
            algorithm=RealSpaceMultislice(order=order, expansion_scope="full"),
            return_backscattered=True,
        )
        dt = time.perf_counter() - t0

        bwd_arr = to_numpy(bwd.array)
        norm = float(np.abs(bwd_arr).sum())

        # Entrance backscattered should be zero
        first_zero = False
        if bwd_arr.ndim >= 3 and bwd_arr.shape[0] > 1:
            first_zero = np.abs(bwd_arr[0]).sum() < 1e-10

        # Check depth growth
        depth_ok = "N/A"
        if bwd_arr.ndim >= 3 and bwd_arr.shape[0] >= 2:
            norms = [float(np.abs(bwd_arr[i]).sum()) for i in range(bwd_arr.shape[0])]
            if norms[-1] >= norms[0]:
                depth_ok = "✓"
            else:
                depth_ok = f"✗ ({norms[0]:.2e}→{norms[-1]:.2e})"

        print(f"{ep:>4} {order:>6} {dt:>8.1f} {norm:>12.3e} "
              f"{str(first_zero):>10} {depth_ok:>14}")

        results[ep] = {"time_s": dt, "bwd_norm": norm,
                        "first_slice_zero": first_zero}

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Mode: stress
# ═══════════════════════════════════════════════════════════════════════════

def run_stress(system):
    """Divergence guard stress test: thick slices, low energies."""
    print(f"\n{'─'*65}")
    print("Divergence guard stress test")
    print(f"{'─'*65}")
    print(f"{'dz(Å)':>7} {'E(keV)':>8} {'order':>6} {'status':>25} {'time_s':>8}")
    print(f"{'─'*65}")

    from abtem.finite_difference import DivergedError, NotConvergedError

    results = []
    for dz in [0.5, 1.0, 2.0, 4.0]:
        for energy in [30e3, 15e3, 5e3]:
            try:
                sys = build_system(system["atoms"], gpts=system["gpts"],
                                   slice_thickness=dz, exit_planes=0,
                                   energy=energy, device=system["device"])
                t0 = time.perf_counter()
                arr = _forward(sys, order=4)
                dt = time.perf_counter() - t0

                if check_finite(arr):
                    status = "✓ finite"
                else:
                    status = "✗ NON-FINITE (IEEE-754 gap!)"

            except DivergedError:
                dt = time.perf_counter() - t0
                status = "✗ DivergedError"
            except NotConvergedError:
                dt = time.perf_counter() - t0
                status = "✗ NotConvergedError"
            except Exception as e:
                dt = 0
                status = f"✗ {type(e).__name__}"

            print(f"{dz:>7.1f} {energy/1e3:>8.1f} {4:>6} {status:>25} {dt:>8.2f}")
            results.append({"dz": dz, "energy": energy, "status": status})

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(all_data, output_dir):
    if not HAS_MPL:
        print("matplotlib not available — skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Forward convergence: NCC vs order
    for mat_name, modes in all_data.items():
        if "forward" in modes:
            data = modes["forward"]
            orders = sorted(data.keys())
            nccs = [data[o]["ncc"] for o in orders]
            rmsds = [data[o]["rmsd"] for o in orders]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(orders, nccs, "o-")
            ax1.set_xlabel("Expansion order"); ax1.set_ylabel("NCC vs highest order")
            ax1.set_title(f"{mat_name}: NCC convergence"); ax1.grid(alpha=0.3)
            ax2.semilogy(orders, rmsds, "s-")
            ax2.set_xlabel("Expansion order"); ax2.set_ylabel("RMSD vs highest order")
            ax2.set_title(f"{mat_name}: RMSD convergence"); ax2.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"{mat_name}_forward.png", dpi=150)
            print(f"  Saved: {output_dir / f'{mat_name}_forward.png'}")
            plt.close(fig)

    # BSC: ratio vs order
    for mat_name, modes in all_data.items():
        if "bsc" in modes:
            data = modes["bsc"]
            orders = sorted(data.keys())
            ratios = [data[o]["ratio"] for o in orders]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.semilogy(orders, ratios, "s-")
            ax.set_xlabel("Expansion order"); ax.set_ylabel("|BSC| / |forward|")
            ax.set_title(f"{mat_name}: BSC magnitude vs order"); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"{mat_name}_bsc.png", dpi=150)
            print(f"  Saved: {output_dir / f'{mat_name}_bsc.png'}")
            plt.close(fig)

    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Convergence diagnostics for dev's fully_corrected + BSC")
    p.add_argument("--mode", default="all",
                   choices=["forward", "bsc", "backprop", "stress", "all"])
    p.add_argument("--materials", default="STO",
                   help="Comma-separated: STO,Au,Si")
    p.add_argument("--orders", type=int, nargs="*",
                   default=[1, 2, 3, 4, 6],
                   help="Expansion orders (default: 1 2 3 4 6)")
    p.add_argument("--energy", type=float, default=30e3,
                   help="Electron energy in eV (default: 30000)")
    p.add_argument("--stage", choices=["A", "B"], default="A",
                   help="Grid: A=128² (default), B=512²")
    p.add_argument("--gpu", action="store_true", help="Run on GPU")
    p.add_argument("--output", "-o", default="test_cvdms/diagnostics")
    p.add_argument("--plot", action="store_true", help="Generate plots")
    p.add_argument("--json", action="store_true", help="Save JSON results")
    args = p.parse_args()

    device = "gpu" if args.gpu else "cpu"
    stage = args.stage
    gpts = (128, 128) if stage == "A" else (512, 512)
    material_names = [m.strip() for m in args.materials.split(",")]
    modes = ["forward", "bsc", "backprop", "stress"] if args.mode == "all" else [args.mode]

    all_data = {}

    for mat_name in material_names:
        print(f"\n{'='*65}")
        print(f" Material: {mat_name}  |  {device.upper()}  |  Stage {stage} ({gpts[0]}²)")
        print(f"{'='*65}")

        atoms = MATERIALS[mat_name]()
        # Build with exit_planes for BSC modes
        sys = build_system(atoms, gpts=gpts, exit_planes=1 if "bsc" in modes or "backprop" in modes else 0,
                           energy=args.energy, device=device)
        all_data[mat_name] = {}

        for mode in modes:
            if mode == "forward":
                all_data[mat_name]["forward"] = run_forward(sys, args.orders)
            elif mode == "bsc":
                all_data[mat_name]["bsc"] = run_bsc(sys, args.orders)
            elif mode == "backprop":
                all_data[mat_name]["backprop"] = run_backprop(sys, args.orders)
            elif mode == "stress":
                all_data[mat_name]["stress"] = run_stress(sys)

    # Plots
    if args.plot:
        plot_results(all_data, args.output)

    # JSON output
    if args.json:
        import json as _json
        class _Encoder(_json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                return super().default(o)
        out_path = Path(args.output) / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json.dumps(all_data, cls=_Encoder, indent=2))
        print(f"\n  JSON saved: {out_path}")

    print(f"\n{'='*65}")
    print("Done.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
