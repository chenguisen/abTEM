#!/usr/bin/env python3
"""
CVDMS Comprehensive Benchmark Suite — main entry point.

Usage:
    # Fast mode (256x256, quick exploration)
    python benchmarks/comprehensive/run_benchmark.py --mode fast --sweeps voltage fp

    # Full mode (627x627 for voltage + sampling, 256x256 for rest)
    python benchmarks/comprehensive/run_benchmark.py --mode full

    # Report-only (regenerate from cache)
    python benchmarks/comprehensive/run_benchmark.py --mode report-only

    # Force recomputation
    python benchmarks/comprehensive/run_benchmark.py --no-cache --sweeps voltage
"""
import argparse
import json
import os
import sys
import time
import numpy as np

# Add abTEM root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABTEM_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
if ABTEM_ROOT not in sys.path:
    sys.path.insert(0, ABTEM_ROOT)

from benchmarks.comprehensive._parameters import (
    Baseline, SWEEPS, ALGORITHMS, ALGORITHM_LABELS,
    resolve_sweep_params, format_value, SweepDef,
)
from benchmarks.comprehensive._sweep import (
    SweepEngine, make_cache_key, cache_paths,
)
from benchmarks.comprehensive._metrics import (
    compute_all_metrics, ncc, rmsd,
)


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="CVDMS Comprehensive Benchmark Suite")
    p.add_argument("--mode", choices=["fast", "full", "report-only"],
                   default="fast", help="Execution mode")
    p.add_argument("--sweeps", nargs="+",
                   choices=[s.name for s in SWEEPS] + ["all"],
                   default=["all"],
                   help="Which sweeps to run (default: all)")
    p.add_argument("--no-cache", action="store_true",
                   help="Force recomputation")
    p.add_argument("--cache-dir",
                   default=os.path.join(SCRIPT_DIR, "cache"),
                   help="Cache directory")
    p.add_argument("--output",
                   default=os.path.join(SCRIPT_DIR, "report.html"),
                   help="Output HTML report path")
    p.add_argument("--figures-dir",
                   default=os.path.join(SCRIPT_DIR, "figures"),
                   help="Figure output directory")
    return p


def main():
    args = build_arg_parser().parse_args()
    fast_mode = (args.mode == "fast")
    report_only = (args.mode == "report-only")
    no_cache = args.no_cache or report_only  # report-only doesn't run sims

    # Sweep selection
    if "all" in args.sweeps:
        selected_sweeps = SWEEPS
    else:
        selected_sweeps = [s for s in SWEEPS if s.name in args.sweeps]

    print("=" * 60)
    print("CVDMS Comprehensive Benchmark Suite")
    print("=" * 60)
    print(f"Mode:    {args.mode}")
    print(f"Sweeps:  {[s.name for s in selected_sweeps]}")
    print(f"Cache:   {args.cache_dir}")
    print(f"Output:  {args.output}")
    print(f"Figures: {args.figures_dir}")
    print()

    # Prepare
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    all_results = []

    # Phase 1: Run simulations (or skip for report-only)
    if not report_only:
        engine = SweepEngine(
            cache_dir=args.cache_dir,
            baseline=Baseline(),
            fast_mode=fast_mode,
            no_cache=no_cache,
        )

        for sweep in selected_sweeps:
            print(f"\n{'─'*50}")
            print(f"Sweep: {sweep.name} ({sweep.label})")
            print(f"{'─'*50}")
            t0 = time.time()
            results = engine.run_sweep(sweep)
            dt = time.time() - t0
            print(f"  → {len(results)} configs in {dt:.0f}s")

            # Add reference metrics (NCC vs Fourier)
            results = _add_reference_metrics(results, args.cache_dir, sweep)
            all_results.extend(results)

        # Save results summary to JSON
        summary_path = os.path.join(args.figures_dir, "results_summary.json")
        _save_summary(all_results, summary_path)
        print(f"\nSummary saved to {summary_path}")

    else:
        # Report-only: load from saved summary
        summary_path = os.path.join(args.figures_dir, "results_summary.json")
        if os.path.exists(summary_path):
            all_results = _load_summary(summary_path)
            print(f"Loaded {len(all_results)} cached results from {summary_path}")
        else:
            print(f"No cached summary found at {summary_path}")
            print("Run with --mode fast or --mode full first.")
            return 1

        # Recompute reference metrics (NCC vs Fourier) from cache NPZ files.
        # This ensures that any missing NCC values are filled in, e.g. if the
        # summary was reconstructed from cache before reference metrics were computed.
        for sweep in selected_sweeps:
            sweep_results = [r for r in all_results if r["sweep"] == sweep.name]
            _add_reference_metrics(sweep_results, args.cache_dir, sweep)
        # Persist updated metrics back to summary file
        _save_summary(all_results, summary_path)
        print(f"Reference metrics recomputed and summary re-saved to {summary_path}")

    # Phase 2: Generate figures
    print(f"\n{'─'*50}")
    print("Generating figures...")
    print(f"{'─'*50}")

    # Load fresh CBED arrays for figures that need them
    _prep_cache_for_figures(all_results, args.cache_dir, selected_sweeps)

    from benchmarks.comprehensive._figures import set_output_dir, generate_all
    set_output_dir(args.figures_dir)
    figures = generate_all(args.cache_dir, all_results)

    # Phase 3: Generate report
    print(f"\n{'─'*50}")
    print("Building HTML report...")
    print(f"{'─'*50}")
    sweep_times = {s.name: sum(r.get("time", 0) for r in all_results
                               if r["sweep"] == s.name)
                   for s in selected_sweeps}

    from benchmarks.comprehensive._report import build_full_report
    build_full_report(all_results, figures, args.output, sweep_times)

    # Phase 4: Print summary
    n_cached = sum(1 for r in all_results if r.get("cached", False))
    n_new = len(all_results) - n_cached
    total_time = sum(r.get("time", 0) for r in all_results)

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Configurations: {len(all_results)} total "
          f"({n_cached} cached, {n_new} new runs)")
    print(f"Total compute:  {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Report:         {args.output}")
    print(f"Figures:        {args.figures_dir}")
    print()

    return 0


def _add_reference_metrics(results: list, cache_dir: str, sweep: SweepDef):
    """Post-process: compute NCC/RMSD vs Fourier reference.

    Tries two methods to locate the cache NPZ files:
    1. From stored _cache_key (reliable, always available)
    2. From stored meta.params (fallback, may be empty)
    """
    fft_results = [r for r in results if r["algorithm"] == "fourier"]

    for r in results:
        if r["algorithm"] == "fourier":
            continue
        # Find matching Fourier result
        fourier = None
        for fr in fft_results:
            if fr["value"] == r["value"]:
                fourier = fr
                break
        if fourier is None:
            continue

        # Strategy 1: Use stored _cache_key (preferred, always present)
        fkey = fourier.get("_cache_key")
        rkey = r.get("_cache_key")

        # Strategy 2: Fallback to reconstructing from meta.params
        if not fkey or not rkey:
            fparams = fourier.get("meta", {}).get("params", {})
            rparams = r.get("meta", {}).get("params", {})
            if fparams and rparams:
                fkey = make_cache_key(sweep.name, "fourier", fparams)
                rkey = make_cache_key(sweep.name, r["algorithm"], rparams)

        if not fkey or not rkey:
            continue

        npz_f, _ = cache_paths(cache_dir, sweep.name, fkey)
        npz_c, _ = cache_paths(cache_dir, sweep.name, rkey)

        if not (os.path.exists(npz_f) and os.path.exists(npz_c)):
            continue

        data_f = np.load(npz_f)
        data_c = np.load(npz_c)

        cbed_f = data_f.get("cbed")
        cbed_c = data_c.get("cbed")
        if cbed_f is None or cbed_c is None:
            continue

        # Ensure same shape
        if cbed_f.shape != cbed_c.shape:
            from scipy.ndimage import zoom
            sy, sx = cbed_f.shape
            ty, tx = cbed_c.shape
            sc_y, sc_x = sy / ty, sx / tx
            cbed_c = zoom(cbed_c, (sc_y, sc_x), order=1)

        # Compute metrics
        ncc_val = ncc(cbed_f, cbed_c)
        rmsd_val = rmsd(cbed_f, cbed_c)

        m = r.get("metrics", {})
        m["ncc_vs_reference"] = ncc_val
        m["rmsd_vs_reference"] = rmsd_val

        # Persist back to cache metadata so future summary rebuilds
        # don't lose these cross-algorithm metrics
        _, json_path = cache_paths(cache_dir, sweep.name, rkey)
        if os.path.exists(json_path):
            with open(json_path) as jf:
                cache_meta = json.load(jf)
            cache_meta.setdefault("metrics", {})
            cache_meta["metrics"]["ncc_vs_reference"] = ncc_val
            cache_meta["metrics"]["rmsd_vs_reference"] = rmsd_val
            with open(json_path, "w") as jf:
                json.dump(cache_meta, jf, indent=2, ensure_ascii=False)

    return results


def _prep_cache_for_figures(results: list, cache_dir: str,
                             sweeps: list[SweepDef]):
    """Ensure cached arrays exist for figure generation.

    For figures that need CBED arrays, we ensure the cache structure
    is accessible. The figure functions load arrays directly from
    the cache directory.
    """
    pass  # Figures load directly from cache via _iter_sweep_results


def _save_summary(results: list, path: str):
    """Save results summary to JSON (metrics + test conditions, no arrays)."""
    summary = []
    for r in results:
        meta = r.get("meta", {})
        p = meta.get("params", {})
        entry = {
            "sweep": r["sweep"],
            "value": r["value"],
            "value_label": r.get("value_label", str(r["value"])),
            "algorithm": r["algorithm"],
            "metrics": r.get("metrics", {}),
            "conditions": {
                "energy": p.get("energy"),
                "gpts": p.get("_gpts"),
                "sampling": p.get("sampling"),
                "frozen_phonons": p.get("frozen_phonons"),
                "slice_thickness": p.get("slice_thickness"),
                "total_thickness": p.get("total_thickness"),
                "supercell_z": p.get("supercell_z"),
                "exit_planes": p.get("exit_planes"),
                "order": p.get("order"),
                "max_terms": p.get("max_terms"),
                "convergence_threshold": p.get("convergence_threshold"),
                "backend": p.get("backend"),
            },
            "time": r.get("time", 0),
            "cached": r.get("cached", False),
            "_cache_key": r.get("_cache_key", ""),
        }
        summary.append(entry)

    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _load_summary(path: str) -> list:
    """Load results summary from JSON."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    sys.exit(main())
