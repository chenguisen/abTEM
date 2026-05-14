"""
Sweep engine: parameter sweep orchestration with file-based caching.

Each simulation result is cached as:
  cache/{sweep_name}/{hash}.npz  — compressed numpy arrays
  cache/{sweep_name}/{hash}.json — metadata
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ._parameters import (
    Baseline, SweepDef, ALGORITHMS, resolve_sweep_params, format_value,
)
from ._metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def make_cache_key(sweep_name: str, algorithm: str, params: dict) -> str:
    """Deterministic hash from sweep + algorithm + params."""
    d = {k: _serialize(v) for k, v in sorted(params.items())}
    d["sweep"] = sweep_name
    d["algorithm"] = algorithm
    raw = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _serialize(v):
    if isinstance(v, (list, tuple)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.12e}"
    return str(v)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------
def cache_paths(cache_dir: str, sweep_name: str, key: str):
    """Return (npz_path, json_path) for a cache entry."""
    base = Path(cache_dir) / sweep_name
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"{key}.npz"), str(base / f"{key}.json")


def save_cache(cache_dir: str, sweep_name: str, key: str, params: dict,
               algorithm: str, result: dict):
    """Save simulation result to cache."""
    npz_path, json_path = cache_paths(cache_dir, sweep_name, key)

    # Save arrays
    arrays = {}
    for k in ("exit_wave", "cbed", "intensity_map"):
        if k in result and result[k] is not None:
            v = np.asarray(result[k])
            if v.size < 1e8:  # skip extremely large arrays
                arrays[k] = v

    np.savez_compressed(npz_path, **arrays)

    # Save metadata (not arrays)
    meta = {
        "algorithm": algorithm,
        "params": {k: _serialize(v) for k, v in params.items()},
        "time": float(result.get("time", 0)),
        "I0": float(result.get("I0", 0)),
        "diagnostics": result.get("diagnostics", {}),
        "metrics": result.get("metrics", {}),
        "arrays": list(arrays.keys()),
        "complete": True,
        "timestamp": time.time(),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Clean up large arrays from result dict (keep only metadata)
    for k in ("exit_wave", "cbed", "intensity_map"):
        result.pop(k, None)
    result["meta"] = meta


def load_cache(cache_dir: str, sweep_name: str, key: str):
    """Load cached result. Returns dict with 'meta' and loaded arrays, or None."""
    npz_path, json_path = cache_paths(cache_dir, sweep_name, key)

    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        meta = json.load(f)

    if not meta.get("complete", False):
        return None

    result = {"meta": meta}

    # Load arrays
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        for k in meta.get("arrays", []):
            if k in data:
                result[k] = data[k]

    return result


# ---------------------------------------------------------------------------
# Sweep Engine
# ---------------------------------------------------------------------------
class SweepEngine:
    """Orchestrate parameter sweeps with caching."""

    def __init__(self, cache_dir: str, baseline: Optional[Baseline] = None,
                 fast_mode: bool = True, no_cache: bool = False):
        self.cache_dir = cache_dir
        self.baseline = baseline or Baseline()
        self.fast_mode = fast_mode
        self.no_cache = no_cache
        self.runner = None  # lazy import

    def _get_runner(self):
        if self.runner is None:
            from ._simulation import SimulationRunner
            self.runner = SimulationRunner(verbose=True)
        return self.runner

    def run_sweep(self, sweep: SweepDef, partial=False):
        """Run all parameter combinations for a sweep.

        Returns list of result dicts, each with:
            sweep: sweep name
            value: parameter value
            algorithm: algorithm name
            metrics: dict of metric values
            time: wall-clock time
            meta: cached metadata
        """
        from tqdm import tqdm

        results = []
        values = sweep.values
        if partial:
            values = values[:3]  # only first 3 for quick check

        for val in tqdm(values, desc=f"Sweep [{sweep.name}]", unit="cfg"):
            for algo in ALGORITHMS:
                r = self._run_one(sweep, val, algo)
                if r:
                    results.append(r)
        return results

    def _run_one(self, sweep: SweepDef, value, algorithm: str):
        """Run single parameter point or load from cache."""
        params = resolve_sweep_params(self.baseline, sweep, value)

        # Override grid and FP for fast mode
        if self.fast_mode:
            if sweep.name == "sampling":
                # Sampling sweep MUST use sampling-derived grid to vary
                # reciprocal-space resolution. Fixed gpts would make all
                # sampling values produce identical CBED patterns.
                from ._parameters import sampling_gpts
                params["_gpts"] = sampling_gpts(params["sampling"])
            elif not sweep.full_resolution:
                from ._parameters import fast_gpts
                gpts = fast_gpts(params["sampling"])
                params["_gpts"] = gpts
            else:
                params["_gpts"] = self.baseline.gpts
            # Reduce FP for non-fp sweeps in fast mode
            if sweep.name != "fp":
                from ._parameters import FAST_FROZEN_PHONONS
                params["frozen_phonons"] = FAST_FROZEN_PHONONS
        else:
            params["_gpts"] = self.baseline.gpts

        key = make_cache_key(sweep.name, algorithm, params)

        # Try cache
        if not self.no_cache:
            cached = load_cache(self.cache_dir, sweep.name, key)
            if cached is not None:
                return {
                    "sweep": sweep.name,
                    "value": value,
                    "value_label": format_value(sweep, value),
                    "algorithm": algorithm,
                    "metrics": cached["meta"].get("metrics", {}),
                    "time": cached["meta"].get("time", 0),
                    "meta": cached["meta"],
                    "cached": True,
                    "_cache_key": key,
                }

        # Run simulation
        runner = self._get_runner()
        atoms = runner.build_structure(
            supercell_xy=params["supercell_xy"],
            supercell_z=params["supercell_z"],
            material=params["material"],
            spacegroup=params["spacegroup"],
            lattice_constant=params["lattice_constant"],
        )
        potential = runner.build_potential(
            atoms,
            sampling=params["sampling"],
            slice_thickness=params["slice_thickness"],
            gpts=params["_gpts"],
            frozen_phonons=params["frozen_phonons"],
            exit_planes=params["exit_planes"],
        )
        probe = runner.build_probe(
            potential,
            energy=params["energy"],
            semiangle_cutoff=params["semiangle_cutoff"],
        )

        result = runner.run_algorithm(
            potential, probe,
            algorithm=algorithm,
            convergence_threshold=params["convergence_threshold"],
            max_terms=params["max_terms"],
            order=params["order"],
            backend=params["backend"],
        )
        runner.cleanup()

        # Compute metrics using Fourier result as reference
        # For this we need the reference — handled externally via
        # post_process_metrics
        cbed = result.get("cbed")
        if cbed is not None:
            metrics = compute_all_metrics(
                exit_wave=result.get("exit_wave", np.array([])),
                cbed_pattern=cbed,
                I0=result.get("I0"),
            )
            result["metrics"] = metrics

        # Save to cache
        save_cache(self.cache_dir, sweep.name, key, params, algorithm, result)

        return {
            "sweep": sweep.name,
            "value": value,
            "value_label": format_value(sweep, value),
            "algorithm": algorithm,
            "metrics": result.get("metrics", {}),
            "time": result.get("time", 0),
            "meta": result.get("meta", {}),
            "cached": False,
            "_cache_key": key,
        }

    def add_reference_metrics(self, results: list):
        """Post-process: add NCC/RMSD vs Fourier reference for each sweep point.

        Mutates results in-place.
        """
        # Group by (sweep, value), find fourier result
        groups = {}
        for r in results:
            key = (r["sweep"], r["value"])
            groups.setdefault(key, {})[r["algorithm"]] = r

        for key, group in groups.items():
            fourier = group.get("fourier")
            if fourier is None:
                continue
            # We need CBED from fourier — load from cache if needed
            fb = fourier.get("meta", {}).get("arrays", [])
            if not fb:
                continue

            for algo in ["cvdms_fd", "cvdms_bsc"]:
                entry = group.get(algo)
                if entry is None:
                    continue
                m = entry.get("metrics", {})
                fm = fourier.get("metrics", {})

                # Copy symmetry metrics from the entry itself
                if "symmetry_h" not in m and "symmetry_h" in fm:
                    m["symmetry_h"] = fm["symmetry_h"]
                    m["symmetry_v"] = fm["symmetry_v"]
                    m["symmetry_pass"] = fm["symmetry_pass"]

                # NCC/RMSD can only be computed with both arrays loaded
                # For report-only mode we compute these during figure generation

    def load_cbed_for_metrics(self, results: list, sweep_name: str):
        """Load cached CBED arrays for metric computation.

        Returns dict: (value, algorithm) -> cbed array
        """
        arrays = {}
        for r in results:
            if r["sweep"] != sweep_name:
                continue
            key = (r["value"], r["algorithm"])
            meta = r.get("meta", {})
            npz_key = self._find_npz(meta)
            if npz_key:
                cached = load_cache(self.cache_dir, sweep_name, npz_key)
                if cached and "cbed" in cached:
                    arrays[key] = cached["cbed"]
                elif cached and "exit_wave" in cached:
                    ew = cached["exit_wave"]
                    arrays[key] = self._compute_cbed(ew)
        return arrays

    def _find_npz(self, meta: dict):
        """Find the npz cache key from meta."""
        # meta is the json content; extract key from params hash
        return None  # handled differently in figure generation

    @staticmethod
    def _compute_cbed(exit_wave):
        if exit_wave.ndim == 3:
            f = np.fft.fft2(exit_wave, axes=(-2, -1))
            cbed = np.mean(np.abs(f) ** 2, axis=0)
        else:
            f = np.fft.fft2(exit_wave)
            cbed = np.abs(f) ** 2
        return np.fft.fftshift(cbed)
