"""
metrics.py — Numerical metrics for convergence testing.

Shared metric functions used by pytest tests and standalone scripts.
All functions accept numpy arrays (use ``to_numpy()`` first for GPU arrays).
"""

import numpy as np


def to_numpy(array):
    """Convert array to numpy, handling both CPU (numpy) and GPU (CuPy) arrays."""
    if hasattr(array, "get"):  # CuPy array
        return np.asarray(array.get())
    return np.asarray(array)


def check_finite(arr):
    """Return True if all values are finite (no NaN, no Inf)."""
    return bool(np.all(np.isfinite(arr)))


def max_diff(ref, test):
    """Maximum absolute difference between two arrays."""
    return float(np.max(np.abs(ref - test)))


def rmsd(ref, test):
    """Root-Mean-Square Deviation between two arrays."""
    ref = np.asarray(ref)
    test = np.asarray(test)
    return float(np.sqrt(np.mean(np.abs(ref - test) ** 2)))


def ncc(ref, test):
    """Normalized Cross-Correlation.

    Returns a scalar in [0, 1].  1 = perfect match, 0 = orthogonal.

    NCC = |Σ ref* · conj(test)| / sqrt(Σ|ref|² · Σ|test|²)
    """
    ref = np.asarray(ref)
    test = np.asarray(test)
    num = np.abs(np.sum(np.conj(ref) * test))
    denom = np.sqrt(np.sum(np.abs(ref) ** 2) * np.sum(np.abs(test) ** 2))
    if denom == 0:
        return 1.0 if num == 0 else 0.0
    return float(num / denom)


def relative_error(ref, test):
    r"""Relative L2 error: ||ref - test||₂ / ||ref||₂."""
    ref = np.asarray(ref)
    test = np.asarray(test)
    ref_norm = np.sqrt(np.sum(np.abs(ref) ** 2))
    if ref_norm == 0:
        return 0.0
    diff_norm = np.sqrt(np.sum(np.abs(ref - test) ** 2))
    return float(diff_norm / ref_norm)


def intensity(array_or_waves):
    """Total intensity (sum of |ψ|²) of a wave field or numpy array."""
    arr = array_or_waves
    # Handle Waves/Probe objects — .build() materialises lazy arrays
    if hasattr(arr, "build"):
        arr = arr.build(lazy=False)
    if hasattr(arr, "array"):
        arr = arr.array
    arr = to_numpy(arr)
    return float(np.sum(np.abs(arr) ** 2))


def intensity_conservation(wave, I0=None):
    """Relative intensity deviation: |I - I0| / I0.

    If I0 is None, the initial intensity is not known — returns the
    raw intensity instead (useful for relative comparisons).
    """
    I = intensity(wave)
    if I0 is None:
        return I
    return float(np.abs(I - I0) / I0)


def amplitude_rms(ref, test):
    """RMS relative amplitude difference, global phase removed.

    Ref: CVDMS branch verification scripts — aligns the global phase
    of ``test`` to ``ref`` before computing the RMS of |ref - test| / |ref|.
    """
    ref = np.asarray(ref).ravel()
    test = np.asarray(test).ravel()

    # Remove global phase offset
    phase_offset = np.angle(np.sum(np.conj(ref) * test))
    test_aligned = test * np.exp(1.0j * phase_offset)

    amp_ref = np.abs(ref)
    mask = amp_ref > 0
    if not np.any(mask):
        return 0.0

    return float(np.sqrt(np.mean(
        (np.abs(ref[mask] - test_aligned[mask]) / amp_ref[mask]) ** 2
    )))


def phase_rms(ref, test):
    """RMS phase difference in radians, amplitude-weighted.

    Global phase is removed before comparison.
    """
    ref = np.asarray(ref).ravel()
    test = np.asarray(test).ravel()

    # Remove global phase
    phase_offset = np.angle(np.sum(np.conj(ref) * test))
    test_aligned = test * np.exp(1.0j * phase_offset)

    # Amplitude weight
    weights = np.abs(ref)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0

    phase_diff = np.angle(test_aligned * np.conj(ref))
    # Wrap to [-π, π]
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

    return float(np.sqrt(np.sum(weights * phase_diff ** 2) / total_weight))
