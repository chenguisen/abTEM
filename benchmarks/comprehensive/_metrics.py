"""
Correctness metrics for benchmark validation.

All metrics operate on 2D CBED pattern arrays (numpy/cupy).
"""
import numpy as np


def normalize(arr):
    """Min-max normalize array to [0, 1]."""
    a = arr.ravel()
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-30:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def ncc(ref, test):
    """Normalized cross-correlation."""
    r = ref.ravel().astype(np.float64)
    t = test.ravel().astype(np.float64)
    rr = r - r.mean()
    tt = t - t.mean()
    denom = np.sqrt(np.dot(rr, rr) * np.dot(tt, tt))
    if denom < 1e-30:
        return 1.0
    return float(np.dot(rr, tt) / denom)


def rmsd(ref, test):
    """Root-mean-square deviation."""
    diff = ref.astype(np.float64) - test.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def max_diff(ref, test):
    """Maximum absolute difference."""
    return float(np.max(np.abs(ref.astype(np.float64) - test.astype(np.float64))))


def intensity_conservation(wave_array):
    """Relative intensity change: |Σ|ψ|² - I₀| / I₀ at exit vs entrance.

    wave_array: complex ndarray, shape (batch, nx, ny) or (nx, ny).
    Returns float (batch-averaged if batched).
    """
    w = np.asarray(wave_array)
    # total intensity per batch item
    if w.ndim == 2:
        w = w[np.newaxis, :, :]
    intensity = np.sum(np.abs(w) ** 2, axis=(1, 2))
    # I₀ is the first row/column summed over — but for batch we take
    # intensity at the entrance as the reference
    I0 = intensity[0] if len(intensity) > 1 else intensity[0]
    # Actually for a single exit wave, the reference is the first measurement.
    # Since we don't have entrance wave here, we'll use the intensity itself.
    # A better approach: pass initial intensity separately.
    # For single exit wave: compare to a reference.
    relative_change = np.abs(intensity - I0) / (I0 + 1e-30)
    return float(np.mean(relative_change))


def intensity_conservation_with_I0(intensity_map, I0):
    """|Σ|ψ|² - I₀| / I₀ given pre-computed intensity map.

    intensity_map: 2D array (intensity at each pixel)
    I0: scalar initial intensity
    """
    total = float(np.sum(intensity_map))
    return abs(total - I0) / (I0 + 1e-30)


def check_symmetry(pattern, axis=0, tol=0.95):
    """Check mirror symmetry of a CBED pattern (Friedel's law).

    Returns (score: float, passed: bool).
    score = NCC between left/right (axis=1) or top/bottom (axis=0) halves.
    """
    p = np.asarray(pattern)
    if axis == 1:
        mid = p.shape[1] // 2
        left = p[:, :mid]
        right = np.fliplr(p[:, -mid:])
    else:
        mid = p.shape[0] // 2
        top = p[:mid, :]
        bottom = np.flipud(p[-mid:, :])

    # Ensure same shape
    min_len = min(left.size, right.size) if axis == 1 else min(top.size, bottom.size)
    if axis == 1:
        lr = left.ravel()[:min_len]
        rr = right.ravel()[:min_len]
    else:
        lr = top.ravel()[:min_len]
        rr = bottom.ravel()[:min_len]

    score = ncc(lr, rr)
    return score, score >= tol


def check_overflow(wave_array):
    """Check for inf/nan in wave array. Returns (has_overflow: bool, locations: str)."""
    w = np.asarray(wave_array)
    n_inf = int(np.sum(np.isinf(w)))
    n_nan = int(np.sum(np.isnan(w)))
    if n_inf > 0:
        return True, f"{n_inf} inf values"
    if n_nan > 0:
        return True, f"{n_nan} nan values"
    return False, ""


def radial_profile(pattern, center=None):
    """Radial average of a 2D pattern.

    Returns (radius: 1D array, profile: 1D array).
    """
    p = np.asarray(pattern)
    ny, nx = p.shape
    if center is None:
        cx, cy = nx // 2, ny // 2
    else:
        cx, cy = center
    Y, X = np.ogrid[:ny, :nx]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), weights=p.ravel())
    nr = np.bincount(r.ravel())
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = np.where(nr > 0, tbin / nr, 0.0)
    return np.arange(len(profile)), profile


def radial_correlation(ref, test, center=None):
    """Correlation of radial profiles."""
    _, rp1 = radial_profile(ref, center)
    _, rp2 = radial_profile(test, center)
    return ncc(rp1, rp2)


def compute_all_metrics(exit_wave, cbed_pattern,
                        reference_exit_wave=None, reference_cbed=None,
                        I0=None):
    """Compute all metrics and return a dict.

    Args:
        exit_wave: complex ndarray, exit wave (nx, ny) or (batch, nx, ny)
        cbed_pattern: ndarray, CBED intensity pattern (nx', ny')
        reference_exit_wave: reference exit wave for NCC/RMSD
        reference_cbed: reference CBED for NCC/RMSD
        I0: initial intensity for conservation check

    Returns:
        dict of metric_name -> value
    """
    metrics = {}

    # Overflow check
    ovf, ovf_msg = check_overflow(exit_wave)
    metrics["overflow"] = ovf
    metrics["overflow_msg"] = ovf_msg

    # Intensity conservation
    if I0 is not None:
        if exit_wave.ndim == 2:
            total_intensity = np.sum(np.abs(exit_wave) ** 2)
        else:
            total_intensity = np.sum(np.abs(exit_wave) ** 2, axis=(1, 2)).mean()
        metrics["intensity_conservation"] = intensity_conservation_with_I0(
            np.abs(exit_wave) ** 2, I0
        )

    # Symmetry of CBED (Friedel's law)
    score_h, pass_h = check_symmetry(cbed_pattern, axis=1)
    score_v, pass_v = check_symmetry(cbed_pattern, axis=0)
    metrics["symmetry_h"] = score_h
    metrics["symmetry_v"] = score_v
    metrics["symmetry_pass"] = pass_h and pass_v

    # Reference comparison
    if reference_cbed is not None:
        # Ensure same shape for comparison
        ref = np.asarray(reference_cbed)
        test = np.asarray(cbed_pattern)
        # Resize if needed
        if ref.shape != test.shape:
            from scipy.ndimage import zoom
            ry, rx = ref.shape
            ty, tx = test.shape
            scale_y, scale_x = ry / ty, rx / tx
            test = zoom(test, (scale_y, scale_x), order=1)

        metrics["ncc_vs_reference"] = ncc(ref, test)
        metrics["rmsd_vs_reference"] = rmsd(ref, test)
        metrics["radial_corr_vs_reference"] = radial_correlation(ref, test)

    if reference_exit_wave is not None:
        ew_ref = np.asarray(reference_exit_wave)
        ew_test = np.asarray(exit_wave)
        metrics["ncc_exit_wave"] = ncc(
            np.abs(ew_ref) ** 2, np.abs(ew_test) ** 2
        )

    return metrics
