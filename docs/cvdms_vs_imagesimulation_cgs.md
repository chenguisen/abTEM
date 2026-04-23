# CVDMS Algorithm: abTEM vs ImageSimulation_CGS

## Overview

CVDMS (Coupled-Wave Dynamical Multislice) is a multislice algorithm that accounts for
backscattering coupling between adjacent slices. It was originally implemented in the
[ImageSimulation_CGS](https://github.com/chenguisen/ImageSimulation_CGS) project
(C++/CUDA) and has been ported to abTEM (Python/CuPy).

Both implementations are based on:
J.H. Chen, D. Van Dyck, *Accurate multislice theory for elastic electron scattering in
transmission electron microscopy*.

## Alignment History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-04-23 | Initial port from `transmitSmallProbe_propCVDMS_CGS_BSC` |
| v1.1 | 2026-04-23 | Inner K-series divergence fix: pixel-by-pixel convergence + divergence detection |
| v1.2 | 2026-04-23 | Full backscattered wave backward propagation via `conj` trick |
| v1.3 | 2026-04-23 | Laplace operator: default 6→8 (9-point stencil), added FFT method |
| v1.4 | 2026-04-23 | Numerical stability: NaN truncation, stagnation detection, `fully_corrected` fix, non-fatal outer loop |

## Common Algorithm Core

Both implementations share the same mathematical structure:

### Forward Scattering (波函数向前传播)

```
phi_j = exp(i * K * dz) * phi_{j-1}
```

where `K = ∇²/(4πK₀) + V(r)` is the multislice operator, and the exponential is
evaluated via Taylor series expansion.

### Backscattering Correction (背散射修正)

```
BSC = (k_j - k_{j-1}) / (2 * k_j)
phi_corrected = phi_forward - BSC(phi_forward)
```

where `k_j` and `k_{j-1}` are the multislice operators for the current and next slice.

### Dual Loop Structure (双层循环)

Both implementations use two nested Taylor series:

- **Outer loop (指数展开)**: Expands `exp(i·K·dz)` as `Σ (i·K·dz)ⁿ / n!`
- **Inner loop (平方根展开)**: Expands the K-operator itself, with higher-order terms
  scaled by `(0.5 - n + 1) · λ / (π · n)`

---

## Key Differences

### 1. Taylor Series Inner Loop

| Aspect | ImageSimulation_CGS | abTEM |
|--------|--------------------|-------|
| Inner loop control | `while (nTaylorSqrt_count < waveSize_)` — pixel-by-pixel convergence | `while True` — pixel-by-pixel convergence (v1.1+) with stagnation detection (v1.4+) |
| Inner loop exit condition | All wavefront pixels below `cut_off_value` | All pixels below `convergence_threshold`, OR unconverged pixel count stops decreasing, OR NaN/Inf detected |
| Divergence detection | `fcms_taylor_max_iter()` with dynamic limit (stops when counter exceeds threshold) | Detects when `n_above >= prev_n_above` (v1.4+) — unconverged pixel count stops decreasing, indicating stagnation or divergence |
| NaN handling | Not explicit (counter-based limit prevents extreme orders) | On NaN/Inf: break and return partial sum (v1.4+) |
| Max iterations | `fcms_taylor_max_iter()` (dynamic, depends on wavefront state) | `max_inner_iter=100` (safety cap) |

**Status**: ✅ **Aligned.** Both implementations detect when the series has reached its
optimal truncation point. The abTEM v1.4+ `>=` check also catches stagnation (where pixel
count plateaus), which corresponds to the oscillatory limit cycle that would eventually
produce NaN in either implementation.

### 2. Convergence Control

| Aspect | ImageSimulation_CGS | abTEM |
|--------|--------------------|-------|
| Outer loop check | `applyThread()` counts pixels > `cut_off_value` per block, then reduces across blocks | `xp.sum(xp.abs(working) > convergence_threshold) == 0` — pixel-by-pixel check, identical logic (v1.1+) |
| Inner loop check | Same pixel-by-pixel method as outer loop | Same pixel-by-pixel method (v1.1+) with stagnation detection (v1.4+) |
| Divergence detection | Counter exceeds `fcms_taylor_max_iter()` | Outer: amplitude ratio `> 2.0`. Inner: unconverged pixel count stagnates or grows |
| Non-convergence handling | Implicit partial sum (counter limit) | Warning + best-effort partial sum (v1.4+) |
| Cutoff granularity | Per-pixel (each pixel independently compared) | Per-pixel (each pixel independently compared) (v1.1+) |

**Status**: ✅ **Aligned.** Both implementations use pixel-by-pixel convergence
checks for both inner and outer loops. The abTEM v1.4+ diverged/convergence warnings
(Warning instead of hard error) match the C++ behavior of accepting partial convergence.

### 3. Laplace Operator

| Aspect | ImageSimulation_CGS | abTEM |
|--------|--------------------|-------|
| Real-space method | `propFCMS_LaplaceNinePoint_1dthread` — custom 9-point finite difference kernel | `LaplaceOperator` — centered finite difference with configurable accuracy (default 8, i.e. 9-point stencil) |
| Fourier-space method | `MultiCoefInReciprocalSpace` + cuFFT | `_laplace_operator_fft` — FFT-based exact band-limited Laplacian (v1.3+) |
| Selection | Runtime flag `simu.FT_9Point` | `laplace_method` parameter: `"finite-difference"` or `"fft"` |
| Boundary | Not explicitly wrapped | Mode=`"wrap"` (periodic) |

**Status**: ✅ **Aligned.** Both real-space (9-point stencil) and Fourier-space (FFT)
methods are available. The abTEM FFT method uses float64 internally for numerical
stability (see v1.3 notes).

### 4. Backscattering

| Aspect | ImageSimulation_CGS | abTEM |
|--------|--------------------|-------|
| Operator | `calBSC()` — custom CUDA using `calK_forward_back` + `calOneDevideK_forward_back` | `_cvdms_backscattering_correction()` — uses `_cvdms_inner_k_series` + `full_series()` |
| Backscattered wave propagation | Reverse loop from current slice to surface, re-using `calPureForwardScatter` at each slice | Reverse loop using `conj` trick: `conj(forward_scattering(conj(ψ), V(z)))` (v1.2+) |
| Correction scheme | `phi_j = (1 - B_{j+1,j}) · ψ_j` (subtract BSC from forward wave) | Identical scheme (v1.2+) |
| Last-slice handling | N/A (loop structure differs) | `fully_corrected` (v1.4+): returns `(Waves, Waves)` tuple for all slices |
| `expansion_scope` & `fully_corrected` | Not applicable (monolithic function) | `expansion_scope` controls BSC activation; `fully_corrected` controls return type convention |

### How `expansion_scope`, `include_backscattering`, and `fully_corrected` work

These parameters operate at two independent levels — **structural** and **physical**:

- **`expansion_scope`** (structural switch, on `CVDMSMultislice`):
  Controls the multislice loop calling convention.
  - `"propagator"` (default): `multislice_and_detect` passes `next_slice=None`. Single return value. No inter-slice coupling structure — BSC cannot execute regardless of `include_backscattering`.
  - `"full"`: `multislice_and_detect` passes actual `next_slice`. Tuple return `(Waves, Waves)`. Inter-slice coupling structure is set up.

- **`include_backscattering`** (physical switch, on `CVDMSMultislice`):
  Controls whether the BSC mathematical operator is applied at each slice interface.
  - `True`: Compute `BSC(ψ)` and subtract from the forward wave: `ψ_corrected = ψ_forward - BSC`.
  - `False`: Skip the BSC operator; forward wave is used as-is.

- **`fully_corrected`** (internal, not user-visible): Derived from `expansion_scope == "full"`.
  Controls return type of `cvdms_multislice_step`:
  - `True`: Always return `(Waves, Waves)` tuple (backscattered wave is zero-padded for the last slice).
  - `False`: Return single `Waves` when BSC is not computed.

**Interaction matrix:**

| `expansion_scope` | `include_backscattering` | `next_slice` | BSC applied | Return type | Effect |
|---|---|---|---|---|---|
| `"propagator"` | `True` (default) | `None` | No | single `Waves` | Default thin-sample mode. BSC structural condition (`next_slice is not None`) is False. |
| `"propagator"` | `False` | `None` | No | single `Waves` | Explicitly disabled, same structural path. |
| `"full"` | `True` | actual slice | **Yes** | `(Waves, Waves)` | Full CVDMS with BSC correction. |
| `"full"` | `False` | actual slice | No | `(Waves, Waves)` | Structural convention enabled, physical operator skipped. |

**Key takeaway**: When `expansion_scope="propagator"`, `include_backscattering` has no effect because `next_slice=None` prevents the BSC condition from being satisfied.

**Status**: ✅ **Aligned.** Both implementations now perform full backward propagation
of backscattered waves through all preceding slices (since abTEM v1.2). The `conj` trick
is functionally equivalent to re-using the forward scattering kernel in reverse. The v1.4
`fully_corrected` fix ensures correct tuple return type for the last slice.

### 5. Memory & GPU

| Aspect | ImageSimulation_CGS | abTEM |
|--------|--------------------|-------|
| GPU backend | Direct CUDA (cuFFT, custom kernels) | CuPy (array-based, automatic GPU mapping) |
| Memory management | Manual `cudaMalloc`/`cudaMemcpy` with explicit staging buffers | Python garbage-collected; uses `copy_to_device()` |
| Kernels | Hand-tuned CUDA kernels (`multiplyComplex_i_CGS`, `addArray_1dthread`, etc.) | Generic NumPy/CuPy array operations |
| CPU fallback | Not available (GPU-only) | Automatic via `get_array_module()` |

**Impact**: abTEM's CuPy-based approach is more portable and easier to maintain but
may have higher per-operation overhead compared to fused CUDA kernels.

### 6. Integration

| Aspect | ImageSimulation_CGS | abTEM |
|--------|--------------------|-------|
| Calling convention | Standalone `CalCBED_CVDMS_FP(calTem, output)` | `cvdms_multislice_step(waves, potential_slice, ...)` |
| Parameter passing | Through `ImageCalTEM` struct with global config | Explicit Python function arguments + dataclass |
| Extensibility | Monolithic function, hard to extend | Composable: integrates with `multislice_and_detect()`, detectors, scans |
| Testing | Manual validation | `pytest` with 16 test cases |

---

## Summary Table

| Feature | ImageSimulation_CGS | abTEM | Match? |
|---------|--------------------|-------|--------|
| Forward scattering outer loop | ✅ Taylor series with convergence | ✅ Taylor series with convergence | ✅ |
| Forward scattering inner loop | ✅ Taylor series with convergence | ✅ Taylor series with convergence (v1.1+) | ✅ |
| Pixel-by-pixel convergence | ✅ `applyThread` | ✅ `xp.sum(|term| > threshold)` (v1.1+) | ✅ |
| Divergence detection | ✅ `fcms_taylor_max_iter()` | ✅ Amplitude ratio + stagnation/growth detection (v1.4+) | ⚠️ Different heuristic |
| Non-convergence handling | ✅ Implicit partial sum | ✅ Warning + best-effort partial sum (v1.4+) | ✅ |
| Backscattering operator | ✅ Custom `calBSC` | ✅ Uses `_cvdms_inner_k_series` + `full_series` | ✅ |
| Backscattered wave back-propagation | ✅ Full multi-slice loop | ✅ Full multi-slice loop via `conj` trick (v1.2+) | ✅ |
| Last-slice tuple return | ✅ N/A (different structure) | ✅ `fully_corrected` returns `(Waves, Waves)` (v1.4+) | ✅ |
| Laplace: 9-point stencil | ✅ `propFCMS_LaplaceNinePoint` | ✅ Configurable accuracy, default 8 (v1.3+) | ✅ |
| Laplace: Fourier method | ✅ `MultiCoefInReciprocalSpace` + cuFFT | ✅ `_laplace_operator_fft` (v1.3+) | ✅ |
| GPU | ✅ Native CUDA | ✅ CuPy (automatic) | Different backend |
| CPU fallback | ❌ Not supported | ✅ Automatic | Different |
| API style | Monolithic C++ struct | Modular Python dataclass | Different |
| Detector integration | ❌ Internal image writing | ✅ `WavesDetector`, `PixelatedDetector` | Different |

## Remaining Differences

Despite the close alignment, the following differences remain:

1. **Divergence heuristic**: The original's `fcms_taylor_max_iter()` uses a counter-based
   threshold, while abTEM detects whether the unconverged pixel count decreases or
   stagnates (`>=` check, v1.4+). Both truncate the series at the optimal point; the
   exact cutoff may differ slightly near the stability boundary.

2. **Outer loop non-convergence**: abTEM v1.4+ issues a warning and returns the best
   approximation when the outer Taylor series does not fully converge within `max_terms`.
   The C++ version implicitly accepts partial convergence via its dynamic counter limits.
   Behavior is functionally equivalent.

3. **GPU backend**: Hand-tuned CUDA kernels vs. CuPy array operations. The CuPy approach
   may introduce negligible overhead from repeated kernel launches that are fused in
   the original.

4. **Backscattering `1/k` correction**: The original's `calOneDevideK_forward_back` is a
   dedicated CUDA kernel; abTEM uses `full_series()` from `finite_difference.py` with
   custom prefactors. The mathematical result is equivalent.

## When to Use Which

- **ImageSimulation_CGS** is the reference implementation for production-grade CBED
  simulation where maximum numerical fidelity to the CVDMS theory is required.

- **abTEM CVDMS** is suitable for:
  - Exploratory CBED simulations
  - Integration with abTEM's broader ecosystem (detectors, scans, frozen phonons)
  - CPU-based workflows (no GPU required)
  - Comparing CVDMS with Fourier/RealSpace multislice within the same framework
