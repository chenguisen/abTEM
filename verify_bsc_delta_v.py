"""
BSC ΔV 方案验证 — 简化版

使用 CuPy/GPU 路径，找出 I/I₀ > 1 的真实来源。

诊断策略：
1. 遍历所有相邻切片对，测量 K_series 差分的条件数
2. 测量 1/k 修正是否放大误差
3. 完整 BSC 多层扫描测 I/I₀
"""

import warnings
import numpy as np
import abtem
abtem.core.config.config['fft'] = 'numpy'
from ase.spacegroup import crystal
from abtem.core.energy import energy2wavelength
from abtem.cvdms import (
    _cvdms_inner_k_series,
    _cvdms_backscattering_correction,
    _cvdms_forward_scattering,
)
from abtem.finite_difference import LaplaceOperator
import ase

warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore")


def make_srtio3():
    a = 3.905
    return crystal(('Sr', 'Ti', 'O'),
                   basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                   spacegroup=221, cellpar=[a, a, a, 90, 90, 90])


def get_potential_slices(atoms, gpts, sampling, slice_thickness, z_mult=4):
    atoms_z = atoms * (1, 1, z_mult)
    pot = abtem.Potential(atoms_z, gpts=gpts, slice_thickness=slice_thickness,
                          sampling=sampling)
    arr = pot.build()
    slices = np.asarray(arr.array).squeeze()
    if slices.ndim == 2:
        slices = slices[np.newaxis, :, :]
    return slices


def diagnose_condition_number():
    """遍历所有相邻切片，测量 wave_2 - wave_1 的条件数

    核心问题：BSC 中的减法 wave_2 - wave_1 是否因 catastrophic cancellation 导致精度丢失？
    条件数 κ = |wave| / |Δ|
    κ > 10⁶ (float32) → 精度完全丢失
    """
    print("=" * 70)
    print("Condition number diagnostic: scan all adjacent slice pairs")
    print("=" * 70)

    atoms = make_srtio3()
    slices = get_potential_slices(atoms, gpts=(128, 128), sampling=0.05,
                                  slice_thickness=0.4, z_mult=6)
    print(f"  Total slices: {slices.shape[0]}")
    non_zero = [i for i in range(slices.shape[0]) if slices[i].max() > 0]
    print(f"  Non-zero slices: {len(non_zero)}/{slices.shape[0]}")

    try:
        import cupy as cp
        xp = cp
        device = "gpu"
    except ImportError:
        xp = np
        device = "cpu"

    for voltage in [30e3, 80e3, 300e3]:
        wavelength = energy2wavelength(voltage)
        K0 = 1.0 / wavelength
        nx, ny = slices.shape[-2:]

        op = LaplaceOperator(accuracy=8, method="finite-difference")
        laplace = op._get_new_stencil(("dummy", (0.05, 0.05)), device=device)

        max_cond = 0.0
        max_delta_ratio = 0.0
        worst_pair = None
        cond_list = []

        for i in range(slices.shape[0] - 1):
            V_cur = xp.asarray(slices[i])
            V_next = xp.asarray(slices[i + 1])

            # Skip if both zero
            if float(xp.max(xp.abs(V_cur))) == 0 and float(xp.max(xp.abs(V_next))) == 0:
                continue

            psi = xp.ones((nx, ny), dtype=xp.complex64)

            ks_cur = _cvdms_inner_k_series(
                psi, V_cur, laplace, wavelength,
                convergence_threshold=1e-16, use_fused_kernel=False)

            ks_next = _cvdms_inner_k_series(
                psi, V_next, laplace, wavelength,
                convergence_threshold=1e-16, use_fused_kernel=False)

            wave_1 = ks_cur / (2 * np.pi) + psi * K0
            wave_2 = ks_next / (2 * np.pi) + psi * K0
            delta = wave_2 - wave_1

            mean_wave = float(xp.mean(xp.abs(wave_1)))
            mean_delta = float(xp.mean(xp.abs(delta)))
            cond = mean_wave / max(mean_delta, 1e-30)
            delta_ratio = mean_delta / max(mean_wave, 1e-30)

            cond_list.append(cond)
            if cond > max_cond:
                max_cond = cond
                max_delta_ratio = delta_ratio
                worst_pair = i

        # Statistics
        cond_arr = np.array(cond_list)
        print(f"\n  {voltage/1e3:.0f}keV ({device}):")
        print(f"    Pairs analyzed: {len(cond_list)}")
        print(f"    Condition number κ = |wave|/|Δ|:")
        print(f"      max  κ = {max_cond:.2e} (pair {worst_pair}-{worst_pair+1})")
        print(f"      mean κ = {float(np.mean(cond_arr)):.2e}")
        print(f"      min  κ = {float(np.min(cond_arr)):.2e}")
        print(f"      > 10⁶: {(cond_arr > 1e6).sum()}/{len(cond_arr)} pairs")
        print(f"      > 10³: {(cond_arr > 1e3).sum()}/{len(cond_arr)} pairs")
        print(f"    |Δ|/|wave| ratio at worst pair: {max_delta_ratio:.6f}")

        # Show worst pair details
        if worst_pair is not None:
            i = worst_pair
            V_cur = xp.asarray(slices[i])
            V_next = xp.asarray(slices[i + 1])
            psi = xp.ones((nx, ny), dtype=xp.complex64)
            ks_cur = _cvdms_inner_k_series(
                psi, V_cur, laplace, wavelength,
                convergence_threshold=1e-16, use_fused_kernel=False)
            ks_next = _cvdms_inner_k_series(
                psi, V_next, laplace, wavelength,
                convergence_threshold=1e-16, use_fused_kernel=False)
            wave_1 = ks_cur / (2 * np.pi) + psi * K0
            wave_2 = ks_next / (2 * np.pi) + psi * K0
            delta = wave_2 - wave_1

            print(f"    Worst pair details (slices {i}, {i+1}):")
            print(f"      |V_cur|_max  = {float(xp.max(xp.abs(V_cur))):.2e}")
            print(f"      |V_next|_max = {float(xp.max(xp.abs(V_next))):.2e}")
            print(f"      |ψ·K₀|_max   = {float(xp.max(xp.abs(psi * K0))):.2e}")
            print(f"      |K_series(cur)|_max  = {float(xp.max(xp.abs(ks_cur))):.2e}")
            print(f"      |K_series(next)|_max = {float(xp.max(xp.abs(ks_next))):.2e}")
            print(f"      |ΔK_series|_max      = {float(xp.max(xp.abs(ks_next - ks_cur))):.2e}")
            print(f"      |wave_1|_max  = {float(xp.max(xp.abs(wave_1))):.2e}")
            print(f"      |delta|_max   = {float(xp.max(xp.abs(delta))):.2e}")
            print(f"      κ_mean        = {max_cond:.2e}")

    print()
    return True


def diagnose_1k_correction():
    """诊断 1/k 修正对 I/I₀ 的影响

    通过比较：
    - 完整 BSC（含 1/k 修正收敛循环）
    - 截断 BSC（单步，不含 1/k 修正）
    """
    print("=" * 70)
    print("1/k correction diagnostic")
    print("=" * 70)

    atoms = ase.build.bulk("Si", cubic=True)
    slices = get_potential_slices(atoms, gpts=(64, 64), sampling=0.1,
                                  slice_thickness=0.5, z_mult=4)

    try:
        import cupy as cp
        xp = cp
        device = "gpu"
    except ImportError:
        xp = np
        device = "cpu"

    print(f"  Device: {device}, slices: {slices.shape[0]}")

    for voltage in [300e3, 80e3, 30e3]:
        wavelength = energy2wavelength(voltage)
        K0 = 1.0 / wavelength
        nx, ny = slices.shape[-2:]

        op = LaplaceOperator(accuracy=8, method="finite-difference")
        laplace = op._get_new_stencil(("dummy", (0.1, 0.1)), device=device)

        psi_full = xp.ones((nx, ny), dtype=xp.complex64)
        psi_no1k = xp.ones((nx, ny), dtype=xp.complex64)
        I0 = float(xp.sum(xp.abs(psi_full) ** 2))

        max_ratio_full = 0.0
        max_ratio_no1k = 0.0
        n_ok = 0
        overflow = False

        from abtem.cvdms import _cvdms_backscattering_correction

        for i in range(slices.shape[0] - 1):
            V_cur = xp.asarray(slices[i].astype(np.float32))
            V_next = xp.asarray(slices[i + 1].astype(np.float32))

            if float(xp.max(V_cur)) == 0 and float(xp.max(V_next)) == 0:
                continue

            # Forward
            psi_fwd, _ = _cvdms_forward_scattering(
                psi_full, V_cur, laplace, wavelength, 0.5,
                max_terms=50, convergence_threshold=1e-7,
                divergence_ratio=5.0, use_fused_kernel=False,
                return_diagnostics=True)

            if np.any(np.isnan(float(xp.sum(xp.abs(psi_fwd))))):
                overflow = True
                break

            # Full BSC (with 1/k correction loop)
            bsc_full = _cvdms_backscattering_correction(
                psi_fwd, V_cur, V_next, laplace, wavelength, 0.5,
                convergence_threshold=1e-16, max_inner_iter=100,
                use_fused_kernel=False)

            # Also compute BSC at first order only (no 1/k correction)
            # The 1/k correction is the loop at lines 800-838 of cvdms.py
            # which iteratively improves backscatter.
            # We can approximate "no 1/k" by setting convergence_threshold=1e-1
            # (stops after first iteration)
            bsc_no1k = _cvdms_backscattering_correction(
                psi_fwd, V_cur, V_next, laplace, wavelength, 0.5,
                convergence_threshold=1e-1, max_inner_iter=1,
                use_fused_kernel=False)

            psi_full = psi_fwd - bsc_full
            psi_no1k = psi_fwd - bsc_no1k

            r_full = float(xp.sum(xp.abs(psi_full) ** 2)) / I0
            r_no1k = float(xp.sum(xp.abs(psi_no1k) ** 2)) / I0

            max_ratio_full = max(max_ratio_full, r_full)
            max_ratio_no1k = max(max_ratio_no1k, r_no1k)
            n_ok += 1

        label = f"{voltage/1e3:.0f}keV"
        if overflow:
            print(f"  {label}: OVERFLOW at slice {n_ok}")
        else:
            print(f"  {label}:")
            print(f"    Full BSC max I/I₀ = {max_ratio_full:.6f}")
            print(f"    No 1/k  max I/I₀ = {max_ratio_no1k:.6f}")
            print(f"    1/k effect       = {max_ratio_full - max_ratio_no1k:+.6f}")

    print()
    return True


def test_bsc_intensity():
    """完整多层扫描，测 I/I₀"""
    print("=" * 70)
    print("BSC intensity conservation test")
    print("=" * 70)

    atoms = ase.build.bulk("Si", cubic=True)
    slices = get_potential_slices(atoms, gpts=(64, 64), sampling=0.1,
                                  slice_thickness=0.5, z_mult=4)

    try:
        import cupy as cp
        xp = cp
        device = "gpu"
    except ImportError:
        xp = np
        device = "cpu"

    print(f"  Device: {device}, slices: {slices.shape[0]}")

    for voltage in [300e3, 200e3, 80e3, 30e3]:
        wavelength = energy2wavelength(voltage)
        nx, ny = slices.shape[-2:]

        op = LaplaceOperator(accuracy=8, method="finite-difference")
        laplace = op._get_new_stencil(("dummy", (0.1, 0.1)), device=device)

        psi = xp.ones((nx, ny), dtype=xp.complex64)
        I0 = float(xp.sum(xp.abs(psi) ** 2))
        n_ok = 0
        max_I_ratio = 0.0
        overflow = False

        for i in range(slices.shape[0] - 1):
            V_cur = xp.asarray(slices[i].astype(np.float32))
            V_next = xp.asarray(slices[i + 1].astype(np.float32))

            if float(xp.max(V_cur)) == 0 and float(xp.max(V_next)) == 0:
                continue

            # Forward
            psi_fwd, diag = _cvdms_forward_scattering(
                psi, V_cur, laplace, wavelength, 0.5,
                max_terms=50, convergence_threshold=1e-7,
                divergence_ratio=5.0, use_fused_kernel=False,
                return_diagnostics=True)

            if np.any(np.isnan(float(xp.sum(xp.abs(psi_fwd))))) or \
               np.any(np.isinf(float(xp.sum(xp.abs(psi_fwd))))):
                overflow = True
                break

            # BSC
            bsc = _cvdms_backscattering_correction(
                psi_fwd, V_cur, V_next, laplace, wavelength, 0.5,
                convergence_threshold=1e-16, max_inner_iter=100,
                use_fused_kernel=False)

            psi = psi_fwd - bsc
            if np.any(np.isnan(float(xp.sum(xp.abs(psi))))) or \
               np.any(np.isinf(float(xp.sum(xp.abs(psi))))):
                overflow = True
                break

            I_ratio = float(xp.sum(xp.abs(psi) ** 2)) / I0
            max_I_ratio = max(max_I_ratio, I_ratio)
            n_ok += 1

        label = f"{voltage/1e3:.0f}keV"
        if overflow:
            print(f"  {label}: OVERFLOW at slice {n_ok}")
        else:
            status = "✅" if max_I_ratio <= 1.001 else f"⚠️  I/I₀={max_I_ratio:.6f}"
            print(f"  {label}: max I/I₀={max_I_ratio:.6f}, slices={n_ok} {status}")

    print()


def run_ablation_study():
    """消融实验：逐步隔离 I/I₀ > 1 的来源

    对单步 BSC 做逐步消融：
    1. forward only (no BSC) → 基线 I/I₀
    2. forward + BSC (full)  → 当前实现
    3. forward + BSC(no 1/k) → 看看 1/k 修正的影响
    4. forward + BSC(manual delta) → 手动计算 delta 而非用 K_series 减法
    """
    print("=" * 70)
    print("Ablation study: isolate I/I₀ > 1 source")
    print("=" * 70)

    atoms = ase.build.bulk("Si", cubic=True)
    slices = get_potential_slices(atoms, gpts=(64, 64), sampling=0.1,
                                  slice_thickness=0.5, z_mult=4)

    try:
        import cupy as cp
        xp = cp
        device = "gpu"
    except ImportError:
        xp = np
        device = "cpu"

    print(f"  Device: {device}, slices: {slices.shape[0]}")
    voltage = 30e3  # worst case
    wavelength = energy2wavelength(voltage)
    nx, ny = slices.shape[-2:]

    op = LaplaceOperator(accuracy=8, method="finite-difference")
    laplace = op._get_new_stencil(("dummy", (0.1, 0.1)), device=device)

    # Track I/I₀ at each step for all variants
    I0_val = float(nx * ny)  # |ψ|=1 everywhere

    # --- Single step analysis ---
    # Pick a pair of adjacent non-zero slices
    for i_test in range(slices.shape[0] - 1):
        if slices[i_test].max() > 0 and slices[i_test + 1].max() > 0:
            break
    else:
        print("  No adjacent non-zero pair found!")
        return

    V_cur = xp.asarray(slices[i_test].astype(np.float32))
    V_next = xp.asarray(slices[i_test + 1].astype(np.float32))

    print(f"\n  Analyzing step {i_test} (30keV):")

    # Step 1: forward only
    psi = xp.ones((nx, ny), dtype=xp.complex64)
    psi_fwd, _ = _cvdms_forward_scattering(
        psi, V_cur, laplace, wavelength, 0.5,
        max_terms=50, convergence_threshold=1e-7,
        divergence_ratio=5.0, use_fused_kernel=False,
        return_diagnostics=True)
    I_fwd = float(xp.sum(xp.abs(psi_fwd) ** 2)) / I0_val
    print(f"    Forward only:           I/I₀ = {I_fwd:.6f}")

    # Step 2: forward + BSC (full)
    bsc = _cvdms_backscattering_correction(
        psi_fwd, V_cur, V_next, laplace, wavelength, 0.5,
        convergence_threshold=1e-16, max_inner_iter=100,
        use_fused_kernel=False)
    psi_bsc = psi_fwd - bsc
    I_bsc = float(xp.sum(xp.abs(psi_bsc) ** 2)) / I0_val
    print(f"    Forward + BSC (full):   I/I₀ = {I_bsc:.6f}")

    # Step 3: forward + BSC (no 1/k)
    bsc_no1k = _cvdms_backscattering_correction(
        psi_fwd, V_cur, V_next, laplace, wavelength, 0.5,
        convergence_threshold=1e-1, max_inner_iter=1,
        use_fused_kernel=False)
    psi_no1k = psi_fwd - bsc_no1k
    I_no1k = float(xp.sum(xp.abs(psi_no1k) ** 2)) / I0_val
    print(f"    Forward + BSC (no 1/k): I/I₀ = {I_no1k:.6f}")

    # Step 4: manual delta (compute wave_1, wave_2 directly)
    K0 = 1.0 / wavelength
    ks_cur = _cvdms_inner_k_series(
        psi_fwd, V_cur, laplace, wavelength,
        convergence_threshold=1e-16, use_fused_kernel=False)
    ks_next = _cvdms_inner_k_series(
        psi_fwd, V_next, laplace, wavelength,
        convergence_threshold=1e-16, use_fused_kernel=False)

    wave_1 = ks_cur / (2 * np.pi) + psi_fwd * K0
    wave_2 = ks_next / (2 * np.pi) + psi_fwd * K0
    delta_manual = wave_2 - wave_1
    psi_manual = psi_fwd - delta_manual
    I_manual = float(xp.sum(xp.abs(psi_manual) ** 2)) / I0_val
    print(f"    Forward + delta manual: I/I₀ = {I_manual:.6f}")

    # Compare delta from BSC vs manual
    delta_diff = float(xp.max(xp.abs(bsc - delta_manual)))
    delta_bsc_norm = float(xp.max(xp.abs(bsc)))
    delta_man_norm = float(xp.max(xp.abs(delta_manual)))
    print(f"    |BSC delta|_max  = {delta_bsc_norm:.2e}")
    print(f"    |manual delta|_max = {delta_man_norm:.2e}")
    print(f"    |BSC - manual|_max = {delta_diff:.2e}")

    # Step 5: What if we just renormalize the BSC output intensity?
    bsc_frac = float(xp.sum(xp.abs(bsc) ** 2)) / float(xp.sum(xp.abs(psi_fwd) ** 2))
    print(f"    |BSC|²/|ψ_fwd|² = {bsc_frac:.6f}")

    print()


if __name__ == "__main__":
    diagnose_condition_number()
    diagnose_1k_correction()
    run_ablation_study()
    test_bsc_intensity()
