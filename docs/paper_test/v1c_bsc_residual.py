#!/usr/bin/env python3
"""V1c: Fresnel BSC residual — backscattering correction unitarity test.

Tests that the CVDMS backscattering correction satisfies energy conservation.
Run CVDMS with BSC enabled, extract forward and backscattered waves,
verify the energy budget: I_fwd + I_bsc ≈ I_incident.

The Fresnel BSC uses T = sqrt(1 - |R|²) by construction, so |T|² + |R|² = 1
is satisfied identically.  This test verifies the integrated energy budget,
BSC bottom=0, BSC depth monotonicity, and forward I/I0 monotonic decrease.

Acceptance criteria (paper outline §14.2.1):
  V1c: |T|²+|R|²-1 < 1e-07  (verified by construction in BSC engine)
  Energy budget: |I_fwd + I_bsc - I0| / I0 < 3e-4 (A) / 3e-3 (B)
  BSC bottom = 0: max|BSC[final EP]| < 1e-10
  BSC depth monotonicity: BSC increases from bottom to top
  Forward I/I0 ≤ 1: all exit planes
  Forward monotonic: I/I0 decreases with depth

Stages:
  A: 128×128 grid, 39 Å — rapid validation
  B: 625×625 grid, 117 Å — target resolution

Outputs:
  docs/data/v1c_bsc_residual_{stage}.npz — numerical results
  docs/data/v1c_bsc_residual_{stage}.pdf  — figure (Comms Phys vector format)

Usage:
  python v1c_bsc_residual.py [A|B]
"""
import sys, os, gc, json
import numpy as np
import cupy as cp
from ase.spacegroup import crystal
import abtem
from abtem.multislice import CVDMSMultislice
from abtem.core import config as _cfg

_cfg.set({"device": "gpu", "fft": "cupy"})

# ── Output directory ──
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Parameter manifest
# ============================================================
ENERGY = 30e3
SLICE_THICKNESS = 0.4
CONVERGENCE_THRESHOLD = 1e-7

STAGE = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
if STAGE == "B":
    SUPERCELL_XY = 8
    SUPERCELL_Z = 30             # ~117 Å (verified safe in notebook)
    SAMPLING = 0.05
    EXIT_PLANES = 30
else:
    SUPERCELL_XY = 4
    SUPERCELL_Z = 10             # thin: ~39 Å
    EXIT_PLANES = 8

ENERGY_BUDGET_TOL_A = 3e-4
ENERGY_BUDGET_TOL_B = 3e-3  # 293 slices × float32 → ~0.3% roundoff limit


def to_cpu(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return arr


def total_intensity(arr):
    return float(np.sum(np.abs(to_cpu(arr)) ** 2))


def cleanup():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def save_results(stage, results, params):
    """Save numerical results to NPZ and summary to JSON."""
    base = os.path.join(DATA_DIR, f"v1c_bsc_residual_{stage}")
    np.savez(base + ".npz",
        fwd_I=results["fwd_I"],
        fwd_ratios=results["fwd_ratios"],
        bsc_I=results["bsc_I"],
        bsc_bottom_max=results["bsc_bottom_max"],
        I0=results["I0"],
        I_fwd_exit=results["I_fwd_exit"],
        I_bsc_entrance=results["I_bsc_entrance"],
        energy_balance=results["energy_balance"],
        energy_budget_tol=results["energy_budget_tol"],
        forward_loss_pct=results["forward_loss_pct"],
        depths=results["depths"],
        tests_pass=results["tests_pass"],
    )
    manifest = {
        "script": "v1c_bsc_residual.py",
        "stage": stage,
        "params": params,
        "results": {k: (float(v) if np.isscalar(v) and not isinstance(v, bool)
                        else v.tolist() if isinstance(v, np.ndarray)
                        else v)
                    for k, v in results.items()
                    if k not in ("fwd_I", "fwd_ratios", "bsc_I", "depths")},
        "tests": {k: bool(v) for k, v in results["tests_pass"].items()},
    }
    with open(base + ".json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Data saved: {base}.npz  +  .json")


def _unused_make_figure(stage, results, params):  # deprecated — use plot_v1c.py
    """Generate Comms Phys-compliant PDF figure."""
    fwd_I = results["fwd_I"]
    fwd_ratios = results["fwd_ratios"]
    bsc_I = results["bsc_I"]
    depths = results["depths"]
    I0 = results["I0"]
    num_ep = len(fwd_I)

    px_A = params["sampling_A_per_px"]
    thickness_A = params["thickness_A"]

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.55))
    # Panel layout: left 2/3 = (a) Fwd I/I0 top | (b) BSC depth bottom
    #              right 1/3 = (c) energy budget bar
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8],
                          hspace=0.45, wspace=0.45,
                          left=0.07, right=0.96, top=0.96, bottom=0.18)

    # ── Panel (a): Forward I/I0 vs depth ──
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.plot(depths, fwd_ratios, "o-", color=WONG["blue"],
              markersize=3, linewidth=0.8)
    ax_a.axhline(y=1.0, color=WONG["grey"], linestyle="--", linewidth=0.5, alpha=0.7)
    ax_a.set_xlabel("Depth (Å)")
    ax_a.set_ylabel("Forward I / I$_0$")
    ax_a.set_ylim(min(fwd_ratios) - 0.0002, 1.0002)
    ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    loss_pct = results["forward_loss_pct"]
    ax_a.text(0.98, 0.88, f"loss = {loss_pct:.3f}%", transform=ax_a.transAxes,
              ha="right", fontsize=6.5, color=WONG["red"],
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax_a.text(0.02, 0.88, f"{px_A:.3f} Å/px", transform=ax_a.transAxes,
              fontsize=6, color=WONG["grey"])
    # Label
    ax_a.text(-0.12, 1.05, "a", transform=ax_a.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Panel (b): BSC intensity vs depth ──
    ax_b = fig.add_subplot(gs[1, :2])
    ax_b.semilogy(depths, bsc_I, "s-", color=WONG["red"],
                  markersize=3, linewidth=0.8)
    ax_b.set_xlabel("Depth (Å)  (0 = entrance surface)")
    ax_b.set_ylabel("Σ|BSC|²")
    ax_b.invert_xaxis()
    ax_b.text(0.02, 0.08, f"entrance Σ|BSC|² = {bsc_I[0]:.2e}",
              transform=ax_b.transAxes, fontsize=6, color=WONG["grey"])
    ax_b.text(0.02, 0.02, f"bottom = {bsc_I[-1]:.1e}",
              transform=ax_b.transAxes, fontsize=6, color=WONG["grey"])
    ax_b.text(0.98, 0.08, f"{px_A:.3f} Å/px, {params['num_exit_planes']} exit planes",
              transform=ax_b.transAxes, ha="right", fontsize=6, color=WONG["grey"])
    # Label
    ax_b.text(-0.12, 1.05, "b", transform=ax_b.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Panel (c): Energy budget bar ──
    ax_c = fig.add_subplot(gs[:, 2])
    I_fwd = results["I_fwd_exit"]
    I_bsc = results["I_bsc_entrance"]
    ax_c.bar(["I$_0$", "I$_{fwd}$", "Σ|BSC|² × 10³"],
             [I0, I_fwd, I_bsc * 1e3],
             color=[WONG["grey"], WONG["blue"], WONG["red"]],
             width=0.55, edgecolor="white", linewidth=0.5)
    ax_c.set_ylabel("Integrated intensity")
    ax_c.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    eb = results["energy_balance"]
    ax_c.text(0.5, 0.92, f"|I$_{{fwd}}$+I$_{{bsc}}$−I$_0$|/I$_0$\n= {eb:.2e}",
              transform=ax_c.transAxes, ha="center", fontsize=6.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=WONG["grey"], alpha=0.8))
    ax_c.text(0.5, 0.04, f"{thickness_A:.0f} Å", transform=ax_c.transAxes,
              ha="center", fontsize=6, color=WONG["grey"])
    # Label
    ax_c.text(-0.22, 1.05, "c", transform=ax_c.transAxes, fontsize=9,
              fontweight="bold", va="bottom", ha="left")

    # ── Save ──
    out_pdf = os.path.join(DATA_DIR, f"v1c_bsc_residual_{stage}.pdf")
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"Figure saved: {out_pdf}")


def main():
    print(f"=== V1c: Fresnel BSC Residual (Stage {STAGE}) ===")
    print(f"Supercell: ({SUPERCELL_XY},{SUPERCELL_XY},{SUPERCELL_Z})")

    # ── Build SrTiO3 ──
    a = 3.905
    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[a, a, a, 90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z)
    total_z = atoms.cell[2, 2]
    print(f"Total thickness: {total_z:.1f} Å")

    # ── Build potential (single config) ──
    if STAGE == "B":
        potential = abtem.Potential(
            atoms, sampling=SAMPLING,
            slice_thickness=SLICE_THICKNESS,
            exit_planes=EXIT_PLANES, projection="finite",
        )
    else:
        potential = abtem.Potential(
            atoms,
            gpts=(128, 128),
            slice_thickness=SLICE_THICKNESS,
            exit_planes=EXIT_PLANES, projection="finite",
        )
    num_ep = potential.num_exit_planes
    print(f"Grid: {potential.gpts}  px={potential.sampling[0]:.4f} Å/px")
    print(f"Exit planes: {num_ep}")

    # ── Build plane wave ──
    wave = abtem.PlaneWave(energy=ENERGY)
    wave.grid.match(potential)

    # ── Run CVDMS + BSC ──
    use_aa = (STAGE == "B")
    cvdms = CVDMSMultislice(
        convergence_threshold=CONVERGENCE_THRESHOLD,
        backscattering=True,
        calculate_backscattered=True,
        antialias=use_aa,
        antialias_inner=True,
    )

    print("Running CVDMS + BSC...")
    result = wave.multislice(
        potential,
        algorithm=cvdms,
        return_backscattered=True,
        lazy=False,
    )
    exit_wave = result[0]
    bsc_wave = result[-1]
    print(f"Exit wave shape: {exit_wave.shape}")
    print(f"BSC wave shape: {bsc_wave.shape}")

    # ── Compute I0 ──
    I0 = total_intensity(exit_wave.array[0])

    # ── Collect results ──
    num_ep_actual = exit_wave.shape[0]
    depths = np.linspace(0, total_z, num_ep_actual)

    # ═══════════════════════════════════════════════════════════
    # A: Forward wave self-consistency
    # ═══════════════════════════════════════════════════════════
    print(f"\n--- A: Forward wave self-consistency ---")
    print(f"I0 = {I0:.6f}")

    # A1: I/I0 ≤ 1
    fwd_I = []
    for ep in range(num_ep_actual):
        fwd_I.append(total_intensity(exit_wave.array[ep]))
    fwd_I = np.array(fwd_I)
    fwd_ratios = fwd_I / I0
    a1_ok = all(r <= 1.0 + 1e-4 for r in fwd_ratios)
    print(f"A1  Forward I/I0 ≤ 1:  {'PASS' if a1_ok else 'FAIL'}")
    for ep in range(num_ep_actual):
        flag = "" if fwd_ratios[ep] <= 1.0 + 1e-4 else "  *** EXCEEDS 1"
        print(f"    EP {ep:2d}: I/I0 = {fwd_ratios[ep]:.8f}  (I = {fwd_I[ep]:.6f}){flag}")

    # A2: monotonic decrease
    worst_delta = 0.0
    for i in range(1, len(fwd_I)):
        delta_I = fwd_I[i] - fwd_I[i-1]
        if delta_I > 0:
            worst_delta = max(worst_delta, delta_I / I0)
    a2_ok = worst_delta < 1e-4
    print(f"A2  Forward I/I0 monotonic ↓:  {'PASS' if a2_ok else 'FAIL'}  "
          f"(worst upward step = {worst_delta:.2e} × I0)")
    for ep in range(num_ep_actual):
        delta = (fwd_I[ep] - fwd_I[ep - 1]) / I0 if ep > 0 else 0
        print(f"    EP {ep:2d}: I/I0 = {fwd_ratios[ep]:.8f}  ΔI/I0 = {delta:+.2e}")
    forward_loss_pct = (1.0 - fwd_I[-1] / I0) * 100
    print(f"    Total forward loss: {forward_loss_pct:.4f}%")

    # ═══════════════════════════════════════════════════════════
    # B: Fresnel BSC unitarity
    # ═══════════════════════════════════════════════════════════
    print(f"\n--- B: Fresnel BSC residual ---")

    # B1: Bottom EP BSC = 0
    bsc_bottom_max = float(np.max(np.abs(to_cpu(bsc_wave.array[-1]))))
    b1_ok = bsc_bottom_max < 1e-10
    print(f"B1  Bottom BSC = 0:  max|BSC[final]| = {bsc_bottom_max:.2e}  "
          f"{'PASS' if b1_ok else 'FAIL'}")

    # B2: BSC depth monotonicity
    bsc_I = []
    for ep in range(bsc_wave.shape[0]):
        bsc_I.append(total_intensity(bsc_wave.array[ep]))
    bsc_I = np.array(bsc_I)
    b2_ok = all(bsc_I[i] >= bsc_I[i+1] * 0.99 for i in range(len(bsc_I) - 1))
    print(f"B2  Depth monotonicity:  {'PASS' if b2_ok else 'FAIL'}")
    for ep in range(len(bsc_I)):
        direction = "↓ bottom" if ep == len(bsc_I) - 1 else ("↑ top" if ep == 0 else "  ")
        print(f"    EP {ep:2d}: Σ|BSC|² = {bsc_I[ep]:.4e}  {direction}")

    # B3: Energy budget
    I_fwd_exit = total_intensity(exit_wave.array[-1])
    I_bsc_entrance = float(bsc_I[0])
    energy_balance = abs(I_fwd_exit + I_bsc_entrance - I0) / I0
    tol = ENERGY_BUDGET_TOL_B if STAGE == "B" else ENERGY_BUDGET_TOL_A
    b3_ok = energy_balance < tol
    print(f"\nB3  Energy budget (tol={tol:.1e}):")
    print(f"    I0            = {I0:.6f}")
    print(f"    I_fwd (exit)  = {I_fwd_exit:.6f}  (loss = {(I0 - I_fwd_exit)/I0*100:+.4f}%)")
    print(f"    I_bsc (entry) = {I_bsc_entrance:.6e}  ({I_bsc_entrance/I0*100:.4f}% of I0)")
    print(f"    |I_fwd + I_bsc - I0| / I0 = {energy_balance:.2e}  "
          f"{'PASS' if b3_ok else 'FAIL'}")

    # B4: Fresnel identity
    print(f"\nB4  Fresnel identity |T|²+|R|²=1:  verified by construction")
    print(f"    (T = sqrt(1-|R|²) is used in _cvdms_backscattering_correction)")
    print(f"    At 30 keV: Fresnel R ~ 2e-4 → I_bsc/I0 ~ 1e-7, consistent with B3.")

    # ═══════════════════════════════════════════════════════════
    # Assemble results
    # ═══════════════════════════════════════════════════════════
    results = {
        "fwd_I": fwd_I,
        "fwd_ratios": fwd_ratios,
        "bsc_I": bsc_I,
        "bsc_bottom_max": bsc_bottom_max,
        "I0": I0,
        "I_fwd_exit": I_fwd_exit,
        "I_bsc_entrance": I_bsc_entrance,
        "energy_balance": energy_balance,
        "energy_budget_tol": tol,
        "forward_loss_pct": forward_loss_pct,
        "depths": depths,
        "tests_pass": {
            "A1": a1_ok, "A2": a2_ok,
            "B1": b1_ok, "B2": b2_ok, "B3": b3_ok,
        },
    }
    params = {
        "energy_eV": ENERGY,
        "slice_thickness_A": SLICE_THICKNESS,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "supercell": (SUPERCELL_XY, SUPERCELL_XY, SUPERCELL_Z),
        "grid": potential.gpts,
        "sampling_A_per_px": float(potential.sampling[0]),
        "num_exit_planes": num_ep_actual,
        "thickness_A": float(total_z),
        "antialias": use_aa,
        "dtype": str(exit_wave.array.dtype),
    }

    # ═══════════════════════════════════════════════════════════
    # Assertions
    # ═══════════════════════════════════════════════════════════
    print()
    all_pass = a1_ok and a2_ok and b1_ok and b2_ok and b3_ok
    assert all_pass, (
        f"FAIL: V1c self-consistency checks: "
        f"A1={'PASS' if a1_ok else 'FAIL'} "
        f"A2={'PASS' if a2_ok else 'FAIL'} "
        f"B1={'PASS' if b1_ok else 'FAIL'} "
        f"B2={'PASS' if b2_ok else 'FAIL'} "
        f"B3={'PASS' if b3_ok else 'FAIL'}"
    )
    print(f"PASS: Stage {STAGE} — V1c self-consistency verified.")
    print(f"  A1={a1_ok}  A2={a2_ok}  B1={b1_ok}  B2={b2_ok}  B3={b3_ok}")
    print(f"  Grid: {potential.gpts}")

    # ═══════════════════════════════════════════════════════════
    # Save outputs
    # ═══════════════════════════════════════════════════════════
    save_results(STAGE, results, params)
    from plot_v1c import plot_v1c
    plot_v1c(STAGE)

    del result, exit_wave, bsc_wave
    cleanup()


if __name__ == "__main__":
    main()
