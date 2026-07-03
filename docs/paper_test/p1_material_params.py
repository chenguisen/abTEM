#!/usr/bin/env python3
"""P1: Material parameter tabulation for the (ρ,η) phase diagram.

Computes projected-potential characterisation for 9 material–voltage
combinations: SrTiO3 [001], Si [110], Au [001] at 30, 100, 300 keV.

Outputs:
  docs/data/p1_material_params.npz + .json
  docs/data/p1_material_params.tex  (Table 1 for manuscript)

Physics quantities:
  V_peak, V_rms     — peak and RMS projected potential (eV·Å)
  w_col             — column FWHM (Å)
  V_g               — Fourier coefficient at lowest g (eV·Å)
  λ                 — electron wavelength (Å)
  r_F               — Fresnel zone radius √(λ·Δz) (Å)
  ℓ_mfp             — scattering mean free path 1/(σ·V_rms) (Å)
  ξ_g               — extinction distance π/(σ·V_g) (Å)
  ρ                 — Δz / ℓ_mfp
  η                 — r_F / w_col
"""

import sys, os, json
import numpy as np
from ase.spacegroup import crystal
import abtem
from abtem.core import config as _cfg
from abtem.core.energy import energy2wavelength, energy2sigma

_cfg.set({"device": "gpu", "fft": "cupy"})
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ============================================================
# Parameter manifest
# ============================================================
SLICE_THICKNESS = 0.4  # Å — production Δz
SAMPLING = 0.02         # Å/px — fine enough for column profiling
SUPERCELL_XY = 4        # lateral supercell repetitions

MATERIALS = {
    "SrTiO3": {
        "symbols": ["Sr", "Ti", "O"],
        "basis": [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        "spacegroup": 221,
        "a": 3.905,
        "orientation": [0, 0, 1],
        "B": {"Sr": 0.5, "Ti": 0.4, "O": 0.4},  # isotropic DW factors (Å²)
    },
    "Si": {
        "symbols": ["Si"],
        "basis": [(0, 0, 0), (0.25, 0.25, 0.25)],
        "spacegroup": 227,
        "a": 5.431,
        "orientation": [1, 1, 0],
        "B": {"Si": 0.45},
    },
    "Au": {
        "symbols": ["Au"],
        "basis": [(0, 0, 0)],
        "spacegroup": 225,
        "a": 4.078,
        "orientation": [0, 0, 1],
        "B": {"Au": 0.6},
    },
}

VOLTAGES = [30e3, 100e3, 300e3]


# ============================================================
# Helpers
# ============================================================
def build_potential(material_key, sampling=None, use_dw=True):
    """Build projected potential for a material."""
    mat = MATERIALS[material_key]
    atoms = crystal(
        mat["symbols"], basis=mat["basis"],
        spacegroup=mat["spacegroup"],
        cellpar=[mat["a"]] * 3 + [90, 90, 90],
    )
    atoms *= (SUPERCELL_XY, SUPERCELL_XY, 1)

    kwargs = dict(
        sampling=sampling or SAMPLING,
        slice_thickness=mat["a"],  # one full unit cell → projected V in-plane
        exit_planes=1,
        projection="finite",
    )
    if use_dw and mat["B"]:
        # abtem builds frozen phonons; for static V^(B), use sigmas
        # We'll compute thermal smearing manually in reciprocal space
        pass

    potential = abtem.Potential(atoms, **kwargs)
    V_proj = potential.build(lazy=False)
    return potential, V_proj


def compute_column_width(V_2d, px_A):
    """FWHM of the strongest atomic column peak in the projected potential."""
    from scipy.ndimage import maximum_filter
    # Find the peak
    V_cpu = V_2d.get() if hasattr(V_2d, "get") else np.asarray(V_2d)
    peak_val = V_cpu.max()
    # Find pixels above half max
    half = peak_val / 2
    mask = V_cpu >= half
    # Approximate as circular: w_col = 2 * sqrt(area / π)
    area_px = mask.sum()
    area_A2 = area_px * px_A**2
    w_col = 2 * np.sqrt(area_A2 / np.pi)
    return w_col, peak_val


def compute_gradient_stats(V_2d, px_A):
    """Max |∇V| and ∇²V at the column peak."""
    V_cpu = V_2d.get() if hasattr(V_2d, "get") else np.asarray(V_2d)
    gy, gx = np.gradient(V_cpu, px_A)
    grad_mag = np.sqrt(gx**2 + gy**2)
    lap = np.gradient(gx, px_A, axis=1) + np.gradient(gy, px_A, axis=0)

    peak_idx = np.unravel_index(V_cpu.argmax(), V_cpu.shape)
    return float(grad_mag.max()), float(lap[peak_idx])


def compute_fourier_coeff(V_2d, px_A, g_hkl):
    """Compute V_g for a given reciprocal vector (approximate)."""
    V_cpu = V_2d.get() if hasattr(V_2d, "get") else np.asarray(V_2d)
    # Use FFT
    V_k = np.fft.fft2(V_cpu)
    ny, nx = V_cpu.shape
    # Reciprocal space coordinates in 1/Å
    dkx = 1.0 / (nx * px_A)
    dky = 1.0 / (ny * px_A)
    # Find pixel nearest to g_hkl
    # For simplicity, take the magnitude of the strongest Fourier peak
    V_k_abs = np.abs(V_k)
    # Zero-order
    V_000 = V_k_abs[0, 0] / (nx * ny)
    # Find first-order peak (exclude DC)
    V_k_abs[0, 0] = 0
    # The strongest non-DC peak corresponds to the lowest g
    idx = np.unravel_index(V_k_abs.argmax(), V_k_abs.shape)
    V_g_val = float(V_k_abs[idx] / (nx * ny))
    # g magnitude
    ky = idx[0] * dky if idx[0] <= ny // 2 else (idx[0] - ny) * dky
    kx = idx[1] * dkx if idx[1] <= nx // 2 else (idx[1] - nx) * dkx
    g_mag = np.sqrt(kx**2 + ky**2)
    return V_g_val, g_mag


# ============================================================
# Main
# ============================================================
def main():
    print("=== P1: Material Parameter Tabulation ===")
    results = []

    for mat_name in ["SrTiO3", "Si", "Au"]:
        print(f"\n--- {mat_name} ---")
        mat = MATERIALS[mat_name]

        # Build potential (no DW first)
        potential, V_data = build_potential(mat_name, use_dw=False)
        V_2d = V_data.array[0] if hasattr(V_data.array, "__len__") else V_data.array
        px_A = potential.sampling[0]

        # Column characterisation
        w_col, V_peak = compute_column_width(V_2d, px_A)
        V_cpu = V_2d.get() if hasattr(V_2d, "get") else np.asarray(V_2d)
        V_rms = float(np.std(V_cpu))
        grad_max, lap_at_col = compute_gradient_stats(V_2d, px_A)

        # Fourier characterisation — lowest two g
        V_g1, g1 = compute_fourier_coeff(V_2d, px_A, None)

        print(f"  V_peak = {V_peak:.2f} eV·Å, V_rms = {V_rms:.2f} eV·Å")
        print(f"  w_col = {w_col:.3f} Å")
        print(f"  |∇V|_max = {grad_max:.2e} eV, ∇²V|_col = {lap_at_col:.2e} eV/Å²")
        print(f"  V_g1 = {V_g1:.2f} eV·Å at g = {g1:.3f} Å⁻¹")

        for voltage in VOLTAGES:
            wavelength = energy2wavelength(voltage)
            sigma = energy2sigma(voltage)
            r_F = np.sqrt(wavelength * SLICE_THICKNESS)
            ell_mfp = 1.0 / (sigma * V_rms) if V_rms > 0 else np.inf
            xi_g = np.pi / (sigma * V_g1) if V_g1 > 0 else np.inf
            rho = SLICE_THICKNESS / ell_mfp
            eta = r_F / w_col

            entry = {
                "material": str(mat_name),
                "voltage_keV": float(voltage / 1000),
                "wavelength_A": float(wavelength),
                "sigma_eVA": float(sigma),
                "V_peak_eVA": float(V_peak),
                "V_rms_eVA": float(V_rms),
                "w_col_A": float(w_col),
                "grad_max_eV": float(grad_max),
                "lap_at_col_eV_A2": float(lap_at_col),
                "V_g1_eVA": float(V_g1),
                "g1_1A": float(g1),
                "r_F_A": float(r_F),
                "ell_mfp_A": float(ell_mfp),
                "xi_g_A": float(xi_g),
                "rho": float(rho),
                "eta": float(eta),
                "Delta_z_A": float(SLICE_THICKNESS),
            }
            results.append(entry)
            print(f"  {voltage/1000:.0f} keV: λ={wavelength:.4f} Å, r_F={r_F:.3f} Å, "
                  f"ℓ_mfp={ell_mfp:.1f} Å, ξ_g={xi_g:.1f} Å, ρ={rho:.3f}, η={eta:.3f}")

    # ── Save ──
    base = os.path.join(DATA_DIR, "p1_material_params")
    # Save structured arrays for NPZ
    keys = sorted(results[0].keys())
    arrays = {}
    for k in keys:
        vals = [r[k] for r in results]
        if isinstance(vals[0], str):
            arrays[k] = np.array(vals)
        else:
            arrays[k] = np.array(vals, dtype=np.float64)
    np.savez(base + ".npz", **arrays, materials=np.array([r["material"] for r in results]))
    with open(base + ".json", "w") as f:
        json.dump({"script": "p1_material_params.py",
                   "params": {"slice_thickness_A": SLICE_THICKNESS,
                              "sampling_A_per_px": SAMPLING,
                              "supercell_xy": SUPERCELL_XY},
                   "results": results}, f, indent=2)
    print(f"\nData saved: {base}.npz + .json")

    # ── Generate LaTeX table ──
    tex = _generate_latex_table(results)
    tex_path = os.path.join(DATA_DIR, "p1_material_params.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"LaTeX table saved: {tex_path}")


def _generate_latex_table(results):
    """Generate Table 1: Material parameters and control variables."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \caption{Material parameters and dimensionless control variables "
        r"for the three test systems at 30--300~keV. $\Delta z=0.4$~\AA\ throughout.}",
        r"  \label{tab:material_params}",
        r"  \small",
        r"  \begin{tabular}{lcccccccccc}",
        r"    \toprule",
        r"    Material & $E$ (keV) & $\lambda$ (\AA) & $V_\text{peak}$ (eV·\AA) & "
        r"$V_\text{rms}$ (eV·\AA) & $w_\text{col}$ (\AA) & "
        r"$\ell_\text{mfp}$ (\AA) & $\xi_g$ (\AA) & $\rho$ & $\eta$ \\",
        r"    \midrule",
    ]
    for r in results:
        lines.append(
            f"    {r['material']} & {r['voltage_keV']:.0f} & "
            f"{r['wavelength_A']:.4f} & {r['V_peak_eVA']:.1f} & "
            f"{r['V_rms_eVA']:.1f} & {r['w_col_A']:.2f} & "
            f"{r['ell_mfp_A']:.0f} & {r['xi_g_A']:.0f} & "
            f"{r['rho']:.3f} & {r['eta']:.3f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
