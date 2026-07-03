# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: py4dstem
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CVDMS HRTEM Simulation
#
# This notebook demonstrates HRTEM (high-resolution transmission electron microscopy)
# simulation using the CVDMS algorithm in abTEM, with backscattering (BSC) correction
# and frozen phonon averaging.
#
# ## Contents
# 1. Setup and crystal structure
# 2. Reference: Fourier multislice HRTEM
# 3. CVDMS HRTEM without backscattering
# 4. CVDMS HRTEM with backscattering (conj mode)
# 5. Frozen phonon HRTEM
# 6. CTF application and image formation
# 7. Backscattered wave analysis
# 8. Comparison and discussion

# %% [markdown]
# ## 1. Setup

# %%
import abtem
import ase
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from ase.spacegroup import crystal
from abtem import (
    PlaneWave, Potential, CTF, FrozenPhonons,
    Waves, show_atoms, standardize_cell,
)
from abtem.multislice import (
    CVDMSMultislice, FourierMultislice,
    multislice_and_detect,
)
from abtem.core import config as _cfg

# Device and FFT configuration
_cfg.set({"device": "gpu", "fft": "cupy"})

# GPU memory helper: call between heavy cells to prevent OOM in notebook
def _cleanup_gpu():
    import gc
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

# %% [markdown]
# ## 2. Crystal Structure
#
# We use SrTiO₃ (100) as a test specimen. A supercell is created for
# the HRTEM simulation.

# %%
# Build SrTiO₃ supercell
atoms = crystal(
    ["Sr", "Ti", "O"],
    basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
    spacegroup=221,
    cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
)
atoms *= (6, 6, 30)  # supercell for HRTEM

# Visualize the structure (use the supercell for display)
atoms_display = atoms.copy()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
show_atoms(atoms_display, plane="xy", ax=ax1, title="Beam view (plan view)")
show_atoms(atoms_display, plane="yz", ax=ax2, title="Side view (cross-section)")
fig.suptitle("SrTiO₃ Crystal Structure", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Potential Setup

# %%
# Note: BSC back-propagation requires at least 2 exit planes
# (entrance surface + final). Set exit_planes to an integer > 0.
# With exit_planes=50, exit planes are placed every ~20 Å.
# Larger spacing reduces GPU memory for thick specimens.
EXIT_PLANES = 30

# Potential with frozen phonon support
frozen_phonons = FrozenPhonons(atoms, num_configs=8, sigmas=0.085)
potential = Potential(
    frozen_phonons,
    sampling=0.05,         # Å/pixel
    slice_thickness=0.4,   # Å/slice
    exit_planes=EXIT_PLANES,
)

print(f"Number of frozen phonon configurations: {frozen_phonons.num_configs}")
print(f"Number of potential slices: {len([sl for sl in potential])}")
print(f"Number of exit planes: {potential.num_exit_planes}")
print(f"Total thickness: {len([sl for sl in potential]) * 0.4:.1f} Å")

# Also create a single-configuration potential for quick tests
single_potential = Potential(
    atoms,
    sampling=0.05,
    slice_thickness=0.4,
    exit_planes=EXIT_PLANES,
)

# %% [markdown]
# ## 4. Plane Wave Setup

# %%
plane_wave = PlaneWave(energy=30e3)  # 30 keV
print(f"Plane wave energy: {plane_wave.energy:.0f} eV")
print(f"Wavelength: {plane_wave.wavelength:.4f} Å")

# %% [markdown]
# ## 5. Reference: Fourier Multislice HRTEM
#
# First, compute a reference HRTEM exit wave using the standard
# Fourier multislice method, without backscattering.

# %%
print("Computing Fourier reference (no BSC)...")
exit_wave_fresnel = plane_wave.multislice(
    single_potential,
    algorithm=FourierMultislice(),
    lazy=False,
)
print(f"Exit wave shape: {exit_wave_fresnel.shape}")

# Use the final exit plane for display
final_ep = -1
print(f"Final exit plane slice: {exit_wave_fresnel.shape[0] + final_ep}")

# %%
_cleanup_gpu()

# %% [markdown]
# ## 6. CVDMS HRTEM Without Backscattering
#
# Next, compute the HRTEM exit wave using the CVDMS algorithm
# without backscattering correction for direct comparison.

# %%
print("Computing CVDMS reference (no BSC)...")
exit_wave_cvdms = plane_wave.multislice(
    single_potential,
    algorithm=CVDMSMultislice(
        convergence_threshold=1e-7,
    ),
    lazy=False,
)
print(f"Exit wave shape: {exit_wave_cvdms.shape}")

# %%
_cleanup_gpu()

# %% [markdown]
# ## 7. CVDMS HRTEM With Backscattering
#
# Compute the HRTEM exit wave with full CVDMS backscattering correction,
# using both the physically correct conj mode and the CGS-compatible forward mode.
#
# The backscattering correction subtracts the backscattered component from the
# forward wave at each slice interface, accounting for energy loss to
# backscattered electrons.

# %%
print("Computing CVDMS + BSC (conj mode)...")
result_bsc_conj = plane_wave.multislice(
    single_potential,
    algorithm=CVDMSMultislice(
        backscattering=True,
        calculate_backscattered=True,
        convergence_threshold=1e-7,
    ),
    return_backscattered=True,
    lazy=False,
)
exit_wave_bsc_conj = result_bsc_conj[0]     # forward exit wave
bsc_wave_conj = result_bsc_conj[-1]          # backscattered wave
print(f"Exit wave shape: {exit_wave_bsc_conj.shape}")
print(f"BSC wave shape: {bsc_wave_conj.shape}")
print(f"BSC max amplitude: {float(np.abs(bsc_wave_conj.array).max()):.4e}")

# %%
print("Computing CVDMS + BSC (forward mode)...")
result_bsc_fwd = plane_wave.multislice(
    single_potential,
    algorithm=CVDMSMultislice(
        backscattering=True,
        calculate_backscattered=True,
        convergence_threshold=1e-7,
    ),
    return_backscattered=True,
    lazy=False,
)
exit_wave_bsc_fwd = result_bsc_fwd[0]
bsc_wave_fwd = result_bsc_fwd[-1]
print(f"Exit wave shape: {exit_wave_bsc_fwd.shape}")
print(f"BSC wave shape: {bsc_wave_fwd.shape}")
print(f"BSC max amplitude: {float(np.abs(bsc_wave_fwd.array).max()):.4e}")

# %% [markdown]
# ## 8. Compare Exit Waves
#
# Compute the difference between the Fourier, CVDMS (no BSC), and CVDMS + BSC results.
# This shows the effect of the CVDMS algorithm and backscattering correction on
# the exit wave.

# %%
# Compute differences
diff_fresnel_cvdms = np.abs(exit_wave_fresnel.array - exit_wave_cvdms.array)
diff_cvdms_bsc = np.abs(exit_wave_cvdms.array - exit_wave_bsc_conj.array)

print("Exit wave comparisons:")
print(f"  |Fourier - CVDMS (no BSC)| max = {float(diff_fresnel_cvdms.max()):.4e}")
print(f"  |CVDMS (no BSC) - CVDMS + BSC| max = {float(diff_cvdms_bsc.max()):.4e}")

# Visualize exit wave intensity at the final exit plane
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

methods = [
    ("Fourier", exit_wave_fresnel),
    ("CVDMS (no BSC)", exit_wave_cvdms),
    ("CVDMS + BSC conj", exit_wave_bsc_conj),
    ("CVDMS + BSC fwd", exit_wave_bsc_fwd),
]

def get_cpu_intensity(wave):
    """Extract final exit plane intensity as CPU numpy array."""
    arr = wave.array[-1]  # final exit plane
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.abs(arr) ** 2

for (name, wave), ax in zip(methods, axes.ravel()):
    intensity = get_cpu_intensity(wave)
    im = ax.imshow(intensity, cmap="gray")
    ax.set_title(name)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("Exit Wave Intensity at Final Exit Plane", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. CTF Application and HRTEM Image Formation
#
# Apply the contrast transfer function (CTF) to the exit waves to form
# HRTEM images. We include spherical aberration (Cs), objective aperture,
# and partial coherence (focal spread).

# %%
# CTF parameters
Cs = -8e-6 * 1e10  # -8 μm in Å
Cc = 1.2e-3 * 1e10  # 1.2 mm chromatic aberration in Å
energy_spread = 0.35  # eV standard deviation
focal_spread = Cc * energy_spread / plane_wave.energy

# Create CTF
ctf = CTF(
    Cs=Cs,
    energy=plane_wave.energy,
    defocus="scherzer",
    semiangle_cutoff=25,  # mrad
    focal_spread=focal_spread,
)

print(f"Scherzer defocus: {ctf.defocus:.2f} Å")
print(f"Crossover angle: {ctf.crossover_angle:.2f} mrad")
print(f"Point resolution: {ctf.point_resolution:.2f} Å")
print(f"Focal spread: {focal_spread:.2f} Å")

def to_numpy(arr):
    """Convert CuPy array to CPU numpy if needed."""
    if hasattr(arr, 'get'):
        return arr.get()
    return arr

# Apply CTF and compute intensity for all methods
images = {}
for name, wave in methods:
    img = wave.apply_ctf(ctf).intensity()
    images[name] = to_numpy(img.array[-1])

# Visualize HRTEM images
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for (name, img_arr), ax in zip(images.items(), axes.ravel()):
    im = ax.imshow(img_arr, cmap="gray")
    ax.set_title(f"{name}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("HRTEM Images with CTF (Scherzer defocus)", fontsize=14)
plt.tight_layout()
plt.show()

# %%
_cleanup_gpu()

# %% [markdown]
# ## 10. Frozen Phonon HRTEM
#
# Now compute the HRTEM image with frozen phonon averaging, comparing
# three methods:
#   - **Fourier multislice** (reference, the conventional method)
#   - **CVDMS** (no backscattering)
#   - **CVDMS + BSC** (with backscattering correction)
#
# The frozen phonon ensemble averages over multiple atomic configurations,
# accounting for thermal diffuse scattering. This produces more realistic
# HRTEM images.

# %%
print("Computing frozen phonon HRTEM with Fourier multislice (reference)...")
exit_wave_fp_fourier = plane_wave.multislice(
    potential,
    algorithm=FourierMultislice(),
    lazy=False,
)
print(f"Fourier FP ensemble shape: {exit_wave_fp_fourier.shape}")

# FP average: CTF → intensity per config → mean (incoherent)
# Apply CTF to all exit planes, then select the last one for display
img_fp_fourier_ensemble = exit_wave_fp_fourier.apply_ctf(ctf).intensity().mean(0)
img_fp_fourier = to_numpy(img_fp_fourier_ensemble.array[-1])
amp_fp_fourier = float(np.abs(exit_wave_fp_fourier.array[:, -1, :, :]).mean())

# Fourier diffraction: |FFT(ψ_i)|² → mean
fp_fourier_last = exit_wave_fp_fourier._array[:, -1]  # (num_configs, H, W) on GPU
dp_fourier = cp.fft.fftshift(cp.abs(cp.fft.fft2(fp_fourier_last)) ** 2, axes=(1, 2))
dp_fourier_mean = dp_fourier.mean(0)

# Move Fourier FP GPU data to CPU and free
dp_fourier_mean_cpu = to_numpy(dp_fourier_mean)
del exit_wave_fp_fourier, fp_fourier_last, dp_fourier
_cleanup_gpu()

print(f"Fourier FP exit amplitude (last plane): {amp_fp_fourier:.6f}")
print(f"Fourier FP diffraction shape: {dp_fourier_mean.shape}")

# %%
print("Computing frozen phonon HRTEM with CVDMS (no BSC)...")
exit_wave_fp = plane_wave.multislice(
    potential,
    algorithm=CVDMSMultislice(
        convergence_threshold=1e-7,
    ),
    lazy=False,
)
print(f"FP ensemble shape: {exit_wave_fp.shape}")

# Correct frozen phonon HRTEM: apply CTF + intensity per config, then average
# intensities (incoherent sum). This accounts for thermal diffuse scattering.
#   WRONG: exit_wave.mean(0) → CTF → |·|²  (coherent avg of exit waves)
#   RIGHT: for each config ψ_i → CTF → |ψ_i * CTF|² → mean over i (incoherent avg)
img_fp_ensemble = exit_wave_fp.apply_ctf(ctf).intensity().mean(0)
img_fp = to_numpy(img_fp_ensemble.array[-1])

# Keep exit wave mean for BSC reference analysis only (coherent average)
exit_wave_fp_mean = exit_wave_fp.mean(0)
print(f"Mean exit wave shape: {exit_wave_fp_mean.shape}")

# Compute FP (no BSC) diffraction before freeing GPU memory
# Incoherent average: |FFT(ψ_i)|² → mean
print("Computing FP (no BSC) diffraction patterns...")
fp_exit_last = exit_wave_fp._array[:, -1]  # (num_configs, H, W) on GPU
diffraction_patterns = cp.abs(cp.fft.fft2(fp_exit_last)) ** 2
diffraction_patterns = cp.fft.fftshift(diffraction_patterns, axes=(1, 2))
diffraction_pattern_mean = diffraction_patterns.mean(0)

sampling_val = float(potential.sampling[0])
ny, nx = diffraction_pattern_mean.shape
kx = cp.fft.fftshift(cp.fft.fftfreq(nx, d=sampling_val))
ky = cp.fft.fftshift(cp.fft.fftfreq(ny, d=sampling_val))
k_max = float(cp.sqrt(kx**2 + ky[:, None]**2).max())
k_max_np = k_max  # for log scale plots

# Move results to CPU to free GPU memory
diffraction_pattern_mean_np = float(cp.sum(diffraction_pattern_mean))
diffraction_patterns_cpu = to_numpy(diffraction_patterns)
diffraction_pattern_mean_cpu = to_numpy(diffraction_pattern_mean)

print(f"Diffraction pattern shape: {diffraction_pattern_mean.shape}")
print(f"Max spatial frequency: {k_max:.4f} Å⁻¹")

# Free large GPU arrays before FP+BSC section
print("Freeing GPU memory before FP+BSC...")
# Save exit wave stats before deletion for later comparison
amp_fp_cvdms = float(cp.abs(exit_wave_fp._array[:, -1, :, :]).mean())
del exit_wave_fp, fp_exit_last, diffraction_patterns, diffraction_pattern_mean
del exit_wave_fp_mean, kx, ky
_cleanup_gpu()

# %%
print("Computing frozen phonon HRTEM with CVDMS + BSC...")
#
# NOTE: We use return_backscattered=False here because the per-slice BSC
# accumulation stores ALL configs × ALL slices simultaneously on GPU.
# For 16 configs × 489 slices × 625² grid = ~24 GB, exceeding GPU memory.
# The BSC correction is still applied to the forward exit waves during
# propagation; we simply skip storing the backscattered wave ensemble.
# Single-config BSC analysis (above) provides the BSC wave data.
exit_wave_fp_bsc = plane_wave.multislice(
    potential,
    algorithm=CVDMSMultislice(
        backscattering=True,
        calculate_backscattered=True,
        convergence_threshold=1e-7,
    ),
    return_backscattered=False,  # skip per-slice BSC storage to save GPU memory
    lazy=False,
)
print(f"FP+BSC exit shape: {exit_wave_fp_bsc.shape}")

# Correct HRTM: intensity per config → average → select last exit plane
img_fp_bsc_ensemble = exit_wave_fp_bsc.apply_ctf(ctf).intensity().mean(0)
img_fp_bsc = to_numpy(img_fp_bsc_ensemble.array[-1])

# Keep exit wave mean for reference
exit_wave_fp_bsc_mean = exit_wave_fp_bsc.mean(0)

# %%
_cleanup_gpu()

# %% [markdown]
# ## 11. Frozen Phonon Coherent Diffraction
#
# Compute the parallel-beam coherent diffraction pattern from each frozen phonon
# configuration. The diffraction pattern is the squared amplitude of the Fourier
# transform of the exit wave:
#
# $$I_{\text{diff}}(\mathbf{k}) = |\mathcal{FT}[\psi_{\text{exit}}](\mathbf{k})|^2$$
#
# For frozen phonon averaging, we average the **intensities** (incoherent sum),
# which accounts for thermal diffuse scattering (TDS).
#
# The no-BSC diffraction was pre-computed before the FP+BSC section to manage
# GPU memory. Here we compute only the BSC diffraction from the new result.

# %%
# Compute BSC diffraction from exit_wave_fp_bsc
print("Computing FP + BSC diffraction patterns...")
fp_bsc_exit_last = exit_wave_fp_bsc._array[:, -1]
dp_bsc = cp.abs(cp.fft.fft2(fp_bsc_exit_last)) ** 2
dp_bsc = cp.fft.fftshift(dp_bsc, axes=(1, 2))
dp_bsc_mean = dp_bsc.mean(0)
dp_bsc_mean_sum = float(cp.sum(dp_bsc_mean))
print(f"BSC diffraction shape: {dp_bsc_mean.shape}")

# %%
# Display individual configuration diffraction patterns and ensemble average
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Show individual no-BSC config patterns (from pre-computed CPU data)
num_show = min(4, frozen_phonons.num_configs)
for i in range(num_show):
    dp_i = np.log10(diffraction_patterns_cpu[i] + 1)
    axes[0, i].imshow(dp_i, cmap="gray", extent=[-k_max_np, k_max_np, -k_max_np, k_max_np])
    axes[0, i].set_title(f"Config {i} (CVDMS)")
    axes[0, i].set_xlabel("k (Å⁻¹)")
    axes[0, i].set_ylabel("k (Å⁻¹)")

# Ensemble averaged patterns
im4 = axes[1, 0].imshow(np.log10(diffraction_pattern_mean_cpu + 1), cmap="gray",
                         extent=[-k_max_np, k_max_np, -k_max_np, k_max_np])
axes[1, 0].set_title(f"FP avg ({frozen_phonons.num_configs} configs)\nCVDMS (no BSC)")
axes[1, 0].set_xlabel("k (Å⁻¹)")
axes[1, 0].set_ylabel("k (Å⁻¹)")
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

dp_bsc_mean_arr = to_numpy(cp.log10(dp_bsc_mean + 1))
im5 = axes[1, 1].imshow(dp_bsc_mean_arr, cmap="gray",
                         extent=[-k_max_np, k_max_np, -k_max_np, k_max_np])
axes[1, 1].set_title(f"FP avg ({frozen_phonons.num_configs} configs)\nCVDMS + BSC")
axes[1, 1].set_xlabel("k (Å⁻¹)")
axes[1, 1].set_ylabel("k (Å⁻¹)")
plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

# Difference between no-BSC and BSC diffraction patterns
dp_diff = np.log10(diffraction_pattern_mean_cpu + 1) - to_numpy(cp.log10(dp_bsc_mean + 1))
im6 = axes[1, 2].imshow(dp_diff, cmap="RdBu", extent=[-k_max_np, k_max_np, -k_max_np, k_max_np])
axes[1, 2].set_title("Log diff: no BSC vs +BSC")
axes[1, 2].set_xlabel("k (Å⁻¹)")
axes[1, 2].set_ylabel("k (Å⁻¹)")
plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

# Radial profile comparison (including Fourier reference)
ny, nx = diffraction_pattern_mean_cpu.shape
center_y, center_x = ny // 2, nx // 2
radial_bins = np.arange(0, min(ny, nx) // 2, 1)
radial_profile_fourier = np.zeros(len(radial_bins))
radial_profile_nobsc = np.zeros(len(radial_bins))
radial_profile_bsc = np.zeros(len(radial_bins))
y, x = np.ogrid[:ny, :nx]
for i, r in enumerate(radial_bins):
    outer = ((y - center_y) ** 2 + (x - center_x) ** 2) <= (r + 1) ** 2
    inner = ((y - center_y) ** 2 + (x - center_x) ** 2) <= r ** 2
    mask = outer & ~inner
    radial_profile_fourier[i] = float(np.mean(dp_fourier_mean_cpu[mask]))
    radial_profile_nobsc[i] = float(np.mean(diffraction_pattern_mean_cpu[mask]))
    radial_profile_bsc[i] = float(to_numpy(cp.mean(dp_bsc_mean[mask])))

ax7 = axes[1, 3]
ax7.semilogy(radial_bins * sampling_val, radial_profile_fourier,
             label="Fourier", alpha=0.8, linestyle="--", color="black")
ax7.semilogy(radial_bins * sampling_val, radial_profile_nobsc,
             label="CVDMS (no BSC)", alpha=0.8)
ax7.semilogy(radial_bins * sampling_val, radial_profile_bsc,
             label="CVDMS + BSC", alpha=0.8)
ax7.set_xlabel("k (Å⁻¹)")
ax7.set_ylabel("Mean intensity")
ax7.set_title("Radial profile (diffraction)")
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

fig.suptitle("Frozen Phonon Coherent Diffraction Patterns", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Quantitative comparison
print("Diffraction pattern comparison (FP averaged):")
print(f"  Fourier total intensity:               {float(dp_fourier_mean_cpu.sum()):.4e}")
print(f"  CVDMS (no BSC) total intensity:         {diffraction_pattern_mean_np:.4e}")
print(f"  CVDMS + BSC total intensity:            {dp_bsc_mean_sum:.4e}")

# FP-HRTEM image comparison
print()
print("=== FP-Averaged HRTEM Image Comparison ===")
img_fp_np = img_fp
img_fp_bsc_np = img_fp_bsc
img_fp_fourier_np = img_fp_fourier

print(f"Fourier:   mean={img_fp_fourier_np.mean():.6f}  std={img_fp_fourier_np.std():.6f}")
print(f"CVDMS:     mean={img_fp_np.mean():.6f}  std={img_fp_np.std():.6f}")
print(f"CVDMS+BSC: mean={img_fp_bsc_np.mean():.6f}  std={img_fp_bsc_np.std():.6f}")
ncc_fc = np.corrcoef(img_fp_fourier_np.ravel(), img_fp_np.ravel())[0, 1]
ncc_fb = np.corrcoef(img_fp_fourier_np.ravel(), img_fp_bsc_np.ravel())[0, 1]
ncc_cb = np.corrcoef(img_fp_np.ravel(), img_fp_bsc_np.ravel())[0, 1]
print(f"NCC(Fourier, CVDMS):     {ncc_fc:.6f}")
print(f"NCC(Fourier, CVDMS+BSC): {ncc_fb:.6f}")
print(f"NCC(CVDMS, CVDMS+BSC):   {ncc_cb:.6f}")

# FP exit amplitude comparison
# amp_fp_fourier and amp_fp_cvdms were saved before GPU cleanup above
amp_fp_bsc_val = float(cp.abs(exit_wave_fp_bsc._array[:, -1, :, :]).mean())
print()
print("FP exit amplitude (last plane):")
print(f"  Fourier:   {amp_fp_fourier:.6f}")
print(f"  CVDMS:     {amp_fp_cvdms:.6f}")
print(f"  CVDMS+BSC: {amp_fp_bsc_val:.6f}")

# %%
# Visualize FP-averaged HRTEM image comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
methods_fp = [
    ("Fourier", img_fp_fourier_np),
    ("CVDMS (no BSC)", img_fp_np),
    ("CVDMS + BSC", img_fp_bsc_np),
]
for (name, img_arr), ax in zip(methods_fp, axes):
    im = ax.imshow(img_arr, cmap="gray")
    ax.set_title(f"{name}\nmean={img_arr.mean():.3f}  std={img_arr.std():.3f}", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("FP-Averaged HRTEM Images with CTF (Scherzer defocus)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Backscattering Diffraction (Single Config)
#
# Compute the diffraction pattern of the **backscattered wave** at the entrance
# surface (exit plane 0), using the single-configuration CVDMS+BSC result.
# The frozen phonon BSC ensemble is not available due to GPU memory constraints:
# the per-slice BSC storage for 16 configs × 489 slices would require ~24 GB.
#
# The backscattered diffraction is compared with the FP-averaged forward
# diffraction to quantify the BSC loss channel for each scattering angle.

# %%
print("Computing BSC diffraction (single config, conj mode)...")

# Entrance surface BSC: (H, W) single config
bsc_ep0 = bsc_wave_conj.array[0]

# |FFT|²
bsc_diffraction = cp.abs(cp.fft.fft2(bsc_ep0)) ** 2
bsc_diffraction = cp.fft.fftshift(bsc_diffraction)
bsc_diffraction_mean = bsc_diffraction  # single config, no averaging needed

print(f"BSC diffraction shape: {bsc_diffraction_mean.shape}")

# Ratio map: BSC / forward diffraction (use saved CPU array)
eps = 1e-30  # avoid division by zero
ratio_map = np.log10(to_numpy(bsc_diffraction_mean) + eps) - np.log10(diffraction_pattern_mean_cpu + eps)

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# BSC diffraction (log scale)
bsc_dp_log = to_numpy(cp.log10(bsc_diffraction_mean + 1))
im0 = axes[0].imshow(bsc_dp_log, cmap="inferno",
                     extent=[-k_max_np, k_max_np, -k_max_np, k_max_np])
axes[0].set_title("BSC diffraction (single config)\n(entrance surface)")
axes[0].set_xlabel("k (Å⁻¹)")
axes[0].set_ylabel("k (Å⁻¹)")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Forward diffraction (log scale) for comparison (from pre-computed CPU)
dp_fwd_log = np.log10(diffraction_pattern_mean_cpu + 1)
im1 = axes[1].imshow(dp_fwd_log, cmap="inferno",
                     extent=[-k_max_np, k_max_np, -k_max_np, k_max_np])
axes[1].set_title(f"Forward diffraction (FP avg)\n({frozen_phonons.num_configs} configs)")
axes[1].set_xlabel("k (Å⁻¹)")
axes[1].set_ylabel("k (Å⁻¹)")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Ratio: log10(BSC / forward)
vlim = max(abs(ratio_map.min()), abs(ratio_map.max()))
im2 = axes[2].imshow(ratio_map, cmap="RdBu", vmin=-vlim, vmax=vlim,
                     extent=[-k_max_np, k_max_np, -k_max_np, k_max_np])
axes[2].set_title("log₁₀(BSC / forward)")
axes[2].set_xlabel("k (Å⁻¹)")
axes[2].set_ylabel("k (Å⁻¹)")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

# Radial profiles (use CPU numpy for forward data)
hp_x, hp_y = diffraction_pattern_mean_cpu.shape[1] // 2, diffraction_pattern_mean_cpu.shape[0] // 2
radial_bins = np.arange(0, min(diffraction_pattern_mean_cpu.shape) // 2, 1)
radial_fwd = np.zeros(len(radial_bins))
radial_bsc = np.zeros(len(radial_bins))
bsc_diff_mean_cpu = to_numpy(bsc_diffraction_mean)
for i, r in enumerate(radial_bins):
    y, x = np.ogrid[:diffraction_pattern_mean_cpu.shape[0], :diffraction_pattern_mean_cpu.shape[1]]
    outer = ((y - hp_y) ** 2 + (x - hp_x) ** 2) <= (r + 1) ** 2
    inner = ((y - hp_y) ** 2 + (x - hp_x) ** 2) <= r ** 2
    mask = outer & ~inner
    radial_fwd[i] = float(np.mean(diffraction_pattern_mean_cpu[mask]))
    radial_bsc[i] = float(np.mean(bsc_diff_mean_cpu[mask]))

ax3 = axes[3]
ax3.semilogy(radial_bins * sampling_val, radial_fwd, label="Forward (FP avg)", alpha=0.8)
ax3.semilogy(radial_bins * sampling_val, radial_bsc, label="BSC (single config)", alpha=0.8)
ax3.set_xlabel("k (Å⁻¹)")
ax3.set_ylabel("Mean intensity")
ax3.set_title("Radial profile")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

fig.suptitle("Backscattering Diffraction vs Forward Diffraction", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Quantitative comparison
print("BSC vs forward diffraction:")
print(f"  Forward total intensity (FP avg, {frozen_phonons.num_configs} configs): {diffraction_pattern_mean_np:.4e}")
print(f"  BSC total intensity (single config): {float(cp.sum(bsc_diffraction_mean)):.4e}")
bsc_ratio = float(cp.sum(bsc_diffraction_mean) / diffraction_pattern_mean_np)
print(f"  BSC / Forward ratio:     {bsc_ratio:.6f}  ({bsc_ratio*100:.3f}%)")

# %% [markdown]
# ## 14. Compare Frozen Phonon Results (HRTEM)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Single config CVDMS
im0 = axes[0].imshow(images["CVDMS (no BSC)"], cmap="gray")
axes[0].set_title("Single config\nCVDMS (no BSC)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# FP averaged CVDMS
img_fp_arr = img_fp
im1 = axes[1].imshow(img_fp_arr, cmap="gray")
axes[1].set_title(f"FP averaged ({frozen_phonons.num_configs} configs)\nCVDMS (no BSC)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# FP averaged CVDMS + BSC
img_fp_bsc_arr = img_fp_bsc
im2 = axes[2].imshow(img_fp_bsc_arr, cmap="gray")
axes[2].set_title(f"FP averaged ({frozen_phonons.num_configs} configs)\nCVDMS + BSC")
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

fig.suptitle("HRTEM Images: Single Config vs Frozen Phonon Average", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 15. Backscattered Wave Analysis
#
# The backscattered wave at the entrance surface provides information about
# electrons that were backscattered within the specimen. For HRTEM, this
# represents a loss channel that reduces the forward wave intensity.

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 12))

# BSC amplitude and phase for conj mode
bsc_amp_conj = to_numpy(np.abs(bsc_wave_conj.array[0]))  # EP 0 = entrance surface
bsc_phase_conj = to_numpy(np.angle(bsc_wave_conj.array[0]))

im1 = axes[0, 0].imshow(bsc_amp_conj, cmap="viridis")
axes[0, 0].set_title("BSC Amplitude (conj mode)")
axes[0, 0].axis("off")
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

im2 = axes[0, 1].imshow(bsc_phase_conj, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
axes[0, 1].set_title("BSC Phase (conj mode)")
axes[0, 1].axis("off")
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

# BSC Fourier spectrum
bsc_fft_conj = np.fft.fftshift(np.fft.fft2(bsc_amp_conj))
im3 = axes[0, 2].imshow(np.log10(np.abs(bsc_fft_conj) + 1), cmap="gray")
axes[0, 2].set_title("BSC Log-power spectrum (conj)")
axes[0, 2].axis("off")
plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

# BSC at different exit planes (single config conj mode)
bsc_mid_ep = to_numpy(np.abs(bsc_wave_conj.array[len(bsc_wave_conj) // 2]))
bsc_last_ep = to_numpy(np.abs(bsc_wave_conj.array[-1]))

im4 = axes[1, 0].imshow(bsc_amp_conj, cmap="viridis")
axes[1, 0].set_title("BSC Amplitude at EP 0\n(entrance surface, conj)")
axes[1, 0].axis("off")
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

im5 = axes[1, 1].imshow(bsc_mid_ep, cmap="viridis")
axes[1, 1].set_title(f"BSC Amplitude at EP {len(bsc_wave_conj) // 2}\n(mid-depth, conj)")
axes[1, 1].axis("off")
plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

im6 = axes[1, 2].imshow(bsc_last_ep, cmap="viridis")
axes[1, 2].set_title("BSC Amplitude at final EP\n(bottom surface, conj)")
axes[1, 2].axis("off")
plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
axes[1, 2].axis("off")
plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

fig.suptitle("Backscattered Wave Analysis at Entrance Surface", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 16. BSC Depth Profile
#
# The `return_backscattered=True` option returns the backscattered wave at
# each exit plane. With default `exit_planes` (only final exit plane), we get
# the total backscattered wave at the entrance surface.
# For a depth-resolved analysis, set `exit_planes` to an integer > 0.

# %%
# Depth profile: |BSC| at each exit plane (from the existing BSC result)
# EP 0 = entrance surface, EP -1 = final exit plane
depth_values = []
for ep_idx in range(len(bsc_wave_conj)):
    depth_values.append(float(to_numpy(cp.abs(bsc_wave_conj._array[ep_idx])).max()))

print("BSC depth profile (max amplitude per exit plane):")
for ep_idx, val in enumerate(depth_values):
    print(f"  EP {ep_idx}: {val:.4e}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(depth_values)), depth_values, "o-", linewidth=2, markersize=8)
ax.set_xlabel("Exit Plane Index (0 = entrance surface)")
ax.set_ylabel("Max |BSC| amplitude")
ax.set_title("BSC Depth Profile (CVDMS + BSC, conj mode)")
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 17. Comparison: Exit Wave Profiles
#
# Extract line profiles through the HRTEM images for quantitative comparison.

# %%
# Cut line profiles through center of images
def line_profile(image_array, axis=0):
    """Extract line profile through center of image."""
    ny, nx = image_array.shape[-2:]
    if axis == 0:  # horizontal line through center
        return image_array[ny // 2, :]
    else:  # vertical line through center
        return image_array[:, nx // 2]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Collect images to compare
compare_images = {
    "Fourier": images["Fourier"],
    "CVDMS": images["CVDMS (no BSC)"],
    "CVDMS+BSC_conj": images["CVDMS + BSC conj"],
}

sampling = single_potential.sampling[0]
num_pixels = compare_images["CVDMS"].shape[-1]
x_axis = np.arange(num_pixels) * sampling

for name, img_arr in compare_images.items():
    profile = line_profile(img_arr, axis=0)
    axes[0].plot(x_axis, profile, label=name, linewidth=1.5)

axes[0].set_xlabel("Position (Å)")
axes[0].set_ylabel("Intensity")
axes[0].set_title("Horizontal Line Profile (through center)")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for name, img_arr in compare_images.items():
    profile = line_profile(img_arr, axis=1)
    axes[1].plot(x_axis, profile, label=name, linewidth=1.5)

axes[1].set_xlabel("Position (Å)")
axes[1].set_ylabel("Intensity")
axes[1].set_title("Vertical Line Profile (through center)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 18. Summary
#
# This notebook demonstrated:
#
# 1. **HRTEM with Fourier multislice** — standard Fourier method reference
# 2. **HRTEM with CVDMS (no BSC)** — real-space Taylor expansion without backscattering
# 3. **HRTEM with CVDMS + BSC** — full coupled-wave theory with backscattering correction
# 4. **Frozen phonon averaging** — thermal diffuse scattering via ensemble averaging
# 5. **CTF application** — aberration correction, aperture, partial coherence
# 6. **Backscattered wave analysis** — depth profile, amplitude/phase distribution
#
# ### Key parameters:
#
# | Parameter | Value |
# |-----------|-------|
# | Material | SrTiO₃ (100) |
# | Voltage | 30 keV |
# | Supercell | 8 × 8 × 50 |
# | Sampling | 0.05 Å/pixel |
# | Slice thickness | 0.4 Å |
# | Cs | -8 μm |
# | Defocus | Scherzer |
# | Frozen phonon configs | 16 |
# | Exit plane spacing | 50 slices (~20 Å) |
#
# ### Notes:
# - The backscattering correction produces a small but measurable difference in
#   the HRTEM image intensity distribution
# - Frozen phonon averaging smooths out high-frequency noise and produces more
#   realistic images
# - The conj mode provides physically correct time-reversed backward propagation
# - The forward mode matches ImageSimulation_CGS results for cross-validation
