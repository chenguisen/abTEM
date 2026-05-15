# Software Implementations

These papers describe the major open-source and GPU-accelerated multislice
packages used for comparison and validation in the CVDMS paper.

## Key references

### abTEM (Madsen & Susi 2021)
**Citation key:** `MadsenSusi2021`

```
J. Madsen, T. Susi,
"The abTEM code: transmission electron microscopy from first principles,"
Open Research Europe 1, 24 (2021).
DOI: 10.12688/openreseurope.13015.1
```

**Significance:** Python-based open-source multislice and PRISM code. Provides
the Fourier multislice reference baseline for CVDMS validation. The CVDMS
implementation is integrated into an abTEM research fork.

Related: Brown et al. (2020) "A Python Based Open-source Multislice Simulation
Package for Transmission Electron Microscopy." *Microscopy and Microanalysis*
26(S2), 2954–2956. DOI: 10.1017/S1431927620023922 — introduces the abTEM
ecosystem.

---

### MULTEM (Lobato & Van Dyck 2015)
**Citation key:** `LobatoVanDyck2015`

```
I. Lobato, D. Van Dyck,
"MULTEM: a new multislice program to perform accurate and fast
 electron diffraction and imaging simulations using Graphics
 Processing Units with CUDA,"
Ultramicroscopy 156, 9–17 (2015).
DOI: 10.1016/j.ultramic.2015.04.016
```

**Significance:** GPU-accelerated multislice code with higher-order expansion
of the multislice solution. Implements accurate Fresnel propagation with
correct subslicing. The higher-order expansion effectively includes commutator
corrections beyond standard first-order splitting.

---

### PRISM (Ophus 2017)
**Citation key:** `Ophus2017`

```
C. Ophus,
"A fast image simulation algorithm for scanning transmission
 electron microscopy,"
Advanced Structural and Chemical Imaging 3, 13 (2017).
DOI: 10.1186/s40679-017-0046-1
```

**Significance:** The PRISM (Plane-wave Reciprocal-space Interpolated
Scattering Matrix) algorithm. Achieves $\sim f^4$ speedup via Fourier
interpolation of probe wavefunctions. Widely used for 4D-STEM simulation.
Provides the primary STEM simulation reference for the field.

Related: Pryor, Ophus & Miao (2017) "A streaming multi-GPU implementation..."
*Adv. Struct. Chem. Imaging* 3, 15. DOI: 10.1186/s40679-017-0048-z — Prismatic
GPU package.

---

### Ophus (2019) — 4D-STEM review
**Citation key:** `Ophus2019`

```
C. Ophus,
"Four-Dimensional Scanning Transmission Electron Microscopy (4D-STEM):
 From scanning nanodiffraction to ptychography and beyond,"
Microscopy and Microanalysis 25, 563–582 (2019).
DOI: 10.1017/S1431927619000497
```

**Significance:** Comprehensive review article covering the full 4D-STEM
workflow. Establishes the context for why fast and accurate simulation matters
in modern electron microscopy.

---

### py4DSTEM (Savitzky et al. 2021)
**Citation key:** `Savitzky2021`

```
B. H. Savitzky et al.,
"py4DSTEM: a software package for four-dimensional scanning
 transmission electron microscopy data analysis,"
Microscopy and Microanalysis 27, 712–743 (2021).
DOI: 10.1017/S1431927621000477
```

**Significance:** Open-source Python package for 4D-STEM data analysis.
Provides the experimental data-analysis ecosystem that CVDMS diagnostics
interface with (convergence maps as 4D-STEM-compatible outputs).

---

### Pelz et al. (2021) — Partitioned PRISM
**Citation key:** `Pelz2021`

```
P. M. Pelz et al.,
"A fast algorithm for scanning transmission electron microscopy
 imaging and 4D-STEM diffraction simulations,"
Microscopy and Microanalysis 27, 835–848 (2021).
DOI: 10.1017/S1431927621012083
```

**Significance:** "Partitioned PRISM" with beamlet interpolation for further
speed and memory improvements. Represents the state of the art in fast STEM
simulation.
