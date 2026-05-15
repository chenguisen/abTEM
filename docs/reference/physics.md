# Scattering Physics

These papers cover the physical models underpinning the CVDMS diagnostic
framework: projected potential, phonon scattering, axial channeling, and
HOLZ effects.

## Key references

### Lobato & Van Dyck (2014) — Scattering factor parameterization
**Citation key:** `LobatoVanDyck2014`

```
I. Lobato, D. Van Dyck,
"An accurate parameterization for scattering factors, electron densities
 and electrostatic potentials for neutral atoms that obey all physical
 constraints,"
Acta Crystallographica A 70, 636–649 (2014).
DOI: 10.1107/S205327331401643X
```

**Significance:** Provides the electron form-factor parameterization used for
projected potential computation throughout the CVDMS paper. The form factor
satisfies $f_e(q) \propto q^{-2}$ at large $q$, giving rise to the bandlimit
heredity effect discussed in §4.7 and §6.10. Used for both $V^{(0)}$ (cold)
and $V^{(B)}$ (Debye–Waller smoothed) potentials.

---

### Forbes et al. (2010) — Quantum phonon model
**Citation key:** `Forbes2010`

```
B. D. Forbes, A. V. Martin, S. D. Findlay, A. J. D'Alfonso, L. J. Allen,
"Quantum mechanical model for phonon excitation in electron diffraction
 and imaging using a Born–Oppenheimer approximation,"
Physical Review B 82, 104103 (2010).
DOI: 10.1103/PhysRevB.82.104103
```

**Significance:** Rigorous quantum-mechanical treatment of phonon excitation in
electron diffraction. Justifies the frozen-phonon approach used for thermal
diffuse scattering in multislice. The Debye–Waller factors from this framework
are applied in §6.10 for thermal smoothing $V^{(B)}$.

---

### Allen et al. (2003) — Channeling theory
**Citation key:** `Allen2003`

```
L. J. Allen, S. D. Findlay, M. P. Oxley, C. J. Rossouw,
"Lattice-resolution contrast from a focused coherent electron probe.
 Part I,"
Ultramicroscopy 96, 47–63 (2003).
DOI: 10.1016/S0304-3991(02)00380-7
```

**Significance:** Formulates the channeling-state theory underpinning the
commutator-sensitive observable analysis of §4.5. Describes the relationship
between axial channeling Pendellösung, HOLZ Bloch-state coupling, and the
$s$-state model. Provides the theoretical framework for interpreting the
omitted-commutator signatures in CVDMS diagnostics.

---

### Pennycook & Nellist (2011) — STEM textbook, Ch. 2 (Channeling)
**Citation key:** `PennycookNellist2011`

```
S. J. Pennycook, P. D. Nellist (eds.),
"Scanning Transmission Electron Microscopy: Imaging and Analysis,"
Ch. 2, Springer (2011).
```

**Significance:** Chapter 2 covers atomic-column channeling, the $s$-state
model, and Pendellösung in STEM. Used as the reference for extinction-distance
values $\xi_\text{ch}$ in §4.5 channeling comparisons.

---

### Brown, Ciston & Ophus (2019) — Inelastic transitions
**Citation key:** `BrownCistonOphus2019`

```
H. G. Brown, J. Ciston, C. Ophus,
"Linear-scaling algorithm for rapid computation of inelastic transitions
 in the presence of multiple electron scattering,"
Physical Review Research 1, 033186 (2019).
DOI: 10.1103/PhysRevResearch.1.033186
```

**Significance:** Extends multislice-based simulation to inelastic EELS
core-loss with linear scaling. Relevant to future coupling of CVDMS elastic
foundation to inelastic channels (§5.5).
