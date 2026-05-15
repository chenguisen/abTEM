# Numerical Methods

These papers cover operator-splitting theory, frozen-phonon methodology,
Fresnel/Fourier optics, and numerical analysis relevant to the CVDMS algorithm.

## Key references

### Operator splitting

#### Trotter (1959) — Product of semigroups
**Citation key:** `Trotter1959`

```
H. F. Trotter,
"On the product of semi-groups of operators,"
Proceedings of the American Mathematical Society 10, 545–551 (1959).
DOI: 10.1090/S0002-9939-1959-0108732-6
```

**Significance:** The foundational theorem proving that $\exp(t(A+B)) =
\lim_{n\to\infty}[\exp(tA/n)\exp(tB/n)]^n$ for semigroups of operators.
The Lie–Trotter splitting that conventional multislice uses traces directly
to this result. CVDMS avoids this splitting by directly exponentiating
$\hat{K} = A+B$ as a single operator.

---

#### Strang (1968) — Symmetric splitting
**Citation key:** `Strang1968`

```
G. Strang,
"On the construction and comparison of difference schemes,"
SIAM Journal on Numerical Analysis 5, 506–517 (1968).
DOI: 10.1137/0705041
```

**Significance:** Introduces Strang (symmetric, second-order) operator
splitting $\exp(\Delta z(A+B)) \approx \exp(\tfrac{\Delta z}{2}A)\,
\exp(\Delta z B)\,\exp(\tfrac{\Delta z}{2}A)$ with error $\mathcal{O}(\Delta
z^3)$. Provides the numerical analysis context for understanding why
first-order Lie–Trotter splitting is used in standard multislice (rather than
Strang), due to the sequential structure of transmission and propagation.

---

#### McLachlan & Quispel (2002) — Splitting methods review
**Citation key:** `McLachlanQuispel2002`

```
R. I. McLachlan, G. R. W. Quispel,
"Splitting methods,"
Acta Numerica 11, 341–434 (2002).
DOI: 10.1017/S0962492902000053
```

**Significance:** Comprehensive review of splitting methods (Lie–Trotter,
Strang, higher-order) for ordinary and partial differential equations.
Provides the mathematical context for the commutator analysis in §5.2.

---

### Frozen-phonon & thermal diffuse scattering

#### Kirkland, Loane & Silcox (1987) — Original frozen-phonon
**Citation key:** `KirklandLoaneSilcox1987`

```
E. J. Kirkland, R. F. Loane, J. Silcox,
"Simulation of annular dark field STEM images using a modified
 multislice method,"
Ultramicroscopy 23, 77–96 (1987).
DOI: 10.1016/0304-3991(87)90229-4
```

**Significance:** The original frozen-phonon paper within the multislice
framework for ADF-STEM. Demonstrates that thermal vibrations can be
approximated by averaging over static atomic configurations.

---

#### Loane, Xu & Silcox (1991) — Thermal vibrations in CBED
**Citation key:** `LoaneXuSilcox1991`

```
R. F. Loane, P. Xu, J. Silcox,
"Thermal vibrations in convergent-beam electron diffraction,"
Acta Crystallographica A 47, 267–278 (1991).
DOI: 10.1107/S0108767391000375
```

**Significance:** Formalized the frozen-phonon technique for CBED,
demonstrating thermal diffuse background and Kikuchi bands. The Debye–Waller
factors used in §6.10 follow this formulation.

---

#### Van Dyck (2009) — Frozen-phonon adequacy
**Citation key:** `VanDyck2009`

```
D. Van Dyck,
"Is the frozen phonon model adequate to describe inelastic
 phonon scattering?",
Ultramicroscopy 109, 677–682 (2009).
DOI: 10.1016/j.ultramic.2009.01.001
```

**Significance:** Rigorous proof that the frozen-phonon model is equivalent to
a full quantum-mechanical treatment of inelastic phonon scattering. Justifies
the frozen-phonon approach used throughout CVDMS validation.

---

### Fresnel propagation & Fourier optics

#### Goodman (2017) — Fourier Optics
**Citation key:** `Goodman2017`

```
J. W. Goodman,
"Introduction to Fourier Optics,"
4th ed., W. H. Freeman / Macmillan Learning (2017).
ISBN: 9781319119164
```

**Significance:** The canonical reference for scalar diffraction theory, the
angular spectrum method, Fresnel diffraction, and the Fresnel transfer
function. Forms the mathematical foundation of the propagation step in
multislice algorithms and of the antialiasing analysis in §6.4.

---

### Multislice algorithm analysis

#### Wacker & Schroder (2015) — Multislice revisited
**Citation key:** `WackerSchroder2015`

```
C. Wacker, R. R. Schroder,
"Multislice algorithms revisited: Solving the Schrödinger equation
 numerically for imaging with electrons,"
Ultramicroscopy 151, 211–223 (2015).
DOI: 10.1016/j.ultramic.2014.12.008
```

**Significance:** Compares real-space and Fourier-space multislice algorithms,
including the Fresnel propagator treatment, and discusses going beyond the
high-energy (paraxial) approximation. Relevant to the FD vs FFT Laplacian
comparison in §6.5.

---

### Batkai et al. (2009) — Operator splitting convergence
**Citation key:** `Batkai2009`

```
A. Batkai, P. Csomos, G. Nickel,
"Operator splittings and spatial approximations for evolution equations,"
Journal of Evolution Equations 9, 613–636 (2009).
DOI: 10.1007/s00028-009-0028-z
```

**Significance:** Convergence analysis of Lie–Trotter, Strang, and weighted
splitting using the Trotter–Kato theorem. Provides additional numerical
rigour for the commutator-error analysis in §5.2.
