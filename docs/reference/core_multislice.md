# Core Multislice Theory

These papers establish the theoretical foundation of the multislice method for
simulating high-energy electron scattering in crystals.

## Key references

### Chen & Van Dyck (1997) — Accurate multislice theory
**Citation key:** `ChenVanDyck1997`

```
J. H. Chen, D. Van Dyck,
"Accurate multislice theory for elastic electron scattering
 in transmission electron microscopy,"
Ultramicroscopy 70, 29–44 (1997).
DOI: 10.1016/S0304-3991(97)00071-5
```

**Significance:** Derives the slice-transmission-operator (STO) matrix and the
forward-scattering coefficient (FSC) and backscattering coefficient (BSC)
operators that motivate the CVDMS framework. Treats the full square-root
operator $\hat{K}$ without Lie–Trotter splitting. Provides the coefficient
cascade $c_1=1$, $c_n = (0.5-n+1)\lambda/(\pi n)$ used in the present K-series.

---

### Cowley & Moodie (1957) — Original multislice formulation
**Citation key:** `CowleyMoodie1957`

```
J. M. Cowley, A. F. Moodie,
"The scattering of electrons by atoms and crystals.
 I. A new theoretical approach,"
Acta Crystallographica 10, 609–619 (1957).
DOI: 10.1107/S0365110X57002194
```

**Significance:** The seminal paper introducing the multislice method.
Demonstrates that a three-dimensional crystal potential can be sliced into
projected slices with wavefunction propagation between them. Establishes the
physical-optics approximation that is the basis of all modern multislice codes.

---

### Van Dyck (1975) — Path integral formalism
**Citation key:** `VanDyck1975`

```
D. Van Dyck,
"The path integral formalism as a new description for the
 diffraction of high-energy electrons in crystals,"
Physica Status Solidi (b) 72, 321–336 (1975).
DOI: 10.1002/pssb.2220720135
```

**Significance:** Reformulates electron diffraction using Feynman path
integrals, providing an alternative mathematical foundation for the series
expansion of the scattering operator. A conceptual antecedent to the Taylor
cascade used in CVDMS.

---

### Coene & Van Dyck (1984) — Real-space multislice (series)
**Citation key:** `CoeneVanDyck1984a` (Part III, computational algorithm)

Parts I–III of this series develop the real-space method for dynamical electron
diffraction:

- **Part I** (Van Dyck & Coene): Principles of the method. *Ultramicroscopy*
  15, 29–40 (1984). DOI: 10.1016/0304-3991(84)90072-X
- **Part II** (Coene & Van Dyck): Critical analysis of input parameters.
  *Ultramicroscopy* 15, 41–50 (1984). DOI: 10.1016/0304-3991(84)90073-1
- **Part III** (Coene & Van Dyck): Computational algorithm and practical
  applications. *Ultramicroscopy* 15, 287–300 (1984).
  DOI: 10.1016/0304-3991(84)90123-2

**Significance:** Part III provides the computational algorithm for the
real-space multislice method in which finite-difference Laplacians replace
FFT-based propagation. This is the direct algorithmic antecedent of the FD
Laplacian path in the CVDMS implementation.

---

### Ishizuka (1982) — Practical multislice
**Citation key:** `Ishizuka1982`

```
K. Ishizuka,
"A practical approach for the multislice method,"
Ultramicroscopy 9, 235–246 (1982).
DOI: 10.1016/0304-3991(82)90215-7
```

**Significance:** Discusses the accuracy of the standard multislice approach
and the role of the physical-optics approximation (neglect of commutator terms
between transmission and propagation). Provides practical guidance on sampling
and slice-thickness criteria.

---

### Kirkland (2020) — Textbook
**Citation key:** `Kirkland2020`

```
E. J. Kirkland,
"Advanced Computing in Electron Microscopy,"
3rd ed., Springer (2020).
DOI: 10.1007/978-3-030-33260-0
```

**Significance:** The definitive textbook covering multislice theory, frozen
phonon, CBED simulation, and image formation. §6.4.1 derives the formal
operator solution and the split-step (Lie–Trotter) approximation. §6.11
discusses commutator-based corrections via the Baker–Campbell–Hausdorff
expansion.
