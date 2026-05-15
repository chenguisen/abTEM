# CVDMS Paper — Reference Index

All references are collected under `docs/reference/`. The master BibTeX file is
[`references.bib`](references.bib). Thematic markdown files provide annotated
reading lists organized by topic.

## File structure

| File | Description |
|------|-------------|
| [`references.bib`](references.bib) | Master BibTeX library (single source of truth) |
| [`core_multislice.md`](core_multislice.md) | Foundational multislice theory |
| [`implementations.md`](implementations.md) | Software packages (abTEM, MULTEM, PRISM, py4DSTEM) |
| [`physics.md`](physics.md) | Scattering physics (potential, phonon, channeling, HOLZ) |
| [`numerical.md`](numerical.md) | Numerical methods (operator splitting, Fourier optics) |
| [`README.md`](README.md) | This index file |

## Quick reference — key papers mapped to outline sections

| Outline § | Key reference | Citation key |
|-----------|--------------|--------------|
| §3.1–3.2 Multislice background | Cowley & Moodie 1957, Kirkland 2020 | `CowleyMoodie1957`, `Kirkland2020` |
| §3.5 CVDMS (Chen–Van Dyck) | Chen & Van Dyck 1997 | `ChenVanDyck1997` |
| §3.5 K-series coefficients | Chen & Van Dyck 1997, Van Dyck 1975 | `ChenVanDyck1997`, `VanDyck1975` |
| §3.3 Projected potential | Lobato & Van Dyck 2014 | `LobatoVanDyck2014` |
| §3.3 Thermal smearing | Forbes et al. 2010, Loane et al. 1991 | `Forbes2010`, `LoaneXuSilcox1991` |
| §3.5 Operator splitting | Trotter 1959, Strang 1968, McLachlan & Quispel 2002 | `Trotter1959`, `Strang1968`, `McLachlanQuispel2002` |
| §4.5 Channeling | Allen et al. 2003, Pennycook & Nellist 2011 | `Allen2003`, `PennycookNellist2011` |
| §4.5 HOLZ | Chen & Van Dyck 1997 (HOLZ formalism) | `ChenVanDyck1997` |
| §4.6 IPR | Allen et al. 2003 | `Allen2003` |
| §6.4 Antialiasing | Kirkland 2020 | `Kirkland2020` |
| §6.10 Potential param. | Lobato & Van Dyck 2014 | `LobatoVanDyck2014` |
| §6.8 Reference methods | Madsen & Susi 2021 (abTEM), Ophus 2017 (PRISM) | `MadsenSusi2021`, `Ophus2017` |

## LaTeX / BibTeX usage

Add to your preamble:
```latex
\bibliography{docs/reference/references}
```

Then cite as:
```latex
The conventional multislice scheme~\cite{CowleyMoodie1957,Kirkland2020}
relies on Lie–Trotter splitting~\cite{Trotter1959}.
The CVDMS framework~\cite{ChenVanDyck1997} avoids this approximation.
```
