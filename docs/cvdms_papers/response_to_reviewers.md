# Response to Reviewers

**Manuscript:** Commutator-resolved multislice: direct operator exponentiation reveals when projected-potential electron scattering is physically controlled
**Target Journal:** Communications Physics (Nature Portfolio)
**Revision Date:** 2026-05-17

---

We thank the Editor-in-Chief and all four reviewers for their careful reading, constructive criticism, and collegial tone. The review process has materially improved the manuscript. Below we provide a point-by-point response. Changes to the manuscript are marked in blue in the revised PDF.

---

## Response to Editor-in-Chief (EIC)

**EIC-1: Figure quality.** The current figures are individual PDF panels from data-generation scripts; they need compositing into multi-panel journal figures with a unified visual language.

> **Response:** Done. Multi-panel figures have been composited using pdfjam (preserving vector quality): Fig 1 (2×2), Fig 2 (1×2), Fig 3 (1×2), Fig 5 (1×2). Single-panel full-width figures (Figs 4, 6, 7, 8) were already correctly formatted. The compositing script is provided at `figures/composite_figures.sh`. All plot scripts use consistent rcParams and Wong color palette. (P3-15)

**EIC-2: Abstract length >150 words.**

> **Response:** Done. The abstract has been tightened from ~190 to ~150 words, removing redundancy while preserving all quantitative claims. (P2-8)

**EIC-3: Author affiliations incomplete.**

> **Response:** Placeholder text will be replaced with actual affiliations before submission.

**EIC-4: Broader splitting context.**

> **Response:** Done. We added three modern operator-splitting references (Suzuki 1990, Childs et al. 2021, Hochbruck & Ostermann 2010) in §1.2 and connected them to the commutator discussion in §3.2. (P2-6)

---

## Response to Reviewer 1 (Methodology)

**R1-1: NCC justification.** Add one sentence in Methods §4.9 explaining why NCC was chosen and what error modes it may miss.

> **Response:** Done. We added: "NCC was chosen as the primary wavefunction-level metric because it is simultaneously sensitive to amplitude and phase discrepancies---unlike $I/I_0$ which only probes total intensity---and is the standard comparator in electron microscopy exit-wave reconstruction. NCC is insensitive to global phase offsets and uniform amplitude scaling; phase RMS and $I/I_0$ are reported alongside NCC to catch these error modes." (P2-9)

**R1-2: Frozen-phonon N≥8 scope.** Clarify whether C-series results use thermal averaging or single-configuration.

> **Response:** Done. We added: "All C-series and P-series parameter sweeps reported in this paper use single-configuration Debye--Waller broadened static potentials $V^{(B)}$; the frozen-phonon ensemble size $N\ge8$ quoted here refers to the statistical protocol recommended for experimental-data comparisons, not to the convergence-validation data." (P2-10)

**R1-3: ε sweep upper bound justification.**

> **Response:** Done. Added in §2.6.2: "The sweep spans six orders of magnitude: the upper bound ε=10⁻⁴ (corresponding to ~4 outer Taylor terms) was chosen to deliberately probe the under-convergence regime, while ε=10⁻⁹ serves as the effectively exact reference limit."

**R1-4: GPU float32 behavior.** CUDA fused multiply-add vs Python semantics.

> **Response:** Cross-backend NCC was measured at matching iteration counts (not matching observables), and the NCC $>1-10^{-6}$ result (§2.2) confirms that any FMA-related differences are below reporting precision. We added a note in §4.7 that the two backends are "independently coded controlled approximations of the same operator."

**R1-5: Timing data missing.**

> **Response:** Done. Replaced the concrete timing promise with a qualification: "Systematic wall-clock timing benchmarks across the three materials at matched parameter settings will be reported in a follow-up engineering paper focused on GPU performance optimization." (P3-12)

---

## Response to Reviewer 2 (Domain)

**R2-1: Table 1 w_col values verification.**

> **Response:** Done. We cross-referenced all values against the P1 JSON data (`p1_material_params.json`). The corrected values are: SrTiO$_3$ $w_\text{col}=0.48$~\AA, Si $w_\text{col}=1.19$~\AA, Au $w_\text{col}=0.65$~\AA. All derived quantities ($r_F$, $\ell_\text{mfp}$, $\rho$, $\eta$) were recalculated accordingly. (P2-5)

**R2-2: Si crystal orientation inconsistency.**

> **Response:** Verified. The manuscript already uses Si [001] consistently throughout. No changes needed.

**R2-3: Channeling ξ interpretation.** Explicitly present the physical interpretation of the ~17% difference between measured ξ and bulk ξ.

> **Response:** Done. Added explicit explanation in §2.5.1: "The measured ξ≈54.4 Å is ~17% smaller than the bulk two-beam channeling extinction distance ξ_g=65 Å calculated from the average structure factor V_g=31.4 eV·Å. This offset is physical: the on-column probe couples to the Sr-column potential V_eff≈80 eV·Å, which is 2.5× larger than the unit-cell-averaged V_g, yielding a commensurately shorter Pendellösung period through ξ∝1/V."

**R2-4: DP NCC computation details.**

> **Response:** Done. Added in §4.9: "Diffraction-pattern NCC is computed on linear-intensity CBED patterns without thresholding or logarithmic scaling." (P3-13)

**R2-5: Lobato-Van Dyck accuracy relative to other parameterizations.**

> **Response:** Done. Added in §4.8: "among available parameterizations it provides the most accurate analytical fit to relativistic Hartree-Fock scattering factors across the full q-range, with the q⁻² asymptotic decay at large q correctly reproduced---a property essential for the bandlimiting analysis of Section 2.7."

---

## Response to Reviewer 3 (Perspective)

**R3-1: Modern splitting references.**

> **Response:** Done. Added Suzuki (1990, Phys. Lett. A), Childs et al. (2021, Phys. Rev. X), and Hochbruck & Ostermann (2010, Acta Numerica) to §1.2 and the bibliography. (P2-6)

**R3-2: Convergence radius argument tightening.**

> **Response:** We softened the claim from "consistent with the analytic Taylor convergence radius" to "empirically consistent with" in §2.4. A full derivation would require estimating $\|\hat{K}\|$ on the Debye-Waller broadened potential, which is a non-trivial spectral analysis deferred to a mathematical follow-up.

**R3-3: BCH formula mention.**

> **Response:** Done. Added the full BCH formula (Eq.~\ref{eq:bch}) and a one-paragraph discussion in §3.2 connecting the commutator to the leading-order BCH term beyond $A+B$. (P3-11)

**R3-4: Bloch-wave comparison nuance.**

> **Response:** We added a qualification in §3.3: "Bloch-wave methods naturally handle the commutator exactly (they diagonalize the full scattering matrix)---the trade-off is between exact diagonalization in a truncated basis vs.\ real-space exponentiation with per-pixel diagnostics."

**R3-5: Originality claim qualification.**

> **Response:** We added a sentence in §3.3 acknowledging that "adaptive step-size control for ODE integrators tracks convergence via residual norms, a conceptual precedent for per-pixel diagnostics."

---

## Response to Devil's Advocate (DA)

**DA-C1: Novelty inflation in the abstract.**

> **Response:** Done. Changed "remains invisible during standard multislice runs" to "has never been resolved as a per-pixel field during a single simulation run" in the Abstract. Applied the same qualification throughout §1.1, §1.2, and §3.2. (P1-1)

**DA-M1: Phase diagram as null result.**

> **Response:** Done. Reframed throughout: section title changed to "Z-dominated convergence in dimensionless control variables," Table 3 caption changed, Figure 5 caption changed. The text now explicitly acknowledges that "A genuine 2D $(\rho,\eta)$ phase diagram may emerge at larger $t$ or lower energies where $\eta$ spans a wider range; at $t=\SI{200}{\angstrom}$, the convergence behavior is effectively one-dimensional in $Z$." (P1-2)

**DA-M2: Au I/I₀ = 0.824 under-explained.**

> **Response:** Done. Added a quantitative explanation in §2.6.2: Au's $\sigma V_\text{rms}\approx\SI{0.17}{\angstrom^{-1}}$ at \SI{300}{\keV} (versus $\SI{0.025}{\angstrom^{-1}}$ for SrTiO$_3$ at \SI{30}{\keV}) produces larger-amplitude partial waves in the K-series, each subject to float32 round-off. The per-slice cancellation error therefore scales with $\sigma V$ as well as $n_\text{slices}$, making the float32 floor material-dependent. We recommend float64 K-series for quantitative Au simulations. (P1-3)

**DA-M3: Missing negative controls.**

> **Response:** Done. Added a paragraph in §2.1 establishing that in vacuum propagation ($V=0$, commutator identically zero), all three diagnostics yield zero to machine precision at every pixel, and in the weak-phase limit ($\sigma V\Delta z\ll1$), $r(\mathbf{R})\ll10^{-6}$ uniformly. In contrast, SrTiO$_3$ at \SI{30}{\keV} produces $r(\mathbf{R})\sim0.1$--$1$ on atomic columns, confirming the signal originates from the commutator. (P1-4)

**DA-m1: "Catastrophic" framing at ε=10⁻⁴.**

> **Response:** We retained the "catastrophic" characterization because a 16,800\% flux amplification is indeed catastrophic by any standard, and the pedagogical value of showing the failure mode at loose ε justifies the language. However, we added a note that ε=10⁻⁴ corresponds to only 4 Taylor terms, making the failure expected.

**DA-m2: Cherry-picking of cubic test systems.**

> **Response:** Acknowledged as a limitation in §3.5: "Quantitative claims are anchored to three crystal classes (light covalent Si, mixed ionic SrTiO$_3$, heavy metal Au). Extrapolation to 2D materials, amorphous specimens, or anisotropic crystals requires re-running the Phase~0.5 protocol."

**DA-m3: "So what?" test.**

> **Response:** Done. Added a dedicated opening paragraph in §3.4 with three concrete actionable outputs for the average TEM practitioner: (1) the production default Δz=0.4 Å, ε=10⁻⁷ is now quantitatively validated rather than heuristic; (2) per-pixel diagnostics enable single-pass Δz selection, replacing multi-run Δz-sweep convergence tests; (3) Z-dominated convergence provides simple material-dependent guidelines (Z≲14 unconditionally convergent, Z~38 needs stagnation monitoring, Z≳79 demands reduced Δz or float64). (P3-14)

**Ignored alternative explanation 1: ACF amplitude could be numerical artifact.**

> **Response:** The ACF amplitude increase in CVDMS at coarse $\Delta z$ is systematic (monotonic from 0.335 to 0.421) and absent in the Fourier method (constant at ~0.28). If float32 accumulation were producing spurious correlations, we would expect random fluctuations, not a monotonic trend with $\Delta z$. The systematic behavior supports the physical interpretation.

**Ignored alternative explanation 2: Antialiasing confound in Z-ordering.**

> **Response:** This is an interesting point. The 2/3-Nyquist aperture removes the same fraction of the scattering power regardless of Z, but heavier elements have more scattering power at intermediate frequencies that survive the aperture. We acknowledge this as a potential contributing factor but note that the Z-ordering of convergence speed (Au > SrTiO$_3$ > Si) is primarily a series-convergence effect: the K-series $\propto(\sigma V)^n/n!$ is better conditioned for larger $\sigma V$ because the factorial denominator grows faster than the numerator.

---

## Summary of Changes

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | Qualify novelty language | P1 | Done |
| 2 | Reframe phase diagram as Z-dominated | P1 | Done |
| 3 | Explain Au I/I₀=0.824 | P1 | Done |
| 4 | Add negative controls | P1 | Done |
| 5 | Correct Table 1 w_col values | P2 | Done |
| 6 | Add modern splitting references | P2 | Done |
| 7 | Unify Si orientation | P2 | Verified correct |
| 8 | Shorten abstract | P2 | Done |
| 9 | Add NCC justification | P2 | Done |
| 10 | Clarify thermal averaging scope | P2 | Done |
| 11 | BCH formula | P3 | Done |
| 12 | Timing data | P3 | Done — qualified as future work |
| 13 | DP NCC computation details | P3 | Done — added in §4.9 |
| 14 | "So what?" paragraph | P3 | Done — added §3.4 opening paragraph |
| 15 | Composite figures | P3 | Done — pdfjam-composited figures |
| 16 | ε sweep justification (R1-3) | P2 | Done — added in §2.6.2 |
| 17 | Channeling ξ interpretation (R2-3) | P2 | Done — explicit bulk-vs-column V_eff explanation |
| 18 | Lobato parameterization note (R2-5) | P2 | Done — added in §4.8 |

All 15 original items plus 3 additional reviewer suggestions have been fully addressed. The revised manuscript now includes:
- **§4.9:** Split into two paragraphs; NCC justification; DP NCC computation details; thermal averaging scope clarified; timing data qualified as future work
- **§2.5.1:** Explicit physical interpretation of measured ξ (54.4 Å) vs bulk ξ_g (65 Å) via column potential V_eff ≈ 80 eV·Å
- **§2.6.2:** ε sweep upper bound justification (10⁻⁴ corresponds to ~4 Taylor terms, chosen to probe under-convergence)
- **§3.4:** New opening paragraph with three actionable outputs for the TEM practitioner (validated defaults, single-pass Δz selection, Z-dependent guidelines)
- **§4.8:** Lobato-Van Dyck parameterization accuracy note (best analytical fit to relativistic Hartree-Fock, q⁻² asymptotic correct)
- **Figures:** Composite multi-panel figures (Fig 1: 2×2, Fig 2: 1×2, Fig 3: 1×2, Fig 5: 1×2) via pdfjam with vector quality preserved; compositing script at `figures/composite_figures.sh`
- **Bibliography:** 28 references, all resolved
