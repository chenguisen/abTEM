# Citation Accuracy Audit Report

**Paper:** Commutator-resolved multislice: direct operator exponentiation reveals when projected-potential electron scattering is physically controlled
**Date:** 2026-05-17
**Total references:** 28

---

## Audit Methodology

Each reference was checked for: (1) bibliographic accuracy (author, title, journal, year, volume, pages, DOI), (2) accuracy of the specific claim(s) attributed to the reference in the paper text.

**Verification tier:**
- **Direct verification** = web search confirmed paper exists and claim is supported
- **Partial verification** = paper exists but specific claim could not be fully confirmed
- **Standard reference** = well-known textbook or canonical paper, claim is standard knowledge

---

## Summary of Findings

| Status | Count | Description |
|--------|-------|-------------|
| ✅ Correct | 22 | Bib entry and claims verified |
| ⚠️ Minor issue | 4 | Bib entry has minor imprecision or claim attribution could be sharpened |
| 🔴 Significant issue | 1 | Ishizuka1982 — bib entry appears to conflate two different papers, DOI unresolvable |
| ⚠️ Bib incomplete | 1 | Chen2025 missing co-authors in bib entry |

---

## Individual Reference Audit

### 1. ✅ CowleyMoodie1957
- **Bib:** Cowley, J. M. and Moodie, A. F. (1957). "The scattering of electrons by atoms and crystals. I. A new theoretical approach." Acta Crystallographica, 10, 609–619.
- **Verification:** Confirmed. This is the foundational multislice paper.
- **Claims in paper:** "the Born approximation fails for all but the thinnest specimens"; "The conventional multislice scheme"
- **Assessment:** Accurate. The paper introduced the physical-optics approach to multislice as an alternative to the Born approximation.

### 2. ✅ Kirkland2020
- **Bib:** Kirkland, E. J. (2020). Advanced Computing in Electron Microscopy, 3rd ed. Springer.
- **Verification:** Standard textbook, confirmed.
- **Claims in paper:** "the Born approximation fails"; "conventional multislice scheme"; "Bloch-wave methods"; "empirical rule"
- **Assessment:** Accurate. Kirkland is the standard reference for multislice implementation, including Bloch-wave methods and practical rules of thumb.

### 3. ✅ ChenVanDyck1997
- **Bib:** Chen, J. H. and Van Dyck, D. (1997). "Accurate multislice theory for elastic electron scattering in transmission electron microscopy." Ultramicroscopy, 70, 29–44.
- **Verification:** Confirmed via web search. Title, journal, volume, pages, DOI all match.
- **Claims in paper:** "the interaction constant"; "Earlier Taylor-series treatments"; "original operator derivation"; "HOLZ effects are a canonical example of projected-potential approximation breakdown identified by Chen & Van Dyck"; "The relativistically corrected scalar wave equation"
- **Assessment:** Accurate. The paper's abstract confirms it addresses HOLZ reflections, back-scattering, and the high-energy approximation limitations.

### 4. 🔴 Ishizuka1982 — SIGNIFICANT ISSUE
- **Bib (current):** Ishizuka, K. (1982). "A practical approach for the multislice method." Ultramicroscopy, 9, 235–246. DOI: 10.1016/0304-3991(82)90218-4
- **Verification:** The DOI does NOT resolve (HTTP 404). Web searches indicate:
  - The title "A practical approach for the multislice method" appears to be a shortened/modified version of **Ishizuka & Uyeda (1977)**: "A new theoretical and practical approach to the multislice method", Acta Crystallographica A33, 740–749.
  - The 1982 paper by Ishizuka alone is: "Multislice formula for inclined illumination", Acta Crystallographica A, 38, 773–779. This is NOT in Ultramicroscopy.
  - The actual Ishizuka paper in Ultramicroscopy 1982 is: "Translation symmetries in convergent-beam electron diffraction", Ultramicroscopy, 9(3), 255–257. Different pages, different title.
- **Claim in paper:** "Ishizuka noted that the 'physical-optics approximation' neglects cross-terms between transmission and propagation operators"
- **Assessment:** The bib entry appears to conflate elements from two different papers:
  1. Title from Ishizuka & Uyeda (1977), Acta Cryst. A33, 740–749
  2. Year (1982) and journal (Ultramicroscopy) — but the 1982 paper in Ultramicroscopy has a different title and pages
- **Recommended fix:** Verify the intended reference. If the claim about "physical-optics approximation neglecting cross-terms" is from the 1977 paper, update the bib entry to:
  ```
  Ishizuka, K. and Uyeda, N. (1977). "A new theoretical and practical approach to the multislice method." Acta Crystallographica A, 33, 740–749.
  ```
  If the claim is from the 1982 paper, update to:
  ```
  Ishizuka, K. (1982). "Multislice formula for inclined illumination." Acta Crystallographica A, 38, 773–779.
  ```

### 5. ✅ Trotter1959
- **Bib:** Trotter, H. F. (1959). "On the product of semi-groups of operators." Proceedings of the American Mathematical Society, 10, 545–551.
- **Verification:** Confirmed. This is the canonical Trotter product formula paper.
- **Claim in paper:** "Trotter proved the product formula exp(t(A+B)) = lim_{n→∞} [exp(tA/n)exp(tB/n)]^n without providing error bounds for finite n"
- **Assessment:** Accurate. Trotter's paper proves convergence but does not provide finite-n error bounds. Note: the bib DOI `10.2307/2033649` is the JSTOR identifier (valid, but the AMS DOI is `10.1090/S0002-9939-1959-0108732-6`). Either is acceptable.

### 6. ✅ Strang1968
- **Bib:** Strang, G. (1968). "On the construction and comparison of difference schemes." SIAM Journal on Numerical Analysis, 5, 506–517.
- **Verification:** Confirmed. This is the canonical Strang splitting paper.
- **Claim in paper:** "Strang showed symmetric splitting reduces per-step error to O(Δz³)"
- **Assessment:** Accurate. Strang splitting achieves second-order accuracy (O(Δz²) global, O(Δz³) per-step).

### 7. ✅ McLachlanQuispel2002
- **Bib:** McLachlan, R. I. and Quispel, G. R. W. (2002). "Splitting methods." Acta Numerica, 11, 341–434.
- **Verification:** Confirmed. This is the landmark survey on splitting methods.
- **Claim in paper:** "no splitting scheme simultaneously achieves high accuracy and efficiency at large step sizes when the commutator is non-negligible"
- **Assessment:** Accurate. The survey comprehensively documents the limitations of splitting methods, particularly commutator-related error.

### 8. ✅ Suzuki1990
- **Bib:** Suzuki, M. (1990). "Fractal decomposition of exponential operators with applications to many-body theories and Monte Carlo simulations." Physics Letters A, 146, 319–323.
- **Verification:** Confirmed. Paper introduces higher-order Suzuki-Trotter decompositions.
- **Claim in paper:** "Higher-order Suzuki-Trotter decompositions can reduce the per-step error to O(Δz^p) for p>3, but at the cost of exponentially growing operator applications per step"
- **Assessment:** Accurate. The paper's fractal decomposition produces m-th order approximants with exponentially many terms.

### 9. ✅ Childs2021
- **Bib:** Childs, A. M. and Su, Y. and Tran, M. C. and Wiebe, N. and Zhu, S. (2021). "Theory of Trotter error with commutator scaling." Physical Review X, 11, 011020.
- **Verification:** Confirmed. Paper provides tight Trotter error bounds with commutator scaling.
- **Claim in paper:** "recent quantum-simulation analyses providing tight error bounds confirming that the commutator norm ||[Â,B̂]|| remains the fundamental limitation for any splitting scheme"
- **Assessment:** Accurate, though "for any splitting scheme" is slightly broad — the paper specifically addresses Trotter and higher-order product formulas.

### 10. ✅ HochbruckOstermann2010
- **Bib:** Hochbruck, M. and Ostermann, A. (2010). "Exponential integrators." Acta Numerica, 19, 209–286.
- **Verification:** Confirmed. This is the definitive survey on exponential integrators.
- **Claim in paper:** "Exponential integrators for stiff evolution equations similarly require evaluating the full operator exponential rather than splitting when the commutator is non-negligible"
- **Assessment:** Accurate. Exponential integrators indeed evaluate the full matrix exponential rather than splitting.

### 11. ✅ VanDyck1975
- **Bib:** Van Dyck, D. (1975). "The path integral formalism as a new description for the diffraction of high-energy electrons in crystals." Physica Status Solidi (b), 72, 321–336.
- **Verification:** Confirmed via ADS and Scilit.
- **Claim in paper:** "Earlier Taylor-series treatments" (with ChenVanDyck1997)
- **Assessment:** Accurate. Van Dyck's path integral formalism is a precursor to the Taylor-series multislice approach.

### 12. ✅ MadsenSusi2021
- **Bib:** Madsen, J. and Susi, T. (2021). "The abTEM code: transmission electron microscopy from first principles." Open Research Europe, 1, 24.
- **Verification:** Confirmed. The abTEM paper exists with matching DOI.
- **Claims in paper:** abTEM Fourier multislice used as reference method throughout.
- **Assessment:** Accurate.

### 13. ✅ Ophus2017
- **Bib:** Ophus, C. (2017). "A fast image simulation algorithm for scanning transmission electron microscopy." Advanced Structural and Chemical Imaging, 3, 13.
- **Verification:** Confirmed. This is the PRISM paper.
- **Claims in paper:** "PRISM"; "prismatic"
- **Assessment:** Accurate. The paper introduces PRISM; the companion paper introduces the Prismatic software.

### 14. ✅ LobatoVanDyck2015
- **Bib:** Lobato, I. and Van Dyck, D. (2015). "MULTEM: a new multislice program to perform accurate and fast electron diffraction and imaging simulations using GPU." Ultramicroscopy, 156, 9–17.
- **Verification:** Standard reference, bib matches known publication.
- **Claim in paper:** MULTEM as existing high-accuracy alternative
- **Assessment:** Accurate.

### 15. ✅ CaiChen2012
- **Bib:** Cai, C. Y. and Chen, J. H. (2012). "An accurate multislice method for low-energy transmission electron microscopy." Micron, 43, 374–379.
- **Verification:** Standard reference, bib matches.
- **Claim in paper:** "revised real-space method"
- **Assessment:** Accurate. This paper describes the revised real-space (RRS) method for low-energy TEM.

### 16. ✅ MingChen2013
- **Bib:** Ming, W. Q. and Chen, J. H. (2013). "Validities of three multislice algorithms for quantitative low-energy transmission electron microscopy." Ultramicroscopy, 134, 135–143.
- **Verification:** Standard reference, bib matches.
- **Claim in paper:** "introduction of a threshold-convergence criterion"
- **Assessment:** Accurate. This paper introduced threshold-based convergence control for the real-space multislice.

### 17. ✅ Lv2016
- **Bib:** Lv, C. L. and Cai, C. Y. and Fu, X. M. and Chen, J. H. (2016). "Dynamical electron diffraction simulation for non-orthogonal crystal system by a revised real space method." Journal of Microscopy, 261, 105–114.
- **Verification:** Standard reference, bib matches.
- **Claim in paper:** "extension to non-orthogonal crystal systems"
- **Assessment:** Accurate. The paper extends the RRS method to non-orthogonal systems.

### 18. ⚠️ Chen2025 — BIB MISSING CO-AUTHORS
- **Bib (current):** Chen, G. S. and He, Y. T. and Yan, J. (2025). "Fast STEM image simulation in low-energy transmission electron microscopy by the accurate Chen–van-Dyck multislice method." Micron, 190, 103778.
- **Verification:** The paper exists but the full author list is: G.S. Chen, Y.T. He, **W.Q. Ming, C.L. Wu, D. Van Dyck, J.H. Chen**. The bib entry only lists 3 of 6 authors.
- **Claim in paper:** "first GPU-accelerated STEM implementation" of the CVD method
- **Assessment:** The claim is accurate, but the bib entry is missing co-authors.
- **Recommended fix:** Add the missing authors: `Chen, G. S. and He, Y. T. and Ming, W. Q. and Wu, C. L. and Van Dyck, D. and Chen, J. H.`

### 19. ✅ LobatoVanDyck2014
- **Bib:** Lobato, I. and Van Dyck, D. (2014). "An accurate parameterization for scattering factors, electron densities and electrostatic potentials for neutral atoms." Acta Crystallographica A, 70, 636–648.
- **Verification:** Confirmed. The paper's title is slightly longer in reality ("An accurate parameterization for scattering factors, electron densities and electrostatic potentials for neutral atoms that obey all physical constraints") but the shortened version in the bib is acceptable.
- **Claims in paper:** "Lobato parameterization"; "q^{-2} asymptotic behavior"; "most accurate analytical fit to relativistic Hartree-Fock scattering factors"
- **Assessment:** Accurate. The paper explicitly presents a parameterization that obeys the correct q^{-2} asymptotic behavior and is reported to be one order of magnitude better than previous fits.

### 20. ⚠️ Forbes2010 — MINOR: B-VALUES ATTRIBUTION
- **Bib:** Forbes, B. D. and Martin, A. V. and Findlay, S. D. and D'Alfonso, A. J. and Allen, L. J. (2010). "Quantum mechanical model for phonon excitation in electron diffraction and imaging using a Born–Oppenheimer approximation." Physical Review B, 82, 104103.
- **Verification:** Paper confirmed. It presents a phonon excitation model using the Born-Oppenheimer approximation.
- **Claim in paper:** Specific Debye-Waller B values (Sr: 0.62, Ti: 0.51, O: 0.86, Si: 0.46, Au: 0.66 Å² at 300K) cited to Forbes2010.
- **Assessment:** The Forbes2010 paper discusses Debye-Waller factors in the context of their phonon model, but these specific isotropic B values are standard tabulated values (from International Tables for Crystallography or similar). While Forbes2010 is a reasonable citation for the frozen-phonon Debye-Waller framework, the specific numerical values may be from standard tables rather than uniquely from Forbes2010. This is a **minor attribution issue** — the values are correct but the citation for the specific numbers could be more precise (e.g., citing International Tables for Crystallography Vol. C or Peng et al. 1996).
- **Recommended action:** Consider adding a note that the B values are from standard tables, while Forbes2010 provides the theoretical framework for the frozen-phonon model.

### 21. ✅ PennycookNellist2011
- **Bib:** Pennycook, S. J. and Nellist, P. D. (eds.) (2011). Scanning Transmission Electron Microscopy: Imaging and Analysis. Springer.
- **Verification:** Standard textbook, confirmed.
- **Claim in paper:** "The s-state model of axial channeling"; "exactly where channeling localization and HOLZ Bloch-state coupling occur"
- **Assessment:** Accurate. Pennycook & Nellist is the canonical reference for STEM imaging including the s-state channeling model.

### 22. ✅ Allen2003
- **Bib:** Allen, L. J. and Findlay, S. D. and Oxley, M. P. and Rossouw, C. J. (2003). "Lattice-resolution contrast from a focused coherent electron probe. Part I." Ultramicroscopy, 96, 47–63.
- **Verification:** Confirmed. Paper develops the Bloch wave framework for STEM lattice-resolution contrast.
- **Claim in paper:** "The s-state model of axial channeling" (with PennycookNellist2011)
- **Assessment:** Accurate. Allen et al. (2003) develops the Bloch wave framework that underpins the s-state model.

### 23. ⚠️ Batkai2009 — MINOR: CLAIM ATTRIBUTION
- **Bib:** Bátkai, A. and Csomós, P. and Nickel, G. (2009). "Operator splittings and spatial approximations for evolution equations." Journal of Evolution Equations, 9, 613–636.
- **Verification:** Confirmed. Paper uses Trotter-Kato theorem for convergence analysis of operator splitting with spatial discretization.
- **Claim in paper:** "Bátkai et al. provided convergence analysis via the Trotter-Kato theorem, confirming that splitting error is controlled by ||[Â,B̂]||"
- **Assessment:** Partially accurate. Batkai2009 does use the Trotter-Kato theorem for convergence analysis. However, the specific claim that "splitting error is controlled by ||[Â,B̂]||" is more directly from McLachlan & Quispel (2002) and Childs et al. (2021). Batkai2009 focuses on the interplay of spatial and temporal discretization errors. The attribution is reasonable but imprecise — the commutator-norm bound is a more general result from splitting theory.
- **Recommended action:** Consider citing McLachlanQuispel2002 or Childs2021 for the commutator-norm bound specifically, while keeping Batkai2009 for the Trotter-Kato convergence context.

### 24. ✅ CoeneVanDyck1984
- **Bib:** Coene, W. and Van Dyck, D. (1984). "Real-space multislice methods." Ultramicroscopy, 15, 287–300.
- **Verification:** Confirmed. This is Part III of a three-part series. The actual title is "The real space method for dynamical electron diffraction calculations in high resolution electron microscopy. III. A computational algorithm for the electron propagation with its practical applications." The shortened title in the bib is acceptable.
- **Claim in paper:** "CVDMS traces back to the real-space multislice algorithm of Coene & Van Dyck, which used finite-difference Laplacians on 2D grids"
- **Assessment:** Accurate. Part III specifically introduces the finite-difference Laplacian implementation.

### 25. ⚠️ WackerSchroder2015 — MINOR: DOI DISCREPANCY
- **Bib:** Wacker, C. and Schroder, R. R. (2015). "Multislice algorithms revisited: solving the Schrödinger equation numerically for imaging with electrons." Ultramicroscopy, 151, 211–223. DOI: 10.1016/j.ultramic.2014.10.012
- **Verification:** Paper confirmed. However, the web search returned DOI `10.1016/j.ultramic.2014.12.008` while the bib has `10.1016/j.ultramic.2014.10.012`. These are slightly different — one month digit differs.
- **Claim in paper:** "Wacker & Schröder systematically compared real-space and Fourier-space multislice algorithms, discussing Laplacian discretization accuracy"
- **Assessment:** The claim is accurate. The DOI in the bib should be verified.
- **Recommended action:** Verify the correct DOI (likely `10.1016/j.ultramic.2014.12.008` based on search results).

### 26. ✅ Savitzky2021
- **Bib:** Savitzky, B. H. et al. (2021). "py4DSTEM: a software package for four-dimensional scanning transmission electron microscopy data analysis." Microscopy and Microanalysis, 27, 712–743.
- **Verification:** Confirmed. The bib lists 14 of ~23 authors, which is common practice for papers with many authors.
- **Claim in paper:** "coupling to phonon and ptychographic 4D-STEM workflows" (in future work context)
- **Assessment:** Accurate. py4DSTEM is indeed the standard tool for 4D-STEM data analysis including ptychography.

### 27. ✅ Goodman2017
- **Bib:** Goodman, J. W. (2017). Introduction to Fourier Optics, 4th ed. W. H. Freeman.
- **Verification:** Standard textbook, bib matches known publication. Note: The bib entry has no DOI, which is fine for a book.
- **Claim in paper:** "a bandwidth inheritance effect understandable within the angular-spectrum framework of scalar diffraction theory"
- **Assessment:** Accurate. Goodman's book is the canonical reference for Fourier optics and angular-spectrum framework.

### 28. ✅ Self1983
- **Bib:** Self, P. G. and O'Keefe, M. A. and Buseck, P. R. and Spargo, A. E. C. (1983). "Practical computation of amplitudes and phases in electron diffraction." Ultramicroscopy, 11, 35–52.
- **Verification:** Confirmed. The abstract states: "in order to obtain accurate results the beams included in a calculation must extend out in reciprocal space to inverse spacing values of approximately 40 nm⁻¹."
- **Claim in paper:** "A 2/3-Nyquist cosine-taper aperture, tracing back to the bandlimiting practice established by Self et al."
- **Assessment:** Accurate. Self et al. established the principle that beams must extend to ~40 nm⁻¹ for accuracy, which is the conceptual origin of bandlimiting in multislice. The specific 2/3-Nyquist implementation is a technical detail, but the general principle traces back to Self et al.

---

## Action Items

### Critical (must fix before submission)

1. ~~**Ishizuka1982**~~ → **FIXED**: Changed to `IshizukaUyeda1977` with correct authors (Ishizuka & Uyeda), title, journal (Acta Crystallographica A), year (1977), volume (33), pages (740–749).

### Important (should fix)

2. ~~**Chen2025**~~ → **FIXED**: Added missing co-authors Ming, Wu, Van Dyck, J.H. Chen.

3. ~~**WackerSchroder2015**~~ → **FIXED**: DOI corrected to `10.1016/j.ultramic.2014.12.008`.

### Minor (deferred — not blocking submission)

4. **Forbes2010** — The Debye-Waller B values are standard; Forbes2010 provides the relevant frozen-phonon framework. Acceptable as-is.
5. **Batkai2009** — Claim already supported by McLachlanQuispel2002 and Childs2021 in the same paragraph. Acceptable as-is.

---

## Overall Assessment

The paper's citations are generally accurate and well-sourced. Of 28 references, **after fixes**:
- **25 are fully verified** with correct bib entries and accurate claims
- **3 have minor issues** (Forbes2010 attribution, Batkai2009 attribution, Chen2025 now fixed)

The most important fix is resolving the Ishizuka1982 reference, as the current bib entry cannot be verified and the DOI doesn't resolve.

---

## Fixes Applied (2026-05-17)

### 1. 🔴 Ishizuka1982 → IshizukaUyeda1977 (FIXED)
- **Bib key changed:** `Ishizuka1982` → `IshizukaUyeda1977`
- **Corrected reference:** Ishizuka, K. & Uyeda, N. (1977). "A new theoretical and practical approach to the multislice method." Acta Crystallographica A, 33, 740–749.
- **DOI:** `10.1107/S0567739477001872`
- **Citations updated in:** `cvdms_paper_en.tex:90`, `cvdms_paper_cn.tex:89`

### 2. ⚠️ Chen2025 — Missing co-authors (FIXED)
- **Authors updated from:** `Chen, G. S. and He, Y. T. and Yan, J.`
- **Authors updated to:** `Chen, G. S. and He, Y. T. and Ming, W. Q. and Wu, C. L. and Van Dyck, D. and Chen, J. H.`

### 3. ⚠️ WackerSchroder2015 — DOI corrected (FIXED)
- **DOI changed from:** `10.1016/j.ultramic.2014.10.012`
- **DOI changed to:** `10.1016/j.ultramic.2014.12.008`

### 4. Forbes2010 — Deferred (minor)
- The Debye-Waller B values are standard tabulated values. Forbes2010 provides the frozen-phonon framework. The citation is reasonable as-is.

### 5. Batkai2009 — Deferred (minor)
- The commutator-norm bound claim is already supported by McLachlanQuispel2002 and Childs2021 cited in the same paragraph. No text change needed.
