# Peer Review Report

**Manuscript:** Commutator-resolved multislice: direct operator exponentiation reveals when projected-potential electron scattering is physically controlled
**Target Journal:** Communications Physics (Nature Portfolio)
**Review Date:** 2026-05-17
**Review Type:** Full (5-reviewer panel)

---

## Reviewer 1 — EIC (Editor-in-Chief, Communications Physics)

### Summary

This manuscript presents a computational method (CVDMS) that directly exponentiates the projected-slice scattering operator $\hat{K}=\sigma V+\nabla_\perp^2/(4\pi K_0)$ via a Taylor series with per-pixel convergence tracking, exposing the commutator $[\sigma V,\nabla_\perp^2/(4\pi K_0)]$ that Lie-Trotter splitting silently discards. The method is validated across three crystal classes (SrTiO₃, Si, Au) at 30-300 keV against analytic limits, Δz-refined Fourier multislice, and a Bloch-wave reference. Three per-pixel diagnostics (divergence ratio, stagnation count, BSC unitarity residual) are introduced as physically meaningful probes of the commutator.

### Strengths

1. **Clear physical narrative.** The paper structures itself around a single physical problem — the discarded commutator in Lie-Trotter splitting — and builds every section toward making it measurable. This is rare in computational methods papers and aligns well with Communications Physics' editorial bar of "significant advances providing new insight."

2. **Systematic validation architecture.** The Phase 0/0.5/C-series protocol (vacuum→analytic→Δz→ε→material sweeps→phase diagram→precision floor) is methodologically rigorous and logically progressive. Each validation tier builds on the previous.

3. **Honest reporting of limitations.** The paper explicitly states: the phase diagram collapses to a Z-axis (not a rich 2D map), float32 diffusion places a lower bound on Δz, Au I/I₀ never reaches 1.0 even at ε=10⁻⁹. This transparency strengthens credibility.

4. **Practical utility.** The ε≤10⁻⁷ universal threshold and the two-sided optimal Δz window (0.6-0.7 Å for SrTiO₃ at 30 keV) are immediately actionable for TEM practitioners.

### Weaknesses

1. **Figure quality is pre-submission.** The current figures are individual PDF panels from data-generation scripts; they need compositing into multi-panel journal figures. More critically, several figures referenced in the text (Figs. 1d, 2b) refer to panels from the V-series scripts that were generated at different grid resolutions and visual styles. The figures need a unified visual language before submission.

2. **Abstract length.** The current abstract is ~190 words. Communications Physics specifies ≤150 words. Needs tightening by ~40 words.

3. **Author affiliations and metadata are incomplete.** Placeholder text ("Affiliation 1, to be completed") remains.

4. **The paper is self-contained but somewhat insular.** While the Chen-Van Dyck tradition is well-cited, the paper could benefit from one paragraph connecting to the broader scientific computing literature on operator splitting (e.g., quantum simulation with Trotter errors, geometric integration).

### Decision Recommendation

**Minor Revision.** The core physics and validation are sound. The required changes are presentational (figures, abstract length, metadata) and one modest content addition (broader splitting context). I do not require new calculations.

**Score:** 78/100
- Originality: 8/10 — First per-pixel measurement framework for this commutator
- Significance: 7/10 — Actionable for TEM community; niche outside
- Evidence strength: 9/10 — Multi-tier validation
- Presentation: 6/10 — Figures and metadata incomplete
- Journal fit: 8/10 — Comms Phys appropriate

---

## Reviewer 2 — R1 (Methodology Reviewer)

### Expertise
Numerical methods for computational physics; convergence analysis; floating-point error analysis; reproducible computational science.

### Summary

This paper presents a direct Taylor exponentiation method for the multislice electron scattering operator, with per-pixel diagnostics replacing global convergence criteria. My review focuses on numerical methodology, error analysis, and reproducibility.

### Strengths

1. **Three-pathway convergence verification (§2.2).** Demonstrating convergence through independent variables (Δz, ε, backend) is methodologically sound. The O(Δz²) scaling confirmation is particularly valuable — it proves the method converges to the correct limit with the expected rate.

2. **Float32 precision floor analysis (§2.7.2, C8).** The C8 study is exemplary: 6 Δz × 5 thicknesses = 30 systematically varied points, all showing I/I₀ < 1 (diffusive, not amplificative). The collapse of 1−I/I₀ against n_slices independent of Δz is a strong confirmatory signal. The derived α ≈ 2.7×10⁻⁵ per slice provides a quantitative bound. This is the kind of careful numerical analysis that most computational TEM papers lack.

3. **Bandwidth inheritance argument (§2.7.1).** The explanation of $\widehat{V\psi}=\hat{V}*\hat{\psi}$ doubling bandwidth, combined with the $q^{-2}$ asymptotic of Lobato form factors making antialiasing physically necessary, is logically coherent and well-supported by the C4 data (AA=OFF → immediate overflow).

4. **Scaled cascade implementation (Eq. 7).** The $\text{cur}_n = c_n \hat{K}(\text{cur}_{n-1})$ formulation exploiting $\hat{K}$'s linearity to propagate coefficient damping is a genuinely clever numerical insight.

### Weaknesses

1. **NCC as the primary metric needs justification.** The paper uses NCC (normalized cross-correlation) throughout as the primary accuracy metric. While NCC is standard in electron microscopy, it is known to be insensitive to certain classes of error (global phase offsets, uniform amplitude scaling). The paper partially addresses this by also reporting phase RMS and I/I₀, but Section 2.2's convergence claims rely heavily on NCC. **Recommendation:** Add one sentence in Methods §4.9 explaining why NCC was chosen and what error modes it may miss.

2. **Frozen-phonon ensemble size.** §4.9 states N≥8 for frozen-phonon ensembles in uncertainty reporting, but the actual paper reports single-configuration results (no temperature averaging) in all tables. If frozen-phonon averaging was not used in the C-series sweeps, this should be stated explicitly. The Debye-Waller broadened potential V^(B) is static — clarify what N≥8 refers to.

3. **ε sweep range justification.** The ε sweep spans 10⁻⁴ to 10⁻⁹, which is appropriate, but the choice of 10⁻⁴ as the upper bound is not justified. Is 10⁻⁴ the point where the method catastrophically fails for Si (I/I₀=168), or was it chosen as a round number? A brief justification would strengthen the experimental design narrative.

4. **GPU float32 behavior.** The paper states complex64 is the default precision and that C8 quantifies the float32 floor, but does not discuss whether the C++/CUDA path uses the same floating-point semantics as CuPy. CUDA's fused multiply-add can produce slightly different results from Python's separate multiply-then-add. Was cross-backend NCC measured at matching iteration counts or at matching observables?

5. **Statistical reporting of timing.** §4.9 promises "5 repeated runs, mean ± std" for wall-clock timing, but no timing data appears in the manuscript. Either add a timing table or remove this promise from Methods.

### Recommendation

**Minor Revision.** The numerical methodology is among the strongest I've seen in computational TEM papers. The requested clarifications are minor and do not require re-running any simulations.

**Score:** 82/100
- Numerical rigor: 9/10
- Error analysis: 8/10
- Reproducibility: 7/10 — Scripts referenced but not yet deposited
- Clarity of methods: 8/10

---

## Reviewer 3 — R2 (Domain Reviewer)

### Expertise
Transmission electron microscopy simulation; multislice methods; abTEM/MULTEM/prismatic development; channeling physics; quantitative CBED.

### Summary

This manuscript introduces per-pixel diagnostics that expose the otherwise-hidden commutator error in multislice simulations. As someone who has spent considerable time debugging multislice convergence by reducing Δz and re-running, I find the concept of in-situ, pixel-resolved convergence maps genuinely appealing.

### Strengths

1. **Channeling Pendellösung as commutator probe (§2.5.1).** Using the axial channeling period as a physical observable that discriminates between CVDMS and Fourier multislice is clever. The data in Table 4 are compelling: Δξ=0 at Δz≤0.4 Å, Δξ=3.2 Å at 0.8 Å. That ACF amplitude systematically increases in CVDMS (0.335→0.421) while remaining constant in Fourier (~0.28) is a clean signal.

2. **Cross-material Z-scaling (Table 5).** The extension to Si and Au transforms the channeling result from a single-material anecdote into a Z-scaling law: commutator signal ∝ Z/v. This is physically expected from [σV, ∇²/(4πK₀)] ∝ σV ∝ Z but had not been quantitatively demonstrated before.

3. **HOLZ contrast ratio.** The finding that FOLZ ring position is preserved (geometric) while ring contrast differs by 1.55-1.80× between CVDMS and Fourier is physically insightful. It cleanly separates geometric from dynamical HOLZ effects.

4. **Phase diagram Z-collapse (§2.4).** The result that the (ρ,η) phase diagram reduces to a Z-axis at t=200 Å is initially disappointing (no rich 2D phase structure) but is reported honestly and interpreted correctly: η's dynamic range is too narrow at these parameters. The paper deserves credit for reporting a "negative" structural result without trying to oversell it.

5. **Connection to experimental observables.** The repeated emphasis on CBED disk intensities I_g(t) as the "primary experimentally measurable observables" (§2.6, §3.4) grounds the method in experiment rather than letting it float as pure numerics.

### Weaknesses

1. **Table 1 — w_col values need verification.** The column 1/e widths in Table 1 appear inconsistent: SrTiO₃ w_col=0.50 Å (Sr column) is reasonable, but Si w_col=0.55 Å contradicts the dumbbell structure of Si [001] where columns are separated by 1.36 Å with each column having w_col ≈ 0.7-0.8 Å at 300 K with Debye-Waller broadening. Please verify against P1 JSON data (which reports w_col=1.19 Å for Si at 30 keV, 0.48 Å for SrTiO₃ at 30 keV).

2. **Si crystal orientation inconsistency.** The Introduction references Si [110] in the potential field discussion but the Results and C-series data all use Si [001]. The notation should be unified to [001] throughout.

3. **Channeling ξ interpretation.** The measured ξ ≈ 54.4 Å for SrTiO₃ is reported alongside ξ_bulk = 65 Å. The ~17% difference is attributed to "physical column potential V_eff ≈ 80 eV·Å vs average V_g = 31.4 eV·Å" (from the plan file). This physical interpretation is interesting but needs explicit presentation in the paper — it currently appears only in the plan file, not the manuscript.

4. **Missing discussion of detector effects.** The statement that DP NCC > 0.97 throughout the thickness sweep (§2.6.1) is valuable, but NCC of diffraction patterns depends on the dynamic range of the detector. A brief note on how DP NCC was computed (log-scale? thresholded?) would aid reproducibility.

5. **Literature gap: Lobato & Van Dyck 2014.** The paper relies heavily on Lobato-Van Dyck scattering factors but does not discuss their accuracy relative to other parameterizations (e.g., Doyle-Turner, Weickenmeier-Kohl, Peng). A one-sentence note would suffice.

### Recommendation

**Minor Revision.** The physical content is strong and the validation is thorough. The requested clarifications concern presentation and documentation, not new calculations.

**Score:** 80/100
- Physical insight: 9/10
- Literature coverage: 7/10
- Experimental relevance: 7/10
- Data quality: 9/10
- Clarity of physical claims: 8/10

---

## Reviewer 4 — R3 (Perspective Reviewer)

### Expertise
Quantum dynamics; operator splitting methods; Lie-Trotter/Suzuki-Trotter decompositions; geometric numerical integration; path-integral methods.

### Summary

This manuscript approaches the multislice electron scattering problem from the perspective of operator exponentiation, framing the conventional Fourier multislice as a Lie-Trotter splitting and the proposed CVDMS as direct Taylor exponentiation. My review examines the mathematical framing through the lens of the broader operator-splitting and quantum simulation literature.

### Strengths

1. **Correct identification of the commutator.** The commutator $[\sigma V,\nabla_\perp^2/(4\pi K_0)] \propto \nabla_\perp V\cdot\nabla_\perp + \frac{1}{2}\nabla_\perp^2 V$ is correctly derived, and its spatial localization at atomic columns is a genuine physical insight that is not obvious from the algebraic form alone.

2. **Historical grounding.** Citing Ishizuka (1982), Trotter (1959), Strang (1968), and McLachlan & Quispel (2002) situates the commutator problem within 70 years of mathematical understanding, which strengthens the claim that "no one has made it locally measurable."

3. **Coefficient-scaled cascade (Eq. 7).** The $\text{cur}_n = c_n \hat{K}(\text{cur}_{n-1})$ construction is essentially a Horner-like scheme for operator polynomials that exploits the linearity of $\hat{K}$. This is numerically sound and could be of interest to the broader quantum simulation community where similar exponentiation problems arise (e.g., exponential integrators for the Schrödinger equation).

4. **Honest positioning relative to splitting.** The paper explicitly states CVDMS is "a different controlled approximation, not a replacement for splitting or Bloch-wave methods" (§3.3). This intellectual honesty is commendable and uncommon in methods papers.

### Weaknesses

1. **Missing connection to modern splitting literature.** The paper cites Strang (1968) and McLachlan & Quispel (2002) but does not engage with more recent developments: higher-order Suzuki-Trotter decompositions, randomized Trotter schemes (Childs et al., 2019), or the extensive quantum simulation literature on Trotter error bounds. **Suggest adding 2-3 references** connecting to: (a) Suzuki's higher-order decompositions (Suzuki, Phys. Lett. A, 1990); (b) the modern theory of Trotter error in quantum simulation (Childs et al., PRX 2021); (c) exponential integrators for stiff PDEs (Hochbruck & Ostermann, Acta Numerica, 2010).

2. **The "convergence radius" argument needs tightening.** §2.4 states ρ_c ≈ 0.12 is "consistent with the analytic Taylor convergence radius |σVΔz| < π after c_n damping." This is too hand-wavy for a claim that anchors the phase boundary. Either provide a brief derivation in the Supplementary Information or soften the claim to "empirically consistent with."

3. **No discussion of Baker-Campbell-Hausdorff.** The BCH formula $\exp(A)\exp(B) = \exp(A+B+\frac{1}{2}[A,B]+\frac{1}{12}[A,[A,B]]-\frac{1}{12}[B,[A,B]]+\cdots)$ is the fundamental object behind splitting error. The commutator the paper studies is precisely the leading BCH term beyond A+B. A one-equation mention would strengthen the mathematical narrative.

4. **Comparison to Bloch-wave methods is superficial.** §3.3 dismisses Bloch-wave methods as "require periodic potentials and become computationally expensive at HRTEM-scale beam counts." This is true but incomplete. A more nuanced comparison would note that Bloch-wave methods naturally handle the commutator exactly (they diagonalize the full scattering matrix) — the trade-off is between exact diagonalization in a truncated basis vs. real-space exponentiation with per-pixel diagnostics.

5. **The originality claim needs one qualification.** The paper claims "first per-pixel measurement framework for the commutator." While the per-pixel aspect is novel, the concept of tracking multislice convergence via residual norms has precedent in adaptive step-size control for ODE integrators. A brief acknowledgment would strengthen rather than weaken the novelty claim by better defining its boundaries.

### Recommendation

**Minor Revision.** The mathematical framing is fundamentally sound and the physical insights are genuine. The requested additions would broaden the paper's appeal beyond the TEM community without requiring new calculations.

**Score:** 75/100
- Mathematical soundness: 8/10
- Literature engagement (splitting): 5/10 — Needs modern splitting references
- Cross-disciplinary appeal: 6/10
- Novelty positioning: 7/10

---

## Reviewer 5 — DA (Devil's Advocate)

### Mandate
Challenge core arguments, identify the strongest counter-arguments, detect logical fallacies, identify alternative explanations, and test whether claims survive skeptical scrutiny.

### Strongest Counter-Argument

**"The commutator was never missing — it was always measurable by reducing Δz."**

The paper's central claim is that CVDMS makes the commutator "locally measurable for the first time." But the standard practice in the TEM community — reducing Δz until results converge — *is* a measurement of the commutator's effect. When a user runs Fourier multislice at Δz=0.8 Å and then at Δz=0.4 Å and observes different channeling periods, they have measured the commutator's integrated effect. The CVDMS innovation is making this measurement *per-pixel*, *in a single run*, and *before the simulation completes* — not making it measurable *at all*. The paper needs to sharpen this distinction; otherwise, a skeptical reader will dismiss the central claim as semantic inflation.

**Recommendation:** Change "never been made locally measurable" to "never been resolved as a per-pixel field during a single multislice run" throughout.

### Issue List

**CRITICAL**

1. **C1 — Novelty inflation in the abstract.** The abstract claims the commutator "remains invisible during standard multislice runs." This is misleading: the commutator's *effect* on observables (channeling, HOLZ) is visible to anyone who varies Δz. What's invisible is its *spatial distribution*. The distinction matters because overclaiming in the abstract invites rejection at the editor's desk.

**MAJOR**

2. **M1 — The (ρ,η) phase diagram is a null result presented as a finding.** §2.4's central result is that the 2D phase diagram collapses to a 1D Z-axis because η's dynamic range is too narrow. This is interesting as a physical observation, but presenting it as a "phase diagram" implies a richer structure than actually exists. A skeptical reader would say: "You set out to find a 2D phase diagram, found a 1D line, and still called it a phase diagram." Consider reframing as "Z-dominated convergence at t=200 Å" rather than "(ρ,η) phase diagram."

3. **M2 — The Au I/I₀ = 0.824 result is under-explained.** §2.6.2 states Au I/I₀ = 0.824 at ε=10⁻⁹ is "the float32 precision floor." But 18% flux loss is large — much larger than the per-slice ε ≈ 2.7×10⁻⁵ measured in C8 for SrTiO₃. Back-of-envelope: 500 slices × 2.7×10⁻⁵ = 1.35% loss, not 18%. The Au loss mechanism is therefore different from (or additional to) the SrTiO₃ C8 mechanism. A skeptical reader would ask: Is this really float32 diffusion, or is it an unconverged BSC correction, or an incorrect potential parameterization? This needs direct investigation.

4. **M3 — Missing negative controls for the three diagnostics.** The paper presents r(R), s(R), and U(R) as "commutator probes" but never demonstrates what these diagnostics look like for a simulation that is *known* to have zero commutator error (e.g., vacuum propagation, or the analytic weak-phase limit). Without such negative controls, the reader cannot distinguish "commutator signal" from "generic numerical noise."

**MINOR**

5. **m1 — The "catastrophic under-convergence" narrative.** §2.6.2 describes Si I/I₀=168 at ε=10⁻⁴ as "catastrophic" — but ε=10⁻⁴ is absurdly loose (only 4 terms). No practitioner would use this. The finding is correct but the dramatic framing weakens credibility.

6. **m2 — Cherry-picking of test systems.** The three materials (Si, SrTiO₃, Au) span Z=14 to Z=79 but all have cubic symmetry. Would the commutator behave differently in anisotropic materials (hexagonal, monoclinic) where ∇V has directional structure? This is an unstated limitation.

7. **m3 — "So what?" test.** For the average TEM practitioner, the actionable output is: "use Δz=0.4 Å and ε=10⁻⁷, which you're already doing." The paper needs to articulate more clearly what *new decision* a user would make based on the diagnostics that they wouldn't make based on existing rules of thumb.

### Ignored Alternative Explanations

1. **The ACF amplitude increase in CVDMS at coarse Δz (Table 4) could be a numerical artifact** — float32 accumulation producing spurious correlations — rather than genuine preservation of channeling coherence. The paper attributes it to physics but does not rule out a numerical origin.

2. **The Z-ordering of ε-convergence (Au > SrTiO₃ > Si, Table 3) is attributed to series convergence.** An alternative explanation: heavier elements have larger σV, which means the antialiasing aperture (2/3 Nyquist) removes a larger fraction of the scattering power, effectively making the problem easier. The paper does not discuss this confound.

### Missing Stakeholder Perspectives

- **Experimental TEM users** who want to know: "Does this change my interpretation of existing data, or only affect future simulations?"
- **Code maintainers** (abTEM, MULTEM) who would need to integrate CVDMS diagnostics as optional outputs.

### Observations (Non-Defects)

- The paper's intellectual honesty (reporting negative results, quantitative limitations) is refreshing.
- The C8 float32 analysis is the strongest single section and deserves to be a standalone contribution.
- The connection between the commutator's spatial structure (∇V·∇ + ½∇²V) and channeling localization is physically deep and well-articulated.

### DA Verdict

The core physical claim — that direct exponentiation preserves a commutator that splitting discards, and this has measurable consequences — **survives skeptical scrutiny**. However, the novelty framing needs tightening (CRITICAL C1), the Au anomaly needs investigation (MAJOR M2), and the phase diagram should be reframed as a Z-ordering result rather than a 2D map (MAJOR M1).

---

# Editorial Synthesis

## Cross-Reviewer Consensus Matrix

| Issue | EIC | R1 | R2 | R3 | DA | Consensus |
|-------|-----|----|----|----|-----|-----------|
| Figures need compositing | ✓ | — | — | — | — | **Agreed (presentational)** |
| Abstract >150 words | ✓ | — | — | — | — | **Agreed (presentational)** |
| Novelty framing too strong | — | — | — | ✓ | ✓ | **Agreed (MAJOR)** |
| Au I/I₀=0.824 unexplained | — | — | — | — | ✓ | **Needs response** |
| Phase diagram as null result | — | — | — | — | ✓ | **Needs reframing** |
| Missing negative controls | — | — | — | — | ✓ | **Needs response** |
| Modern splitting references | — | — | — | ✓ | — | **Agreed (MINOR)** |
| w_col values in Table 1 | — | — | ✓ | — | — | **Needs verification** |
| Frozen-phonon N≥8 | — | ✓ | — | — | — | **Needs clarification** |
| BCH formula mention | — | — | — | ✓ | — | **Optional** |
| Timing data missing | — | ✓ | — | — | — | **Agreed (MINOR)** |

## Failure Condition Evaluation

| Condition | Severity | Triggered? | Evidence |
|-----------|----------|------------|----------|
| F1: Critical logical flaw | CRITICAL | **NO** | DA finds core claim survives scrutiny |
| F2: Missing essential validation | CRITICAL | **NO** | Validated across 3 materials × 3 energies |
| F3: Unsupported central claim | MAJOR | **YES** | Novelty framing (C1) needs qualification |
| F4: Incomplete methodology | MAJOR | **YES** | Negative controls missing (M3) |
| F5: Misleading data presentation | MAJOR | **BORDERLINE** | Phase diagram framing (M1), Au anomaly (M2) |
| F6: Insufficient literature | MINOR | **YES** | Modern splitting literature missing |

## Editorial Decision: **MINOR REVISION**

The paper presents a physically insightful, systematically validated computational method. The five reviewers converge on the scientific soundness of the core claims. No reviewer recommends Reject or Major Revision. The required changes are:

### Priority 1 (Required before acceptance)

1. **Sharpen novelty claims.** Replace "never been made locally measurable" → "never been resolved as a per-pixel field during a single multislice run." (DA-C1, R3)

2. **Reframe the phase diagram.** Present as "Z-dominated convergence at t=200 Å" rather than as a 2D (ρ,η) phase diagram. Acknowledge that η's narrow dynamic range prevents observing 2D phase structure at currently valid parameters. (DA-M1)

3. **Address the Au I/I₀ = 0.824 anomaly.** Either (a) provide a quantitative explanation for why Au's 18% loss exceeds the C8 per-slice ε prediction, or (b) add a caveat that the Au float32 floor mechanism may differ from SrTiO₃'s and requires further study. (DA-M2)

4. **Add or discuss negative controls.** Show (or explain) what r(R), s(R), U(R) produce for a known-zero-commutator case (vacuum, weak-phase limit). (DA-M3)

### Priority 2 (Recommended)

5. **Verify and correct Table 1 w_col values** against P1 JSON data. (R2-1)
6. **Add 2-3 modern operator-splitting references** (Suzuki, Childs et al., Hochbruck & Ostermann). (R3-1)
7. **Unify Si crystal orientation** to [001] throughout. (R2-2)
8. **Tighten abstract to ≤150 words.** (EIC-2)
9. **Add brief NCC justification** in Methods §4.9. (R1-1)
10. **Clarify frozen-phonon N≥8** — are C-series results thermal averages or single-configuration? (R1-2)

### Priority 3 (Optional)

11. Add BCH formula mention for mathematical narrative. (R3-3)
12. Add timing data or remove the promise from §4.9. (R1-5)
13. Discuss detector dynamic range effects on DP NCC computation. (R2-4)
14. Add "So what?" paragraph for the average TEM practitioner. (DA-m3)
15. Composite figures into unified multi-panel format. (EIC-1)

## Revision Roadmap

| # | Action | Priority | Effort | Section |
|---|--------|----------|--------|---------|
| 1 | Qualify novelty language throughout | P1 | Text edit | Abstract, §1.1, §1.2, §3.2 |
| 2 | Reframe phase diagram as Z-dominated | P1 | Text edit | §2.4, Abstract |
| 3 | Explain or caveat Au I/I₀=0.824 | P1 | Analysis + text | §2.6.2 |
| 4 | Add negative control discussion | P1 | Text addition | §2.1 or §2.3 |
| 5 | Correct w_col in Table 1 | P2 | Data verification | Table 1, §1.4 |
| 6 | Add modern splitting references | P2 | Lit search + 1 para | §1.2, §3.2 |
| 7 | Fix Si orientation notation | P2 | Text edit | §1.4, Table 1 |
| 8 | Shorten abstract to ≤150 words | P2 | Text edit | Abstract |
| 9 | Add NCC justification | P2 | 1 sentence | §4.9 |
| 10 | Clarify thermal averaging scope | P2 | 2 sentences | §4.9 |
| 11 | BCH formula mention | P3 | 1 equation + 1 sentence | §1.2 or §3.2 |
| 12 | Add timing data | P3 | Table | New or §4.7 |
| 13 | DP NCC computation details | P3 | 1 sentence | §2.6 or §4.9 |
| 14 | "So what?" paragraph | P3 | 3-4 sentences | §3.4 |
| 15 | Composite figures | P3 | Scripting | All figures |
