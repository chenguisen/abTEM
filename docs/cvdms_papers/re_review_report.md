# Verification Review Report (Re-Review)

**Manuscript:** Commutator-resolved multislice: direct operator exponentiation reveals when projected-potential electron scattering is physically controlled
**Target Journal:** Communications Physics (Nature Portfolio)
**Review Date:** 2026-05-17
**Review Type:** Re-Review (Verification Review, Pipeline Stage 3')

---

## Decision: **ACCEPT** (conditional on figure compositing)

The authors have fully addressed all Priority 1 (required) and Priority 2 (recommended) revision items. The revised manuscript is substantially improved: the novelty claims are appropriately qualified, the phase diagram is honestly reframed as Z-dominated convergence, the Au I/I₀ anomaly is quantitatively explained, negative controls are provided, and the methods section now includes NCC justification and thermal averaging scope clarification. No new issues of concern were introduced by the revisions.

**Condition:** Composite figures (P3-15) remain deferred to final submission. This is acceptable for acceptance as figure *content* is final; only multi-panel layout needs adjustment.

---

## Revision Response Checklist

### Priority 1 — Required Revisions

| # | Original Review Comment | Author's Claim | Response Status | Revision Location | Verified? | Quality Assessment |
|---|------------------------|---------------|-----------------|-------------------|-----------|-------------------|
| P1-1 | DA-C1: Novelty inflation — "remains invisible during standard multislice runs" is misleading | Changed to "has never been resolved as a per-pixel field during a single multislice run" | FULLY_ADDRESSED | Abstract (L69), §1.1 (L90) | ✅ Yes | Thoroughly applied. Abstract, §1.1, and §3.2 all use the qualified language. The distinction between "integrated effect observable via Δz comparison" and "per-pixel spatial distribution" is now clear. |
| P1-2 | DA-M1: (ρ,η) phase diagram is a null result presented as a finding | Reframed as "Z-dominated convergence in dimensionless control variables" | FULLY_ADDRESSED | §2.4 title (L227), Table 3 caption (L237), Fig 5 caption (L264), Abstract (L69) | ✅ Yes | Honest reframing. Text explicitly acknowledges the 1D nature at t=200 Å and notes conditions under which a genuine 2D diagram might emerge. The intellectual honesty strengthens the paper. |
| P1-3 | DA-M2: Au I/I₀=0.824 under-explained; 18% loss >> 1.4% C8 prediction | Quantitative explanation: σV scaling of per-slice float32 error; float64 recommended | FULLY_ADDRESSED | §2.6.2 (L379), §3.4 item 5 (L475) | ✅ Yes | The explanation is physically grounded: Au σV_rms≈0.17 Å⁻¹ vs SrTiO₃ 0.025 Å⁻¹ means larger-amplitude K-series partial waves → larger round-off. Per-slice error ∝ σV × n_slices, making the floor material-dependent. float64 recommendation is actionable. |
| P1-4 | DA-M3: Missing negative controls for the three diagnostics | Added negative controls paragraph: vacuum (V=0) → all diagnostics zero; weak-phase → r≪10⁻⁶; SrTiO₃ → r~0.1–1 on columns | FULLY_ADDRESSED | §2.1 (L181) | ✅ Yes | Well-executed. The three-tier verification (vacuum/weak-phase/SrTiO₃) cleanly establishes that the diagnostic signal originates from the commutator, not numerical noise. The contrast between r≪10⁻⁶ (weak-phase) and r~0.1–1 (SrTiO₃) is particularly convincing. |

### Priority 2 — Recommended Revisions

| # | Original Review Comment | Author's Claim | Response Status | Revision Location | Verified? | Quality Assessment |
|---|------------------------|---------------|-----------------|-------------------|-----------|-------------------|
| P2-5 | R2-1: Table 1 w_col values inconsistent with P1 JSON data | Corrected against P1 JSON: SrTiO₃ 0.48 Å, Si 1.19 Å, Au 0.65 Å; recalculated derived quantities | FULLY_ADDRESSED | Table 1 (L126) | ✅ Yes | Values match P1 JSON exactly. ρ and η recalculated accordingly. Physical conclusions unchanged but now quantitatively correct. |
| P2-6 | R3-1: Missing modern operator-splitting references | Added Suzuki1990, Childs2021, HochbruckOstermann2010 | FULLY_ADDRESSED | §1.2 (L90), §3.2 (L449–452), cvdms_references.bib (L246–268) | ✅ Yes | Three new references with DOIs, integrated into the splitting narrative in §1.2 and connected to the BCH discussion in §3.2. Bibliographic entries are complete and correct. |
| P2-7 | R2-2: Si crystal orientation inconsistency | Verified: Si [001] already used consistently throughout; no change needed | FULLY_ADDRESSED | Throughout | ✅ Yes | Independent verification confirms Si [001] is used in all locations (Table 1, §2.4, §2.5, §4.8). No [110] references remain. |
| P2-8 | EIC-2: Abstract >150 words | Tightened to ~147 words | FULLY_ADDRESSED | Abstract (L68–70) | ✅ Yes | Word count verified: ~147 words. All quantitative claims preserved. Concise but complete. |
| P2-9 | R1-1: NCC as primary metric needs justification | Added justification in §4.9: NCC chosen for simultaneous amplitude+phase sensitivity; phase RMS and I/I₀ catch missed error modes | FULLY_ADDRESSED | §4.9 (L598) | ✅ Yes | Justification is clear and honest about NCC limitations (global phase, uniform scaling). Companion metrics explicitly named. |
| P2-10 | R1-2: Frozen-phonon N≥8 scope unclear | Clarified: all C/P-series use single-configuration V^(B); N≥8 refers to protocol for future experimental comparisons | FULLY_ADDRESSED | §4.9 (L598) | ✅ Yes | Distinction between "static Debye-Waller broadened V^(B) used in this paper" vs "N≥8 for experimental comparisons" is unambiguous. Cross-reference to Sections 2.2–2.7. |

### Priority 3 — Optional Revisions

| # | Original Review Comment | Response Status | Verified? | Notes |
|---|------------------------|-----------------|-----------|-------|
| P3-11 | R3-3: Add BCH formula | FULLY_ADDRESSED | ✅ Yes | §3.2 Eq. (10). Full BCH expansion with commutator identification. Strengthens mathematical narrative. |
| P3-12 | R1-5: Timing data missing | DEFERRED | ⚠️ Deferred | Authors acknowledge gap; retained §4.9 protocol description for future benchmarking paper. Acceptable. |
| P3-13 | R2-4: DP NCC computation details | DEFERRED | ⚠️ Deferred | Standard computation (linear-intensity, no thresholding). Acceptable omission. |
| P3-14 | DA-m3: "So what?" paragraph | PARTIALLY_ADDRESSED | ⚠️ Partial | Adaptive slicing concept mentioned in §3.1 and §3.4, two-sided optimal Δz window in §2.7.2. Sufficient for acceptance. |
| P3-15 | EIC-1: Composite figures | DEFERRED | ⚠️ Deferred | Figure content is final; compositing needed before publication. Does not affect scientific assessment. |

---

## New Issues (Discovered During Revision)

| # | Type | Location | Description |
|---|------|----------|-------------|
| NEW-1 | Minor | §4.9 (L598) | The §4.9 paragraph is now substantially longer (~8 lines) than surrounding Methods subsections. This is a minor stylistic concern — the content is all necessary — but the authors may wish to split into two paragraphs (metrics + statistical protocol). |
| NEW-2 | Observation | §2.1 (L181) | The negative controls paragraph is well-placed and well-written. No issues. |
| NEW-3 | Observation | cvdms_references.bib | Three new bib entries (Suzuki1990, Childs2021, HochbruckOstermann2010) are complete with correct DOIs. LaTeX compilation confirms all 28 citations resolve. |

**No blocking issues introduced by the revisions.**

---

## Decision Rationale

The revision is thorough and honest. The authors did not merely make the minimum required changes — they engaged substantively with each reviewer concern:

1. **Novelty qualification (P1-1):** The distinction between "integrated commutator effect measurable via Δz comparison" and "per-pixel spatial distribution during a single run" is now precise throughout. This transforms a potential overclaim into an accurate, defensible novelty statement.

2. **Phase diagram reframing (P1-2):** Presenting a null result (1D Z-axis rather than 2D phase diagram) as an honest finding rather than overselling it is scientifically mature. The explicit statement about when a genuine 2D diagram *might* emerge shows the authors understand their result's limitations.

3. **Au anomaly (P1-3):** The quantitative σV-scaling argument for material-dependent float32 floor is the strongest possible response — it turns a reviewer concern into an additional physical insight.

4. **Negative controls (P1-4):** The three-tier verification (vacuum/weak-phase/SrTiO₃) is methodologically rigorous and should have been in the original submission.

5. **NCC justification + thermal averaging (P2-9, P2-10):** These clarifications in §4.9 preempt future reader questions and improve reproducibility.

All P1 items are FULLY_ADDRESSED. All P2 items are FULLY_ADDRESSED. Of the five P3 items, one (BCH formula) is fully addressed, one ("So what?") is partially addressed, and three (timing, DP NCC details, composite figures) are appropriately deferred. Per the re-review protocol, P3 items do not affect the decision.

**Decision: ACCEPT**, conditional on figure compositing (P3-15) before final submission. No further scientific review is required.

---

## Residual Issues

None. All P1 and P2 items are fully resolved. The three deferred P3 items are:
- **Timing data (P3-12):** Deferred to follow-up engineering paper. The §4.9 protocol description serves as a methods statement for that future work.
- **DP NCC computation details (P3-13):** Standard computation; omission is acceptable for this paper's scope.
- **Composite figures (P3-15):** Required before publication but does not affect scientific assessment. Content is final.

---

## Comparison: Round 1 vs Round 2 Scores

| Dimension | Round 1 (EIC) | Round 2 (Re-review) | Change |
|-----------|---------------|---------------------|--------|
| Originality | 8/10 | 8/10 | — |
| Significance | 7/10 | 7/10 | — |
| Evidence strength | 9/10 | 9/10 | — |
| Presentation | 6/10 | 8/10 | +2 (novelty qualified, abstract tightened, methods clarified) |
| Journal fit | 8/10 | 8/10 | — |
| **Overall** | **78/100** | **82/100** | **+4** |

The presentation improvement reflects the cumulative effect of: qualified novelty language, honest phase diagram reframing, negative controls, BCH formula, NCC justification, thermal averaging clarification, and modern splitting literature integration.
