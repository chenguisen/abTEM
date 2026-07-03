#!/usr/bin/env bash
# Composite individual PDF panels into multi-panel journal figures.
# Uses pdfjam (TeX Live) to preserve vector quality.
# Run from: docs/cvdms_papers/figures/

set -e
cd "$(dirname "$0")"

echo "=== Compositing multi-panel figures ==="

# Fig 1: Self-tests (2×2 grid)
echo "  Fig 1: self_tests (2×2)"
pdfjam v1a_vacuum.pdf v1b_unitarity.pdf v1c_bsc_residual.pdf v2_divergence.pdf \
    --nup 2x2 --delta "0.05cm 0.05cm" --scale 0.97 \
    --outfile fig1_self_tests.pdf

# Fig 2: Convergence (1×2)
echo "  Fig 2: convergence (1×2)"
pdfjam v3_slice_A.pdf v4_convergence_A.pdf \
    --nup 2x1 --delta "0.1cm 0cm" --scale 0.97 \
    --outfile fig2_convergence.pdf

# Fig 3: Channeling + HOLZ (1×2)
echo "  Fig 3: channeling_holz (1×2)"
pdfjam p2_channeling.pdf p3_holz.pdf \
    --nup 2x1 --delta "0.1cm 0cm" --scale 0.97 \
    --outfile fig3_channeling_holz.pdf

# Fig 5: Thickness + epsilon (1×2)
echo "  Fig 5: thickness_epsilon (1×2)"
pdfjam c2_thickness.pdf c3_epsilon.pdf \
    --nup 2x1 --delta "0.1cm 0cm" --scale 0.97 \
    --outfile fig5_thickness_epsilon.pdf

# Crop all composites to remove A4 whitespace from pdfjam
echo "  Cropping to content bounding box..."
for f in fig1_self_tests fig2_convergence fig3_channeling_holz fig5_thickness_epsilon; do
    pdfcrop --margins 5 "${f}.pdf" "${f}_cropped.pdf"
done

echo "=== Done ==="
ls -lh fig*_cropped.pdf
