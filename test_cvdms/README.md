# test_cvdms — Convergence test suite for DEV's fully_corrected + BSC

Systematic convergence validation for DEV branch's real-space multislice with
``expansion_scope="full"`` (fully-corrected operator) and backscattering.

**Methodology inspired by CVDMS branch (feat/cgs_cvdms), API uses DEV only.**

## Quick start

```bash
conda activate abtem-env
cd F:/abTEM

# All Stage A tests (fast, CI-friendly, ~1-3 min)
python -m pytest test_cvdms/ -v --tb=short

# Exclude Stage B tests (CI)
python -m pytest test_cvdms/ -v --tb=short -k "not stage_b"

# Run standalone diagnostic scan
python test_cvdms/run_convergence.py --mode all --materials STO,Au --orders 1,2,4,6
```

## Test structure

| File | Focus | Key questions |
|---|---|---|
| `test_forward_convergence.py` | Forward scattering | Does NCC approach 1 as order increases? Does RMSD decrease monotonically? |
| `test_conservation.py` | Conservation laws | Is intensity conserved? Does vacuum propagation preserve amplitude? |
| `test_analytical_limits.py` | Analytic solutions | Vacuum Fresnel, homogeneous V, weak phase limit correctness |
| `test_bsc_convergence.py` | BSC correction | Does \|BSC\|/\|fwd\| converge? Is energy budget approximately closed? |
| `test_backpropagation.py` | Backscattered waves | Is entrance-plane BSC zero? Does intensity grow with depth? |
| `run_convergence.py` | Standalone scan | Sweep orders, materials, energies; plots + JSON output |

## Shared infrastructure

| File | Role |
|---|---|
| `metrics.py` | NCC, RMSD, max_diff, intensity_conservation, phase_rms, amplitude_rms |
| `conftest.py` | Multi-material fixtures (STO, Au, Si), Stage A/B grids, GPU marker |

## Stage A vs Stage B

- **Stage A** (default): gpts=(128,128), fast iteration. All pytest tests run here.
- **Stage B**: gpts=(512,512). Mark tests with `@pytest.mark.stage_b` for manual validation.

## Materials

| Material | Z | Scattering | Use |
|---|---|---|---|
| SrTiO₃ [001] | mixed (38/22/8) | moderate | Primary test system |
| Au [001] | 79 | strong | BSC sensitivity, convergence stress |
| Si [001] | 14 | medium | Light-element baseline |

## Energy

Default test energy is **30 keV** — most challenging regime for convergence
and BSC effects. Higher energies (100, 300 keV) converge faster and produce
smaller BSC signals.

## GPU tests

```bash
python -m pytest test_cvdms/ -v -k "gpu"  # GPU only, if available
```
