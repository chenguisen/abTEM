#!/usr/bin/env python
"""Standalone runner for the remaining visualization figures."""
import sys, os, warnings
sys.path.insert(0, "/media/chenguisen/WD_BLACK/cgs/cgs/program/multem_cgs/abTEM")
os.chdir("/media/chenguisen/WD_BLACK/cgs/cgs/program/multem_cgs/abTEM")
warnings.filterwarnings("ignore")

from diag_cvdms_visualization import (
    fig_cbed_side_by_side, fig_cbed_grid, fig_thick_sample_stress,
)

print("=== Fig 3: CBED side-by-side ===")
try:
    fig_cbed_side_by_side()
    print("OK: fig_cbed_side_by_side.png")
except Exception as e:
    print(f"FAIL: {e}")

print("\n=== Fig 1-2: CBED grids ===")
try:
    fig_cbed_grid()
    print("OK: fig_cbed_log.png, fig_cbed_linear.png")
except Exception as e:
    print(f"FAIL: {e}")

print("\n=== Fig 7: Thick sample stress test ===")
try:
    fig_thick_sample_stress()
    print("OK: fig_thick_sample_stress.png")
except Exception as e:
    print(f"FAIL: {e}")

print("\n=== All done ===")
