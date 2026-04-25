#!/bin/bash
# Complete CVDMS visualization + commit workflow
# Run this from: /media/chenguisen/WD_BLACK/cgs/cgs/program/multem_cgs/abTEM
set -e
cd "$(dirname "$0")"
echo "=== Step 1: Generate remaining figures ==="
python _run_viz_figs.py
echo ""
echo "=== Step 2: Stage all changes ==="
git add -f abtem/cvdms.py
git add diag_cvdms_visualization.py _run_viz_figs.py _run_viz_figs.sh
git add -f docs/figures/fig_taylor_convergence.png
git add -f docs/figures/fig_critical_frequency.png
git add -f docs/figures/fig_intensity_conservation.png
git add -f docs/cvdms_divergence_analysis.md
# Add remaining figures if generated
for f in docs/figures/fig_cbed_log.png docs/figures/fig_cbed_linear.png \
         docs/figures/fig_cbed_side_by_side.png \
         docs/figures/fig_thick_sample_stress.png; do
    [ -f "$f" ] && git add -f "$f"
done
echo ""
echo "=== Step 3: Commit ==="
git commit -m "$(cat <<'COMMIT_MSG'
feat: CVDMS 诊断回传、可视化与CBED正确性验证

### 新增：`_cvdms_forward_scattering` 诊断回传
- `return_diagnostics=True` 返回收敛历史：
  n_terms_used, ratios_per_order, n_above_per_order,
  overflow_detected, divergence_truncated, max_amplitude
- 默认行为不变（向后兼容）

### 新增：`diag_cvdms_visualization.py`
- 7 张出版级可视化图表：
  Fig 1-2: CBED 对比网格 (log/linear)
  Fig 3:   CVDMS vs Fourier 并排 CBED 对比
  Fig 4:   泰勒级数收敛 (term amplitude vs order)
  Fig 5:   临界频率热力图 (k_critical vs Nyquist)
  Fig 6:   强度守恒 vs 厚度
  Fig 7:   50nm 厚样品压力测试
- 英文字体 14-16pt, 300 DPI, 科学配色

### 更新：`docs/cvdms_divergence_analysis.md`
- 新增 Section 7: 可视化分析
  - 泰勒收敛行为的物理解读
  - 临界频率与 Nyquist 比值解释
  - 强度守恒对比分析
  - CBED 对比说明
  - 厚样品压力测试说明
- 更新已完成修改清单

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
COMMIT_MSG
)"
echo ""
echo "=== Done ==="
git log --oneline -3
