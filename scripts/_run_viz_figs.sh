#!/bin/bash
cd /media/chenguisen/WD_BLACK/cgs/cgs/program/multem_cgs/abTEM
python _run_viz_figs.py > _run_viz_figs.log 2>&1
echo "Exit: $?" >> _run_viz_figs.log
