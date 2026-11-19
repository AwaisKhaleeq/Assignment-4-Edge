#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end pipeline (CPU budget configuration used for the submitted results).
# Run from the repository root:   bash scripts/run_all.sh
# Every step writes its console output to results/**/logs/*_console.txt
# Increase the epoch counts on a GPU machine (see README "Full-budget run").
# ---------------------------------------------------------------------------
set -eo pipefail
mkdir -p results/q1/logs results/q2/logs

STUDENT_EPOCHS=${STUDENT_EPOCHS:-5}
TEACHER_EPOCHS=${TEACHER_EPOCHS:-2}

# ---- Q1: knowledge distillation -------------------------------------------
if [ ! -f results/q1/teacher_summary.json ]; then
  python -m q1_kd.train_teacher --epochs $TEACHER_EPOCHS --res 64 2>&1 | tee results/q1/logs/teacher_console.txt
fi
python -m q1_kd.train_baseline --epochs $STUDENT_EPOCHS 2>&1 | tee results/q1/logs/baseline_console.txt
python -m q1_kd.train_kd --epochs $STUDENT_EPOCHS --T 4 --alpha 0.9 --beta 1.0 --kd-mode both 2>&1 \
  | tee results/q1/logs/kd_console.txt

# ---- Q2: hardware-aware NAS -----------------------------------------------
python -m q2_nas.search --population 8 --sample 3 --budget 20 --epochs 3 --train-subset 10000 2>&1 \
  | tee results/q2/logs/search_console.txt
python -m q2_nas.train_best --epochs 15 2>&1 | tee results/q2/logs/train_best_console.txt

# ---- Deployment: ONNX export, ORT inference, benchmarking -----------------
python -m deploy.export_onnx --models all 2>&1 | grep -v -i warn | tee results/onnx_export_console.txt
python -m deploy.onnx_infer --models all --samples 3 2>&1 | tee results/onnx_infer_console.txt
python -m deploy.benchmark --models all --runs 100 2>&1 | tee results/benchmark_console.txt
python -m q2_nas.plots 2>&1 | tee results/q2/logs/plots_console.txt
echo "PIPELINE DONE"
