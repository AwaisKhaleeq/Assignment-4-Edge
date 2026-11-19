#!/usr/bin/env bash
# Resume the pipeline after the teacher + baseline stages (used after the run was interrupted).
# Each step writes straight to its own log file (no pipes), so steps don't depend on a live terminal.
set -e
python -u -m q1_kd.train_kd --epochs 5 --T 4 --alpha 0.9 --beta 1.0 --kd-mode both > results/q1/logs/kd_console.txt 2>&1
python -u -m q2_nas.search --population 8 --sample 3 --budget 20 --epochs 3 --train-subset 10000 > results/q2/logs/search_console.txt 2>&1
python -u -m q2_nas.train_best --epochs 15 > results/q2/logs/train_best_console.txt 2>&1
python -u -m deploy.export_onnx --models all > results/onnx_export_console.txt 2>&1
python -u -m deploy.onnx_infer --models all --samples 3 > results/onnx_infer_console.txt 2>&1
python -u -m deploy.benchmark --models all --runs 100 > results/benchmark_console.txt 2>&1
python -u -m q2_nas.plots > results/q2/logs/plots_console.txt 2>&1
echo "PIPELINE DONE"
