"""
Automated benchmarking of every trained model (PyTorch and ONNX Runtime, CPU).

For each model it measures
  * accuracy        : top-1 / top-5 on the test set with PyTorch and with ONNX Runtime
  * complexity      : parameters, MFLOPs (batch 1)
  * storage         : fp32 state-dict size, ONNX file size
  * latency         : batch-1 median / p95 latency, 1 thread (edge-like) for PyTorch and ORT,
                      plus ORT with all threads
  * throughput      : ORT images/s at batch 64 (all threads)
  * peak memory     : peak extra RSS during batch-1 inference (fresh subprocess), PyTorch & ORT
and writes
  results/benchmark/benchmark.csv          - one row per model
  results/benchmark/kd_vs_baseline.csv     - KD gain per student architecture
  results/figures/*.png                    - accuracy/latency/size plots

    python -m deploy.benchmark --models all
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from common.profiling import (file_size_mb, make_ort_session, model_storage_mb, ort_cpu_latency,
                              torch_cpu_latency)
from common.utils import count_flops, count_params
from deploy.infer import resolve_keys
from deploy.predictor import OnnxPredictor, TorchPredictor, evaluate_predictor


def peak_mem(key, backend):
    try:
        out = subprocess.run([sys.executable, "-m", "deploy.mem_probe", key, backend],
                             capture_output=True, text=True, timeout=300)
        return json.loads(out.stdout.strip().splitlines()[-1])["peak_mem_mb"]
    except Exception as e:
        print(f"   peak-memory probe failed for {key}/{backend}: {e}")
        return None


def bench_model(key, runs, eval_limit):
    tp = TorchPredictor(key)
    model, info, meta = tp.model, tp.info, tp.meta
    r = info["resolution"]
    row = {"model": key, "family": info["family"], "dataset": info["dataset"], "input": f"{r}x{r}",
           "params_m": count_params(model) / 1e6, "mflops": count_flops(model, (1, 3, r, r)) / 1e6,
           "torch_size_mb": model_storage_mb(model)}
    # accuracy
    acc_t = evaluate_predictor(tp, eval_limit)
    row.update({"torch_top1": acc_t["top1"], "torch_top5": acc_t["top5"]})
    # latency (PyTorch, 1 thread)
    lt = torch_cpu_latency(model, (1, 3, r, r), runs=runs, threads=1)
    row.update({"torch_lat_ms_1t": lt["latency_median_ms"], "torch_lat_p95_ms_1t": lt["latency_p95_ms"]})

    if os.path.exists(info["onnx"]):
        op = OnnxPredictor(key)
        acc_o = evaluate_predictor(op, eval_limit)
        row.update({"onnx_size_mb": file_size_mb(info["onnx"]), "ort_top1": acc_o["top1"], "ort_top5": acc_o["top5"]})
        s1 = make_ort_session(info["onnx"], threads=1)
        l1 = ort_cpu_latency(s1, (1, 3, r, r), runs=runs)
        sN = make_ort_session(info["onnx"])
        lN = ort_cpu_latency(sN, (1, 3, r, r), runs=runs)
        row.update({"ort_lat_ms_1t": l1["latency_median_ms"], "ort_lat_p95_ms_1t": l1["latency_p95_ms"],
                    "ort_lat_ms_all": lN["latency_median_ms"]})
        lb = ort_cpu_latency(sN, (64, 3, r, r), warmup=3, runs=max(5, runs // 5))
        row["ort_throughput_b64"] = 64 * 1000 / lb["latency_median_ms"]
        row["ort_peak_mem_mb"] = peak_mem(key, "onnx")
    row["torch_peak_mem_mb"] = peak_mem(key, "pytorch")
    return row


def make_plots(df, out_dir="results/figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    q1 = df[df.dataset == "cifar100"]

    # accuracy vs latency (Q1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"teacher": "#444444", "student_baseline": "#3b7dd8", "student_kd": "#e4572e"}
    for _, rw in q1.iterrows():
        ax.scatter(rw.ort_lat_ms_1t, rw.torch_top1, s=40 + 25 * rw.params_m, color=colors.get(rw.family, "gray"),
                   alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.annotate(rw.model.replace("_", " "), (rw.ort_lat_ms_1t, rw.torch_top1), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    for fam, c in colors.items():
        ax.scatter([], [], color=c, label=fam.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("ONNX Runtime CPU latency, batch 1, 1 thread (ms, log)")
    ax.set_ylabel("CIFAR-100 test top-1 (%)")
    ax.set_title("Q1: accuracy vs. latency (marker size ~ parameters)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/q1_accuracy_vs_latency.png", dpi=160)
    plt.close(fig)

    # KD vs baseline bars
    kd = pd.read_csv("results/benchmark/kd_vs_baseline.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(kd))
    ax.bar(x - 0.2, kd.baseline_top1, 0.4, label="baseline (no KD)", color="#3b7dd8")
    ax.bar(x + 0.2, kd.kd_top1, 0.4, label="distilled (KD)", color="#e4572e")
    for i, rw in kd.iterrows():
        ax.text(i + 0.2, rw.kd_top1 + 0.5, f"{rw.kd_gain:+.2f}", ha="center", fontsize=8)
    t = df[df.model == "teacher_resnet50"]
    if len(t):
        ax.axhline(t.torch_top1.iloc[0], ls="--", color="#444", label="ResNet-50 teacher")
    ax.set_xticks(x)
    ax.set_xticklabels(kd.student)
    ax.set_ylabel("CIFAR-100 test top-1 (%)")
    ax.set_title("Q1: effect of knowledge distillation")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/q1_kd_vs_baseline.png", dpi=160)
    plt.close(fig)

    # PyTorch vs ORT latency for all models
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.torch_lat_ms_1t, 0.4, label="PyTorch (1 thread)", color="#8d99ae")
    ax.bar(x + 0.2, df.ort_lat_ms_1t, 0.4, label="ONNX Runtime (1 thread)", color="#2a9d8f")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df.model, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("latency, batch 1 (ms, log)")
    ax.set_title("CPU inference latency: PyTorch vs ONNX Runtime")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/latency_torch_vs_ort.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Benchmark all trained models on CPU")
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--runs", type=int, default=100, help="timed runs per latency measurement")
    p.add_argument("--eval-limit", type=int, default=None, help="limit #test images for accuracy")
    a = p.parse_args()
    os.makedirs("results/benchmark", exist_ok=True)
    rows = []
    for key in resolve_keys(a.models):
        t0 = time.time()
        row = bench_model(key, a.runs, a.eval_limit)
        rows.append(row)
        print(f"[bench] {key:<28s} top1 {row['torch_top1']:6.2f}% (ort {row.get('ort_top1', float('nan')):6.2f}%) | "
              f"{row['params_m']:.3f}M | {row['mflops']:7.1f} MFLOPs | torch {row['torch_lat_ms_1t']:7.2f}ms | "
              f"ort {row.get('ort_lat_ms_1t', float('nan')):7.2f}ms | onnx {row.get('onnx_size_mb', float('nan')):6.2f}MB | "
              f"mem {row.get('ort_peak_mem_mb')}MB | {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).round(4).to_csv("results/benchmark/benchmark.csv", index=False)
    df = pd.DataFrame(rows)

    # KD vs baseline table
    kd_rows = []
    for s in ["mobilenetv2", "squeezenet", "efficientnet_lite0"]:
        b, k = df[df.model == f"{s}_baseline"], df[df.model == f"{s}_kd"]
        if len(b) and len(k):
            kd_rows.append({"student": s, "params_m": b.params_m.iloc[0], "baseline_top1": b.torch_top1.iloc[0],
                            "kd_top1": k.torch_top1.iloc[0], "kd_gain": k.torch_top1.iloc[0] - b.torch_top1.iloc[0],
                            "baseline_top5": b.torch_top5.iloc[0], "kd_top5": k.torch_top5.iloc[0],
                            "ort_lat_ms_1t": k.ort_lat_ms_1t.iloc[0] if "ort_lat_ms_1t" in k else None})
    if kd_rows:
        pd.DataFrame(kd_rows).round(3).to_csv("results/benchmark/kd_vs_baseline.csv", index=False)
    if "ort_lat_ms_1t" in df and kd_rows:
        make_plots(df)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[["model", "torch_top1", "ort_top1", "params_m", "mflops", "onnx_size_mb", "torch_lat_ms_1t",
                  "ort_lat_ms_1t", "ort_throughput_b64", "ort_peak_mem_mb"]].round(2).to_string(index=False))


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 4)
    main()
