"""Figures for the NAS experiment (reads results/q2/*.csv|json, writes results/figures/q2_*.png)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "results/figures"


def main():
    os.makedirs(OUT, exist_ok=True)
    h = pd.read_csv("results/q2/search_history.csv")
    best = json.load(open("results/q2/best_arch.json"))
    cons = best["constraints"]
    tr, rej = h[h.status == "trained"], h[h.status == "rejected"]

    # 1) latency vs proxy accuracy, constraint line, rejected candidates
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = ax[0].scatter(tr.latency_ms, tr.val_top1, c=tr.id, cmap="viridis", s=20 + 150 * tr.size_mb,
                       edgecolor="black", linewidth=0.5, label="trained (colour = search step)")
    ax[0].scatter(best["latency_ms"], best["val_top1_proxy"], marker="*", s=350, color="#e4572e",
                  edgecolor="black", label=f"best #{best['best_id']}")
    ax[0].axvline(cons["max_latency_ms"], ls="--", color="red", lw=1, label="latency constraint")
    ax[0].set_xlabel("measured CPU latency, batch 1, 1 thread (ms)")
    ax[0].set_ylabel("proxy validation accuracy (%)")
    ax[0].set_title("Trained candidates (marker size ~ storage size)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    fig.colorbar(sc, ax=ax[0], label="candidate id")

    ax[1].scatter(tr.latency_ms, tr.size_mb, color="#2a9d8f", label=f"accepted ({len(tr)})", edgecolor="black", lw=0.5)
    ax[1].scatter(rej.latency_ms, rej.size_mb, color="#e63946", marker="x", label=f"rejected ({len(rej)})")
    ax[1].axvline(cons["max_latency_ms"], ls="--", color="red", lw=1)
    ax[1].axhline(cons["max_size_mb"], ls="--", color="red", lw=1)
    ax[1].set_xlabel("measured CPU latency (ms)")
    ax[1].set_ylabel("storage size (MB)")
    ax[1].set_title("Hardware-aware filtering of all sampled candidates")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/q2_search_space_scatter.png", dpi=160)
    plt.close(fig)

    # 2) search progress: fitness of each trained candidate and running best
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(tr.id, tr.fitness, "o", color="#3b7dd8", label="candidate fitness")
    ax.plot(tr.id, tr.fitness.cummax(), "-", color="#e4572e", lw=2, label="best so far")
    for _, r in tr[tr.status == "trained"].iterrows():
        if r.origin == "random":
            ax.axvspan(r.id - 0.5, r.id + 0.5, color="gray", alpha=0.08)
    ax.set_xlabel("candidate id (grey = random initial population)")
    ax.set_ylabel("fitness = acc x (lat/T)^-0.07")
    ax.set_title("Evolutionary search progress")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/q2_search_progress.png", dpi=160)
    plt.close(fig)

    # 3) training curve of the best architecture (from scratch)
    if os.path.exists("results/q2/best_arch_training.csv"):
        t = pd.read_csv("results/q2/best_arch_training.csv")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(t.epoch, t.train_top1, "-o", ms=3, label="train")
        ax.plot(t.epoch, t.val_top1, "-o", ms=3, label="validation")
        ax.plot(t.epoch, t.test_top1, "-o", ms=3, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("top-1 accuracy (%)")
        ax.set_title("Best NAS architecture trained from scratch on CIFAR-10")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{OUT}/q2_best_training_curve.png", dpi=160)
        plt.close(fig)
    print("NAS figures written to", OUT)


if __name__ == "__main__":
    main()
