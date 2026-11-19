"""
Q2 - Re-build the best discovered architecture (results/q2/best_arch.json) and train it
FROM SCRATCH on the full CIFAR-10 training set (45k train / 5k validation), then report
test accuracy and save the checkpoint.

Example
    python -m q2_nas.train_best --epochs 15
"""
import argparse

from common.data import get_loaders
from common.utils import (CSVLogger, banner, count_params, evaluate, get_device, load_json,
                          save_checkpoint, save_json, set_seed)
from q2_nas.generator import build_model
from q2_nas.search_space import describe
from q2_nas.trainer import train_and_validate


def parse_args():
    p = argparse.ArgumentParser(description="Train the best NAS architecture from scratch")
    p.add_argument("--arch", default="results/q2/best_arch.json")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--out", default="checkpoints/nas_best.pt")
    p.add_argument("--data", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    dev = get_device(a.device)
    best = load_json(a.arch)
    cfg = best["config"]
    model = build_model(cfg, num_classes=10)
    banner(f"Training best NAS architecture from scratch | {describe(cfg)} | "
           f"{count_params(model) / 1e3:.0f}K params")

    train, val, test = get_loaders("cifar10", a.data, a.batch_size, val_size=a.val_size,
                                   resolution=cfg["resolution"], seed=a.seed)
    log = CSVLogger("results/q2/best_arch_training.csv",
                    ["epoch", "train_loss", "train_top1", "val_loss", "val_top1",
                     "test_loss", "test_top1", "test_top5", "time_s"])
    state = {"best_val": -1}

    def on_epoch(ep, row):
        log.log(row)
        if row["val_top1"] >= state["best_val"]:            # model selection on validation, not test
            state["best_val"] = row["val_top1"]
            save_checkpoint(a.out, model, {"arch": "nas_best", "config": cfg, "epoch": ep,
                                           "val_top1": row["val_top1"], "test_top1": row["test_top1"]})

    res = train_and_validate(model, train, val, dev, a.epochs, lr=a.lr, verbose=True,
                             test_loader=test, epoch_callback=on_epoch)
    from common.utils import load_checkpoint
    meta = load_checkpoint(a.out, model)
    final = evaluate(model, test, dev)
    summary = {"arch": describe(cfg), "config": cfg, "epochs": a.epochs, "params": count_params(model),
               "best_epoch": meta["epoch"], "val_top1": meta["val_top1"],
               "test_top1": final["top1"], "test_top5": final["top5"],
               "proxy_val_top1_during_search": best.get("val_top1_proxy"),
               "train_time_min": res["train_time_s"] / 60, "checkpoint": a.out}
    save_json(summary, "results/q2/best_arch_final.json")
    print(f"[nas-best] test top1 {final['top1']:.2f}% (epoch {meta['epoch']}) -> {a.out}")


if __name__ == "__main__":
    main()
