"""
Q1 - Step 1: train the ResNet-50 teacher on CIFAR-100.

Example
    python -m q1_kd.train_teacher --epochs 2 --res 64          # CPU budget run (what was used here)
    python -m q1_kd.train_teacher --epochs 30 --res 128        # full run on a GPU
"""
import argparse
import os
import time

import torch
import torch.nn as nn

from common.data import NUM_CLASSES, get_loaders
from common.utils import (CSVLogger, banner, build_optimizer, build_scheduler, count_params,
                          evaluate, get_device, save_checkpoint, save_json, set_seed, accuracy)
from q1_kd.teacher import ResNet50Teacher


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet-50 teacher on CIFAR-100")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--res", type=int, default=64, help="internal input resolution of the teacher")
    p.add_argument("--no-pretrained", action="store_true", help="train from random init")
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="checkpoints/teacher_resnet50.pt")
    p.add_argument("--log", default="results/q1/logs/teacher_resnet50.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    dev = get_device(a.device)
    train, _, test = get_loaders("cifar100", a.data, a.batch_size, train_subset=a.train_subset, seed=a.seed)

    model = ResNet50Teacher(NUM_CLASSES["cifar100"], input_res=a.res, pretrained=not a.no_pretrained).to(dev)
    banner(f"Teacher ResNet-50 | params={count_params(model) / 1e6:.2f}M | res={a.res} | device={dev}")

    opt = build_optimizer(model.parameters(), "sgd", a.lr, a.wd)
    sched = build_scheduler(opt, a.lr, a.epochs, len(train))
    crit = nn.CrossEntropyLoss()
    log = CSVLogger(a.log, ["epoch", "train_loss", "train_top1", "test_loss", "test_top1", "test_top5", "lr", "time_s"])

    best = 0.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tl, ta, n = time.time(), 0.0, 0.0, 0
        for i, (x, y) in enumerate(train):
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * y.size(0)
            ta += accuracy(out, y)[0] * y.size(0)
            n += y.size(0)
            if i % 50 == 0:
                print(f"  ep {ep} [{i:4d}/{len(train)}] loss {tl / n:.3f} acc {ta / n:5.2f}% "
                      f"({n / (time.time() - t0):.0f} img/s)", flush=True)
        ev = evaluate(model, test, dev)
        row = {"epoch": ep, "train_loss": tl / n, "train_top1": ta / n, "test_loss": ev["loss"],
               "test_top1": ev["top1"], "test_top5": ev["top5"], "lr": opt.param_groups[0]["lr"],
               "time_s": time.time() - t0}
        log.log(row)
        print(f"[teacher] epoch {ep}/{a.epochs} train_acc {ta / n:.2f}% | test top1 {ev['top1']:.2f}% "
              f"top5 {ev['top5']:.2f}% | {row['time_s']:.0f}s", flush=True)
        if ev["top1"] >= best:
            best = ev["top1"]
            # fp16 storage keeps the checkpoint < 50 MB (GitHub friendly); it is upcast on load.
            save_checkpoint(a.out, model, {"arch": "resnet50", "res": a.res, "epoch": ep,
                                           "test_top1": ev["top1"], "test_top5": ev["top5"]}, half=True)
    save_json({"model": "resnet50_teacher", "best_test_top1": best, "epochs": a.epochs, "res": a.res,
               "params": count_params(model)}, "results/q1/teacher_summary.json")
    print(f"[teacher] done. best test top1 = {best:.2f}% -> {a.out}")


if __name__ == "__main__":
    main()
