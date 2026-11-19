"""Generic training / evaluation helpers shared by Q1 (KD) and Q2 (NAS)."""
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref="auto"):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, input_shape):
    """Forward-pass FLOPs (multiply-add counted as 2) via torch's FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode
    model = model.eval()
    x = torch.randn(*input_shape, device=next(model.parameters()).device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, val, k=1):
        self.sum += val * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


def accuracy(logits, target, topk=(1,)):
    """Return list of top-k accuracies (percent) for the batch."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)
    correct = pred.t().eq(target.view(1, -1))
    return [correct[:k].reshape(-1).float().sum().item() * 100.0 / target.size(0) for k in topk]


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """Top-1 / Top-5 accuracy and loss of `model` on `loader`."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_m, t1, t5 = AverageMeter(), AverageMeter(), AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_m.update(criterion(out, y).item(), y.size(0))
        k5 = min(5, out.size(1))
        a1, a5 = accuracy(out, y, (1, k5))
        t1.update(a1, y.size(0))
        t5.update(a5, y.size(0))
    return {"loss": loss_m.avg, "top1": t1.avg, "top5": t5.avg}


def build_optimizer(params, opt="sgd", lr=0.05, wd=5e-4):
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, lr, epochs, steps_per_epoch):
    """One-cycle schedule: short warm-up followed by cosine annealing (per step)."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.15, anneal_strategy="cos", div_factor=10, final_div_factor=100)


def save_checkpoint(path, model, meta=None, half=False):
    """Save a state dict (+ metadata). `half=True` stores fp16 weights to keep the repo small."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sd = model.state_dict()
    if half:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save({"state_dict": sd, "meta": meta or {}}, path)


def load_checkpoint(path, model, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()}
    model.load_state_dict(sd)
    return ck.get("meta", {})


class CSVLogger:
    """Append-only CSV logger (writes the header on first use)."""

    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0


def banner(text):
    line = "=" * max(60, len(text) + 4)
    print(f"\n{line}\n  {text}\n{line}")
