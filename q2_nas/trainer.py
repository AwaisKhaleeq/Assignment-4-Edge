"""Automated training / validation pipeline for NAS candidates (CIFAR-10)."""
import time

import torch.nn as nn

from common.utils import accuracy, build_optimizer, build_scheduler, evaluate, AverageMeter


def train_and_validate(model, train_loader, val_loader, device, epochs, lr=0.1, wd=5e-4,
                       label_smoothing=0.0, verbose=False, test_loader=None, epoch_callback=None):
    """Train `model` from scratch for `epochs` epochs and return validation metrics.

    epoch_callback(epoch, row) is called after every epoch (used for logging / checkpointing).
    """
    model.to(device)
    opt = build_optimizer(model.parameters(), "sgd", lr, wd)
    sched = build_scheduler(opt, lr, epochs, len(train_loader))
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    t0 = time.time()
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        lm, am = AverageMeter(), AverageMeter()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            lm.update(loss.item(), y.size(0))
            am.update(accuracy(out, y)[0], y.size(0))
            if verbose and i % 100 == 0:
                print(f"    ep {ep} [{i:4d}/{len(train_loader)}] loss {lm.avg:.3f} acc {am.avg:5.2f}%", flush=True)
        val = evaluate(model, val_loader, device)
        row = {"epoch": ep, "train_loss": lm.avg, "train_top1": am.avg,
               "val_loss": val["loss"], "val_top1": val["top1"], "time_s": time.time() - t0}
        if test_loader is not None:
            te = evaluate(model, test_loader, device)
            row.update({"test_loss": te["loss"], "test_top1": te["top1"], "test_top5": te["top5"]})
        hist.append(row)
        if epoch_callback:
            epoch_callback(ep, row)
        if verbose:
            extra = f" test {row['test_top1']:.2f}%" if "test_top1" in row else ""
            print(f"  epoch {ep}/{epochs} train {am.avg:.2f}% val {val['top1']:.2f}%{extra} "
                  f"({row['time_s']:.0f}s)", flush=True)
    return {"val_top1": hist[-1]["val_top1"], "val_loss": hist[-1]["val_loss"],
            "train_top1": hist[-1]["train_top1"], "train_time_s": time.time() - t0, "history": hist}
