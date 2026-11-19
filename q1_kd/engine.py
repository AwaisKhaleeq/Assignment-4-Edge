"""
Training engine shared by the baseline pipeline and the distillation pipeline.

Several students can be trained *in lock-step*: each mini-batch is fed to every
student. In distillation mode the teacher forward pass is computed only once per
batch and its outputs are reused by all students - this cuts the cost of KD on a
CPU by ~3x and guarantees that all students see exactly the same data stream
(fair comparison between architectures and between baseline and KD).
"""
import time

import torch
import torch.nn as nn

from common.utils import (AverageMeter, CSVLogger, accuracy, build_optimizer, build_scheduler,
                          count_params, evaluate, save_checkpoint)
from q1_kd.distill import FeatureAdapter


class StudentRun:
    """Bundles a student model with its optimiser, scheduler, (optional) adapter and logger."""

    def __init__(self, name, model, device, lr, wd, epochs, steps, log_path, ckpt_path,
                 teacher_channels=None):
        self.name, self.model, self.ckpt = name, model.to(device), ckpt_path
        self.adapter = (FeatureAdapter(model.feature_channels, teacher_channels).to(device)
                        if teacher_channels else None)
        params = list(self.model.parameters()) + (list(self.adapter.parameters()) if self.adapter else [])
        self.opt = build_optimizer(params, "sgd", lr, wd)
        self.sched = build_scheduler(self.opt, lr, epochs, steps)
        self.log = CSVLogger(log_path, ["epoch", "train_loss", "train_top1", "ce", "kd", "feat",
                                        "test_loss", "test_top1", "test_top5", "lr", "time_s"])
        self.best = 0.0
        self.history = []
        self.reset_meters()

    def reset_meters(self):
        self.m = {k: AverageMeter() for k in ("loss", "top1", "ce", "kd", "feat")}
        self.time = 0.0

    def train(self):
        self.model.train()
        if self.adapter:
            self.adapter.train()


def train_students(runs, train_loader, test_loader, device, epochs, teacher=None, kd_loss=None,
                   tag="baseline"):
    """Train a list of StudentRun objects (baseline if teacher is None, otherwise KD)."""
    ce = nn.CrossEntropyLoss()
    if teacher is not None:
        teacher.eval()                                # frozen teacher (BN in inference mode)
        for p in teacher.parameters():
            p.requires_grad_(False)

    for ep in range(1, epochs + 1):
        for r in runs:
            r.train()
            r.reset_meters()
        t_ep = time.time()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if teacher is not None:
                with torch.no_grad():
                    f_t = teacher.features(x)         # intermediate representation (B, 2048, 2, 2)
                    z_t = teacher.classifier(f_t)     # teacher logits
            for r in runs:
                t0 = time.time()
                f_s = r.model.features(x)
                z_s = r.model.classifier(f_s)
                if teacher is None:
                    loss = ce(z_s, y)
                    parts = {"ce": loss.item()}
                else:
                    f_s_proj = r.adapter(f_s, f_t.shape[-2:]) if r.adapter else None
                    loss, parts = kd_loss(z_s, z_t, y, f_s_proj, f_t)
                r.opt.zero_grad(set_to_none=True)
                loss.backward()
                r.opt.step()
                r.sched.step()
                b = y.size(0)
                r.m["loss"].update(loss.item(), b)
                r.m["top1"].update(accuracy(z_s, y)[0], b)
                for k, v in parts.items():
                    r.m[k].update(v, b)
                r.time += time.time() - t0
            if i % 50 == 0:
                msg = " | ".join(f"{r.name}: L {r.m['loss'].avg:.3f} acc {r.m['top1'].avg:5.2f}%" for r in runs)
                print(f"  [{tag}] ep {ep} [{i:4d}/{len(train_loader)}] {msg}", flush=True)

        # ---- end of epoch: evaluate, log, checkpoint ------------------------
        for r in runs:
            ev = evaluate(r.model, test_loader, device)
            row = {"epoch": ep, "train_loss": r.m["loss"].avg, "train_top1": r.m["top1"].avg,
                   "ce": r.m["ce"].avg, "kd": r.m["kd"].avg, "feat": r.m["feat"].avg,
                   "test_loss": ev["loss"], "test_top1": ev["top1"], "test_top5": ev["top5"],
                   "lr": r.opt.param_groups[0]["lr"], "time_s": r.time}
            r.log.log(row)
            r.history.append(row)
            print(f"[{tag}] {r.name:<20s} epoch {ep}/{epochs} | train {r.m['top1'].avg:5.2f}% | "
                  f"test top1 {ev['top1']:5.2f}% top5 {ev['top5']:5.2f}%", flush=True)
            if ev["top1"] >= r.best:
                r.best = ev["top1"]
                save_checkpoint(r.ckpt, r.model, {"arch": r.name, "mode": tag, "epoch": ep,
                                                  "test_top1": ev["top1"], "test_top5": ev["top5"],
                                                  "params": count_params(r.model)})
        print(f"[{tag}] epoch {ep} finished in {time.time() - t_ep:.0f}s", flush=True)
    return {r.name: {"best_test_top1": r.best, "final": r.history[-1], "params": count_params(r.model),
                     "checkpoint": r.ckpt} for r in runs}
