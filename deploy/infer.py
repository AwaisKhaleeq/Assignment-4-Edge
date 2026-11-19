"""
Unified inference pipeline for ALL trained models (3 baseline students, 3 distilled
students, the NAS model and the teacher), with a selectable backend.

Examples
    # classify an image file with every student model
    python -m deploy.infer --models students --image samples/cat.png

    # show predictions for 5 random test images
    python -m deploy.infer --models mobilenetv2_kd nas_best --samples 5

    # evaluate accuracy on the test set (PyTorch or ONNX Runtime backend)
    python -m deploy.infer --models all --eval
    python -m deploy.infer --models all --eval --backend onnx
"""
import argparse
import os
import time

import numpy as np

from common.data import CLASS_NAMES, STATS, prepare_dataset, preprocess_image
from deploy.predictor import evaluate_predictor, make_predictor
from deploy.registry import all_keys, model_info, student_keys


def resolve_keys(names):
    keys = []
    for n in names:
        if n == "all":
            keys += all_keys()
        elif n == "students":
            keys += student_keys()
        else:
            keys.append(n)
    keys = [k for k in dict.fromkeys(keys)]
    missing = [k for k in keys if not os.path.exists(model_info(k)["checkpoint"])]
    for k in missing:
        print(f"  [skip] {k}: checkpoint not found")
    return [k for k in keys if k not in missing]


def random_test_images(dataset, n, resolution, seed=0):
    import torch
    import torch.nn.functional as F
    _, _, xte, yte = prepare_dataset(dataset, "data")
    idx = np.random.RandomState(seed).choice(len(yte), n, replace=False)
    x = torch.from_numpy(xte[idx]).permute(0, 3, 1, 2).float() / 255.0
    mean, std = STATS[dataset]
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    if resolution != 32:
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return x.numpy(), yte[idx]


def main(default_backend="pytorch"):
    p = argparse.ArgumentParser(description="Unified inference for all trained models")
    p.add_argument("--models", nargs="+", default=["all"],
                   help="model keys, or 'all' / 'students'. Keys: " + ", ".join(all_keys()))
    p.add_argument("--backend", choices=["pytorch", "onnx"], default=default_backend)
    p.add_argument("--image", help="path of an image file to classify")
    p.add_argument("--samples", type=int, default=0, help="classify N random test images")
    p.add_argument("--eval", action="store_true", help="evaluate accuracy on the test set")
    p.add_argument("--limit", type=int, default=None, help="evaluate on the first N test images only")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--threads", type=int, default=None)
    a = p.parse_args()
    if not (a.image or a.samples or a.eval):
        a.samples = 3

    rows = []
    for key in resolve_keys(a.models):
        pred = make_predictor(key, a.backend, a.threads)
        print(f"\n=== {key}  [{pred.backend}, {pred.info['dataset']}, input {pred.resolution}x{pred.resolution}] ===")
        if a.image:
            x = preprocess_image(a.image, pred.info["dataset"], pred.resolution).numpy()
            t0 = time.perf_counter()
            res = pred.classify(x, a.topk)[0]
            print(f"  {os.path.basename(a.image)} -> " +
                  ", ".join(f"{c} ({p * 100:.1f}%)" for c, p in res) +
                  f"   [{(time.perf_counter() - t0) * 1000:.2f} ms]")
        if a.samples:
            x, y = random_test_images(pred.info["dataset"], a.samples, pred.resolution)
            for xi, yi, res in zip(x, y, pred.classify(x, a.topk)):
                ok = "OK " if res[0][0] == CLASS_NAMES[pred.info["dataset"]][yi] else "ERR"
                print(f"  [{ok}] true: {CLASS_NAMES[pred.info['dataset']][yi]:<14s} pred: " +
                      ", ".join(f"{c} ({p * 100:.1f}%)" for c, p in res))
        if a.eval:
            t0 = time.perf_counter()
            r = evaluate_predictor(pred, a.limit)
            dt = time.perf_counter() - t0
            print(f"  test top1 {r['top1']:.2f}% | top5 {r['top5']:.2f}% | {r['n']} images in {dt:.1f}s "
                  f"({r['n'] / dt:.0f} img/s)")
            rows.append((key, r["top1"], r["top5"]))
    if len(rows) > 1:
        print(f"\n{'model':<30s}{'top1':>8s}{'top5':>8s}")
        for k, t1, t5 in rows:
            print(f"{k:<30s}{t1:8.2f}{t5:8.2f}")


if __name__ == "__main__":
    main()
