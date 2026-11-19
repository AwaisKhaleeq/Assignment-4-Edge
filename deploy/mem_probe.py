"""
Peak inference memory probe for a registered model, run in a FRESH process.

    python -m deploy.mem_probe <model_key> <pytorch|onnx> [batch]

Prints a JSON line {"peak_mem_mb": ...}: the increase of resident memory while running
inference (after the model / ORT session has been loaded).
"""
import json
import sys

import numpy as np
import torch

from common.profiling import PeakRSSMonitor
from deploy.predictor import make_predictor


def main():
    key, backend = sys.argv[1], sys.argv[2]
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    torch.set_num_threads(1)
    pred = make_predictor(key, backend, threads=1)
    r = pred.resolution
    x = np.random.randn(batch, 3, r, r).astype(np.float32)
    # baseline is taken AFTER loading the model, so the peak covers activations,
    # kernel workspaces and output buffers allocated during (first and repeated) inference
    with PeakRSSMonitor(interval=0.0002) as mon:
        for _ in range(10):
            pred.predict(x)
    print(json.dumps({"peak_mem_mb": round(mon.peak_delta_mb, 3)}))


if __name__ == "__main__":
    main()
