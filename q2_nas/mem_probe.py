"""
Peak-inference-memory probe (executed in a fresh process by hw_profiler.py).

    python -m q2_nas.mem_probe '<config json>' [runs]

Prints one JSON line: {"peak_mem_mb": ..., "baseline_rss_mb": ...}
The model is built *before* the baseline is taken, so the reported value is the
extra memory needed to run inference (activations + workspace), i.e. the dynamic part.
"""
import json
import sys

import torch

from common.profiling import PeakRSSMonitor
from q2_nas.generator import build_model


def main():
    cfg = json.loads(sys.argv[1])
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    torch.set_num_threads(1)
    model = build_model(cfg).eval()
    r = cfg["resolution"]
    x = torch.randn(1, 3, r, r)
    with torch.no_grad(), PeakRSSMonitor(interval=0.0002) as mon:
        for _ in range(runs):
            model(x)
    print(json.dumps({"peak_mem_mb": round(mon.peak_delta_mb, 3),
                      "baseline_rss_mb": round(mon.baseline / 2 ** 20, 2)}))


if __name__ == "__main__":
    main()
