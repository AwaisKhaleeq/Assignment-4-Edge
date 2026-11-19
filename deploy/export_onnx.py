"""
Export trained models to ONNX and verify numerical parity with PyTorch.

    python -m deploy.export_onnx --models all

For each model:
  * export with a dynamic batch dimension (opset 17),
  * validate the graph with onnx.checker,
  * run the same random batch through PyTorch and ONNX Runtime and report the max abs
    difference of the logits and the top-1 agreement.
Results -> results/onnx_export.csv
"""
import argparse
import os

import numpy as np
import onnx
import torch

from common.profiling import file_size_mb, make_ort_session
from common.utils import CSVLogger
from deploy.infer import resolve_keys
from deploy.registry import load_model


def export(key, opset=17):
    model, info, _ = load_model(key)
    r = info["resolution"]
    dummy = torch.randn(1, 3, r, r)
    os.makedirs(os.path.dirname(info["onnx"]), exist_ok=True)
    torch.onnx.export(model, dummy, info["onnx"], input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(info["onnx"]))

    # parity check PyTorch vs ONNX Runtime
    x = torch.randn(16, 3, r, r)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = make_ort_session(info["onnx"])
    out = sess.run(None, {"input": x.numpy()})[0]
    return {"model": key, "onnx_path": info["onnx"], "onnx_size_mb": round(file_size_mb(info["onnx"]), 4),
            "input_shape": f"Nx3x{r}x{r}", "opset": opset,
            "max_abs_diff": float(np.abs(ref - out).max()),
            "top1_agreement": float((ref.argmax(1) == out.argmax(1)).mean() * 100)}


def main():
    p = argparse.ArgumentParser(description="Export trained models to ONNX")
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--opset", type=int, default=17)
    a = p.parse_args()
    log = CSVLogger("results/onnx_export.csv", ["model", "onnx_path", "onnx_size_mb", "input_shape", "opset",
                                                "max_abs_diff", "top1_agreement"])
    for key in resolve_keys(a.models):
        r = export(key, a.opset)
        log.log(r)
        print(f"[onnx] {key:<30s} -> {r['onnx_path']:<42s} {r['onnx_size_mb']:8.3f} MB | "
              f"max|diff| {r['max_abs_diff']:.2e} | top1 agree {r['top1_agreement']:.0f}%")


if __name__ == "__main__":
    main()
