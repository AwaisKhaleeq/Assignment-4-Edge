"""
Model registry for the unified inference / export / benchmarking pipeline.

Every deployable model is described by a key, e.g.
    mobilenetv2_baseline, mobilenetv2_kd, squeezenet_baseline, squeezenet_kd,
    efficientnet_lite0_baseline, efficientnet_lite0_kd, teacher_resnet50, nas_best
and can be loaded with `load_model(key)`.
"""
import os

from common.data import NUM_CLASSES
from common.utils import load_checkpoint
from q1_kd.students import STUDENTS, build_student

STUDENT_NAMES = list(STUDENTS)
VARIANTS = ["baseline", "kd"]


def model_info(key):
    """Return dict(dataset, resolution, checkpoint, onnx) for a model key."""
    if key == "teacher_resnet50":
        return {"dataset": "cifar100", "resolution": 32, "checkpoint": "checkpoints/teacher_resnet50.pt",
                "onnx": "onnx_models/teacher_resnet50.onnx", "family": "teacher"}
    if key == "nas_best":
        res = 32
        if os.path.exists("results/q2/best_arch.json"):
            from common.utils import load_json
            res = load_json("results/q2/best_arch.json")["config"]["resolution"]
        return {"dataset": "cifar10", "resolution": res, "checkpoint": "checkpoints/nas_best.pt",
                "onnx": "onnx_models/nas_best.onnx", "family": "nas"}
    name, variant = key.rsplit("_", 1)
    if name not in STUDENTS or variant not in VARIANTS:
        raise KeyError(f"unknown model key '{key}'. Available: {all_keys()}")
    return {"dataset": "cifar100", "resolution": 32, "checkpoint": f"checkpoints/students/{key}.pt",
            "onnx": f"onnx_models/{key}.onnx", "family": f"student_{variant}"}


def student_keys():
    return [f"{n}_{v}" for n in STUDENT_NAMES for v in VARIANTS]


def all_keys():
    return student_keys() + ["nas_best", "teacher_resnet50"]


def load_model(key):
    """Build the architecture for `key` and load its trained weights (CPU, eval mode)."""
    info = model_info(key)
    if key == "teacher_resnet50":
        from q1_kd.teacher import ResNet50Teacher
        model = ResNet50Teacher(NUM_CLASSES["cifar100"], pretrained=False)
        meta = load_checkpoint(info["checkpoint"], model)
        model.input_res = meta.get("res", 64)
    elif key == "nas_best":
        import torch
        from q2_nas.generator import build_model
        ck = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
        model = build_model(ck["meta"]["config"], NUM_CLASSES["cifar10"])
        meta = load_checkpoint(info["checkpoint"], model)
    else:
        name = key.rsplit("_", 1)[0]
        model = build_student(name, NUM_CLASSES["cifar100"])
        meta = load_checkpoint(info["checkpoint"], model)
    return model.eval(), info, meta
