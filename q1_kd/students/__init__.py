"""Student model registry: build any student by name."""
from .efficientnet_lite import EfficientNetLite0
from .mobilenetv2 import MobileNetV2
from .squeezenet import SqueezeNet

STUDENTS = {
    "mobilenetv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efficientnet_lite0": EfficientNetLite0,
}


def build_student(name, num_classes=100):
    if name not in STUDENTS:
        raise ValueError(f"unknown student '{name}', choose from {list(STUDENTS)}")
    return STUDENTS[name](num_classes=num_classes)
