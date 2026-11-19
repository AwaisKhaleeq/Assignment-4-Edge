"""
CPU inference with ONNX Runtime for the exported models (students + NAS + teacher).

Same interface as deploy/infer.py, but the backend is fixed to ONNX Runtime
(CPUExecutionProvider) and the models are loaded from onnx_models/*.onnx - no PyTorch
model code is needed at inference time.

Examples
    python -m deploy.onnx_infer --models students --eval
    python -m deploy.onnx_infer --models nas_best --samples 5
    python -m deploy.onnx_infer --models mobilenetv2_kd --image my_photo.jpg
"""
from deploy.infer import main

if __name__ == "__main__":
    main(default_backend="onnx")
