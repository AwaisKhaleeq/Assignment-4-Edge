"""
Unified predictor abstraction used by every inference script.

    TorchPredictor(key)  - PyTorch eager model on CPU
    OnnxPredictor(key)   - ONNX Runtime CPUExecutionProvider

Both expose the same API:
    predict(x)            -> probabilities (numpy, N x C) for a normalised NCHW batch
    classify(x, topk=5)   -> list of [(class_name, prob), ...] per image
"""
import numpy as np
import torch

from common.data import CLASS_NAMES
from deploy.registry import load_model, model_info


class _Base:
    def __init__(self, key):
        self.key = key
        self.info = model_info(key)
        self.classes = CLASS_NAMES[self.info["dataset"]]
        self.resolution = self.info["resolution"]

    def classify(self, x, topk=5):
        probs = self.predict(x)
        out = []
        for p in probs:
            idx = np.argsort(-p)[:topk]
            out.append([(self.classes[i], float(p[i])) for i in idx])
        return out


class TorchPredictor(_Base):
    backend = "pytorch"

    def __init__(self, key, threads=None):
        super().__init__(key)
        if threads:
            torch.set_num_threads(threads)
        self.model, _, self.meta = load_model(key)

    @torch.no_grad()
    def predict(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.softmax(self.model(x), 1).numpy()


class OnnxPredictor(_Base):
    backend = "onnxruntime"

    def __init__(self, key, threads=None, onnx_path=None):
        super().__init__(key)
        from common.profiling import make_ort_session
        self.session = make_ort_session(onnx_path or self.info["onnx"], threads)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        logits = self.session.run(None, {self.input_name: x})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def make_predictor(key, backend="pytorch", threads=None):
    return (OnnxPredictor if backend == "onnx" else TorchPredictor)(key, threads)


def evaluate_predictor(pred, limit=None, batch_size=256):
    """Top-1 / top-5 accuracy of a predictor on the matching CIFAR test set."""
    from common.data import get_loaders
    _, _, test = get_loaders(pred.info["dataset"], "data", 128, test_subset=limit,
                             resolution=pred.resolution if pred.resolution != 32 else None)
    c1 = c5 = n = 0
    for x, y in test:
        p = pred.predict(x.numpy())
        top5 = np.argsort(-p, 1)[:, :5]
        y = y.numpy()
        c1 += (top5[:, 0] == y).sum()
        c5 += (top5 == y[:, None]).any(1).sum()
        n += len(y)
    return {"top1": 100.0 * c1 / n, "top5": 100.0 * c5 / n, "n": n}
