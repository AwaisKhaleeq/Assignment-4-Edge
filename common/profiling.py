"""
Hardware profiling utilities (CPU latency, storage size, peak inference memory).

These are used by
  * the Q1 benchmarking script (teacher vs. baseline vs. distilled students), and
  * the Q2 hardware-aware NAS loop (constraint checking of every candidate).

All measurements are *actual measurements* on the machine running the code,
not analytical proxies.
"""
import gc
import io
import os
import threading
import time

import numpy as np
import psutil
import torch


def model_storage_mb(model):
    """Size of the serialised fp32 state dict in MB (what would be shipped to a device)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)


def file_size_mb(path):
    """Size of a file on disk, including an ONNX external-data file if present."""
    size = os.path.getsize(path)
    if os.path.exists(path + ".data"):
        size += os.path.getsize(path + ".data")
    return size / (1024 ** 2)


class PeakRSSMonitor:
    """Samples the process resident-set-size in a background thread and records the peak.

    Peak inference memory = peak RSS during inference - RSS right before inference.
    This works on Windows, Linux and macOS (psutil). On CUDA, torch.cuda.max_memory_allocated
    would be used instead; on CPU this RSS-based measurement is the portable option.
    """

    def __init__(self, interval=0.0005):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __enter__(self):
        gc.collect()
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def peak_delta_mb(self):
        return (self.peak - self.baseline) / (1024 ** 2)


def _summ(times_ms):
    t = np.array(times_ms)
    return {"latency_mean_ms": float(t.mean()), "latency_median_ms": float(np.median(t)),
            "latency_p95_ms": float(np.percentile(t, 95)), "latency_std_ms": float(t.std())}


@torch.no_grad()
def torch_cpu_latency(model, input_shape, warmup=10, runs=50, threads=None):
    """Measure PyTorch CPU inference latency (ms) for a fixed input shape.

    threads: if given, temporarily pin torch intra-op threads (e.g. 1 thread to
             emulate a single-core edge CPU and to make measurements stable).
    """
    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = torch.randn(*input_shape)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(old)
    return _summ(times)


@torch.no_grad()
def torch_peak_memory_mb(model, input_shape, runs=5, threads=None):
    """Peak extra resident memory (MB) used during PyTorch CPU inference."""
    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = torch.randn(*input_shape)
    with PeakRSSMonitor() as mon:
        for _ in range(runs):
            model(x)
    torch.set_num_threads(old)
    return mon.peak_delta_mb


def make_ort_session(onnx_path, threads=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def ort_cpu_latency(session, input_shape, warmup=10, runs=50):
    """Measure ONNX Runtime CPU latency (ms) for a fixed input shape."""
    name = session.get_inputs()[0].name
    x = np.random.randn(*input_shape).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return _summ(times)


def ort_peak_memory_mb(onnx_path, input_shape, runs=5, threads=None):
    """Peak extra resident memory (MB) for creating an ORT session and running inference."""
    name = None
    x = np.random.randn(*input_shape).astype(np.float32)
    with PeakRSSMonitor() as mon:
        sess = make_ort_session(onnx_path, threads)
        name = sess.get_inputs()[0].name
        for _ in range(runs):
            sess.run(None, {name: x})
    del sess
    return mon.peak_delta_mb
