"""
Hardware-aware profiling of NAS candidates on the actual CPU.

For every candidate architecture we measure (not estimate):
  * CPU inference latency  - batch 1, median over N runs, single intra-op thread
                              (emulates a single-core edge CPU and gives stable numbers),
  * storage size           - bytes of the serialised fp32 state dict,
  * peak inference memory  - peak resident-memory increase during inference, measured
                              in a *fresh subprocess* so that memory already held by the
                              search process (data set, previous models) does not hide it.
and reject the architecture if it violates the deployment constraints.
"""
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass

from common.profiling import model_storage_mb, torch_cpu_latency
from common.utils import count_flops, count_params
from q2_nas.generator import build_model


@dataclass
class DeploymentConstraints:
    max_latency_ms: float = 4.0        # batch-1, 1 CPU thread
    max_size_mb: float = 1.0           # fp32 weights on disk (fits a microcontroller-class flash budget)
    max_peak_mem_mb: float = 16.0      # extra RSS during inference
    max_params_m: float = 0.3          # sanity cap

    def check(self, prof):
        """Return list of violated constraints (empty list -> architecture accepted)."""
        v = []
        if prof["latency_ms"] > self.max_latency_ms:
            v.append(f"latency {prof['latency_ms']:.2f}ms > {self.max_latency_ms}ms")
        if prof["size_mb"] > self.max_size_mb:
            v.append(f"size {prof['size_mb']:.2f}MB > {self.max_size_mb}MB")
        if prof.get("peak_mem_mb") is not None and prof["peak_mem_mb"] > self.max_peak_mem_mb:
            v.append(f"peak_mem {prof['peak_mem_mb']:.1f}MB > {self.max_peak_mem_mb}MB")
        if prof["params_m"] > self.max_params_m:
            v.append(f"params {prof['params_m']:.2f}M > {self.max_params_m}M")
        return v

    def to_dict(self):
        return asdict(self)


def peak_memory_subprocess(cfg, runs=10, timeout=120):
    """Run q2_nas.mem_probe in a clean interpreter and return peak inference memory in MB."""
    try:
        out = subprocess.run([sys.executable, "-m", "q2_nas.mem_probe", json.dumps(cfg), str(runs)],
                             capture_output=True, text=True, timeout=timeout,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return json.loads(out.stdout.strip().splitlines()[-1])["peak_mem_mb"]
    except Exception as e:                       # "where supported": fall back gracefully
        print(f"    [profiler] peak-memory probe failed: {e}")
        return None


def profile_architecture(cfg, latency_runs=50, measure_memory=True, threads=1):
    model = build_model(cfg).eval()
    r = cfg["resolution"]
    lat = torch_cpu_latency(model, (1, 3, r, r), warmup=10, runs=latency_runs, threads=threads)
    prof = {
        "params_m": count_params(model) / 1e6,
        "mflops": count_flops(model, (1, 3, r, r)) / 1e6,
        "latency_ms": lat["latency_median_ms"],
        "latency_p95_ms": lat["latency_p95_ms"],
        "size_mb": model_storage_mb(model),
        "peak_mem_mb": peak_memory_subprocess(cfg) if measure_memory else None,
    }
    return prof
