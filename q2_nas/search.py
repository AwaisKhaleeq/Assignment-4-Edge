"""
Q2 - Hardware-aware evolutionary Neural Architecture Search (regularized / aging evolution).

Algorithm (Real et al., 2019 "Regularized Evolution", made hardware-aware):

    population <- P random architectures that SATISFY the deployment constraints
    while #trained candidates < budget:
        sample S individuals from the population, take the fittest as parent (tournament)
        child  <- mutate(parent)                      (one gene changes)
        profile child on the CPU: latency, storage size, peak memory
        if child violates a constraint -> REJECT (logged, never trained)
        else -> train on CIFAR-10 (proxy budget), validate, compute fitness
                push child into population, remove the OLDEST individual (aging)

Fitness (MnasNet-style soft latency reward on top of the hard constraints):
    fitness = val_acc * (latency / latency_target) ** w,  w = -0.07

Everything (accepted *and* rejected candidates) is logged to CSV + JSON, and the best
architecture is written to results/q2/best_arch.json.

Example
    python -m q2_nas.search --population 8 --sample 3 --budget 24 --epochs 3 --train-subset 10000
"""
import argparse
import collections
import json
import random
import time

import torch

from common.data import get_loaders
from common.utils import banner, get_device, save_json, set_seed, CSVLogger
from q2_nas.generator import build_model
from q2_nas.hw_profiler import DeploymentConstraints, profile_architecture
from q2_nas.search_space import arch_key, describe, mutate, sample_config, space_size
from q2_nas.trainer import train_and_validate

FIELDS = ["id", "status", "origin", "parent_id", "mutation", "arch", "val_top1", "fitness",
          "latency_ms", "latency_p95_ms", "size_mb", "peak_mem_mb", "params_m", "mflops",
          "train_time_s", "reject_reason", "config"]


def parse_args():
    p = argparse.ArgumentParser(description="Hardware-aware evolutionary NAS on CIFAR-10")
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--sample", type=int, default=3, help="tournament size")
    p.add_argument("--budget", type=int, default=24, help="number of candidates to train")
    p.add_argument("--max-attempts", type=int, default=200, help="cap on sampled candidates (incl. rejected)")
    p.add_argument("--epochs", type=int, default=3, help="proxy training epochs per candidate")
    p.add_argument("--train-subset", type=int, default=10000)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--max-latency-ms", type=float, default=4.0)
    p.add_argument("--max-size-mb", type=float, default=1.0)
    p.add_argument("--max-peak-mem-mb", type=float, default=16.0)
    p.add_argument("--max-params-m", type=float, default=0.3)
    p.add_argument("--latency-weight", type=float, default=-0.07)
    p.add_argument("--out", default="results/q2")
    p.add_argument("--data", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


class NASSearch:
    def __init__(self, a):
        self.a = a
        self.rng = random.Random(a.seed)
        self.dev = get_device(a.device)
        self.cons = DeploymentConstraints(a.max_latency_ms, a.max_size_mb, a.max_peak_mem_mb, a.max_params_m)
        self.history, self.seen = [], {}
        self.csv = CSVLogger(f"{a.out}/search_history.csv", FIELDS)
        self.loaders = {}

    def loader(self, res):
        """Data loaders are cached per input resolution (variable-resolution search)."""
        if res not in self.loaders:
            tr, va, _ = get_loaders("cifar10", self.a.data, self.a.batch_size, val_size=self.a.val_size,
                                    train_subset=self.a.train_subset, resolution=res, seed=self.a.seed)
            self.loaders[res] = (tr, va)
        return self.loaders[res]

    def fitness(self, acc, lat):
        return acc * (lat / self.a.max_latency_ms) ** self.a.latency_weight

    def evaluate(self, cfg, origin, parent_id=None, mutation=""):
        """Profile -> constraint check -> (train + validate). Returns the history record."""
        cid = len(self.history)
        rec = {"id": cid, "origin": origin, "parent_id": parent_id, "mutation": mutation,
               "arch": describe(cfg), "config": json.dumps(cfg)}
        key = arch_key(cfg)
        if key in self.seen:                                   # duplicate phenotype: reuse result
            prev = self.history[self.seen[key]]
            rec.update({k: prev[k] for k in FIELDS if k in prev and k not in rec})
            rec["status"] = "duplicate"
            self._log(rec)
            return rec

        prof = profile_architecture(cfg)
        rec.update(prof)
        violations = self.cons.check(prof)
        hw = (f"lat {prof['latency_ms']:.2f}ms | size {prof['size_mb']:.2f}MB | "
              f"mem {prof['peak_mem_mb']}MB | {prof['params_m'] * 1e3:.0f}K params | {prof['mflops']:.1f} MFLOPs")
        if violations:
            rec.update({"status": "rejected", "reject_reason": "; ".join(violations)})
            print(f"[#{cid:03d}] REJECTED  {describe(cfg)}\n         {hw}\n         -> {rec['reject_reason']}")
        else:
            set_seed(self.a.seed + cid)
            tr, va = self.loader(cfg["resolution"])
            res = train_and_validate(build_model(cfg), tr, va, self.dev, self.a.epochs, lr=self.a.lr)
            rec.update({"status": "trained", "val_top1": res["val_top1"], "train_time_s": res["train_time_s"],
                        "fitness": self.fitness(res["val_top1"], prof["latency_ms"])})
            print(f"[#{cid:03d}] TRAINED   {describe(cfg)}\n         {hw}\n"
                  f"         -> val acc {res['val_top1']:.2f}% | fitness {rec['fitness']:.2f} "
                  f"| {res['train_time_s']:.0f}s", flush=True)
        self.seen[key] = cid
        self._log(rec)
        return rec

    def _log(self, rec):
        self.history.append(rec)
        self.csv.log(rec)
        save_json(self.history, f"{self.a.out}/search_history.json")

    def run(self):
        a = self.a
        banner(f"Hardware-aware evolutionary NAS | space size ~{space_size():.2e} | constraints {self.cons.to_dict()}")
        trained = lambda: sum(r["status"] == "trained" for r in self.history)
        population = collections.deque()
        t0 = time.time()

        # 1) initial population: random architectures that satisfy the constraints
        while len(population) < a.population and len(self.history) < a.max_attempts:
            rec = self.evaluate(sample_config(self.rng), "random")
            if rec["status"] == "trained":
                population.append(rec)

        # 2) evolution loop (tournament selection + mutation + aging)
        while trained() < a.budget and len(self.history) < a.max_attempts:
            tournament = self.rng.sample(list(population), min(a.sample, len(population)))
            parent = max(tournament, key=lambda r: r["fitness"])
            child_cfg, desc = mutate(json.loads(parent["config"]), self.rng)
            rec = self.evaluate(child_cfg, "mutation", parent["id"], desc)
            if rec["status"] == "trained":
                population.append(rec)
                population.popleft()                           # remove the oldest (regularisation)

        # 3) summary
        valid = [r for r in self.history if r["status"] == "trained"]
        best = max(valid, key=lambda r: r["fitness"])
        n_rej = sum(r["status"] == "rejected" for r in self.history)
        summary = {
            "best_id": best["id"], "best_arch": best["arch"], "config": json.loads(best["config"]),
            "val_top1_proxy": best["val_top1"], "fitness": best["fitness"],
            "latency_ms": best["latency_ms"], "size_mb": best["size_mb"], "peak_mem_mb": best["peak_mem_mb"],
            "params_m": best["params_m"], "mflops": best["mflops"],
            "constraints": self.cons.to_dict(), "search_args": vars(a),
            "n_candidates": len(self.history), "n_trained": len(valid), "n_rejected": n_rej,
            "search_time_min": (time.time() - t0) / 60}
        save_json(summary, f"{a.out}/best_arch.json")
        banner("NAS finished")
        print(f"candidates: {len(self.history)} | trained: {len(valid)} | rejected: {n_rej} | "
              f"time: {summary['search_time_min']:.1f} min")
        print(f"BEST #{best['id']}: {best['arch']}\n  proxy val acc {best['val_top1']:.2f}% | "
              f"latency {best['latency_ms']:.2f} ms | size {best['size_mb']:.3f} MB | "
              f"peak mem {best['peak_mem_mb']} MB -> {a.out}/best_arch.json")
        return summary


if __name__ == "__main__":
    args = parse_args()
    NASSearch(args).run()
