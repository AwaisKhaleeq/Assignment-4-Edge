"""
Q1 - Baseline pipeline: train the student architectures WITHOUT knowledge distillation
(plain cross-entropy on the hard labels). These are the reference models KD is compared to.

Example
    python -m q1_kd.train_baseline --students mobilenetv2 squeezenet efficientnet_lite0 --epochs 5
"""
import argparse

from common.data import NUM_CLASSES, get_loaders
from common.utils import banner, count_params, get_device, save_json, set_seed
from q1_kd.engine import StudentRun, train_students
from q1_kd.students import STUDENTS, build_student


def parse_args():
    p = argparse.ArgumentParser(description="Baseline (no-KD) training of student models on CIFAR-100")
    p.add_argument("--students", nargs="+", default=list(STUDENTS), choices=list(STUDENTS))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--data", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    dev = get_device(a.device)
    train, _, test = get_loaders("cifar100", a.data, a.batch_size, train_subset=a.train_subset, seed=a.seed)
    runs = []
    for name in a.students:
        m = build_student(name, NUM_CLASSES["cifar100"])
        print(f"  {name:<20s} params = {count_params(m) / 1e6:.2f}M")
        runs.append(StudentRun(name, m, dev, a.lr, a.wd, a.epochs, len(train),
                               f"results/q1/logs/{name}_baseline.csv",
                               f"checkpoints/students/{name}_baseline.pt"))
    banner(f"BASELINE training (no KD) | {a.students} | epochs={a.epochs} | device={dev}")
    summary = train_students(runs, train, test, dev, a.epochs, tag="baseline")
    save_json({"config": vars(a), "results": summary}, "results/q1/baseline_summary.json")
    for k, v in summary.items():
        print(f"[baseline] {k:<20s} best test top1 = {v['best_test_top1']:.2f}%")


if __name__ == "__main__":
    main()
