"""
Q1 - Distillation pipeline: train students with the KD framework (q1_kd/distill.py)
using the trained ResNet-50 teacher.

Example
    python -m q1_kd.train_kd --teacher checkpoints/teacher_resnet50.pt --epochs 5 \
        --T 4 --alpha 0.9 --beta 1.0 --kd-mode both
"""
import argparse

from common.data import NUM_CLASSES, get_loaders
from common.utils import banner, count_params, get_device, load_checkpoint, save_json, set_seed
from q1_kd.distill import DistillationLoss
from q1_kd.engine import StudentRun, train_students
from q1_kd.students import STUDENTS, build_student
from q1_kd.teacher import ResNet50Teacher


def parse_args():
    p = argparse.ArgumentParser(description="Knowledge-distillation training of student models on CIFAR-100")
    p.add_argument("--teacher", default="checkpoints/teacher_resnet50.pt")
    p.add_argument("--students", nargs="+", default=list(STUDENTS), choices=list(STUDENTS))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--T", type=float, default=4.0, help="softmax temperature")
    p.add_argument("--alpha", type=float, default=0.9, help="weight of soft-target loss (1-alpha for CE)")
    p.add_argument("--beta", type=float, default=1.0, help="weight of feature hint loss")
    p.add_argument("--kd-mode", default="both", choices=["logit", "feature", "both"])
    p.add_argument("--suffix", default="kd", help="checkpoint/log suffix, e.g. kd, kd_logit")
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

    teacher = ResNet50Teacher(NUM_CLASSES["cifar100"], pretrained=False)
    meta = load_checkpoint(a.teacher, teacher)
    teacher.input_res = meta.get("res", 64)
    teacher.to(dev).eval()
    print(f"  teacher loaded: {a.teacher} (test top1 {meta.get('test_top1', float('nan')):.2f}%)")

    use_feat = a.kd_mode in ("feature", "both")
    runs = []
    for name in a.students:
        m = build_student(name, NUM_CLASSES["cifar100"])
        print(f"  {name:<20s} params = {count_params(m) / 1e6:.2f}M")
        runs.append(StudentRun(name, m, dev, a.lr, a.wd, a.epochs, len(train),
                               f"results/q1/logs/{name}_{a.suffix}.csv",
                               f"checkpoints/students/{name}_{a.suffix}.pt",
                               teacher_channels=teacher.feature_channels if use_feat else None))
    kd_loss = DistillationLoss(T=a.T, alpha=a.alpha, beta=a.beta, mode=a.kd_mode)
    banner(f"KD training | mode={a.kd_mode} T={a.T} alpha={a.alpha} beta={a.beta} | epochs={a.epochs}")
    summary = train_students(runs, train, test, dev, a.epochs, teacher=teacher, kd_loss=kd_loss, tag=a.suffix)
    save_json({"config": vars(a), "teacher_top1": meta.get("test_top1"), "results": summary},
              f"results/q1/{a.suffix}_summary.json")
    for k, v in summary.items():
        print(f"[{a.suffix}] {k:<20s} best test top1 = {v['best_test_top1']:.2f}%")


if __name__ == "__main__":
    main()
