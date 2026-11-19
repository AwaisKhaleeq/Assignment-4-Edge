# Edge AI: Knowledge Distillation & Hardware-Aware NAS

**Assignment 4 – Edge Computing & Intelligent Systems**
Muhammad Awais Khaleeq · 22i-2039 · Section A

This repository contains two end-to-end frameworks for building models that run on resource-constrained edge CPUs:

| Part | What | Dataset |
|---|---|---|
| **Q1** | Knowledge distillation: ResNet-50 teacher → MobileNetV2, SqueezeNet and EfficientNet-Lite0 students (soft targets with temperature, combined hard + soft loss, feature-based hint loss), plus a baseline pipeline without KD | CIFAR-100 |
| **Q2** | Hardware-aware evolutionary Neural Architecture Search with real CPU latency, storage-size and peak-memory measurements and constraint-based rejection | CIFAR-10 |
| **Deploy** | Unified inference pipeline, ONNX export, ONNX Runtime CPU inference and automated benchmarking for every model | both |

The technical report with all results and screenshots is in [`report/Technical_Report.pdf`](report/Technical_Report.pdf).

---

## 1. Repository layout

```
edge-kd-nas/
├── common/                    shared code
│   ├── data.py                CIFAR loading (auto-download), fast in-memory batch loader + augmentation
│   ├── utils.py               training helpers, metrics, checkpoints, CSV logging, FLOP counting
│   └── profiling.py           CPU latency (PyTorch / ORT), storage size, peak-RSS memory monitor
├── q1_kd/                     QUESTION 1 – knowledge distillation
│   ├── teacher.py             ResNet-50 teacher
│   ├── students/              MobileNetV2, SqueezeNet, EfficientNet-Lite0 (implemented from scratch)
│   ├── distill.py             KD framework: temperature softmax, soft-target KL, hard+soft, feature hints
│   ├── engine.py              lock-step training engine (shared teacher forward pass)
│   ├── train_teacher.py       teacher training script
│   ├── train_baseline.py      baseline pipeline (no KD)
│   └── train_kd.py            distillation pipeline
├── q2_nas/                    QUESTION 2 – hardware-aware NAS
│   ├── search_space.py        searchable space (depth, channels, kernels, DS-conv, activations, resolution)
│   ├── generator.py           config → nn.Module architecture generator
│   ├── hw_profiler.py         latency / size / peak-memory measurement + deployment constraints
│   ├── mem_probe.py           peak-memory probe (fresh process)
│   ├── trainer.py             automated training & validation of candidates (CIFAR-10)
│   ├── search.py              regularized (aging) evolution search
│   ├── train_best.py          re-train the best architecture from scratch
│   └── plots.py               NAS figures
├── deploy/                    inference / deployment (both questions)
│   ├── registry.py            model registry (all students, NAS model, teacher)
│   ├── predictor.py           unified PyTorch / ONNX Runtime predictor
│   ├── infer.py               unified inference CLI (image, random samples, test-set evaluation)
│   ├── export_onnx.py         ONNX export + PyTorch/ORT parity check
│   ├── onnx_infer.py          ONNX Runtime CPU inference CLI
│   ├── mem_probe.py           peak inference memory probe
│   └── benchmark.py           automated benchmark → CSV + figures
├── checkpoints/               trained weights (.pt)
│   ├── teacher_resnet50.pt
│   ├── students/{mobilenetv2,squeezenet,efficientnet_lite0}_{baseline,kd}.pt
│   └── nas_best.pt
├── onnx_models/               exported ONNX models (students + NAS model)
├── results/                   all experimental results
│   ├── q1/                    training logs (CSV per epoch), summaries (JSON), console logs
│   ├── q2/                    search_history.csv/.json, best_arch.json, best-arch training log
│   ├── benchmark/             benchmark.csv, kd_vs_baseline.csv
│   ├── figures/               plots used in the report
│   └── onnx_export.csv        ONNX file sizes + numerical parity
├── report/                    Technical_Report.pdf and its generator
├── scripts/run_all.sh|.ps1    run the complete pipeline
└── requirements.txt
```

## 2. Installation

Requirements: Python 3.10+ (tested with 3.11). A GPU is **optional**; everything runs on CPU.

```bash
git clone <this-repo-url> edge-kd-nas
cd edge-kd-nas
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA GPU, install the matching PyTorch build from https://pytorch.org first; the code picks `cuda` automatically.

**Datasets** are downloaded automatically on first use (torchvision). On first use each dataset is cached as `data/<name>.npz`.
If the official mirror is slow, put the HuggingFace parquet files (`uoft-cs/cifar10`, `uoft-cs/cifar100`) into `data/hf/`
as `c10_train.parquet`, `c10_test.parquet`, `c100_train.parquet`, `c100_test.parquet` and they will be used instead.

All commands below are run **from the repository root**.

## 3. Step-by-step execution

### Quick: everything in one go
```bash
bash scripts/run_all.sh                                          # Linux / macOS / Git-Bash
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1     # Windows PowerShell
```

### Question 1 – Knowledge distillation (CIFAR-100)

```bash
# 1) Teacher: ResNet-50 (ImageNet-initialised, fine-tuned on CIFAR-100, internal input 64x64)
python -m q1_kd.train_teacher --epochs 2 --res 64

# 2) Baseline students WITHOUT distillation (cross-entropy only)
python -m q1_kd.train_baseline --students mobilenetv2 squeezenet efficientnet_lite0 --epochs 5

# 3) Distilled students (soft targets + temperature + hard/soft mix + feature hints)
python -m q1_kd.train_kd --teacher checkpoints/teacher_resnet50.pt --epochs 5 \
       --T 4 --alpha 0.9 --beta 1.0 --kd-mode both
#    --kd-mode logit    -> only soft-target KD (Hinton)
#    --kd-mode feature  -> only feature-based KD (FitNets-style hints)
#    --suffix kd_logit  -> save under a different name for ablations
```

Loss used by the KD framework:

```
L = (1 - α) · CE(z_s, y)  +  α · T² · KL( softmax(z_t / T) ‖ softmax(z_s / T) )  +  β · ‖ norm(r(F_s)) − norm(F_t) ‖²
```

### Question 2 – Hardware-aware NAS (CIFAR-10)

```bash
# 1) Evolutionary search (candidates violating latency/size/memory constraints are rejected before training)
python -m q2_nas.search --population 8 --sample 3 --budget 20 --epochs 3 --train-subset 10000 \
       --max-latency-ms 4 --max-size-mb 1.0 --max-peak-mem-mb 16
#    -> results/q2/search_history.csv, search_history.json, best_arch.json

# 2) Train the best discovered architecture from scratch on the full training set
python -m q2_nas.train_best --epochs 15            # -> checkpoints/nas_best.pt

# 3) Figures
python -m q2_nas.plots
```

### Deployment: inference, ONNX, benchmarking (both questions)

Model keys: `mobilenetv2_baseline`, `mobilenetv2_kd`, `squeezenet_baseline`, `squeezenet_kd`,
`efficientnet_lite0_baseline`, `efficientnet_lite0_kd`, `nas_best`, `teacher_resnet50`
(or the groups `students` / `all`).

```bash
# Unified PyTorch inference
python -m deploy.infer --models students --samples 5            # predictions on random test images
python -m deploy.infer --models mobilenetv2_kd --image cat.jpg  # classify your own image
python -m deploy.infer --models all --eval                      # test-set accuracy of every model

# ONNX export (+ numerical parity check vs. PyTorch)
python -m deploy.export_onnx --models all                       # -> onnx_models/*.onnx, results/onnx_export.csv

# ONNX Runtime CPU inference (no PyTorch model code needed)
python -m deploy.onnx_infer --models students nas_best --eval
python -m deploy.onnx_infer --models nas_best --image ship.png

# Automated benchmark (accuracy, params, FLOPs, size, latency, throughput, peak memory)
python -m deploy.benchmark --models all                         # -> results/benchmark/*.csv, results/figures/*.png

# Rebuild the technical report PDF
python report/build_report.py
```

## 4. Compute budget / reproducing with more epochs

The submitted results were produced on a **4-core laptop CPU (Intel i5-1135G7, no GPU)**, so the training
budgets are intentionally small (teacher 2 epochs, students 5 epochs, 20 NAS candidates × 3 proxy epochs,
best NAS model 15 epochs). The code is not limited to this – for a full-budget run on a GPU, e.g.:

```bash
python -m q1_kd.train_teacher  --epochs 30 --res 128
python -m q1_kd.train_baseline --epochs 120
python -m q1_kd.train_kd       --epochs 120
python -m q2_nas.search        --budget 100 --epochs 10 --train-subset 45000
python -m q2_nas.train_best    --epochs 200
```

## 5. Notes on design choices

* **CIFAR adaptation of the students.** The ImageNet stems (stride 2) are replaced by stride-1 stems so 32×32
  images are not over-downsampled; the final feature maps are 2×2 (MobileNetV2, EfficientNet-Lite0) and
  4×4 (SqueezeNet).
* **Teacher.** ResNet-50 is initialised with ImageNet weights and fine-tuned; inputs are up-sampled to 64×64
  inside the model so that teacher and students consume the same 32×32 batch during distillation.
* **Feature distillation.** A 1×1-conv + BN regressor maps the student's last feature map to the teacher's
  2048-channel layer-4 map; the regressor is used only during training and is not exported.
* **Lock-step training.** All students see exactly the same mini-batch sequence, and in KD mode the teacher
  is evaluated once per batch, which makes baseline vs. KD a controlled comparison.
* **Hardware measurements.** Latency is the median of repeated batch-1 runs with 1 intra-op thread (an
  edge-like single core); storage = serialised fp32 weights; peak memory = peak resident-set increase
  during inference, measured in a fresh process (psutil). Values are machine-specific.
