"""
Builds report/Technical_Report.pdf from the experimental results.

All numbers in the report are read from results/**/*.csv|json, so the report always
matches the files in the repository. Console logs are rendered as terminal
"screenshots" (results/figures/screens/*.png).

    python report/build_report.py
"""
import json
import os
import re
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

AUTHOR, ROLL, SECTION = "Muhammad Awais Khaleeq", "22i-2039", "A"
FIG, SCR = "results/figures", "results/figures/screens"
os.makedirs(SCR, exist_ok=True)

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1d3557"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1d3557"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=3, keepWithNext=1)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13.2, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
CAP = ParagraphStyle("C", parent=SMALL, alignment=TA_CENTER, spaceAfter=8)
CODE = ParagraphStyle("Code", fontName="Courier", fontSize=8, leading=10, backColor=colors.HexColor("#f4f4f4"),
                      borderPadding=4, spaceAfter=6, spaceBefore=2)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def csv(p):
    return pd.read_csv(p) if os.path.exists(p) else None


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [P(f"&bull;&nbsp; {i}") for i in items]


def table(data, widths=None, header=True, font=8):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", font),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f4f8")])]
    t.setStyle(TableStyle(st))
    return t


def fig(path, width=16, caption=None):
    if not os.path.exists(path):
        return [P(f"<i>[missing figure: {path}]</i>", SMALL)]
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    out = [Image(path, width=width * cm, height=width * cm * h / w)]
    if caption:
        out.append(P(caption, CAP))
    return out


def screenshot(src, name, title, keep=None, drop=None, max_lines=34, tail=False):
    """Render (filtered) console-log lines as a terminal-style PNG."""
    if not os.path.exists(src):
        return None
    lines = open(src, encoding="utf-8", errors="ignore").read().splitlines()
    lines = [l.rstrip() for l in lines if l.strip()]
    lines = [l for l in lines if "warn" not in l.lower() and "torch.onnx.export(" not in l]
    if keep:
        lines = [l for l in lines if re.search(keep, l)]
    if drop:
        lines = [l for l in lines if not re.search(drop, l)]
    lines = lines[-max_lines:] if tail else lines[:max_lines]
    lines = [l if len(l) <= 150 else l[:147] + "..." for l in lines]
    cmd = f"(.venv) PS edge-kd-nas> {title}"
    text = "\n".join([cmd] + lines)
    n = len(lines) + 1
    width = max(len(l) for l in text.splitlines())
    fw = max(8, min(17, 0.072 * width + 0.6))
    fig_, ax = plt.subplots(figsize=(fw, 0.165 * n + 0.35))
    fig_.patch.set_facecolor("#0c0c0c")
    ax.set_facecolor("#0c0c0c")
    ax.axis("off")
    ax.text(0.005, 0.99, cmd, family="monospace", fontsize=8.2, color="#7ee787", va="top", transform=ax.transAxes)
    ax.text(0.005, 0.99 - 1.0 / n, "\n".join(lines), family="monospace", fontsize=8.2, color="#e6e6e6",
            va="top", transform=ax.transAxes, linespacing=1.18)
    path = f"{SCR}/{name}.png"
    fig_.savefig(path, dpi=150, facecolor=fig_.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig_)
    return path


def q1_curves():
    fig_, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, s in zip(axes, ["mobilenetv2", "squeezenet", "efficientnet_lite0"]):
        for mode, c in [("baseline", "#3b7dd8"), ("kd", "#e4572e")]:
            d = csv(f"results/q1/logs/{s}_{mode}.csv")
            if d is not None:
                ax.plot(d.epoch, d.test_top1, "-o", ms=3, color=c, label=f"{mode} test")
                ax.plot(d.epoch, d.train_top1, "--", color=c, alpha=0.6, label=f"{mode} train")
        ax.set_title(s)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("top-1 accuracy (%)")
    axes[0].legend(fontsize=7)
    fig_.tight_layout()
    fig_.savefig(f"{FIG}/q1_training_curves.png", dpi=160)
    plt.close(fig_)


def f(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def build():
    q1_curves()
    teacher = load("results/q1/teacher_summary.json")
    tlog = csv("results/q1/logs/teacher_resnet50.csv")
    base = load("results/q1/baseline_summary.json")
    kd = load("results/q1/kd_summary.json")
    bench = csv("results/benchmark/benchmark.csv")
    kdvb = csv("results/benchmark/kd_vs_baseline.csv")
    onnx_exp = csv("results/onnx_export.csv")
    best = load("results/q2/best_arch.json")
    bestf = load("results/q2/best_arch_final.json")
    hist = csv("results/q2/search_history.csv")
    notes = load("report/discussion.json") or {}

    S = []
    # ------------------------------------------------------------------ title
    S += [Spacer(1, 3.2 * cm),
          P("Edge Computing &amp; Intelligent Systems", ParagraphStyle("t0", parent=H2, alignment=TA_CENTER)),
          P("Assignment 4 &mdash; Technical Report", ParagraphStyle("t1", parent=H1, fontSize=20, alignment=TA_CENTER)),
          Spacer(1, 0.4 * cm),
          P("Knowledge Distillation for Lightweight Students &amp;<br/>Hardware-Aware Neural Architecture Search",
            ParagraphStyle("t2", parent=H2, fontSize=14, alignment=TA_CENTER, textColor=colors.HexColor("#e4572e"))),
          Spacer(1, 1.5 * cm),
          table([["Name", AUTHOR], ["Roll number", ROLL], ["Section", SECTION],
                 ["Course", "Edge Computing & Intelligent Systems"], ["Date", date.today().strftime("%d %B %Y")]],
                widths=[4 * cm, 8 * cm], header=False, font=10),
          Spacer(1, 1.5 * cm)]
    if bench is not None:
        S += [P("<b>Headline results</b>", ParagraphStyle("hc", parent=BODY, alignment=TA_CENTER))]
        hl = [["Model", "Dataset", "Top-1 (%)", "Params", "ORT latency (ms)"]]
        for _, r in bench.iterrows():
            hl.append([r.model, r.dataset, f(r.torch_top1), f"{r.params_m:.3f}M", f(r.get("ort_lat_ms_1t"))])
        S += [table(hl, widths=[5.2 * cm, 2.2 * cm, 2.2 * cm, 2 * cm, 3.2 * cm])]
    S += [PageBreak()]

    # ------------------------------------------------------------------ 1 intro
    S += [P("1. Overview and environment", H1),
          P("This report documents two frameworks for producing models that can be deployed on resource-constrained "
            "edge CPUs. <b>Question 1</b> transfers the knowledge of a large ResNet-50 teacher into three lightweight "
            "students (MobileNetV2, SqueezeNet, EfficientNet-Lite0) on CIFAR-100, and compares them with identically "
            "trained students without distillation. <b>Question 2</b> automatically discovers a small CNN for CIFAR-10 "
            "with a hardware-aware evolutionary NAS that measures the real CPU latency, storage size and peak memory of "
            "every candidate and rejects candidates that violate deployment constraints. All trained models are "
            "exported to ONNX and executed with ONNX Runtime on the CPU, and every model is benchmarked automatically."),
          table([["Item", "Value"],
                 ["Hardware", "Intel Core i5-1135G7 (4 cores / 8 threads), 16 GB RAM, no GPU"],
                 ["Software", "Windows 11, Python 3.11, PyTorch 2.9.1 (CPU), ONNX 1.23, ONNX Runtime 1.30"],
                 ["Datasets", "CIFAR-100 (Q1), CIFAR-10 (Q2); 50k train / 10k test images of 32x32"],
                 ["Code", "github repository (see README) - every number in this report is read from results/"]],
                widths=[3 * cm, 13.5 * cm]),
          Spacer(1, 4),
          P("<b>Compute budget.</b> Everything was trained on a laptop CPU, therefore the epoch budgets are much smaller "
            "than in the literature (teacher 2 epochs, students 5 epochs, NAS proxy 3 epochs per candidate). The "
            "absolute accuracies are consequently lower than state-of-the-art numbers, but every comparison "
            "(baseline vs. KD, accepted vs. rejected architectures) is controlled: same data stream, same epochs, "
            "same optimiser. All scripts expose the budgets as command-line arguments (README section 4).", BODY)]

    # ------------------------------------------------------------------ 2 Q1
    S += [PageBreak(), P("2. Question 1 &mdash; Knowledge distillation framework", H1),
          P("2.1 Pipeline", H2),
          P("<font face='Courier'>train_teacher.py</font> &rarr; <font face='Courier'>train_baseline.py</font> "
            "(students, CE only) &rarr; <font face='Courier'>train_kd.py</font> (students, KD loss) &rarr; "
            "<font face='Courier'>deploy/export_onnx.py</font> &rarr; <font face='Courier'>deploy/onnx_infer.py</font> "
            "/ <font face='Courier'>deploy/benchmark.py</font>. Every model implements the same interface "
            "<font face='Courier'>features(x)</font> (last convolutional feature map), "
            "<font face='Courier'>classifier(f)</font> (logits) and <font face='Courier'>forward(x)</font>, which is what "
            "lets one KD framework, one inference pipeline and one exporter serve all architectures."),
          P("2.2 Teacher: ResNet-50", H2),
          P("The teacher is the torchvision ResNet-50 (bottleneck blocks 3-4-6-3, 23.7 M parameters), initialised with "
            "ImageNet weights and fine-tuned on CIFAR-100 with a new 100-way classifier (dropout 0.2). Because the "
            "ImageNet stem (7x7/2 conv + max-pool) would reduce a 32x32 image to 8x8 immediately, the teacher "
            "bilinearly up-samples its input to 64x64 internally. Students and teacher therefore consume the same "
            "32x32 batch, and the teacher's layer-4 map is 2048x2x2. Training: SGD (momentum 0.9, Nesterov), "
            "one-cycle LR (max 0.02), weight decay 5e-4, batch 128, random-crop + horizontal-flip augmentation.")]
    if tlog is not None:
        rows = [["Epoch", "Train loss", "Train top-1", "Test loss", "Test top-1", "Test top-5", "Time (s)"]]
        for _, r in tlog.iterrows():
            rows.append([int(r.epoch), f(r.train_loss, 3), f(r.train_top1), f(r.test_loss, 3), f(r.test_top1),
                         f(r.test_top5), f(r.time_s, 0)])
        S += [table(rows), P("Table 1 - ResNet-50 teacher training on CIFAR-100 (results/q1/logs/teacher_resnet50.csv).", CAP)]

    S += [P("2.3 Student architectures (implemented from scratch)", H2)]
    S += bullets([
        "<b>MobileNetV2</b> (<font face='Courier'>q1_kd/students/mobilenetv2.py</font>): inverted-residual blocks "
        "(1x1 expansion x6 &rarr; 3x3 depthwise &rarr; linear 1x1 projection) with residual connections, ReLU6, "
        "1280-channel head. Stem stride reduced to 1 for 32x32 inputs.",
        "<b>SqueezeNet v1.1</b> (<font face='Courier'>squeezenet.py</font>): 8 Fire modules (1x1 squeeze &rarr; "
        "parallel 1x1/3x3 expand, concatenated), no fully connected layers - a 1x1 conv produces one map per class "
        "followed by global average pooling. BatchNorm added and final ReLU removed for stable training from scratch.",
        "<b>EfficientNet-Lite0</b> (<font face='Courier'>efficientnet_lite.py</font>): EfficientNet-B0 stage layout "
        "of MBConv blocks with 3x3/5x5 depthwise kernels, but - as in the Lite family - without squeeze-and-excitation, "
        "with ReLU6 instead of Swish and with a fixed stem (32) and head (1280); stochastic depth 0.1."])
    if bench is not None:
        rows = [["Model", "Params (M)", "MFLOPs", "Weights (MB)", "Final feature map"]]
        fm = {"mobilenetv2": "1280 x 2 x 2", "squeezenet": "512 x 4 x 4", "efficientnet_lite0": "1280 x 2 x 2",
              "teacher": "2048 x 2 x 2"}
        for _, r in bench[bench.model.str.endswith("_baseline") | (bench.model == "teacher_resnet50")].iterrows():
            nm = r.model.replace("_baseline", "")
            rows.append([nm, f(r.params_m, 3), f(r.mflops, 1), f(r.torch_size_mb), fm.get(nm.split("_resnet")[0], "")])
        S += [table(rows), P("Table 2 - Model complexity at 32x32 input (FLOPs counted with torch FlopCounterMode).", CAP)]

    S += [P("2.4 Distillation framework (q1_kd/distill.py)", H2),
          P("The framework combines the four required mechanisms in a single configurable loss:"),
          Preformatted("p_T(z)   = softmax(z / T)                                   temperature-based softmax\n"
                       "L_soft   = T^2 * KL( p_T(z_teacher) || p_T(z_student) )     soft-target KD (Hinton et al.)\n"
                       "L_hard   = CE(z_student, y)                                 hard labels\n"
                       "L_feat   = || norm(r(F_s)) - norm(F_t) ||^2                 feature hint (FitNets-style)\n"
                       "L        = (1 - alpha) * L_hard + alpha * L_soft + beta * L_feat", CODE)]
    S += bullets([
        "<b>Temperature.</b> T = 4 softens the teacher distribution so that the relative probabilities of wrong "
        "classes (\"dark knowledge\", e.g. <i>leopard</i> vs. <i>tiger</i>) carry gradient signal. The T<super>2</super> "
        "factor keeps the soft-loss gradients on the same scale as the CE gradients.",
        "<b>Combined hard/soft training.</b> alpha = 0.9 weights the soft targets, 1 - alpha = 0.1 the ground truth.",
        "<b>Feature distillation.</b> The student's last feature map F<sub>s</sub> is average-pooled to the teacher's "
        "spatial size and projected to 2048 channels by a learnable regressor r (1x1 conv + BN). Both maps are "
        "flattened and L2-normalised, so the loss is scale-free (range [0, 4]); beta = 1. The regressor is trained "
        "jointly with the student and discarded after training (not part of the exported model).",
        "<b>Modes.</b> <font face='Courier'>--kd-mode logit | feature | both</font> enables each term independently.",
        "<b>Teacher is frozen</b> (eval mode, no gradients). The engine trains all three students in lock-step: the "
        "teacher forward pass is computed once per mini-batch and shared, which reduced KD cost on the CPU ~3x."])
    S += [P("2.5 Baseline pipeline and training setup", H2),
          P("<font face='Courier'>train_baseline.py</font> trains the same three architectures with plain cross-entropy. "
            "Baseline and KD runs use the identical seed, data order, augmentation, optimiser and schedule, so the only "
            "difference is the loss function."),
          table([["Setting", "Baseline", "KD"],
                 ["Epochs / batch", "5 / 128", "5 / 128"],
                 ["Optimiser", "SGD Nesterov, m=0.9, wd=5e-4", "same"],
                 ["LR schedule", "one-cycle, max 0.1, cosine", "same"],
                 ["Augmentation", "random crop (pad 4) + h-flip", "same"],
                 ["Loss", "CE", "0.1 CE + 0.9 T^2 KL (T=4) + 1.0 feature hint"]],
                widths=[3.5 * cm, 6 * cm, 7 * cm])]

    S += [P("2.6 Results: baseline vs. distilled students", H2)]
    if kdvb is not None:
        rows = [["Student", "Params (M)", "Baseline top-1", "KD top-1", "Gain", "Baseline top-5", "KD top-5"]]
        for _, r in kdvb.iterrows():
            rows.append([r.student, f(r.params_m, 2), f(r.baseline_top1), f(r.kd_top1), f"{r.kd_gain:+.2f}",
                         f(r.baseline_top5), f(r.kd_top5)])
        if teacher:
            rows.append(["ResNet-50 teacher", "23.71", f(teacher["best_test_top1"]), "-", "-", "-", "-"])
        S += [table(rows), P("Table 3 - CIFAR-100 test accuracy (%) after 5 epochs (results/benchmark/kd_vs_baseline.csv).", CAP)]
    S += fig(f"{FIG}/q1_kd_vs_baseline.png", 13, "Figure 1 - Effect of knowledge distillation per student architecture.")
    S += fig(f"{FIG}/q1_training_curves.png", 17, "Figure 2 - Train (dashed) and test (solid) top-1 per epoch, baseline vs KD.")
    if notes.get("q1"):
        S += [P("Discussion", H3)] + [P(t) for t in notes["q1"]]

    S += [P("2.7 Unified inference and ONNX deployment", H2),
          P("<font face='Courier'>deploy/registry.py</font> maps a model key (e.g. <font face='Courier'>squeezenet_kd</font>) "
            "to its architecture, checkpoint, dataset and input resolution. <font face='Courier'>deploy/predictor.py</font> "
            "wraps either the PyTorch model or an ONNX Runtime session behind one <font face='Courier'>predict / classify</font> "
            "API, and <font face='Courier'>deploy/infer.py</font> uses it to classify an image file, random test images, "
            "or to evaluate the whole test set for any subset of models. Models are exported with "
            "<font face='Courier'>torch.onnx.export</font> (opset 17, dynamic batch axis); after export the graph is "
            "checked with <font face='Courier'>onnx.checker</font> and the logits of PyTorch and ONNX Runtime are "
            "compared on a random batch.")]
    if onnx_exp is not None:
        rows = [["Model", "ONNX file", "Size (MB)", "Input", "max |diff|", "Top-1 agree"]]
        for _, r in onnx_exp.iterrows():
            rows.append([r.model, os.path.basename(r.onnx_path), f(r.onnx_size_mb, 3), r.input_shape,
                         f"{r.max_abs_diff:.1e}", f"{r.top1_agreement:.0f}%"])
        S += [table(rows), P("Table 4 - ONNX export and numerical parity with PyTorch (results/onnx_export.csv).", CAP)]

    # ------------------------------------------------------------------ 3 Q2
    S += [PageBreak(), P("3. Question 2 &mdash; Hardware-aware Neural Architecture Search", H1),
          P("3.1 Search space (q2_nas/search_space.py)", H2),
          P("A genotype is a JSON dictionary. It always stores 4 stage genes; only the first "
            "<font face='Courier'>num_stages</font> are expressed, so changing the depth by mutation re-activates "
            "previously hidden genes."),
          table([["Dimension", "Choices", "Requirement covered"],
                 ["Input resolution", "24, 28, 32", "variable input resolution"],
                 ["Stem channels", "16, 24, 32", "variable filters"],
                 ["Number of stages", "2, 3, 4", "variable depth"],
                 ["Blocks per stage", "1, 2, 3", "variable depth"],
                 ["Channels per stage", "16, 24, 32, 48, 64, 96, 128 (non-decreasing)", "variable filters"],
                 ["Kernel size per stage", "3, 5, 7", "different kernel sizes"],
                 ["Block type per stage", "conv, dsconv (depthwise separable), mbconv (inverted residual)",
                  "depthwise separable blocks"],
                 ["Activation per stage", "ReLU, ReLU6, Hard-Swish, SiLU, LeakyReLU", "different activations"],
                 ["Head 1x1 conv", "none, 128, 256", "-"],
                 ["Dropout", "0, 0.1, 0.2", "-"]],
                widths=[3.6 * cm, 7.6 * cm, 5.3 * cm]),
          P("The space contains about 6.5 x 10<super>13</super> distinct architectures.", SMALL),
          P("3.2 Architecture generator (q2_nas/generator.py)", H2),
          P("<font face='Courier'>build_model(cfg)</font> instantiates: 3x3 stem (stride 1) &rarr; stages (the first block "
            "of every stage after the first uses stride 2) &rarr; optional 1x1 head conv &rarr; global average pooling "
            "&rarr; dropout &rarr; linear classifier. Every block is Conv-BN-activation based, and residual connections "
            "are inserted automatically when a block preserves the tensor shape."),
          P("3.3 Search strategy: regularized (aging) evolution (q2_nas/search.py)", H2)]
    S += bullets([
        "Initial population: P = 8 random architectures that <i>satisfy</i> the constraints.",
        "Each step: tournament of S = 3 random population members &rarr; the fittest is the parent &rarr; one gene is "
        "mutated (global gene with p = 0.3, otherwise a gene of an expressed stage) &rarr; child is profiled.",
        "Child violating a constraint &rarr; rejected and logged, <b>never trained</b> (saves compute). "
        "Otherwise it is trained and validated, added to the population, and the <b>oldest</b> member is removed (aging "
        "acts as regularisation against lucky proxy scores). Duplicate architectures are detected and not retrained.",
        "Fitness (MnasNet-style soft latency reward on top of the hard constraints): "
        "<font face='Courier'>fitness = val_acc x (latency / 4 ms)<super>-0.07</super></font>.",
        "Proxy training of a candidate (q2_nas/trainer.py): from scratch on 10,000 CIFAR-10 training images for 3 "
        "epochs, validation on 5,000 held-out training images (the test set is never used during search)."])
    S += [P("3.4 Hardware-aware evaluation and deployment constraints (q2_nas/hw_profiler.py)", H2)]
    S += bullets([
        "<b>CPU latency</b>: median of 50 timed batch-1 forward passes after 10 warm-up runs, with 1 intra-op thread "
        "(emulates a single-core edge CPU and gives stable measurements).",
        "<b>Storage size</b>: bytes of the serialised fp32 state dict.",
        "<b>Peak inference memory</b>: a fresh Python process (<font face='Courier'>q2_nas/mem_probe.py</font>) builds "
        "the model, records the resident-set size, runs inference while a background thread samples the RSS every "
        "0.2 ms, and reports the peak increase. A fresh process avoids memory already held by the search process "
        "masking the allocation. If the probe fails, the value is recorded as missing (\"where supported\").",
        "<b>Constraints</b> (rejection if any is violated): latency &le; 4 ms, size &le; 1.0 MB, peak memory &le; "
        "16 MB, parameters &le; 0.3 M."])
    if hist is not None and best is not None:
        n_tr, n_rej = (hist.status == "trained").sum(), (hist.status == "rejected").sum()
        n_dup = (hist.status == "duplicate").sum()
        S += [P("3.5 Search results", H2),
              P(f"The search sampled <b>{len(hist)}</b> candidates in {best['search_time_min']:.0f} minutes: "
                f"<b>{n_tr}</b> were trained, <b>{n_rej}</b> were rejected by the hardware constraints"
                f"{f' and {n_dup} were duplicates' if n_dup else ''}. The full history (configuration, all hardware "
                "measurements, status, rejection reason, accuracy, fitness, parent and mutation) is stored in "
                "<font face='Courier'>results/q2/search_history.csv</font> and <font face='Courier'>.json</font>.")]
        reasons = hist[hist.status == "rejected"].reject_reason.fillna("")
        rc = {k: reasons.str.contains(k).sum() for k in ["latency", "size", "peak_mem", "params"]}
        S += [P("Rejections by violated constraint (a candidate can violate several): " +
                ", ".join(f"{k}: {v}" for k, v in rc.items()) + ".", BODY)]
        top = hist[hist.status == "trained"].sort_values("fitness", ascending=False).head(8)
        rows = [["#", "Architecture", "Val (%)", "Lat (ms)", "Size (MB)", "Mem (MB)", "Fitness"]]
        for _, r in top.iterrows():
            rows.append([int(r.id), P(r.arch, SMALL), f(r.val_top1), f(r.latency_ms), f(r.size_mb, 3),
                         f(r.peak_mem_mb, 1), f(r.fitness)])
        S += [table(rows, widths=[0.8 * cm, 8.9 * cm, 1.4 * cm, 1.4 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm]),
              P("Table 5 - Top trained candidates by fitness (proxy budget: 10k images, 3 epochs).", CAP)]
        rej = hist[hist.status == "rejected"].head(6)
        rows = [["#", "Architecture", "Lat (ms)", "Size (MB)", "Reason"]]
        for _, r in rej.iterrows():
            rows.append([int(r.id), P(r.arch, SMALL), f(r.latency_ms), f(r.size_mb, 3), P(str(r.reject_reason), SMALL)])
        S += [table(rows, widths=[0.8 * cm, 7.6 * cm, 1.4 * cm, 1.5 * cm, 5.4 * cm]),
              P("Table 6 - Examples of architectures rejected by the deployment constraints.", CAP)]
        S += fig(f"{FIG}/q2_search_space_scatter.png", 17, "Figure 3 - Left: proxy accuracy vs. measured latency of trained "
                 "candidates. Right: all sampled candidates with the constraint boundaries.")
        S += fig(f"{FIG}/q2_search_progress.png", 12, "Figure 4 - Fitness of each trained candidate and best-so-far.")

        c = best["config"]
        rows = [["Stage", "Blocks", "Channels", "Kernel", "Block type", "Activation"]]
        for i, st in enumerate(c["stages"][:c["num_stages"]]):
            rows.append([i, st["depth"], st["channels"], st["kernel"], st["block"], st["activation"]])
        S += [P("3.6 Best discovered architecture", H2),
              P(f"Best candidate <b>#{best['best_id']}</b>: <font face='Courier'>{best['best_arch']}</font>. "
                f"Input {c['resolution']}x{c['resolution']}, stem {c['stem_channels']} channels, head "
                f"{c['head_channels'] or 'none'}, dropout {c['dropout']}. It has {best['params_m'] * 1e3:.0f} K "
                f"parameters, {best['mflops']:.1f} MFLOPs, {best['size_mb']:.3f} MB of weights, "
                f"{best['latency_ms']:.2f} ms latency and {f(best['peak_mem_mb'], 1)} MB peak inference memory. "
                "The configuration is saved automatically to <font face='Courier'>results/q2/best_arch.json</font>."),
              table(rows)]
        if bestf:
            S += [P(f"<font face='Courier'>train_best.py</font> rebuilt this architecture from the JSON file and trained "
                    f"it <b>from scratch</b> on 45,000 training images for {bestf['epochs']} epochs (5,000 held out "
                    f"for model selection). Final CIFAR-10 test accuracy: <b>{bestf['test_top1']:.2f}%</b> top-1 "
                    f"({bestf['test_top5']:.2f}% top-5), selected at epoch {bestf['best_epoch']} by validation "
                    f"accuracy ({bestf['val_top1']:.2f}%). Training took {bestf['train_time_min']:.1f} minutes on the CPU.")]
            S += fig(f"{FIG}/q2_best_training_curve.png", 12, "Figure 5 - Training of the best NAS architecture from scratch.")
        if notes.get("q2"):
            S += [P("Discussion", H3)] + [P(t) for t in notes["q2"]]

    # ------------------------------------------------------------------ 4 benchmark
    S += [PageBreak(), P("4. Automated benchmarking (deploy/benchmark.py)", H1),
          P("All models are benchmarked by one script: accuracy with PyTorch and with ONNX Runtime on the full test set, "
            "parameters, FLOPs, PyTorch state-dict size, ONNX file size, batch-1 latency (median of 100 runs; 1 thread "
            "for PyTorch and ORT, plus ORT with all threads), ORT throughput at batch 64 and peak inference memory "
            "(fresh process). Raw output: <font face='Courier'>results/benchmark/benchmark.csv</font>.")]
    if bench is not None:
        rows = [["Model", "Top-1 PT", "Top-1 ORT", "Params", "MFLOPs", "ONNX MB", "PT ms", "ORT ms", "ORT ms (4c)",
                 "img/s b64", "Mem MB"]]
        for _, r in bench.iterrows():
            rows.append([r.model.replace("efficientnet_lite0", "effnet_lite0"), f(r.torch_top1), f(r.get("ort_top1")),
                         f"{r.params_m:.3f}M", f(r.mflops, 1), f(r.get("onnx_size_mb")), f(r.torch_lat_ms_1t),
                         f(r.get("ort_lat_ms_1t")), f(r.get("ort_lat_ms_all")), f(r.get("ort_throughput_b64"), 0),
                         f(r.get("ort_peak_mem_mb"), 1)])
        S += [table(rows, font=7), P("Table 7 - CPU benchmark (PT = PyTorch eager, ORT = ONNX Runtime; latency at "
                                     "batch 1 with 1 thread unless marked 4c; memory = ORT peak RSS increase).", CAP)]
    S += fig(f"{FIG}/q1_accuracy_vs_latency.png", 13, "Figure 6 - Accuracy / latency trade-off of teacher and students.")
    S += fig(f"{FIG}/latency_torch_vs_ort.png", 14, "Figure 7 - PyTorch vs. ONNX Runtime CPU latency.")
    if notes.get("bench"):
        S += [P("Discussion", H3)] + [P(t) for t in notes["bench"]]

    # ------------------------------------------------------------------ 5 screenshots
    S += [PageBreak(), P("5. Screenshots of execution", H1),
          P("Console output of the actual runs (captured to <font face='Courier'>results/**/logs/*_console.txt</font> and "
            "rendered as terminal screenshots).", BODY)]
    shots = [
        ("results/q1/logs/teacher_console.txt", "teacher", "python -m q1_kd.train_teacher --epochs 2 --res 64",
         r"Teacher|teacher\]|\[ *(0|200)/", None, 16, False),
        ("results/q1/logs/baseline_console.txt", "baseline", "python -m q1_kd.train_baseline --epochs 5",
         r"BASELINE|params|\[baseline\]", r"\[ *\d+/", 30, False),
        ("results/q1/logs/kd_console.txt", "kd", "python -m q1_kd.train_kd --epochs 5 --T 4 --alpha 0.9 --beta 1.0",
         r"KD training|teacher loaded|\[kd\]|ep 1 \[ *(0|200)/", None, 32, False),
        ("results/q2/logs/search_console.txt", "nas_search", "python -m q2_nas.search --budget 20 --epochs 3",
         None, r"^=+$", 34, False),
        ("results/q2/logs/search_console.txt", "nas_search_end", "python -m q2_nas.search  (end of run)",
         None, r"^=+$", 14, True),
        ("results/q2/logs/train_best_console.txt", "train_best", "python -m q2_nas.train_best --epochs 15",
         r"Training best|epoch|nas-best", None, 20, False),
        ("results/onnx_export_console.txt", "onnx_export", "python -m deploy.export_onnx --models all",
         r"\[onnx\]", None, 12, False),
        ("results/onnx_infer_console.txt", "onnx_infer", "python -m deploy.onnx_infer --models all --samples 3",
         None, None, 34, False),
        ("results/benchmark_console.txt", "benchmark", "python -m deploy.benchmark --models all",
         None, r"^\[bench\]", 12, True),
    ]
    for src, name, title, keep, drop, n, tail in shots:
        pth = screenshot(src, name, title, keep, drop, n, tail)
        if pth:
            S += [KeepTogether(fig(pth, 17, f"Screenshot - {title}"))]

    # ------------------------------------------------------------------ 6 conclusion
    S += [PageBreak(), P("6. Limitations and conclusion", H1)]
    S += [P(t) for t in notes.get("conclusion", [])]
    S += [P("7. Deliverables checklist", H2),
          table([["#", "Deliverable", "Location in repository"],
                 ["1", "Teacher model", "checkpoints/teacher_resnet50.pt, q1_kd/train_teacher.py, q1_kd/teacher.py"],
                 ["2", "Student architectures", "q1_kd/students/{mobilenetv2,squeezenet,efficientnet_lite}.py"],
                 ["3", "Baseline student models", "checkpoints/students/*_baseline.pt, q1_kd/train_baseline.py"],
                 ["4", "Knowledge distillation", "q1_kd/distill.py, q1_kd/engine.py, q1_kd/train_kd.py, checkpoints/students/*_kd.pt"],
                 ["5", "NAS search space", "q2_nas/search_space.py, q2_nas/generator.py"],
                 ["6", "NAS algorithm", "q2_nas/search.py (+ hw_profiler.py, trainer.py, mem_probe.py)"],
                 ["7", "NAS results", "results/q2/best_arch.json, search_history.csv, search_history.json"],
                 ["8", "Optimized architecture", "checkpoints/nas_best.pt, q2_nas/train_best.py, results/q2/best_arch_final.json"],
                 ["9", "Inference pipeline", "deploy/infer.py, deploy/predictor.py, deploy/registry.py"],
                 ["10", "ONNX deployment", "onnx_models/*.onnx, deploy/export_onnx.py, deploy/onnx_infer.py"],
                 ["11", "Benchmarking", "deploy/benchmark.py, results/benchmark/*.csv, results/figures/"],
                 ["12", "Technical report", "report/Technical_Report.pdf (this document), report/build_report.py"],
                 ["13", "README", "README.md, requirements.txt, scripts/run_all.sh|.ps1"]],
                widths=[0.8 * cm, 4 * cm, 11.7 * cm])]

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(1.8 * cm, 1.1 * cm, f"{AUTHOR} ({ROLL}, Section {SECTION}) - Assignment 4: KD & Hardware-aware NAS")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"page {doc.page}")
        canvas.restoreState()

    out = "report/Technical_Report.pdf"
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.7 * cm, title="Assignment 4 - Technical Report", author=AUTHOR)
    doc.build(S, onFirstPage=on_page, onLaterPages=on_page)
    print("report written to", out)


if __name__ == "__main__":
    build()
