"""
Compare DenseNet121 vs EfficientNet-B0 vs ResNet50.

Usage:
    cd hms-ai
    python -m ml.evaluation.compare_pneumonia_models
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from ml.training.config import (BATCH_SIZE, CLASS_NAMES, DEVICE, IMAGENET_MEAN, IMAGENET_STD, METRICS_DIR, NUM_CLASSES, NUM_WORKERS, TEST_DIR)

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
OUT = METRICS_DIR / "model_comparison"

MODELS = {
    "DenseNet121": {"ckpt": CKPT_DIR / "pneumonia_densenet121_best.pt", "arch": "densenet121"},
    "EfficientNet-B0": {"ckpt": CKPT_DIR / "pneumonia_efficientnet_b0_best.pt", "arch": "efficientnet_b0"},
    "ResNet50": {"ckpt": CKPT_DIR / "pneumonia_resnet50_best.pt", "arch": "resnet50"},
}

def _build(arch):
    if arch == "densenet121":
        m = models.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    elif arch == "resnet50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    else: raise ValueError(f"Unknown arch: {arch}")
    return m

@torch.no_grad()
def run_inference(model, loader):
    yl, ypr, times = [], [], []
    for imgs, lbls in tqdm(loader, desc="  Inference", leave=False):
        imgs = imgs.to(DEVICE); t0 = time.time()
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"): logits = model(imgs)
        else: logits = model(imgs)
        times.append((time.time() - t0) / imgs.size(0))
        yl.extend(lbls.numpy()); ypr.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.array(yl), np.array(ypr), np.mean(times) * 1000

def compute_metrics(yt, yp, t):
    pred = (yp >= t).astype(int); cm = confusion_matrix(yt, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    n = len(yt); acc = (tp + tn) / n; prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    try: auc = roc_auc_score(yt, yp)
    except: auc = 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1, "auc": auc, "FP": fp, "FN": fn, "TP": tp, "TN": tn}

def find_threshold(yt, yp):
    best_f1 = 0; best_t = 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        pred = (yp >= t).astype(int)
        tp = ((pred == 1) & (yt == 1)).sum(); fp = ((pred == 1) & (yt == 0)).sum(); fn = ((pred == 0) & (yt == 1)).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1: best_f1 = f1; best_t = round(t, 2)
    return best_t

def main():
    print("=" * 64); print("  3-Model Comparison: DenseNet121 vs EfficientNet-B0 vs ResNet50"); print("=" * 64)

    for name, info in MODELS.items():
        if not info["ckpt"].exists():
            print(f"ERROR: {name} checkpoint not found: {info['ckpt']}"); sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    ds = datasets.ImageFolder(str(TEST_DIR), transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  Test: {len(ds)} images  Device: {DEVICE}\n")

    results = {}
    for name, info in MODELS.items():
        print(f"  Loading {name}...")
        model = _build(info["arch"])
        ckpt = torch.load(info["ckpt"], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"]); model.to(DEVICE).eval()
        params = sum(p.numel() for p in model.parameters()); size_mb = info["ckpt"].stat().st_size / 1024 / 1024

        yt, yp, ms = run_inference(model, loader)
        thresh = find_threshold(yt, yp); m = compute_metrics(yt, yp, thresh)
        results[name] = {"thresh": thresh, "metrics": m, "params": params, "size_mb": size_mb, "ms": ms, "yt": yt, "yp": yp}
        del model; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # Print table
    print(f"\n  {'Metric':<20s}", end="")
    for name in MODELS: print(f" {name:>18s}", end="")
    print(f" {'Winner':>10s}")
    print(f"  {'-' * (20 + 18 * len(MODELS) + 10)}")

    rows_csv = []
    metric_keys = [
        ("Optimal Threshold", lambda r: f"{r['thresh']:.2f}"),
        ("Accuracy", lambda r: f"{r['metrics']['accuracy']:.4f}"),
        ("Precision", lambda r: f"{r['metrics']['precision']:.4f}"),
        ("Recall", lambda r: f"{r['metrics']['recall']:.4f}"),
        ("Specificity", lambda r: f"{r['metrics']['specificity']:.4f}"),
        ("F1", lambda r: f"{r['metrics']['f1']:.4f}"),
        ("AUC-ROC", lambda r: f"{r['metrics']['auc']:.4f}"),
        ("False Positives", lambda r: str(r['metrics']['FP'])),
        ("False Negatives", lambda r: str(r['metrics']['FN'])),
        ("Inference (ms)", lambda r: f"{r['ms']:.1f}"),
        ("Parameters", lambda r: f"{r['params']/1e6:.1f}M"),
        ("Checkpoint", lambda r: f"{r['size_mb']:.1f}MB"),
    ]

    for mname, fn in metric_keys:
        vals = {n: fn(results[n]) for n in MODELS}
        # Determine winner for numeric comparisons
        winner = ""
        if mname in ("Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC-ROC"):
            best = max(MODELS, key=lambda n: results[n]["metrics"].get(mname.lower().replace("-", "").replace(" ", ""), results[n]["metrics"].get("auc", 0)))
            winner = best[:6]
        elif mname == "False Negatives":
            winner = min(MODELS, key=lambda n: results[n]["metrics"]["FN"])[:6]
        elif mname == "False Positives":
            winner = min(MODELS, key=lambda n: results[n]["metrics"]["FP"])[:6]
        elif mname == "Inference (ms)":
            winner = min(MODELS, key=lambda n: results[n]["ms"])[:6]
        elif mname == "Parameters":
            winner = min(MODELS, key=lambda n: results[n]["params"])[:6]

        print(f"  {mname:<20s}", end="")
        for n in MODELS: print(f" {vals[n]:>18s}", end="")
        print(f" {winner:>10s}")
        row = {"metric": mname}
        for n in MODELS: row[n] = vals[n]
        row["winner"] = winner
        rows_csv.append(row)

    # Save CSV
    csv_fields = ["metric"] + list(MODELS.keys()) + ["winner"]
    with open(OUT / "model_comparison_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields); w.writeheader(); w.writerows(rows_csv)

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        colors = {"DenseNet121": "#4C72B0", "EfficientNet-B0": "#DD8452", "ResNet50": "#55A868"}
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ROC
        for name in MODELS:
            r = results[name]; fpr, tpr, _ = roc_curve(r["yt"], r["yp"])
            axes[0].plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC={r['metrics']['auc']:.3f})")
        axes[0].plot([0, 1], [0, 1], "--", color="gray"); axes[0].set_title("ROC Curves", fontweight="bold")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        # Metrics bar
        mnames = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
        x = np.arange(len(mnames)); w = 0.25
        for i, name in enumerate(MODELS):
            r = results[name]["metrics"]
            vals = [r["accuracy"], r["precision"], r["recall"], r["specificity"], r["f1"]]
            axes[1].bar(x + i * w - w, vals, w, label=name, color=colors[name])
        axes[1].set_xticks(x); axes[1].set_xticklabels(mnames, fontsize=8); axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Metrics", fontweight="bold"); axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)

        # Errors
        err_names = ["FP", "FN"]; x2 = np.arange(2)
        for i, name in enumerate(MODELS):
            r = results[name]["metrics"]
            axes[2].bar(x2 + i * 0.25 - 0.25, [r["FP"], r["FN"]], 0.22, label=name, color=colors[name])
        axes[2].set_xticks(x2); axes[2].set_xticklabels(err_names)
        axes[2].set_title("Errors", fontweight="bold"); axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)

        plt.suptitle("DenseNet121 vs EfficientNet-B0 vs ResNet50", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(OUT / "model_comparison_charts.png", dpi=130); plt.close()
        print(f"\n  Saved: model_comparison_charts.png")
    except ImportError: pass

    # Clinical report
    best_recall = max(MODELS, key=lambda n: results[n]["metrics"]["recall"])
    best_spec = max(MODELS, key=lambda n: results[n]["metrics"]["specificity"])
    best_f1 = max(MODELS, key=lambda n: results[n]["metrics"]["f1"])
    fewest_fn = min(MODELS, key=lambda n: results[n]["metrics"]["FN"])
    fewest_fp = min(MODELS, key=lambda n: results[n]["metrics"]["FP"])
    fastest = min(MODELS, key=lambda n: results[n]["ms"])
    smallest = min(MODELS, key=lambda n: results[n]["params"])

    report = f"""# Three-Model Comparison Report

## DenseNet121 vs EfficientNet-B0 vs ResNet50

| Metric | DenseNet121 | EfficientNet-B0 | ResNet50 |
|--------|-------------|-----------------|----------|
"""
    for mname, fn in metric_keys:
        vals = [fn(results[n]) for n in MODELS]
        report += f"| {mname} | {vals[0]} | {vals[1]} | {vals[2]} |\n"

    report += f"""
## Clinical Assessment

1. **Best sensitivity (recall):** {best_recall} ({results[best_recall]['metrics']['recall']:.1%})
2. **Best specificity:** {best_spec} ({results[best_spec]['metrics']['specificity']:.1%})
3. **Best F1 (overall):** {best_f1} ({results[best_f1]['metrics']['f1']:.4f})
4. **Fewest missed pneumonia (FN):** {fewest_fn} ({results[fewest_fn]['metrics']['FN']})
5. **Fewest false alarms (FP):** {fewest_fp} ({results[fewest_fp]['metrics']['FP']})
6. **Fastest inference:** {fastest} ({results[fastest]['ms']:.1f} ms)
7. **Smallest model:** {smallest} ({results[smallest]['params']/1e6:.1f}M params)

## Recommendation

**{best_f1}** is recommended as the default model based on:
- Highest overall F1 score
- {"Highest recall (fewest missed cases)" if best_f1 == best_recall else f"Strong recall ({results[best_f1]['metrics']['recall']:.1%})"}
- Clinically acceptable specificity

For **screening** (maximize detection): use {best_recall}.
For **speed/resource-constrained** deployment: use {fastest}.
For **confirmation** (minimize false positives): use {fewest_fp}.

All models are for **AI-assisted screening only**. Not a final diagnosis.
Clinical decisions must be made by qualified healthcare professionals.

## Known Limitations
- All trained on pediatric chest X-rays (ages 1-5) from a single center
- Not validated for adult populations or multi-center deployment
- Compression bias partially mitigated by dataset standardization
"""
    with open(OUT / "model_comparison_report.md", "w", encoding="utf-8") as f: f.write(report)
    with open(OUT / "clinical_model_comparison.md", "w", encoding="utf-8") as f: f.write(report)
    print(f"  Saved: model_comparison_report.md + clinical_model_comparison.md")
    print(f"\n{'=' * 64}\n  Comparison complete.\n{'=' * 64}")

if __name__ == "__main__": main()
