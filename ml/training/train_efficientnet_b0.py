"""
Train EfficientNet-B0 for pneumonia detection on the CLEANED dataset.

Usage:
    cd hms-ai
    python -m ml.training.train_efficientnet_b0 --epochs 11
    python -m ml.training.train_efficientnet_b0 --epochs 11 --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from ml.training.config import (
    BATCH_SIZE, CHECKPOINT_DIR, CLASS_NAMES, DATA_DIR, DEVICE,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LEARNING_RATE,
    METRICS_DIR, NUM_CLASSES, NUM_EPOCHS, NUM_WORKERS,
    PATIENCE, RANDOM_SEED, TRAIN_DIR, VAL_DIR, WEIGHT_DECAY,
)

MODEL_VERSION = "pneumonia-efficientnet-b0-v1"
BEST_CKPT = CHECKPOINT_DIR / "pneumonia_efficientnet_b0_best.pt"
OUT_DIR = METRICS_DIR / "efficientnet_b0"


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_dataset() -> None:
    if "chest_xray_cleaned" not in str(DATA_DIR):
        print("ERROR: Must use cleaned dataset."); sys.exit(1)
    for name, d in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        for cls in CLASS_NAMES:
            cd = d / cls
            n = sum(1 for f in cd.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}) if cd.exists() else 0
            print(f"  {name}/{cls}: {n}")


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def compute_class_weights(train_dir: Path) -> torch.Tensor:
    n = sum(1 for f in (train_dir / "NORMAL").iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    p = sum(1 for f in (train_dir / "PNEUMONIA").iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    total = n + p
    w0 = total / (2 * max(n, 1)); w1 = total / (2 * max(p, 1))
    print(f"  Class weights: NORMAL={w0:.4f}  PNEUMONIA={w1:.4f}")
    return torch.tensor([w0, w1], dtype=torch.float32)


def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # EfficientNet-B0 classifier: Sequential(Dropout, Linear(1280, 1000))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    loss_sum = 0.0; correct = 0; total = 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if scaler and DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(images); loss = criterion(out, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            out = model(images); loss = criterion(out, labels)
            loss.backward(); optimizer.step()
        loss_sum += loss.item() * images.size(0)
        correct += out.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0; total = 0
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(images); loss = criterion(out, labels)
        else:
            out = model(images); loss = criterion(out, labels)
        loss_sum += loss.item() * images.size(0)
        probs = torch.softmax(out, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(out.max(1)[1].cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        total += labels.size(0)

    y_true = np.array(all_labels); y_pred = np.array(all_preds); y_prob = np.array(all_probs)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    try: auc = roc_auc_score(y_true, y_prob)
    except ValueError: auc = 0.0
    return {"loss": loss_sum / total, "accuracy": acc, "precision": prec,
            "recall": rec, "specificity": spec, "f1": f1, "auc": auc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 pneumonia classifier")
    parser.add_argument("--epochs", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    _seed(RANDOM_SEED)

    print("=" * 64)
    print("  EfficientNet-B0 Pneumonia Training (Cleaned Dataset)")
    print("=" * 64)
    print(f"  Device:     {DEVICE}")
    if DEVICE.type == "cuda": print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  Dataset:    {DATA_DIR}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Model:      {MODEL_VERSION}\n")

    verify_dataset()

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=get_transforms(True))
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=get_transforms(False))
    assert list(train_ds.class_to_idx.keys()) == CLASS_NAMES

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"\n  Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} images, {len(val_loader)} batches\n")

    model = build_model()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    cw = compute_class_weights(TRAIN_DIR).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    start_epoch = 0; best_f1 = 0.0; no_improve = 0

    if args.resume:
        ck = Path(args.resume)
        if ck.exists():
            ckpt = torch.load(ck, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_f1 = ckpt.get("metrics", {}).get("f1", 0.0)
            print(f"  Resumed at epoch {start_epoch}, best F1: {best_f1:.4f}\n")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    fields = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision",
              "val_recall", "val_specificity", "val_f1", "val_auc", "lr"]

    print("Starting training...\n")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        vm = validate(model, val_loader, criterion)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(vm["f1"])

        print(f"  Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}")
        print(f"  Val   Loss: {vm['loss']:.4f}  Acc: {vm['accuracy']:.4f}  "
              f"F1: {vm['f1']:.4f}  Recall: {vm['recall']:.4f}  AUC: {vm['auc']:.4f}  LR: {lr:.2e}")

        metrics_rows.append({
            "epoch": epoch + 1, "train_loss": round(tr_loss, 4), "val_loss": round(vm["loss"], 4),
            "val_accuracy": round(vm["accuracy"], 4), "val_precision": round(vm["precision"], 4),
            "val_recall": round(vm["recall"], 4), "val_specificity": round(vm["specificity"], 4),
            "val_f1": round(vm["f1"], 4), "val_auc": round(vm["auc"], 4), "lr": lr,
        })

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]; no_improve = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "class_to_idx": train_ds.class_to_idx, "metrics": vm,
                "preprocessing": {"image_size": IMAGE_SIZE, "normalize_mean": IMAGENET_MEAN, "normalize_std": IMAGENET_STD},
                "model_version": MODEL_VERSION, "architecture": "efficientnet_b0",
            }, BEST_CKPT)
            print(f"  >> Saved best checkpoint (F1={best_f1:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{args.patience})")

        if no_improve >= args.patience:
            print(f"\n  Early stopping."); break
        print()

    # Save metrics CSV
    csv_path = OUT_DIR / "training_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(metrics_rows)
    print(f"\n  Saved: {csv_path.name}")

    # Training curves
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ep = [r["epoch"] for r in metrics_rows]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(ep, [r["train_loss"] for r in metrics_rows], "b-o", label="Train")
        axes[0].plot(ep, [r["val_loss"] for r in metrics_rows], "r-o", label="Val")
        axes[0].set_title("Loss", fontweight="bold"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ep, [r["val_f1"] for r in metrics_rows], "b-o", label="F1")
        axes[1].plot(ep, [r["val_recall"] for r in metrics_rows], "r-o", label="Recall")
        axes[1].plot(ep, [r["val_accuracy"] for r in metrics_rows], "g-o", label="Accuracy")
        axes[1].set_title("Metrics", fontweight="bold"); axes[1].legend(); axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3)
        axes[2].plot(ep, [r["val_auc"] for r in metrics_rows], "m-o", label="AUC")
        axes[2].set_title("AUC-ROC", fontweight="bold"); axes[2].legend(); axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)
        plt.suptitle("Training Curves -- EfficientNet-B0", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(OUT_DIR / "training_curves.png", dpi=130); plt.close()
        print(f"  Saved: training_curves.png")
    except ImportError: pass

    elapsed = time.time() - t0
    print(f"\n{'=' * 64}")
    print(f"  Training complete in {elapsed / 60:.1f} minutes")
    print(f"  Best validation F1: {best_f1:.4f}")
    print(f"  Checkpoint: {BEST_CKPT}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
