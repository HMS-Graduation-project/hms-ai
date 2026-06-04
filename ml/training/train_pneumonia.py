"""
Train DenseNet121 for pneumonia detection on the CLEANED dataset.

Usage:
    cd hms-ai
    python -m ml.training.train_pneumonia --epochs 10 --batch-size 16
    python -m ml.training.train_pneumonia --resume ml/checkpoints/pneumonia_densenet121_best.pt
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from ml.training.config import (
    BATCH_SIZE, BEST_MODEL_PATH, CHECKPOINT_DIR, CLASS_NAMES, DATA_DIR,
    DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LEARNING_RATE,
    METRICS_DIR, MODEL_VERSION, NUM_CLASSES, NUM_EPOCHS, NUM_WORKERS,
    PATIENCE, RANDOM_SEED, TEST_DIR, TRAIN_DIR, VAL_DIR, WEIGHT_DECAY,
)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_dataset() -> None:
    """Fail fast if dataset is missing or wrong."""
    # Must use cleaned dataset
    if "chest_xray_cleaned" not in str(DATA_DIR):
        print("ERROR: Config must point to chest_xray_cleaned, not original dataset.")
        sys.exit(1)

    for name, d in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        if not d.exists():
            print(f"ERROR: {name} directory not found: {d}")
            sys.exit(1)
        for cls in CLASS_NAMES:
            cls_dir = d / cls
            if not cls_dir.exists():
                print(f"ERROR: {name}/{cls} not found: {cls_dir}")
                sys.exit(1)
            count = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            print(f"  {name}/{cls}: {count} images")


def get_transforms(train: bool = False) -> transforms.Compose:
    """Cleaned images are already 224x224 RGB. Minimal preprocessing."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
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
    """Compute balanced class weights from actual cleaned training data."""
    normal_count = sum(1 for f in (train_dir / "NORMAL").iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    pneumonia_count = sum(1 for f in (train_dir / "PNEUMONIA").iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    total = normal_count + pneumonia_count
    w_normal = total / (2 * max(normal_count, 1))
    w_pneumonia = total / (2 * max(pneumonia_count, 1))
    print(f"  Class weights: NORMAL={w_normal:.4f}  PNEUMONIA={w_pneumonia:.4f}")
    print(f"  NORMAL gets {w_normal/w_pneumonia:.2f}x gradient weight")
    return torch.tensor([w_normal, w_pneumonia], dtype=torch.float32)


def build_model() -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0; correct = 0; total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if scaler and DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
) -> dict:
    model.eval()
    running_loss = 0.0; total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        total += labels.size(0)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Confusion matrix components
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)  # sensitivity
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return {
        "loss": running_loss / total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DenseNet121 pneumonia classifier")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    _seed_everything(RANDOM_SEED)

    print("=" * 64)
    print("  DenseNet121 Pneumonia Training (Cleaned Dataset)")
    print("=" * 64)
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  Dataset:    {DATA_DIR}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Patience:   {args.patience}")
    print(f"  Model:      {MODEL_VERSION}")
    print()

    verify_dataset()

    # Data
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=get_transforms(train=False))
    assert list(train_ds.class_to_idx.keys()) == CLASS_NAMES, \
        f"Expected {CLASS_NAMES}, got {list(train_ds.class_to_idx.keys())}"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"\n  Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} images, {len(val_loader)} batches\n")

    # Model
    model = build_model()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Loss with class weights
    class_weights = compute_class_weights(TRAIN_DIR).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    best_f1 = 0.0
    epochs_no_improve = 0

    # Resume
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"  Resuming from {ckpt_path} ...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_f1 = ckpt.get("metrics", {}).get("f1", 0.0)
            print(f"  Resumed at epoch {start_epoch}, best F1: {best_f1:.4f}\n")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Metrics CSV
    metrics_path = METRICS_DIR / "training_metrics.csv"
    metrics_fields = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision",
                      "val_recall", "val_specificity", "val_f1", "val_auc", "lr"]
    metrics_rows: list[dict] = []

    # Training loop
    print("Starting training...\n")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics = validate(model, val_loader, criterion)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["f1"])

        print(
            f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}\n"
            f"  Val   Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.4f}  "
            f"F1: {val_metrics['f1']:.4f}  Recall: {val_metrics['recall']:.4f}  "
            f"AUC: {val_metrics['auc']:.4f}  LR: {current_lr:.2e}"
        )

        # Save metrics
        metrics_rows.append({
            "epoch": epoch + 1, "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_auc": round(val_metrics["auc"], 4),
            "lr": current_lr,
        })

        # Save best checkpoint by F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "metrics": val_metrics,
                "preprocessing": {
                    "image_size": IMAGE_SIZE,
                    "normalize_mean": IMAGENET_MEAN,
                    "normalize_std": IMAGENET_STD,
                },
                "model_version": MODEL_VERSION,
            }, BEST_MODEL_PATH)
            print(f"  >> Saved best checkpoint (F1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{args.patience})")

        if epochs_no_improve >= args.patience:
            print(f"\n  Early stopping after {args.patience} epochs without improvement.")
            break

        print()

    # Save metrics CSV
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metrics_fields)
        w.writeheader()
        w.writerows(metrics_rows)
    print(f"\n  Saved: {metrics_path.name}")

    # Training curves plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_range = [r["epoch"] for r in metrics_rows]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs_range, [r["train_loss"] for r in metrics_rows], "b-o", label="Train Loss")
        axes[0].plot(epochs_range, [r["val_loss"] for r in metrics_rows], "r-o", label="Val Loss")
        axes[0].set_title("Loss", fontweight="bold"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[0].set_xlabel("Epoch")

        axes[1].plot(epochs_range, [r["val_accuracy"] for r in metrics_rows], "g-o", label="Accuracy")
        axes[1].plot(epochs_range, [r["val_f1"] for r in metrics_rows], "b-o", label="F1")
        axes[1].plot(epochs_range, [r["val_recall"] for r in metrics_rows], "r-o", label="Recall")
        axes[1].set_title("Validation Metrics", fontweight="bold"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)

        axes[2].plot(epochs_range, [r["val_auc"] for r in metrics_rows], "m-o", label="AUC-ROC")
        axes[2].set_title("AUC-ROC", fontweight="bold"); axes[2].legend(); axes[2].grid(alpha=0.3)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylim(0, 1)

        plt.suptitle("Training Curves -- DenseNet121 Pneumonia", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(METRICS_DIR / "training_curves.png", dpi=130)
        plt.close()
        print(f"  Saved: training_curves.png")
    except ImportError:
        pass

    elapsed = time.time() - t0
    print(f"\n{'=' * 64}")
    print(f"  Training complete in {elapsed / 60:.1f} minutes")
    print(f"  Best validation F1: {best_f1:.4f}")
    print(f"  Checkpoint: {BEST_MODEL_PATH}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
