"""
Pneumonia DenseNet121 Training Script.

Trains a binary classifier (NORMAL vs PNEUMONIA) on chest X-ray images
using a pretrained DenseNet121 backbone with ImageNet weights.

Usage:
    cd hms-ai
    python -m ml.training.train_pneumonia
    python -m ml.training.train_pneumonia --epochs 10 --batch-size 32
    python -m ml.training.train_pneumonia --resume ml/checkpoints/pneumonia_densenet121_best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from ml.training.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    CLASS_WEIGHTS,
    CROP_SIZE,
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
    WEIGHT_DECAY,
)


# ── Transforms ────────────────────────────────────────────────────────────

def get_transforms(train: bool = False) -> transforms.Compose:
    """Build image transform pipeline."""
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── DataLoaders ───────────────────────────────────────────────────────────

def build_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders using ImageFolder."""
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=get_transforms(train=False))
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=get_transforms(train=False))

    # Verify class-to-index mapping matches our expected order
    print(f"  Class mapping: {train_dataset.class_to_idx}")
    assert list(train_dataset.class_to_idx.keys()) == CLASS_NAMES, (
        f"Expected class order {CLASS_NAMES}, got {list(train_dataset.class_to_idx.keys())}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ── Model ─────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> nn.Module:
    """Build DenseNet121 with modified classifier for binary classification."""
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    # DenseNet121 classifier: Linear(1024, 1000) -> Linear(1024, 2)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model.to(DEVICE)


# ── Training loop ─────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Validate. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

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
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DenseNet121 for pneumonia detection",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Number of epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Python version warning
    if sys.version_info.releaselevel != "final":
        print(f"WARNING: Pre-release Python detected ({sys.version})")
        print("Consider using Python 3.11.x stable for best compatibility.\n")

    print("=" * 60)
    print("  Pneumonia DenseNet121 Training")
    print("=" * 60)
    print(f"  Device:      {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  CUDA:        {torch.version.cuda}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Classes:     {CLASS_NAMES}")
    print(f"  Weights:     {CLASS_WEIGHTS.tolist()}")
    print()

    # Validate dataset exists
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not d.exists():
            print(f"ERROR: Dataset directory not found: {d}")
            sys.exit(1)

    # Warn about tiny validation set
    val_count = sum(1 for _ in VAL_DIR.rglob("*") if _.is_file())
    if val_count < 50:
        print(f"WARNING: Validation set has only {val_count} images.")
        print("Validation metrics will be noisy. Use test set for reliable evaluation.\n")

    # Build components
    print("Building dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(batch_size=args.batch_size)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    print("Building model...")
    model = build_model(pretrained=True)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    start_epoch = 0
    best_val_acc = 0.0

    # Resume from checkpoint
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"Resuming from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_acc = checkpoint.get("val_accuracy", 0.0)
            print(f"  Resumed at epoch {start_epoch}, best val acc: {best_val_acc:.4f}\n")
        else:
            print(f"WARNING: Checkpoint not found: {ckpt_path}, training from scratch.\n")

    # Ensure checkpoint directory exists
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(
            f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                },
                BEST_MODEL_PATH,
            )
            print(f"  Saved best checkpoint (val_acc={val_acc:.4f})")

        print()

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"  Training complete in {elapsed / 60:.1f} minutes")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint: {BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
