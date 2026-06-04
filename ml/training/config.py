"""
Pneumonia DenseNet121 Training Configuration.

Central source of truth for all paths, hyperparameters, and constants.
All other ml/ scripts import from here.
"""

import platform
import sys
from pathlib import Path

import torch

# ── Python version check ──────────────────────────────────────────────────

if sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel != "final":
    print(
        "WARNING: You are running a pre-release Python 3.11 "
        f"({sys.version}). PyTorch and MONAI may not install correctly. "
        "Please use Python 3.11.8+ stable from https://python.org."
    )

# ── Paths ─────────────────────────────────────────────────────────────────
# Resolve relative to this file: config.py -> training/ -> ml/ -> hms-ai/

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "app" / "data" / "chest_xray"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "pneumonia_densenet121_best.pt"

# ── Class mapping ─────────────────────────────────────────────────────────

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 2

# ── Hyperparameters ───────────────────────────────────────────────────────

IMAGE_SIZE = 256       # resize target before crop
CROP_SIZE = 224        # final input size for DenseNet121
BATCH_SIZE = 16
NUM_EPOCHS = 5         # start small, increase once verified
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ── ImageNet normalization (DenseNet121 pretrained on ImageNet) ────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Windows compatibility ─────────────────────────────────────────────────

NUM_WORKERS = 0 if platform.system() == "Windows" else 4

# ── Device ────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Class imbalance weights ───────────────────────────────────────────────
# train/NORMAL = 1341, train/PNEUMONIA = 3875 -> ratio 2.89:1
# Inverse frequency weighting: higher weight for minority class (NORMAL)

_NORMAL_COUNT = 1341
_PNEUMONIA_COUNT = 3875
_TOTAL = _NORMAL_COUNT + _PNEUMONIA_COUNT

CLASS_WEIGHTS = torch.tensor(
    [_PNEUMONIA_COUNT / _TOTAL, _NORMAL_COUNT / _TOTAL],
    dtype=torch.float32,
)
# CLASS_WEIGHTS = tensor([0.7430, 0.2570])
# Index 0 = NORMAL (minority, gets higher weight)
# Index 1 = PNEUMONIA (majority, gets lower weight)

# ── Model versioning ─────────────────────────────────────────────────────

MODEL_VERSION = "pneumonia-densenet121-v1"
