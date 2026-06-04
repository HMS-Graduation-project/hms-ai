# HMS-AI: Pneumonia Detection ML Module

Binary classifier (NORMAL vs PNEUMONIA) using DenseNet121 on chest X-ray images.

## Prerequisites

- **Python 3.11.8+ stable** (NOT alpha/beta — check with `python --version`)
- **NVIDIA GPU with CUDA 12.1+** (RTX 4070 Laptop GPU verified)
- Dataset at `app/data/chest_xray/` with `train/`, `val/`, `test/` splits

## Environment Setup (Windows)

```powershell
cd C:\Users\MOHAMEDKHEIR\Desktop\HMS-Graduation-project\hms-ai

# Create virtual environment
py -3.11 -m venv ai-env
.\ai-env\Scripts\activate

# Verify Python version (must be 3.11.x final, NOT alpha)
python --version

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining ML dependencies
pip install -r ml\requirements-ml.txt

# Also install the FastAPI deps so imports work
pip install -r requirements.txt
```

## Verify Installation

```powershell
# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"

# Check MONAI
python -c "import monai; print(f'MONAI {monai.__version__}')"

# Check imports
python -c "from ml.training.config import TRAIN_DIR; print(f'Train dir: {TRAIN_DIR}')"
```

## Usage

All commands run from the `hms-ai/` root directory.

### 1. Exploratory Data Analysis

```powershell
python -m ml.eda.chest_xray_eda
```

Outputs:
- Image count summary per split/class
- Corrupted file detection
- Image dimension statistics
- Plots saved to `ml/eda/outputs/`

### 2. Train the Model

```powershell
# Default settings (5 epochs, batch=16, lr=1e-4)
python -m ml.training.train_pneumonia

# Custom settings
python -m ml.training.train_pneumonia --epochs 15 --batch-size 32 --lr 1e-4

# Resume from checkpoint
python -m ml.training.train_pneumonia --resume ml/checkpoints/pneumonia_densenet121_best.pt --epochs 20
```

Best checkpoint saved to: `ml/checkpoints/pneumonia_densenet121_best.pt`

### 3. Evaluate on Test Set

```powershell
python -m ml.evaluation.evaluate_pneumonia
```

Outputs:
- Accuracy, Precision, Recall, F1, Specificity, ROC-AUC
- Classification report
- Confusion matrix plot → `ml/eda/outputs/confusion_matrix.png`
- ROC curve plot → `ml/eda/outputs/roc_curve.png`

### 4. Run Inference (Python)

```python
from ml.inference.pneumonia_predictor import PneumoniaPredictor

predictor = PneumoniaPredictor()
result = predictor.predict("path/to/chest_xray.jpg")
print(result)
# {
#   "prediction": "PNEUMONIA",
#   "confidence": 0.91,
#   "probability": 0.91,
#   "probabilities": {"NORMAL": 0.09, "PNEUMONIA": 0.91},
#   "modelVersion": "pneumonia-densenet121-v1"
# }
```

## Dataset Structure

```
app/data/chest_xray/
├── train/
│   ├── NORMAL/      (1,341 images)
│   └── PNEUMONIA/   (3,875 images)
├── val/
│   ├── NORMAL/      (8 images)
│   └── PNEUMONIA/   (8 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

**Note:** The validation set is very small (16 images). This is a known limitation of the Kaggle Chest X-Ray dataset. The test set (624 images) is used for reliable evaluation.

## Known Issues

- **Class imbalance:** Training set is 74% PNEUMONIA. Weighted CrossEntropyLoss compensates.
- **Small validation set:** Only 16 images. Validation accuracy will be noisy.
- **Python 3.11 alpha:** If `python --version` shows `3.11.0a1`, install stable Python 3.11.8+.
- **Windows multiprocessing:** `NUM_WORKERS=0` is set automatically on Windows.

## Architecture

- **Model:** DenseNet121 (pretrained on ImageNet), classifier head replaced with `Linear(1024, 2)`
- **Loss:** CrossEntropyLoss with inverse-frequency class weights
- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-5)
- **Mixed precision:** Automatic on CUDA (torch.amp.autocast)
- **Input:** 224x224 RGB (grayscale X-rays converted to 3-channel)
