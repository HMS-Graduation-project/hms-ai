# Chest X-Ray Pneumonia Dataset — Complete EDA Study

**Date:** 2026-06-04
**Analyst:** HMS AI Team
**Dataset:** Kaggle Chest X-Ray Images (Pneumonia) — Guangzhou Women and Children's Medical Center
**Python:** 3.11.4 stable

---

## 1. Executive Summary

This dataset contains **5,856 pediatric chest X-ray images** classified as NORMAL or PNEUMONIA, sourced from the Guangzhou Women and Children's Medical Center. It is the well-known Kaggle Chest X-Ray Images (Pneumonia) dataset.

**Key findings:**

- **Severe class imbalance**: PNEUMONIA outnumbers NORMAL 2.70:1 overall (2.89:1 in train)
- **Critically small validation set**: Only 16 images (8 per class) — statistically useless
- **32 exact duplicate images** within the training set (30 duplicate groups)
- **No cross-split leakage detected** — train/val/test are hash-disjoint
- **Zero corrupted files** across all 5,856 images
- **Extreme file size disparity**: NORMAL images average 538 KB vs PNEUMONIA at 83 KB — a 6.5x difference suggesting different source scanners or export settings
- **Significant dimension variation**: 867 unique widths, 1,089 unique heights — mandatory resize needed
- **283 RGB images mixed with 5,573 grayscale** — channel standardization required
- **Dataset is usable for ML** with preprocessing, but has notable quality risks that must be addressed

**ML Readiness Score: 6.5/10** — Adequate for research/prototyping after preprocessing. Not production-grade without augmentation and validation set expansion.

---

## 2. Dataset Structure Analysis

```
app/data/chest_xray/
├── train/
│   ├── NORMAL/      1,341 images
│   └── PNEUMONIA/   3,875 images
├── val/
│   ├── NORMAL/      8 images
│   └── PNEUMONIA/   8 images
└── test/
    ├── NORMAL/      234 images
    └── PNEUMONIA/   390 images
```

**Structure quality:**
- Correct ImageFolder convention (class-as-folder)
- All expected directories present
- No unexpected files or nested folders
- No missing label folders
- Standard train/val/test split structure

**Critical issue:** The validation split is not a true validation set — it's a placeholder with 16 images.

---

## 3. Dataset Inventory Report

| Split | NORMAL | PNEUMONIA | Total | Pneumonia % | Ratio |
|-------|--------|-----------|-------|-------------|-------|
| **Train** | 1,341 | 3,875 | 5,216 | 74.3% | 2.89:1 |
| **Val** | 8 | 8 | 16 | 50.0% | 1.00:1 |
| **Test** | 234 | 390 | 624 | 62.5% | 1.67:1 |
| **Total** | **1,583** | **4,273** | **5,856** | **73.0%** | **2.70:1** |

**Split proportions:**
- Train: 89.1% of total
- Test: 10.7% of total
- Val: 0.3% of total

---

## 4. Class Distribution Analysis

### Imbalance Assessment

| Metric | Value | Severity |
|--------|-------|----------|
| Train PNEUMONIA:NORMAL ratio | 2.89:1 | **Severe** |
| Test PNEUMONIA:NORMAL ratio | 1.67:1 | **Moderate** |
| Overall PNEUMONIA prevalence | 73.0% | **High** |
| Minority class (NORMAL) share | 27.0% | **Low** |

**Impact:** An untrained model predicting "PNEUMONIA" for every image would achieve 74.3% accuracy on the training set. This makes accuracy a misleading metric. **Weighted loss, F1-score, and balanced accuracy are mandatory evaluation criteria.**

**Recommended class weights for CrossEntropyLoss:**
- NORMAL: `3875 / (1341 + 3875) = 0.743` (upweighted)
- PNEUMONIA: `1341 / (1341 + 3875) = 0.257` (downweighted)

---

## 5. Image Metadata Analysis

### Dimensions

| Metric | Width | Height |
|--------|-------|--------|
| Min | 384 px | 127 px |
| Max | 2,916 px | 2,713 px |
| Mean | 1,328 px | 971 px |
| Median | 1,281 px | 888 px |
| Unique values | 867 | 1,089 |

**Observation:** Extreme variation. The smallest image is 384x127 (highly atypical radiograph) and the largest is 2916x2713. This confirms **mandatory resizing** during preprocessing.

### Color Modes

| Mode | Count | Percentage |
|------|-------|------------|
| Grayscale (L) | 5,573 | 95.2% |
| RGB | 283 | 4.8% |

**Action required:** Convert all images to a consistent channel format. For DenseNet121 (pretrained on ImageNet RGB), convert grayscale to 3-channel via `.convert("RGB")`.

### File Formats

All images are JPEG (`.jpeg` extension). No PNGs, BMPs, or TIFFs.

### File Sizes

| Class | Min | Max | Mean | Median | Total |
|-------|-----|-----|------|--------|-------|
| NORMAL | 45 KB | 2,357 KB | 538 KB | 506 KB | ~831 MB |
| PNEUMONIA | 5 KB | 583 KB | 83 KB | 70 KB | ~345 MB |

**Critical observation:** NORMAL images are **6.5x larger** on average than PNEUMONIA images. This strongly suggests:
1. Different scanner equipment or export settings for each class
2. NORMAL images may be higher resolution originals
3. PNEUMONIA images may be compressed clinical copies

**Risk:** The model could learn to distinguish classes by JPEG compression artifacts rather than clinical features. This is a form of **spurious correlation**.

---

## 6. Resolution Analysis

### Aspect Ratios

| Category | Count | Percentage |
|----------|-------|------------|
| Landscape (>1.05) | 5,709 | 97.5% |
| Square (~1.0) | 123 | 2.1% |
| Portrait (<0.95) | 24 | 0.4% |

| Metric | Value |
|--------|-------|
| Min ratio | 0.835 |
| Max ratio | 3.379 |
| Mean ratio | 1.443 |

**Observation:** Most images are landscape orientation (wider than tall), typical of PA chest radiographs. The extreme ratio of 3.379 indicates at least one severely cropped or improperly formatted image.

**Resizing strategy recommendation:**
1. `Resize(256)` — scale shortest edge to 256px
2. `CenterCrop(224)` — extract center square for DenseNet121
3. This preserves the central lung fields which contain the diagnostic information

---

## 7. Image Quality Assessment

### Brightness (Mean Pixel Intensity, 0-255 scale)

| Split/Class | Mean | Std | Range |
|-------------|------|-----|-------|
| train/NORMAL | 122.4 | 12.9 | 73–170 |
| train/PNEUMONIA | 123.2 | 20.1 | 61–222 |
| test/NORMAL | 124.1 | 16.0 | 80–161 |
| test/PNEUMONIA | 119.2 | 17.3 | 59–159 |
| val/NORMAL | 123.8 | 23.4 | 89–152 |
| val/PNEUMONIA | 126.1 | 16.3 | 99–143 |

**Findings:**
- Mean brightness is consistent across classes (~122–124), which is good
- PNEUMONIA images have **higher brightness variance** (std=20.1 vs 12.9 in train) — likely due to diverse source scanners
- **1 image** with brightness >220 (near white — potentially overexposed or blank)
- **0 images** with brightness <30 (no extremely dark images)

### Contrast (Pixel Intensity Standard Deviation)

| Class | Mean Contrast | Std |
|-------|--------------|-----|
| NORMAL | 61.3 | 5.8 |
| PNEUMONIA | 55.4 | 10.0 |

**Key finding:** NORMAL images have **systematically higher contrast** (61.3 vs 55.4). This is clinically expected — pneumonia opacities reduce lung field contrast by filling air spaces with fluid/consolidation. This is a legitimate diagnostic signal, not an artifact.

### Quality Issues Detected

| Issue | Count | Risk |
|-------|-------|------|
| Very dark (<30 brightness) | 0 | None |
| Very bright (>220 brightness) | 1 | Low |
| Corrupted files | 0 | None |
| Extremely small files (<5 KB) | Present in PNEUMONIA | Review |
| Min file size PNEUMONIA | 5 KB | **Suspiciously small** |

---

## 8. Duplicate Detection Analysis

### Exact Duplicates (MD5 Hash)

| Metric | Value |
|--------|-------|
| Unique image hashes | 5,824 |
| Duplicate groups | 30 |
| Redundant images | 32 |

**All 30 duplicate groups are within the training set.** No duplicates cross split boundaries.

**Pattern:** Duplicates follow the naming pattern `personXXXX_bacteria_YYYY.jpeg` and `personXXXX_bacteria_YYYY+1.jpeg` — the same scan saved with consecutive filenames. Examples:
- `person1372_bacteria_3501.jpeg` = `_3502` = `_3503` (3 identical copies)
- `person258_bacteria_1207.jpeg` = `_1208` = `_1209` (3 identical copies)

**Impact:** 32 redundant images in a 5,216-image training set is negligible (0.6%). However, they slightly inflate PNEUMONIA class counts and should be removed for clean evaluation.

### Filename Duplicates

No filenames appear in multiple locations. Each filename is unique across the entire dataset.

---

## 9. Dataset Leakage Analysis

| Split Pair | Overlapping Images | Status |
|------------|-------------------|--------|
| Train ↔ Val | 0 | **Clean** |
| Train ↔ Test | 0 | **Clean** |
| Val ↔ Test | 0 | **Clean** |

**Leakage risk assessment: LOW.** No identical images exist across splits.

**Patient-level leakage caveat:** The filename convention (`personXXX_virus_YYY.jpeg`) embeds patient IDs. Multiple images from the same patient could appear in different splits. Without the original metadata, we cannot verify patient-level separation. This is a **known limitation of the Kaggle version** of this dataset.

---

## 10. Pixel-Level Analysis

### Intensity Statistics

| Class | Mean Intensity | Median | Std Dev |
|-------|---------------|--------|---------|
| NORMAL | 122.6 | 122.4 | 13.5 |
| PNEUMONIA | 122.8 | 122.9 | 19.9 |

**Observation:** Mean pixel intensity is nearly identical between classes. The discriminative signal lies in **spatial patterns** (lung opacity distribution), not global intensity. This confirms that the model must learn spatial features, not just histogram statistics.

### Normalization Recommendation

Use **ImageNet normalization** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for transfer learning with DenseNet121 pretrained on ImageNet. This is standard practice and outperforms dataset-specific normalization when using pretrained backbones.

### Scaling Strategy
- Convert pixel values to [0, 1] via `ToTensor()` (automatic in torchvision)
- Then apply ImageNet normalization
- No dataset-specific scaling needed

---

## 11. Medical Imaging Assessment

### Clinical Suitability

| Criterion | Assessment |
|-----------|-----------|
| Image modality | Chest PA radiographs — appropriate for pneumonia detection |
| Patient population | Pediatric (ages 1-5) — results may not generalize to adults |
| Labeling quality | Verified by 2 expert physicians, graded by third (per original paper) |
| Clinical relevance | Binary NORMAL/PNEUMONIA — clinically meaningful but simplified |
| Pneumonia subtypes | Includes both bacterial and viral pneumonia (mixed in one class) |

### Radiography Quality Observations

1. **Consistent positioning:** Most images show standard PA chest positioning
2. **Varying exposure:** Brightness variation (std=12.9 to 23.4) indicates different scanner settings — typical for multi-center datasets
3. **Contrast difference between classes is clinically meaningful:** Pneumonia reduces lung field contrast via consolidation/ground-glass opacities
4. **File size discrepancy is NOT clinically meaningful:** The 6.5x size difference between NORMAL and PNEUMONIA suggests different digital export pipelines, not diagnostic significance

### Labeling Concerns

1. **Binary label simplification:** Pneumonia has subtypes (bacterial, viral, fungal) with different radiographic presentations. Merging them reduces clinical utility.
2. **Filename-encoded subtypes:** `_bacteria_` and `_virus_` in filenames reveal the original 3-class labeling. This metadata could support subtype analysis.
3. **Pediatric-only population:** Models trained on this data should NOT be deployed for adult chest X-ray screening without validation on adult datasets.

---

## 12. Data Cleaning Recommendations

| Step | Action | Rationale | Priority |
|------|--------|-----------|----------|
| 1 | Remove 32 exact duplicate images | Prevents training bias from repeated samples | High |
| 2 | Investigate the 5 KB PNEUMONIA image | Suspiciously small — may be corrupt or improperly exported | High |
| 3 | Review the 1 overexposed image (brightness >220) | May be a blank or improperly scanned image | Medium |
| 4 | Review the 24 portrait-orientation images | Atypical orientation for chest X-rays — may be rotated or cropped | Medium |
| 5 | Review the image with aspect ratio 3.379 | Severely non-standard — likely a cropping artifact | Medium |
| 6 | Expand validation set | 16 images is statistically useless — carve 10-15% from training set | **Critical** |

---

## 13. Data Augmentation Recommendations

| Augmentation | Purpose | Recommended Parameters |
|-------------|---------|----------------------|
| **Random Horizontal Flip** | Chest X-rays are approximately symmetric; doubles effective dataset | p=0.5 |
| **Random Rotation** | Accounts for slight patient positioning variation | ±10° |
| **Brightness Jitter** | Accounts for scanner exposure variation (std=13-20 across classes) | ±0.1 |
| **Contrast Jitter** | Accounts for scanner contrast variation | ±0.1 |
| **Random Affine (translate)** | Accounts for patient centering variation | translate=(0.05, 0.05) |
| **Random Resized Crop** | Simulates zoom variation and forces model to focus on local patterns | scale=(0.85, 1.0) |

**Not recommended:**
- Vertical flip — chest X-rays are never upside-down in clinical practice
- Heavy color jitter — radiographs have clinically meaningful intensity patterns
- Elastic deformation — can distort anatomical structures unrealistically
- Cutout/erasing — risk obscuring the pneumonia opacity itself

---

## 14. Preprocessing Recommendations

| Step | Implementation | Rationale |
|------|---------------|-----------|
| 1. Channel standardization | `.convert("RGB")` on all images | DenseNet121 expects 3-channel input; 4.8% of images are already RGB |
| 2. Resize | `Resize(256)` (shortest edge) | Standardize variable dimensions (384-2916px range) |
| 3. Center crop | `CenterCrop(224)` | DenseNet121 input size; preserves central lung fields |
| 4. Tensor conversion | `ToTensor()` | Scales [0,255] → [0,1] |
| 5. Normalization | `Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])` | ImageNet stats for pretrained backbone |
| 6. Train augmentation | Flip + Rotation + ColorJitter (after step 2, before step 4) | Only for training DataLoader |

---

## 15. ML Readiness Assessment

### Scoring Methodology

Each dimension scored 1-10 based on industry standards for medical imaging ML.

| Dimension | Score | Justification |
|-----------|-------|--------------|
| **Data Quality** | 8/10 | Zero corrupted files, consistent format, good labeling provenance (expert-verified). Deducted for file size disparity and 283 mixed-mode images. |
| **Leakage Risk** | 9/10 | No image-level leakage detected. Deducted 1 point for unverifiable patient-level leakage. |
| **Class Balance** | 4/10 | Severe 2.89:1 imbalance in training set. Minority class (NORMAL) is the clinically critical class (missing pneumonia is worse than false alarm). |
| **Split Quality** | 3/10 | Validation set with 16 images is unusable. Test set is reasonable (624 images). |
| **Dimension Consistency** | 5/10 | 867 unique widths — extreme variation requires mandatory resize. |
| **Clinical Validity** | 7/10 | Expert-labeled pediatric CXRs from a single center. Limited generalizability. |

### Overall ML Readiness Score: **6.5 / 10**

**Interpretation:** The dataset is suitable for **research and prototyping** after preprocessing. For production deployment, it would need:
- Expanded and properly stratified validation set
- Adult population data
- Multi-center validation
- Class balance remediation

---

## 16. Final Recommendations

### Must-Do Before Training

1. **Expand the validation set** — randomly sample ~520 images (10%) from training set, stratified by class
2. **Remove 32 duplicate images** from training set
3. **Apply weighted loss** or stratified sampling to address 2.89:1 class imbalance
4. **Standardize channels** — convert all images to RGB
5. **Resize to 256px + CenterCrop 224px** — mandatory given dimension variance

### Should-Do for Robust Results

6. **Use F1, balanced accuracy, and AUC-ROC** as primary metrics — NOT accuracy
7. **Apply moderate augmentation** (flip, rotation, brightness jitter) to training set only
8. **Monitor for file-size-based spurious correlation** — the 6.5x size gap between classes could leak signal through JPEG artifact patterns
9. **Track per-class precision and recall separately** — missing pneumonia (false negative) is clinically worse than false alarm

### Known Limitations to Document

10. **Pediatric-only population** — do not generalize to adults
11. **Single-center data** — may not transfer to other hospitals
12. **Binary classification** — does not distinguish bacterial vs viral pneumonia
13. **Patient-level leakage unverifiable** — multiple scans from the same patient may exist across splits

---

*Report generated from HMS-AI EDA analysis pipeline.*
*Dataset: Kaggle Chest X-Ray Images (Pneumonia)*
*Source: Kermany et al., 2018 — Guangzhou Women and Children's Medical Center*
