# Three-Branch Normal-Only Image Anomaly Detection

## Overview

This project implements a **normal-only image anomaly detection pipeline** using three complementary feature-extraction branches:

1. **DINOv2** — captures global structure and semantic representation.
2. **ConvNeXt** — captures hierarchical and multi-scale visual patterns.
3. **WideResNet50 + PatchCore-style scoring** — focuses on local patch-level defects.

The system is designed for a **one-class / unsupervised anomaly detection setting**, where the training data contains only **normal images**.

Instead of training a conventional binary classifier, the pipeline learns the feature distribution of normal samples and assigns an anomaly score to unseen images based on how different they are from the learned normal distribution.

---

## Pipeline

```text
                         Input Image
                              |
            +-----------------+------------------+
            |                 |                  |
            v                 v                  v
         DINOv2            ConvNeXt          WideResNet50
            |                 |                  |
      Global / semantic   Multi-scale        Local patch
        features            features          features
            |                 |                  |
          PCA               PCA            PatchCore memory
            |                 |                  |
           kNN               kNN          Nearest normal patch
            |                 |                  |
        DINO score       ConvNeXt score     PatchCore score
            |                 |                  |
            +-----------------+------------------+
                              |
                   Robust score normalization
                              |
                              v
                    Equal-score fusion
                              |
                              v
                  Category-wise threshold
                              |
                     Normal / Anomaly
```

The final anomaly score is:

\[
S_{final} = \frac{Z(S_{DINO}) + Z(S_{ConvNeXt}) + Z(S_{PatchCore})}{3}
\]

where each branch score is robustly normalized before fusion.

---

## Why Three Branches?

### DINOv2

DINOv2 is used for high-level and structural representation. It is useful for detecting anomalies such as:

- abnormal object shape,
- missing or extra components,
- structural changes,
- unusual object arrangement,
- global appearance deviations.

The implementation uses intermediate transformer representations instead of relying only on the final embedding.

### ConvNeXt

ConvNeXt provides hierarchical CNN features at multiple stages. It is useful for:

- texture anomalies,
- surface changes,
- local appearance variations,
- medium-scale structural differences,
- abnormal visual patterns.

Features from several ConvNeXt stages are pooled and concatenated before dimensionality reduction.

### WideResNet50 + PatchCore-style Branch

The third branch focuses on local defects. Intermediate WideResNet50 feature maps are converted into patch embeddings and stored in a normal patch memory bank.

For each test patch, the model searches for the nearest normal patch. This branch is especially useful for detecting:

- scratches,
- small holes,
- local deformations,
- spots,
- tiny missing parts,
- localized texture defects.

The image-level PatchCore score is calculated from the most anomalous local patches rather than a single maximum value.

---

## Normal-Only Training Strategy

The pretrained backbones are **frozen** in the baseline implementation. There is no conventional neural-network training stage.

```text
Normal training images
        |
        v
Feature extraction
        |
        v
Normal feature memory / distribution
        |
        v
Distance-based anomaly scoring
```

This keeps the baseline focused on learning the distribution of normal data and reduces the risk of overfitting artificial anomalies.

---

## 5-Fold Out-of-Fold Validation

Because the training set contains only normal images, the project uses **5-fold out-of-fold (OOF) validation** for each category.

```text
Normal images
     |
     v
5 folds
     |
     +--> 80% normal images -> build normal memory
     |
     +--> 20% unseen normal images -> OOF normal validation
```

The held-out normal images are never included in the memory used to score them. This provides a more realistic estimate of anomaly scores for unseen normal samples and avoids self-neighbor leakage in kNN-based scoring.

---

## Synthetic Anomaly Validation

Synthetic anomalies are generated only from held-out normal images. They are used as a **validation and stress-testing tool**, not as real anomaly ground truth.

Implemented corruption types include:

- CutPaste,
- local masking,
- patch duplication,
- local blur,
- synthetic scratches.

For each fold:

```text
80% normal
   |
   +--> build normal model

20% held-out normal
   |
   +--> original image --------> normal validation
   |
   +--> corrupted copy --------> synthetic anomaly validation
```

This produces a pseudo-validation set containing:

```text
Normal OOF samples        -> label 0
Synthetic anomaly samples -> label 1
```

Synthetic validation is useful for comparing feature extractors, fusion strategies, and threshold robustness, but it is not assumed to perfectly represent hidden real anomalies.

---

## Feature Processing

### DINOv2 and ConvNeXt

For image-level features:

1. Extract pretrained features.
2. Fit PCA using normal training features only.
3. L2-normalize the reduced embeddings.
4. Build a k-nearest-neighbor normal memory.
5. Compute anomaly score from the mean distance to the nearest normal neighbors.

\[
S(x) = \frac{1}{K}\sum_{j \in KNN(x)} d(z_x, z_j)
\]

A larger distance indicates a higher degree of abnormality.

### PatchCore-style Scoring

For the WideResNet50 branch:

1. Extract intermediate feature maps.
2. Align feature-map resolutions.
3. Build local patch embeddings.
4. Apply random dimensionality projection.
5. Store representative normal patches in a memory bank.
6. Find the nearest normal patch for every query patch.
7. Aggregate the most anomalous patch distances into one image-level score.

---

## Robust Score Normalization

The three branches produce scores on different numerical scales, so raw scores are not directly averaged.

For each branch and category:

\[
Z(s) = \frac{s - \operatorname{median}(S_{normal})}{1.4826 \cdot MAD(S_{normal}) + \epsilon}
\]

where `MAD` is the median absolute deviation.

This reduces the influence of outliers and makes the three branch scores comparable before fusion.

---

## Score Fusion

The baseline uses equal-weight score fusion:

\[
S_{final} = \frac{S_{DINO} + S_{ConvNeXt} + S_{PatchCore}}{3}
\]

after robust normalization.

Equal weighting is intentionally used as the default because learned fusion weights can easily overfit synthetic validation data or a public leaderboard.

Future fusion experiments may include:

- weighted averaging,
- logistic-regression stacking,
- leave-one-branch-out ablation,
- category-specific fusion.

---

## Category-Wise Thresholding

Each category is treated independently. A separate threshold is estimated from the normal anomaly-score distribution:

\[
\tau_c = Q_p(S^{normal}_c)
\]

where `Q_p` is a selected percentile.

The default configuration uses the **97.5th percentile**.

Prediction is:

\[
\hat{y}(x) =
\begin{cases}
0, & S(x) < \tau_c \\
1, & S(x) \geq \tau_c
\end{cases}
\]

where:

- `0` = Normal
- `1` = Anomaly

---

## Data Mapping

The pipeline maps images using **`sample_id`**, not `relative_path`.

Expected training structure:

```text
dataset/
└── train/
    ├── images/
    ├── train1_6.csv
    ├── train2_5.csv
    └── train3_4.csv
```

The CSV files must contain at least:

```text
sample_id,category
```

If a `relative_path` column exists, it is ignored.

The loader recursively scans the `images/` directory and matches each image filename stem to its `sample_id`. The same mapping strategy is used for public and private test sets.

---

## Expected Dataset Structure

```text
dataset/
├── train/
│   ├── images/
│   ├── train1_6.csv
│   ├── train2_5.csv
│   └── train3_4.csv
│
├── public_test/
│   ├── images/
│   └── test.csv
│
└── private_test/
    ├── images/
    └── test.csv
```

---

## Main Configuration

```python
N_FOLDS = 5

DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"
CONVNEXT_MODEL = "convnext_tiny.fb_in22k"
PATCH_MODEL = "wide_resnet50_2.tv2_in1k"

PCA_DIM_DINO = 128
PCA_DIM_CONV = 128

KNN_K = 5

PATCH_GRID = 7
PATCH_PROJ_DIM = 128
MEMORY_PATCHES_PER_IMAGE = 8

NORMAL_PERCENTILE = 97.5
```

These values are intended as baseline hyperparameters and are not assumed to be globally optimal.

---

## Output Files

### OOF analysis

```text
oof_scores.csv
fold_metrics.csv
```

These files contain branch-level OOF results and synthetic validation metrics.

### Final calibration

```text
final_thresholds.csv
final_normal_models.joblib
```

These files store category-specific thresholds and the fitted normal-distribution models.

### Prediction details

```text
task2_public_prediction_details.csv
task2_private_prediction_details.csv
```

These files include:

- DINO anomaly score,
- ConvNeXt anomaly score,
- PatchCore anomaly score,
- fused anomaly score,
- category threshold,
- final prediction.

### Submission files

```text
task2_public_output.csv
task2_private_output.csv
```

with the format:

```text
sample_id,category,label
```

---

## Model Comparison

The notebook evaluates:

| Branch | Main Representation |
|---|---|
| DINOv2 | Global / semantic / structural |
| ConvNeXt | Multi-scale texture and structure |
| PatchCore | Local patch-level defects |
| FusionMean | Combined anomaly evidence |

Recommended evaluation criteria are:

- mean Balanced Accuracy,
- mean AUROC,
- fold-to-fold standard deviation,
- robustness across synthetic corruption types,
- per-category consistency.

A branch should not be selected only because it performs well on one synthetic anomaly type or one public submission.

---

## Recommended Experiment Order

| Experiment | DINO | ConvNeXt | PatchCore | Fusion |
|---|---:|---:|---:|---|
| E1 | ✓ |  |  | Single |
| E2 |  | ✓ |  | Single |
| E3 |  |  | ✓ | Single |
| E4 | ✓ | ✓ |  | Mean |
| E5 | ✓ |  | ✓ | Mean |
| E6 |  | ✓ | ✓ | Mean |
| E7 | ✓ | ✓ | ✓ | Equal Mean |
| E8 | ✓ | ✓ | ✓ | Learned / Weighted |

The equal-weight fusion should be established as a stable baseline before introducing learned stacking weights.

---

## Avoiding Public-Leaderboard Overfitting

Repeatedly changing thresholds, fusion weights, feature dimensions, kNN parameters, or synthetic corruption settings based only on public leaderboard feedback can lead to overfitting.

The recommended workflow is:

```text
Train-normal data
      |
      v
OOF normal validation
      +
Synthetic stress testing
      |
      v
Select stable configuration
      |
      v
Freeze model and thresholds
      |
      v
Public / Private inference
```

The public leaderboard should be treated as an external check rather than the primary source of hyperparameter optimization.

---

## Possible Future Improvements

Potential extensions include:

- leave-one-corruption-out validation,
- multi-scale DINO patch features,
- larger DINOv2 backbones,
- larger ConvNeXt variants,
- improved PatchCore coreset sampling,
- Mahalanobis anomaly scoring,
- score-level stacking,
- category-specific branch weighting,
- self-supervised fine-tuning on normal images,
- contrastive learning with normal augmentations and synthetic anomalies.

Any learned fusion or fine-tuning strategy should be validated carefully to avoid overfitting synthetic anomalies.

---

## Requirements

Typical dependencies:

```text
Python
PyTorch
torchvision
timm
scikit-learn
NumPy
Pandas
Pillow
tqdm
joblib
```

Example installation:

```bash
pip install timm scikit-learn pandas pillow tqdm joblib
```

---

## Reproducibility

The notebook fixes random seeds for:

- Python `random`,
- NumPy,
- PyTorch,
- synthetic anomaly generation,
- PCA,
- fold splitting,
- patch sampling.

This helps make feature extraction, OOF validation, synthetic corruption generation, and final predictions reproducible.

---

## Summary

This project follows a simple principle:

> **Learn what normal looks like, then detect samples that deviate from the learned normal representation.**

DINOv2, ConvNeXt, and PatchCore provide complementary views of image normality:

- DINOv2 captures global structure,
- ConvNeXt captures hierarchical texture and appearance,
- PatchCore captures localized defects.

Their anomaly scores are normalized and fused to form a final category-specific anomaly decision without requiring labeled anomaly samples during training.
