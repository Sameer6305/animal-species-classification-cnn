# Dataset & Preprocessing Notes — Activity 2

## 1. Dataset Identification (Step 2)

| Field | Value |
|---|---|
| Dataset name | Animal-80 |
| Source | Kaggle — [antoreepjana/animals-detection-images-dataset](https://www.kaggle.com/antoreepjana/animals-detection-images-dataset) (Open-Images-derived) |
| Number of classes | 80 |
| Raw images (train) | 22,566 |
| Raw images (test) | 6,505 |
| **Total raw images** | **29,071** |
| Annotation format | Plain-text bounding boxes per image, stored in a `Label/` subfolder within each class folder. Format: `ClassName xmin ymin xmax ymax` (absolute pixel coordinates). One line per object instance; images with multiple animals have multiple lines. |
| Raw image dimensions | Highly variable. Width: 522–6000 px, Height: 289–4000 px (median ≈ 1024×768, sampled n=3,131). This variance is the primary motivation for the resize step below. |

**Note on discrepancy with the source paper:** Ukwuoma et al. (2022) report Animal-80 as containing 45,132 training and 13,010 test images. The current Kaggle release contains 22,566 train / 6,505 test — roughly half the volume described in the paper. This is documented here as an observed limitation of using a public dataset that has since been revised/reduced by its maintainer, not an error in our pipeline.

### Dataset Description
Animal-80 is a real-world, photograph-based image dataset covering 80 distinct animal species (mammals, birds, reptiles, fish, and invertebrates alike — e.g. Lion, Eagle, Snake, Fish, Spider). It was originally compiled for object-detection tasks, so every image ships with one or more bounding-box annotations marking the animal(s) present, rather than being a simple one-label-per-image classification set. Images are sourced from a mix of wildlife photography, stock photos, and — as we discovered while inspecting samples — occasionally illustrations or heavily stylized shots. Because it was built for detection rather than classification, image sizes, framing, and background content vary considerably (see raw dimension stats below), which is why the preprocessing pipeline described in Section 2 (crop-to-bbox, resize, normalize) exists.

### Class distribution
Severely imbalanced: largest class (Butterfly, 1,875 train images) is ~268× larger than the smallest (Seahorse, 7 train images). Full distribution is in `outputs/figures/class_distribution.png`.

Rarest training classes (<50 images): Seahorse (7), Squid (15), Turtle (24), Shrimp (30), Red panda (45).

### Train / Validation / Test split
- **Test set (7,576 crops after box-extraction)**: taken directly from the dataset's original `test/` folder, left untouched as a clean holdout.
- **Train set**: the original `train/` folder was further split into **train** and **val** per class (stratified), with a minimum of 2 validation images guaranteed per class even for the smallest categories, rather than a proportional split that would round some rare classes down to 0 validation images.
- Final split sizes (post-crop, pre-augmentation): train 23,982 / val 4,232 / test 7,576.

---

## 2. Preprocessing Pipeline (Step 3)

Pipeline: **Raw dataset → cleaning → crop-to-bbox → resize → normalization → augmentation → split**

### 2.1 Data Cleaning
All 29,071 source images were scanned for two failure conditions: missing annotation files, and unreadable/corrupt image files. **Result: 0 images skipped for either reason** — the dataset was found to be structurally clean, with a valid annotation file present for every image.

### 2.2 Crop-to-Bounding-Box
Each image was cropped to its annotated bounding box(es) rather than classified as a whole image. **Justification:** since Animal-80 images may contain background clutter, multiple objects, or the animal occupying only a small part of the frame, cropping isolates the classification-relevant region and removes irrelevant context — appropriate for a pure classification task, even though the source paper used these boxes for object detection. Images containing multiple annotated animals contribute one crop per bounding box.

### 2.3 Resize
All crops resized to **224×224** using bilinear interpolation, matching the standard input size expected by ImageNet-pretrained backbones (ResNet50, EfficientNet, ConvNeXt). Note: the source paper used 356×356, but that figure was tuned for their FPN/anchor-based detection stage and is not relevant to a classification-only pipeline.

### 2.4 Normalization
Per-channel mean and standard deviation were computed directly from a sample of the processed training crops (n=2,000), rather than assuming ImageNet defaults, since these are tightly cropped animal photos rather than general natural images:

- **Mean (RGB):** [0.4629, 0.4413, 0.3891]
- **Std (RGB):** [0.2658, 0.2520, 0.2598]

These values are applied at training time by all three models (baseline, efficient, modern CNN) for consistency.

### 2.5 Handling Class Imbalance
Given the 268× imbalance ratio, all training classes with fewer than 100 images were augmented up to a target of 150 images each (27 classes affected, e.g. Seahorse 7→150, Squid 15→150, Turtle 24→150). This was done only on the **train** split — validation and test sets were left unaugmented to preserve honest evaluation.

### 2.6 Data Augmentation
Applied to rare-class training images only:
- **Rotation** (±25°)
- **Horizontal flip** (50% probability)
- **Zoom** (crop to 80–95% of original size, then resize back)
- **Translation** (random shift up to 10% of image dimensions)
- **Shear** (affine shear transform, factor ±0.15)

**Vertical flip was deliberately excluded.** Animals do not naturally appear upside-down in real-world photographs; including vertical flips would introduce unrealistic training examples that do not reflect the true deployment distribution, potentially harming rather than helping generalization.

Example before/after augmentation comparisons are in `outputs/figures/augmentation_examples.png`.

### 2.7 Output Artifacts
- `data/manifest.csv` — master index (`filepath, label, split, source_image`) covering all train/val/test crops, including augmented images
- `outputs/figures/class_distribution.png`
- `outputs/figures/sample_crops_grid.png`
- `outputs/figures/augmentation_examples.png`

---

## 3. Observed Dataset Limitations

While generating sample visualizations, the following data quality issues were observed in the source dataset (not introduced by our pipeline, but worth noting given their potential effect on model performance):

- **Label noise:** at least one instance of an image labeled "Bear" that visually appears to be a red panda.
- **Non-photographic content:** at least one "Elephant"-labeled image is an illustration/engraving rather than a photograph.
- **Image quality variance:** some crops are affected by motion blur (e.g. Sparrow, Eagle) or heavy stylized lighting/vignette effects (e.g. Lion, Horse) present in the original source photos.
- **Severe class imbalance**, partially mitigated via augmentation (Section 2.5/2.6) but not fully eliminated — rare classes remain underrepresented relative to common ones even after augmentation to 150 images.

These are consistent with known quality issues in large-scale, scraped/crowd-sourced image datasets and are documented here for transparency rather than corrected, since manual relabeling was out of scope for this activity.