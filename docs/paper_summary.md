# Research Paper Summary

**Title:** Animal species detection and classification framework based on modified multi-scale attention mechanism and feature pyramid network
**Authors:** Chiagoziem C. Ukwuoma, Zhiguang Qin, Sophyani B. Yussif, Monday N. Happy, Grace U. Nneji, Gilbert C. Urama, Chibueze D. Ukwuoma, Nimo B. Darkwa, Harriet Agobah
**Journal:** Scientific African (Elsevier), Vol. 16, 2022, e01151
**DOI:** 10.1016/j.sciaf.2022.e01151

## Problem Statement & Objective
Detecting and classifying animal species accurately is important for wildlife conservation, reducing human–animal conflict (predator identification, roadway collisions), and monitoring biodiversity. Traditional machine-learning approaches (SIFT, local binary patterns, SVM) require manually cropped, known-shape animal images and generalize poorly. The paper's objective is to build a deep-learning framework that reliably detects and classifies animal species — including small or partially visible animals — despite high variation in size, shape, color, and behavior across species.

## Proposed Approach
The authors propose a **two-stage network**:
1. **Detection stage** — a ResNet-50 backbone extracts multi-scale feature maps (c1–c5), which pass through a novel **modified multi-scale attention mechanism** (dot-product similarity + softmax weighting across scales) before feeding a **Feature Pyramid Network (FPN)**. This combination sharpens the model's sensitivity to small animals without significantly increasing model complexity. A Region Proposal Network (RPN) then generates bounding-box proposals using custom anchor sizes tuned for small targets.
2. **Classification stage** — a **DenseNet** classification head processes the detected regions, using dense connectivity to boost accuracy while keeping parameter count lower than conventional CNNs.
**Focal Loss** is used throughout to address class imbalance and emphasize hard-to-classify examples.

## Dataset
Two datasets were used:
- **African Wildlife Dataset**: 4 classes (Buffalo, Elephant, Rhino, Zebra), 3,008 images total, roughly balanced.
- **Animal-80 Dataset**: 80 animal classes, reported as 45,132 training / 13,010 test images (the publicly available Kaggle version used for this activity was found to contain fewer images than reported — see `dataset_notes.md`). Both datasets use bounding-box annotations (PascalVOC / YOLO-derived formats).

## Preprocessing (as done in the paper)
Images were resized to 356×356. Bounding-box annotations were recomputed after resizing. The African Wildlife dataset's annotations were converted from YOLO format to PascalVOC format. No data augmentation was applied by the original authors.

## Training Setup
SGD optimizer, learning rate 1e-4, weight decay 0.0001, momentum 0.9, batch size 8, 100 epochs, single GPU (NVIDIA GTX 1080Ti). Backbone pretrained on ImageNet-1k.

## Evaluation Metrics & Results
The paper uses **Average Precision (AP)** per class and **mean Average Precision (mAP)** overall (standard object-detection metrics, not classification accuracy).
- On the African Wildlife dataset: the attention-based model achieved **mAP 0.87** vs. **0.85** without attention, and outperformed YOLOv3, RetinaNet, Faster R-CNN, and Cascade R-CNN baselines (all below 0.86 mAP) on a matched 4-class comparison.
- On Animal-80: attention-based mAP outperformed the non-attention baseline by roughly **+0.1%**, with individual class AP gains of 5–20%, though performance on rare classes (e.g. Seahorse, Squid, Turtle) remained low due to severe class imbalance and no augmentation being applied.

## Reported Limitations (acknowledged by the authors)
No data augmentation was used for Animal-80, so heavily underrepresented classes performed poorly. The model was not tested on scenes containing multiple animal classes simultaneously. Sensitivity of the attention mechanism's internal parameters was not explored.

## Relevance to This Activity
This paper's task (two-stage detection + classification) differs from our activity's scope (pure classification using ResNet50/EfficientNet/ConvNeXt-family backbones). We are using the same **Animal-80** dataset and building on **ResNet50** (also used as the paper's backbone) as our baseline classifier, while adapting the task to whole-object classification via bounding-box crops rather than full detection — a deliberate scope adjustment explained further in `dataset_notes.md`.