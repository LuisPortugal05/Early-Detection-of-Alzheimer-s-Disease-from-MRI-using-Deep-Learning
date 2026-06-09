# Early Detection of Alzheimer's Disease from MRI

**A Comparative Study of a Baseline CNN, ResNet-50, and a Vision Transformer**

Luis Miguel Portugal Kegel · AI Engineering (Undergraduate), Universidad Panamericana

---

## Overview

This repository contains the full, reproducible code for a four-class
Alzheimer's disease staging task from brain MRI. Three deep learning
architectures of increasing capacity — a baseline CNN trained from scratch, a
fine-tuned ResNet-50, and a fine-tuned Vision Transformer (ViT-B/16) — are
compared under a single, leakage-free experimental protocol.

Because a missed early diagnosis is the costliest error in this setting, models
are evaluated not only on overall accuracy but, primarily, on **per-class recall
for the clinically critical Very Mild Demented stage**.

| Model | Accuracy | Recall (Very Mild) | Macro-F1 | Params | Inference (962 img) |
|-------|:--------:|:------------------:|:--------:|:------:|:-------------------:|
| Baseline CNN | 94.28% | 0.9018 | 0.9564 | 6.45M | 2.54 s |
| ResNet-50 | 95.63% | 0.9345 | 0.9646 | 24.03M | 3.47 s |
| **ViT-B/16** | **96.99%** | **0.9732** | **0.9766** | 86.0M | 9.93 s |

Accuracy increases monotonically with capacity, and the transformer's largest
gain falls precisely on the hardest, most clinically important class.

---

## Dataset

[Alzheimer's Dataset (4 class of images)](https://www.kaggle.com/datasets/preetpalsingh25/alzheimers-dataset-4-class-of-images)
— grayscale brain MRI slices labelled into four stages: Non Demented, Very Mild
Demented, Mild Demented, and Moderate Demented.

> **Data-leakage warning.** The original source ships the *same* images across
> its `train/` and `test/` folders. Using that split as-is leaks test images
> into training and produces misleadingly optimistic scores. This project
> **discards the provided test folder entirely** and rebuilds a clean,
> stratified 70/15/15 split from the unique training images only
> (see `01_Data_Prep.ipynb`). A filename-overlap check confirms zero
> intersection between the resulting partitions.

---

## Repository structure

| File | Description |
|------|-------------|
| `01_Data_Prep.ipynb` | Builds the clean, leakage-free dataset: pools the unique training images, performs a per-class stratified 70/15/15 split with a fixed seed, and applies offline augmentation to balance the training classes. |
| `02_Baseline_CNN.ipynb` | A lightweight CNN trained from scratch — the performance floor. **94.28%** test accuracy. |
| `03_ResNet50.ipynb` | Transfer learning with an ImageNet-pretrained ResNet-50; `layer4` and a custom head are fine-tuned. **95.63%** test accuracy. |
| `04_ViT.ipynb` | Fine-tuning of an ImageNet-pretrained ViT-B/16; the last three encoder blocks and the head are unfrozen. Best model, strongest recall on the critical class. **96.99%** test accuracy. |
| `05_Testing_and_Results.ipynb` | Final evaluation of all three models on the held-out test set. Produces accuracy, precision, recall, F1, confusion matrices, comparison plots, and qualitative interpretability maps (Grad-CAM for the CNNs, attention rollout for the ViT). |
| `requirements.txt` | Python dependencies. |

---

## Installation

Requires Python 3.10+ and (for training) a CUDA-capable GPU.

```bash
git clone https://github.com/LuisPortugal05/Early-Detection-of-Alzheimer-s-Disease-from-MRI-using-Deep-Learning.git
cd Early-Detection-of-Alzheimer-s-Disease-from-MRI-using-Deep-Learning

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Then download the dataset from the Kaggle link above and place it where
`01_Data_Prep.ipynb` expects it (see the path at the top of that notebook).

---

## How to run

Run the notebooks **in numerical order** (`01` → `05`):

1. **`01_Data_Prep.ipynb`** — builds and freezes the clean train/val/test split.
   *Run this first; everything else depends on its output.*
2. **`02`–`04`** — train each model. Each saves its best checkpoint
   (`baseline_cnn.pth`, `resnet50.pth`, `vit.pth`).
3. **`05_Testing_and_Results.ipynb`** — loads all three checkpoints and produces
   every metric, table, and figure used in the paper.

---

## Experimental protocol

To ensure a fair comparison, the data pipeline, normalisation, loss function,
and training budget are held constant across all three models:

- **Input** — 3-channel, 224×224, ImageNet normalisation.
- **Loss** — class-weighted cross-entropy (weights inversely proportional to
  class frequency) to counter severe class imbalance.
- **Training** — batch size 16, automatic mixed precision, gradient-norm
  clipping at 1.0, 30 epochs.
- **Checkpointing** — the weights with the lowest *validation loss* are kept,
  not the final-epoch weights (both pretrained models over-fit within a few
  epochs).
- **Reproducibility** — a fixed random seed (42) governs the split and training.

---

## Citation

If you use this work, please cite the accompanying paper:

> L. M. Portugal Kegel, "Early Detection of Alzheimer's Disease from MRI:
> A Comparative Study of a Baseline CNN, ResNet-50, and a Vision Transformer,"
> Universidad Panamericana, 2025.

**Contact:** luismi.porke@gmail.com
