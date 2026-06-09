================================================================
Early Detection of Alzheimer's Disease from MRI
A Comparative Study of a Baseline CNN, ResNet-50, and a Vision Transformer
================================================================

Author: Luis Miguel Portugal Kegel
AI Engineering, Universidad Panamericana

This repository contains the code for a four-class Alzheimer's disease
staging task from brain MRI, comparing three deep learning architectures.

----------------------------------------------------------------
FILES
----------------------------------------------------------------

01_Data_Prep.ipynb
    Builds a clean, leakage-free dataset. The original Kaggle source
    duplicates the same images across its train/ and test/ folders, so
    that split is discarded entirely. All unique images from the training
    folder are pooled and re-split per class into a stratified
    70/15/15 train/validation/test partition. Offline data augmentation
    is applied to balance the training classes.

02_Baseline_CNN.ipynb
    A lightweight CNN trained from scratch on the prepared dataset.
    Serves as the performance floor (baseline) for the comparison.
    Test accuracy: 94.28%.

03_ResNet50.ipynb
    Transfer learning with a ResNet-50 pretrained on ImageNet. All layers
    are frozen except layer4 and a custom classification head.
    Test accuracy: 95.63%.

04_ViT.ipynb
    Fine-tuning of a Vision Transformer (ViT-B/16) pretrained on ImageNet.
    The last three encoder blocks and the classification head are unfrozen.
    Best-performing model, with the strongest recall on the clinically
    critical "VeryMild" class. Test accuracy: 96.99%.

05_Testing_and_Results.ipynb
    Final evaluation of all three trained models on the held-out test set
    (built in 01 and never seen during training or validation). Produces
    accuracy, precision, recall, F1-score, confusion matrices, and the
    comparison plots used in the paper.

----------------------------------------------------------------
NOTES
----------------------------------------------------------------
- All notebooks use a fixed random seed (42) for reproducibility.
- Models are trained with mixed precision, gradient clipping, and class
  loss weighting; the best checkpoint is selected by lowest validation loss.
- ImageNet normalisation is applied consistently across all models.
- Run the notebooks in numerical order (01 -> 05).
