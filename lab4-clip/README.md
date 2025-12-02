# Lab 4: CLIP (Contrastive Language-Image Pre-Training)

**Authors:** Nathan Duncan and Daniel Dubinko

## Project Overview

This project implements a CLIP (Contrastive Language-Image Pre-Training) model designed to align image and text representations. The model uses a **ResNet50** backbone (initialized with ImageNet weights) as the image encoder and a frozen, pre-trained **CLIP text encoder** (from Hugging Face's `openai/clip-vit-base-patch32`) as the text encoder.

The model is trained using **InfoNCE loss** (symmetric cross-entropy loss) on the **COCO 2014** dataset to learn a shared embedding space where matching image-text pairs are close together.

The project is divided into two main parts:
1.  **Non-Augmentation**: A baseline implementation without additional data augmentation during training.
2.  **Augmentation**: An enhanced implementation that includes data augmentation techniques to improve model robustness and generalization.

## Directory Structure & File Descriptions

### `non_augmentation/`
This directory contains the baseline implementation.

*   **`dataset.py`**: Defines the `CocoDataset` class for loading COCO 2014 images and captions, and the `CocoCollator` for batch processing with the CLIP processor. **Note:** You may need to update the `BASE_ROOT` variable in this file if your dataset is stored in a different location.
*   **`model.py`**: Defines the `CLIPModel` architecture, including the `ImageEncoder` (ResNet50 + Projection Head) and `TextEncoder` (Frozen CLIP).
*   **`train.py`**: The main training script. It initializes the model, data loaders, and optimizer, and trains the model for a specified number of epochs using InfoNCE loss. It saves checkpoints to `checkpoints/`.
*   **`eval.py`**: The evaluation script. It loads a trained checkpoint, computes quantitative metrics (Recall@1, Recall@5, Recall@10 for Image-to-Text and Text-to-Image retrieval), and generates qualitative visualizations of retrieval results.
*   **`verify_dataset.py`**: A utility script to verify that the dataset and annotations are loading correctly.

### `augmentation/`
This directory contains the implementation with data augmentation.

*   **`dataset_augmentation.py`**: Similar to `dataset.py`, but likely includes data augmentation transforms applied to the images.
*   **`model_augmentation.py`**: The model definition for the augmented version.
*   **`train_augmentation.py`**: The training script for the augmented model. It saves checkpoints to `checkpoints_augmentation/`.
*   **`eval_augmentation.py`**: The evaluation script for the augmented model.

## Usage Instructions

### Prerequisites
*   Python 3.x
*   PyTorch
*   Transformers (Hugging Face)
*   COCO 2014 Dataset (Images and Annotations)

**Important:** If your COCO 2014 dataset is located in a directory other than `../coco2014`, please update the `BASE_ROOT` variable in `non_augmentation/dataset.py` and `augmentation/dataset_augmentation.py` before running any scripts.

### Training the Model

#### Non-Augmentation Model
1.  Navigate to the `non_augmentation` directory:
    ```bash
    cd non_augmentation
    ```
2.  Run the training script:
    ```bash
    python train.py
    ```

#### Augmentation Model
1.  Navigate to the `augmentation` directory:
    ```bash
    cd augmentation
    ```
2.  Run the training script:
    ```bash
    python train_augmentation.py
    ```

### Testing and Evaluation

#### Non-Augmentation Model
1.  Navigate to the `non_augmentation` directory:
    ```bash
    cd non_augmentation
    ```
2.  Run the evaluation script:
    ```bash
    python eval.py
    ```
    *Note: Ensure that trained checkpoints exist in `non_augmentation/checkpoints`.*

#### Augmentation Model
1.  Navigate to the `augmentation` directory:
    ```bash
    cd augmentation
    ```
2.  Run the evaluation script:
    ```bash
    python eval_augmentation.py
    ```
    *Note: Ensure that trained checkpoints exist in `augmentation/checkpoints_augmentation`.*
