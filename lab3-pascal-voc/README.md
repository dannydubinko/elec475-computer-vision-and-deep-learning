# Semantic Segmentation & Knowledge Distillation

Nathan Duncan 20ntd1@queensu.ca

Daniel Dubinko 19dd34@queensu.ca

## Description

This project implements a full knowledge distillation (KD) pipeline for semantic segmentation on the PASCAL VOC 2012 dataset.

A compact "student" model (MobileNetV3-Small + ASPP) is trained in two ways:

1. Standard Training: Trained only on the ground truth labels.

2. Distillation Training: Trained using a "teacher" model (FCN-ResNet50) to guide its learning, using both response-based and feature-based distillation losses.

The goal is to produce a small, fast model (the student) that achieves a segmentation performance close to that of the large, complex model (the teacher).


## Pipeline Summary 
The pipeline is fully functional and capable of:

1. Training a student model without distillation (baseline).

2. Training a student model with a complex, three-part distillation loss.

3. Evaluating all models (Teacher, Student-Baseline, Student-Distilled) to get quantitative mIoU scores.

4. Generating qualitative visual results (image overlays) for all models.

## File Structure

`student_model.py:` Defines your lightweight student model architecture, combining a MobileNetV3-Small backbone with an ASPP (Atrous Spatial Pyramid Pooling) head.

`dataset.py:` A robust data-loading script that handles all training and validation augmentations (cropping, flipping, normalization) for the PASCAL VOC dataset.

`train_kd.py:` The "engine" of your lab. This flexible script loads both student and teacher models and trains the student. By using command-line arguments (--alpha, --beta), it can run in either "standard training" mode (loss = loss_hard) or "distillation" mode (loss = ... + loss_soft + loss_feat).

`test_teacher.py:` A standalone script to benchmark the FCN-ResNet50 teacher model, providing the "upper-bound" mIoU score.

`evaluate.py:` A standalone script to load your saved .pth student models and evaluate their final mIoU on the validation set.

`count_parameters.py:` A utility to verify the small size of your student model against the large teacher model.

`visualize.py:` A helper utility that generates the "Input | Ground Truth | Prediction" overlay images.

`README.md:` Full documentation on how to set up the project and run the entire pipeline.

## Setup

1. **Create an environemnt**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install torch torchvision torchmetrics matplotlib tqdm numpy
    ```
    *On Windows, use: venv\Scripts\activate*

2.  **Dataset:**
    *Download the dataset:*
    Use your web browser to download this file from a mirror: http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar
    *Move the file:*
    * Once the 1.9 GB download is complete, move the
    VOCtrainval_11-May-2012.tar file from your `Downloads` into the `data` folder.
    * `DO NOT EXTRACT IT`. Just move the .tar file as-is.
    
    * Run to automatically extract and setup the Dataset:
        ```bash
        python dataset.py
        ```

3.  **Hardware:** A GPU (CUDA or Apple Silicon MPS) is recommended for faster training, but the code will fall back to CPU if none is available.
---

## Training, Testing and Visualizing Models
* For training please go to `train.txt`
* For testing please go to `test.txt` Additional in testing, you can run visualization commands to print mIoU
* For counting the number of paramters please go to `count_params.txt`