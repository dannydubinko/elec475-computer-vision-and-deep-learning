# SnoutNet: Pet Snout Localization Project

## Description

This project trains and evaluates various convolutional neural network models to locate the snout (nose tip) in images of pets (cats and dogs). It includes a custom CNN architecture (`SnoutNet`), variants using pretrained AlexNet (`SnoutNetAlex`) and VGG16 (`SnoutNetVGG`) backbones, and an ensemble model that averages the predictions of the three base models. The project supports training with or without data augmentation (Random Horizontal Flip and Random Erasing).

---

## File Structure

* `model.py`: Contains the definitions for `SnoutNet`, `SnoutNetAlex`, and `SnoutNetVGG` model architectures.
* `data_loader.py`: Defines the `SnoutDataset` class for loading images and annotations, handling normalization, and applying geometric augmentations (flip).
* `train.py`: Script for training the models. Takes arguments for model type and augmentation. Saves trained model weights (`.pth`) and loss curves.
* `test.py`: Script for evaluating a single trained model. Takes arguments for model type and augmentation status. Saves evaluation metrics, error histograms, and best/worst prediction examples into organized folders.
* `ensemble_test.py`: Script for evaluating ensemble models (averaging custom, AlexNet, VGG16). Automatically tests ensembles using models trained with and without augmentation if the corresponding `.pth` files exist. Saves results into organized folders.
* `visualize.py`: Script for visualizing model predictions on random test images. Supports comparing models side-by-side.
* `train_noses.txt`: Training annotations (image filename, snout coordinates).
* `test_noses.txt`: Testing annotations (image filename, snout coordinates).
* `images-original/`: Directory containing the pet images, organized within a subfolder (e.g., `images/`).
* `results/`: Directory created automatically to store testing outputs (plots, images).
* `SnoutNet.txt`, `SnoutNet-A.txt`, `SnoutNet-V.txt`, `SnoutNet_Ensemble.txt`: Text files containing specific instructions or results related to each model type (as provided).

---

## Setup

1.  **Clone Repository/Download Zip File:** Get the project files.
2.  **Dataset:**
    * Place the `images-original` folder (containing the `images` subfolder with `.jpg` files) in the main project folder.
    * Ensure `train_noses.txt` and `test_noses.txt` are also in the main project folder.
3.  **Dependencies:** Install the required Python libraries. It's recommended to use a virtual environment.
    ```bash
    pip install torch torchvision numpy matplotlib Pillow tqdm
    ```
4.  **Model Weights:** Please go to `Instructions` folder and `Weights_link.txt` and go to the google drive link. Please download the `.pth` files and place into the main project folder. Another option is to train each model.
5.  **Hardware:** A GPU (CUDA or Apple Silicon MPS) is recommended for faster training, but the code will fall back to CPU if none is available.
---

## Training, Testing and Visualizing Models
Please go to the `Instructions folder` to see the commands for `SnoutNet`, `SnoutNet-A`, `SnoutNet-V`, and `SnoutNet_Ensemble`.

Use the `train.py` script to train the different model configurations. Models are saved as `.pth` files in the main directory.

* `--model`: Specify `custom`, `alexnet`, or `vgg16`.
* `--no_augmentation`: Add this flag to disable 'flip' and 'erase' augmentations. If omitted, both are enabled by default.

**Example Commands:**

```bash
# Train Custom SnoutNet (No Augmentation)
python train.py --model custom --no_augmentation

# Train Custom SnoutNet (With Flip + Erase Augmentation)
python train.py --model custom

# Train AlexNet-based model (No Augmentation)
python train.py --model alexnet --no_augmentation

# Train AlexNet-based model (With Flip + Erase Augmentation)
python train.py --model alexnet

# Train VGG16-based model (No Augmentation)
python train.py --model vgg16 --no_augmentation

# Train VGG16-based model (With Flip + Erase Augmentation)
python train.py --model vgg16