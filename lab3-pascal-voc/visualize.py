import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F

# VOC Color Palette (standard for PASCAL VOC)
VOC_PALETTE = torch.tensor([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
    [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
    [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
], dtype=torch.uint8)

# ImageNet Mean and Std (from dataset.py) for denormalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize(tensor):
    """Reverses the ImageNet normalization for display."""
    return (tensor.cpu() * STD + MEAN).clip(0, 1)

def save_overlay(image_tensor, target_tensor, pred_tensor, save_path, index):
    """
    Saves a side-by-side comparison: Input Image | Ground Truth Overlay | Prediction Overlay
    
    IMPORTANT: Assumes image_tensor, target_tensor, and pred_tensor are all
    the same height and width.
    """
    # 1. Prepare Image (C, H, W) uint8 [0-255]
    img_denorm = denormalize(image_tensor)
    img_uint8 = (img_denorm * 255).to(torch.uint8)

    # 2. Prepare Masks (ensure they are on CPU and boolean/long for drawing)
    # Target might have 255 (ignore_index), replace with 0 (background) for visualization
    target_tensor = target_tensor.cpu()
    target_tensor[target_tensor == 255] = 0
    
    pred_tensor = pred_tensor.cpu()

    # --- Full Multi-Class Color Overlay ---
    # Convert masks to RGB images using palette
    def mask_to_rgb(mask):
        h, w = mask.shape
        rgb = torch.zeros((3, h, w), dtype=torch.uint8)
        for c in range(21): # 21 classes
            idx = mask == c
            if idx.any():
                rgb[0, idx] = VOC_PALETTE[c, 0]
                rgb[1, idx] = VOC_PALETTE[c, 1]
                rgb[2, idx] = VOC_PALETTE[c, 2]
        return rgb

    target_rgb = mask_to_rgb(target_tensor)
    pred_rgb = mask_to_rgb(pred_tensor)

    # Blend manually for full control
    alpha = 0.5
    target_blended = (img_uint8 * (1 - alpha) + target_rgb * alpha).to(torch.uint8)
    pred_blended = (img_uint8 * (1 - alpha) + pred_rgb * alpha).to(torch.uint8)

    # 4. Plot and Save
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_uint8.permute(1, 2, 0))
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(target_blended.permute(1, 2, 0))
    axs[1].set_title("Ground Truth Overlay")
    axs[1].axis("off")

    axs[2].imshow(pred_blended.permute(1, 2, 0))
    axs[2].set_title("Prediction Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_path}/result_{index}.png")
    plt.close(fig)