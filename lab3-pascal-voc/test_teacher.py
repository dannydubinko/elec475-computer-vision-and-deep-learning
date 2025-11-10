import torch
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import argparse
import os

import dataset as data_utils
import visualize # <-- IMPORT

def get_device():
    """Get the best available device (MPS, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device

def evaluate_teacher(args):
    """
    Step 2.1: Load and evaluate the pretrained FCN-ResNet50 on PASCAL VOC 2012.
    """
    print("Evaluating Pretrained Teacher FCN-ResNet50...")
    device = get_device()
    
    if args.visualize:
        save_dir = "results/teacher"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving visualizations to {save_dir}")

    # Load Model
    weights = segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = segmentation.fcn_resnet50(weights=weights).to(device)
    model.eval()

    # Load Data
    # These are the v1 transforms specified by the model weights
    image_transforms = weights.transforms()
    # We create a simple v1 target transform just to get the tensor
    target_transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.long, scale=False)])
    
    val_dataset = datasets.VOCSegmentation(
        root="./data", 
        year="2012", 
        image_set="val", 
        download=False, 
        transform=image_transforms, # v1 transform on image
        target_transform=target_transforms # v1 transform on target
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, # Must be 1 for variable-sized images
        shuffle=False, 
        num_workers=2
    )

    metric = JaccardIndex(
        task='multiclass', 
        num_classes=data_utils.NUM_CLASSES, 
        ignore_index=data_utils.IGNORE_INDEX
    ).to(device)

    print("Running evaluation...")
    count = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating Teacher"):
            # images = [1, C, H_img, W_img]
            # targets = [1, 1, H_targ, W_targ] (note: H/W can be different!)
            images, targets = images.to(device), targets.to(device)
            
            # Run model
            outputs = model(images)['out']
            
            # Resize model output to match TARGET size
            outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)
            
            # Get predictions and squeeze targets
            preds = torch.argmax(outputs, dim=1) # preds shape: [1, H_targ, W_targ]
            targets = targets.squeeze(1) # targets shape: [1, H_targ, W_targ]
            
            # Update metric
            metric.update(preds, targets)

            # --- VISUALIZATION ---
            if args.visualize and count < args.num_images:
                # *** THE FIX IS HERE ***
                # We must resize the original image to match the target/pred size
                # for visualization.
                img_for_vis = F.interpolate(images, size=targets.shape[1:], mode='bilinear', align_corners=False)
                
                visualize.save_overlay(
                    img_for_vis[0], # Resized Image [C, H_targ, W_targ]
                    targets[0],     # Target [H_targ, W_targ]
                    preds[0],       # Prediction [H_targ, W_targ]
                    save_dir, 
                    count
                )
                count += 1
            # ---------------------

    miou = metric.compute()
    print(f"\nTeacher mIoU: {miou.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Teacher FCN-ResNet50")
    parser.add_argument('--visualize', action='store_true', help="Save qualitative results")
    parser.add_argument('--num_images', type=int, default=10, help="Number of images to save")
    args = parser.parse_args()
    evaluate_teacher(args)