import torch
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import argparse
import os
import time

import dataset as data_utils # Import our dataset file
import visualize # Import visualization utility

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

    # --- 1. Setup ---
    device = get_device()
    
    # Use batch_size=1 for visualization, 16 for speed test
    eval_batch_size = 16 if args.measure_speed else 1
    
    if args.visualize:
        save_dir = "results/teacher"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving visualizations to {save_dir}")

    # --- 2. Load Model ---
    weights = segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = segmentation.fcn_resnet50(weights=weights).to(device)
    model.eval() # Set model to evaluation mode

    # --- 3. Load Data ---
    # MODIFIED: Use the *same* dataloader as the student for a
    # fair apples-to-apples comparison on 480x480 crops.
    _, val_loader = data_utils.get_dataloaders(batch_size=eval_batch_size)
    print(f"Using Batch Size: {eval_batch_size}")
    print("Dataset loaded.")

    # --- 4. mIoU CALCULATION ---
    print("\n--- Calculating mIoU ---")
    metric = JaccardIndex(
        task='multiclass',
        num_classes=data_utils.NUM_CLASSES,
        ignore_index=data_utils.IGNORE_INDEX
    )
    
    count = 0
    with torch.no_grad(): # Disable gradient calculation
        for images, targets in tqdm(val_loader, desc="Evaluating mIoU"):
            images = images.to(device)
            targets = targets.to(device) # Shape: [B, H, W]

            outputs = model(images)['out'] # Shape: [B, 21, H, W]
            
            # We must still interpolate, as the model output size (480x480)
            # may not perfectly match the target (480x480) due to padding.
            
            # --- FIX: Use shape[1:] to get (H, W) from [B, H, W] target ---
            if outputs.shape[2:] != targets.shape[1:]:
                outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)

            preds = torch.argmax(outputs, dim=1) # Shape: [B, H, W]

            # Move to CPU for metric update
            metric.update(preds.cpu(), targets.cpu())

            # --- VISUALIZATION ---
            if args.visualize and count < args.num_images and eval_batch_size == 1:
                # We need to resize the input image to match the target
                # for visualization, as it was cropped by the dataloader.
                # (This is just for the visual overlay, not for the model)
                img_resized = F.interpolate(images, size=targets.shape[1:], mode='bilinear', align_corners=False)
                visualize.save_overlay(img_resized[0], targets[0], preds[0], save_dir, count)
                count += 1
            elif args.visualize and eval_batch_size > 1 and count == 0:
                print("[Warning] Visualization is disabled when batch_size > 1 for evaluation.")
                count = -1 # Don't print again
            # ---------------------

    # 7. Compute and Print Final mIoU
    miou = metric.compute()
    print(f"\nTeacher FCN-ResNet50 mIoU: {miou.item():.4f}")
    
    
    # --- 5. SPEED TEST ---
    if args.measure_speed:
        print("\n--- Measuring Inference Speed ---")
        device_type = device.type
        
        # A. Warmup
        print("Warming up GPU...")
        with torch.no_grad():
            for _ in range(10):
                dummy_input = torch.randn(eval_batch_size, 3, data_utils.CROP_SIZE, data_utils.CROP_SIZE).to(device)
                _ = model(dummy_input)
                if device_type == 'cuda': torch.cuda.synchronize()
                if device_type == 'mps': torch.mps.synchronize()

        # B. Measurement
        print(f"Running inference speed test over {len(val_loader)} batches...")
        total_time = 0.0
        total_images = 0
        
        if device_type == 'cuda': torch.cuda.synchronize()
        if device_type == 'mps': torch.mps.synchronize()
        
        loop_start_time = time.time()
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                _ = model(images)['out'] # Get teacher output
                total_images += images.shape[0]

        if device_type == 'cuda': torch.cuda.synchronize()
        if device_type == 'mps': torch.mps.synchronize()
        
        loop_end_time = time.time()
        
        # C. Calculate & Print Results
        total_time = loop_end_time - loop_start_time
        avg_time_ms = (total_time / total_images) * 1000
        fps = total_images / total_time
        
        print("\n--- Inference Speed Results ---")
        print(f"Batch Size: {eval_batch_size}")
        print(f"Total Images Processed: {total_images}")
        print(f"Total Time (loop): {total_time:.3f} s")
        print(f"Average Time per Image: {avg_time_ms:.3f} ms")
        print(f"Inference Speed (FPS): {fps:.2f} FPS")
    # -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pretrained Teacher FCN-ResNet50")
    parser.add_argument('--visualize', action='store_true', help="Save qualitative results (requires batch_size=1)")
    parser.add_argument('--num_images', type=int, default=10, help="Number of images to save")
    parser.add_argument('--measure_speed', action='store_true', help="Run inference speed benchmark")
    args = parser.parse_args()
    
    if args.visualize and args.measure_speed:
        print("[Warning] Cannot run visualization and speed test at the same time. Speed test requires batch_size=16, visualization requires batch_size=1. Disabling visualization.")
        args.visualize = False

    evaluate_teacher(args)