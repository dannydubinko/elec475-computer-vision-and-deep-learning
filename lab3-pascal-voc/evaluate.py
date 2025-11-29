import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import argparse
import os
import time  # <-- 1. IMPORT TIME

from student_model import CompactStudentModel
import dataset as data_utils
from test_teacher import get_device
import visualize

def evaluate_model(args):
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    print(f"Evaluating model: {model_name}")
    
    device = get_device()
    # Use batch_size=1 for easier visualization
    # Use batch_size=16 for a more realistic speed test
    eval_batch_size = 16 if args.measure_speed else 1
    
    _, val_loader = data_utils.get_dataloaders(batch_size=eval_batch_size) 
    print(f"Using Batch Size: {eval_batch_size}")

    if args.visualize:
        save_dir = f"results/{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving visualizations to {save_dir}")

    model = CompactStudentModel(num_classes=data_utils.NUM_CLASSES)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 2. mIoU CALCULATION ---
    print("\n--- Calculating mIoU ---")
    metric = JaccardIndex(task='multiclass', num_classes=data_utils.NUM_CLASSES, ignore_index=data_utils.IGNORE_INDEX)
    count = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating mIoU"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Move preds to CPU for metric update to avoid OOM
            metric.update(preds.cpu(), targets.cpu())

            # Visualization
            if args.visualize and count < args.num_images and eval_batch_size == 1:
                visualize.save_overlay(images[0], targets[0], preds[0], save_dir, count)
                count += 1
            elif args.visualize and eval_batch_size > 1 and count == 0:
                print("[Warning] Visualization is disabled when batch_size > 1 for evaluation.")
                count = -1 # Don't print again

    print(f"\nFinal mIoU for {model_name}: {metric.compute().item():.4f}")
    # ----------------------------


    # --- 3. SPEED TEST ---
    if args.measure_speed:
        print("\n--- Measuring Inference Speed ---")
        device_type = device.type
        
        # A. Warmup: Run a few batches to get GPU/MPS clocks up
        print("Warming up GPU...")
        with torch.no_grad():
            for _ in range(10):
                # Use a random tensor of the correct size
                dummy_input = torch.randn(eval_batch_size, 3, data_utils.CROP_SIZE, data_utils.CROP_SIZE).to(device)
                _ = model(dummy_input)
                # Synchronize after each warmup iter
                if device_type == 'cuda': torch.cuda.synchronize()
                if device_type == 'mps': torch.mps.synchronize()

        # B. Measurement: Time the full validation loop
        print(f"Running inference speed test over {len(val_loader)} batches...")
        total_time = 0.0
        total_images = 0
        
        if device_type == 'cuda': torch.cuda.synchronize()
        if device_type == 'mps': torch.mps.synchronize()
        
        loop_start_time = time.time()
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                _ = model(images)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to .pth model file")
    parser.add_argument('--visualize', action='store_true', help="Save qualitative results (requires batch_size=1)")
    parser.add_argument('--num_images', type=int, default=10, help="Number of images to save")
    parser.add_argument('--measure_speed', action='store_true', help="Run inference speed benchmark")
    args = parser.parse_args()
    
    if args.visualize and args.measure_speed:
        print("[Warning] Cannot run visualization and speed test at the same time. Speed test requires batch_size=16, visualization requires batch_size=1. Disabling visualization.")
        args.visualize = False

    evaluate_model(args)