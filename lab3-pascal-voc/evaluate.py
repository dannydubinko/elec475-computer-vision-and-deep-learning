import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import argparse

from student_model import CompactStudentModel
import dataset as data_utils
from test_teacher import get_device

def evaluate_model(model_path):
    print(f"Evaluating model: {model_path}")
    
    # --- 1. Setup ---
    device = get_device()
    
    # --- 2. Data ---
    print("Loading validation dataset...")
    # Use a batch size that fits your MPS memory
    _, val_loader = data_utils.get_dataloaders(batch_size=16) 

    # --- 3. Model ---
    print("Loading student model...")
    model = CompactStudentModel(num_classes=data_utils.NUM_CLASSES)
    
    # Load the saved weights and map to the correct device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 4. Metric ---
    # FIX 1: Keep metric on CPU to match training script and avoid OOM
    metric = JaccardIndex(
        task='multiclass', 
        num_classes=data_utils.NUM_CLASSES, 
        ignore_index=data_utils.IGNORE_INDEX
    )

    # --- 5. Evaluation Loop ---
    print("Running final evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating Student"):
            images = images.to(device)
            targets = targets.to(device)
            
            # FIX 2: Unpack outputs just like in train_kd.py
            outputs = model(images) 
            
            preds = torch.argmax(outputs, dim=1)
            
            # FIX 3: Move to CPU for update to avoid MPS OOM/precision issues
            metric.update(preds.cpu(), targets.cpu())

    # --- 6. Results ---
    miou = metric.compute()
    print("\n--- Final Evaluation Complete ---")
    print(f"Model: {model_path}")
    print(f"Final Student mIoU (VOC 2012 val): {miou.item():.4f}")
    print("---------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained student model")
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="models/student_model_kd.pth", 
        help="Path to the trained student model (.pth) file"
    )
    args = parser.parse_args()
    
    evaluate_model(args.model_path)