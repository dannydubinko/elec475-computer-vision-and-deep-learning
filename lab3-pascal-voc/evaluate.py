import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import argparse
import os

from student_model import CompactStudentModel
import dataset as data_utils
from test_teacher import get_device
import visualize # <-- NEW IMPORT

def evaluate_model(args):
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    print(f"Evaluating model: {model_name}")
    
    device = get_device()
    # Use batch_size=1 for easier visualization handling in loop
    _, val_loader = data_utils.get_dataloaders(batch_size=1) 

    if args.visualize:
        save_dir = f"results/{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving visualizations to {save_dir}")

    model = CompactStudentModel(num_classes=data_utils.NUM_CLASSES)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    metric = JaccardIndex(task='multiclass', num_classes=data_utils.NUM_CLASSES, ignore_index=data_utils.IGNORE_INDEX)

    count = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            metric.update(preds.cpu(), targets.cpu())

            # --- VISUALIZATION ---
            if args.visualize and count < args.num_images:
                visualize.save_overlay(images[0], targets[0], preds[0], save_dir, count)
                count += 1
            # ---------------------

    print(f"Final mIoU for {model_name}: {metric.compute().item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to .pth model file")
    parser.add_argument('--visualize', action='store_true', help="Save qualitative results")
    parser.add_argument('--num_images', type=int, default=10, help="Number of images to save")
    args = parser.parse_args()
    evaluate_model(args)