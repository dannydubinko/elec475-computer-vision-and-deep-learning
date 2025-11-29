import os
import torch
import torchvision.transforms as transforms
# from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import ast
import re
import numpy as np
import random

# --- Import models ---
try:
    from model import SnoutNet, SnoutNetAlex, SnoutNetVGG
except ImportError:
    print("\nError: Could not import models from model.py.")
    exit()

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR_NAME = 'images-original'
TEST_ANNOTATION_FILE_NAME = 'test_noses.txt'
ROOT_DATA_DIR = os.path.join(SCRIPT_DIR, IMAGE_DIR_NAME)
IMAGE_DIR = os.path.join(ROOT_DATA_DIR, 'images')
TEST_ANNOTATION_FILE = os.path.join(SCRIPT_DIR, TEST_ANNOTATION_FILE_NAME)
IMAGE_SIZE = 227


def load_all_test_samples(annotation_file, image_dir):
    """Loads all valid image paths and coordinates from the annotation file."""
    samples = []
    print(f"Loading all samples from: {annotation_file}")
    try:
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip(); parts = line.split(',', 1)
                if not line or len(parts) != 2: continue
                filename, coord_str = parts[0], parts[1].strip().strip('"')
                try:
                    coords = ast.literal_eval(coord_str)
                    image_path = os.path.join(image_dir, filename)
                    if os.path.exists(image_path):
                        samples.append({
                            'path': image_path, 'filename': filename,
                            'coords': torch.tensor(coords, dtype=torch.float32)
                        })
                except (ValueError, SyntaxError): continue
        print(f"Loaded {len(samples)} valid samples.")
        return samples
    except FileNotFoundError: print(f"Error: Annotation file not found: {annotation_file}"); return []
    except Exception as e: print(f"Error reading annotations: {e}"); return []


def load_model_viz(model_path, device):
    """Loads model, infers type and aug status, returns model and label."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None, "Not Found"
    filename = os.path.basename(model_path)
    model_type = "custom" if "custom" in filename else "alexnet" if "alexnet" in filename else "vgg16" if "vgg16" in filename else "unknown"
    aug_status = "noaug" if "noaug" in filename else "flip_erase" if "flip_erase" in filename else "unknown_aug"
    model_label = f"{model_type.capitalize()}\n({aug_status})"
    try:
        model_class = {'custom': SnoutNet, 'alexnet': SnoutNetAlex, 'vgg16': SnoutNetVGG}[model_type]
        model = model_class().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded: {model_path}")
        return model, model_label
    except Exception as e: print(f"Error loading {model_path}: {e}"); return None, model_label


def get_prediction(model, img_tensor_model, device):
    """Gets prediction (normalized) from a loaded model."""
    if model is None: return torch.tensor([[0.5, 0.5]], device=device) # Default center
    with torch.no_grad(): pred_coords_norm = model(img_tensor_model)
    return pred_coords_norm

def main():
    parser = argparse.ArgumentParser(description="Visualize snout prediction(s) for a RANDOM image.")
    # --- MODIFIED: Added 'compare_two' mode ---
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'compare', 'compare_two'],
                        help="'single': one model. 'compare': 2x4 grid. 'compare_two': 1x2 grid.")
    # --- END MODIFIED ---

    # Args for 'single' mode
    parser.add_argument('--model_path', type=str, help="Path for 'single' mode.")

    # --- ADDED: Args for 'compare_two' mode ---
    parser.add_argument('--model_path1', type=str, help="Path for first model in 'compare_two' mode.")
    parser.add_argument('--model_path2', type=str, help="Path for second model in 'compare_two' mode.")
    # --- END ADDED ---

    # Args for 'compare' mode (6 paths)
    parser.add_argument('--custom_noaug_path', type=str, help="Compare: Custom (noaug).")
    parser.add_argument('--alexnet_noaug_path', type=str, help="Compare: AlexNet (noaug).")
    parser.add_argument('--vgg16_noaug_path', type=str, help="Compare: VGG16 (noaug).")
    parser.add_argument('--custom_aug_path', type=str, help="Compare: Custom (aug).")
    parser.add_argument('--alexnet_aug_path', type=str, help="Compare: AlexNet (aug).")
    parser.add_argument('--vgg16_aug_path', type=str, help="Compare: VGG16 (aug).")
    args = parser.parse_args()

    # --- Validate arguments based on mode ---
    if args.mode == 'single' and not args.model_path:
        parser.error("--model_path required for mode 'single'")
    # --- ADDED: Validation for 'compare_two' ---
    if args.mode == 'compare_two' and not (args.model_path1 and args.model_path2):
        parser.error("--model_path1 and --model_path2 required for mode 'compare_two'")
    # --- END ADDED ---
    if args.mode == 'compare' and not all([
        args.custom_noaug_path, args.alexnet_noaug_path, args.vgg16_noaug_path,
        args.custom_aug_path, args.alexnet_aug_path, args.vgg16_aug_path
    ]):
        parser.error("All 6 compare paths required for mode 'compare'")

    # --- Select Random Image ---
    all_samples = load_all_test_samples(TEST_ANNOTATION_FILE, IMAGE_DIR)
    if not all_samples: print("Exiting."); return
    selected_sample = random.choice(all_samples)
    img_path, gt_coords_px_orig, selected_filename = selected_sample['path'], selected_sample['coords'], selected_sample['filename']
    print(f"\n--- Visualizing RANDOM image: {selected_filename} ---")

    # --- Common Setup ---
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # Prepare image transforms
    model_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    vis_transform = transforms.Compose([ transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor() ])
    try:
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor_model = model_transform(img_pil).unsqueeze(0).to(device)
        img_tensor_vis = vis_transform(img_pil)
    except Exception as e: print(f"Error loading/transforming image: {e}"); return

    # Calculate GT coords for resized image
    original_w, original_h = img_pil.size
    gt_coords_resized_px = np.array([
        (gt_coords_px_orig[0] / original_w) * IMAGE_SIZE, (gt_coords_px_orig[1] / original_h) * IMAGE_SIZE
    ])
    img_display_np = img_tensor_vis.permute(1, 2, 0).numpy() # Prepare display image once

    # --- Mode-Specific Logic ---
    if args.mode == 'single':
        print(f"--- Visualizing Single Model: {args.model_path} ---")
        model, model_label = load_model_viz(args.model_path, device)
        if model is None: return
        pred_coords_norm = get_prediction(model, img_tensor_model, device)
        pred_coords_px = pred_coords_norm.squeeze().cpu().numpy() * IMAGE_SIZE

        # Visualize Single
        plt.figure(figsize=(8, 8))
        plt.imshow(img_display_np)
        plt.plot(gt_coords_resized_px[0], gt_coords_resized_px[1], 'go', ms=12, fillstyle='none', mew=2.5, label='Ground Truth')
        plt.plot(pred_coords_px[0], pred_coords_px[1], 'rX', ms=12, mew=2.5, label='Prediction')
        plt.title(f"{selected_filename} | Model: {model_label}")
        plt.legend(); plt.axis('off'); plt.tight_layout(); plt.show()

    # --- ADDED: 'compare_two' mode ---
    elif args.mode == 'compare_two':
        print(f"--- Visualizing Two Models Side-by-Side for {selected_filename} ---")
        model1, label1 = load_model_viz(args.model_path1, device)
        model2, label2 = load_model_viz(args.model_path2, device)

        # Get predictions (even if model loading failed, get_prediction returns default)
        pred_norm1 = get_prediction(model1, img_tensor_model, device)
        pred_norm2 = get_prediction(model2, img_tensor_model, device)
        pred_px1 = pred_norm1.squeeze().cpu().numpy() * IMAGE_SIZE
        pred_px2 = pred_norm2.squeeze().cpu().numpy() * IMAGE_SIZE

        # Create 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8)) # Adjust figsize as needed
        fig.suptitle(f'Model Comparison for {selected_filename}', fontsize=16)

        # Plot for Model 1
        axes[0].imshow(img_display_np)
        axes[0].plot(gt_coords_resized_px[0], gt_coords_resized_px[1], 'go', ms=10, fillstyle='none', mew=2, label='Ground Truth')
        axes[0].plot(pred_px1[0], pred_px1[1], 'rX', ms=10, mew=2, label='Prediction')
        axes[0].set_title(label1)
        axes[0].axis('off')
        axes[0].legend()

        # Plot for Model 2
        axes[1].imshow(img_display_np)
        axes[1].plot(gt_coords_resized_px[0], gt_coords_resized_px[1], 'go', ms=10, fillstyle='none', mew=2, label='Ground Truth')
        axes[1].plot(pred_px2[0], pred_px2[1], 'rX', ms=10, mew=2, label='Prediction')
        axes[1].set_title(label2)
        axes[1].axis('off')
        axes[1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        plt.show()
    # --- END ADDED ---

    elif args.mode == 'compare': # Existing 2x4 compare mode
        print(f"--- Visualizing Comparison Grid for {selected_filename} ---")
        model_paths = {
            'noaug': [args.custom_noaug_path, args.alexnet_noaug_path, args.vgg16_noaug_path],
            'aug': [args.custom_aug_path, args.alexnet_aug_path, args.vgg16_aug_path]
        }
        loaded_models = {'noaug': [], 'aug': []}; model_labels = {'noaug': [], 'aug': []}
        print("Loading NOAUG models..."); # Load models
        for path in model_paths['noaug']: m, lbl = load_model_viz(path, device); loaded_models['noaug'].append(m); model_labels['noaug'].append(lbl)
        print("Loading AUG models...");
        for path in model_paths['aug']: m, lbl = load_model_viz(path, device); loaded_models['aug'].append(m); model_labels['aug'].append(lbl)

        # Get predictions
        preds_norm = {'noaug': [], 'aug': []}; preds_px = {'noaug': [], 'aug': []}
        for status in ['noaug', 'aug']:
            for model in loaded_models[status]:
                pred_norm = get_prediction(model, img_tensor_model, device)
                preds_norm[status].append(pred_norm)
                preds_px[status].append(pred_norm.squeeze().cpu().numpy() * IMAGE_SIZE)
            # Calculate ensemble
            if len(loaded_models[status]) == 3 and all(m is not None for m in loaded_models[status]) and len(preds_norm[status]) == 3:
                stacked_preds = torch.cat([p.unsqueeze(0) for p in preds_norm[status]], dim=0)
                ensemble_pred_norm = torch.mean(stacked_preds, dim=0)
                preds_px[status].append(ensemble_pred_norm.squeeze().cpu().numpy() * IMAGE_SIZE)
                model_labels[status].append(f"Ensemble\n({status})")
            else:
                 preds_px[status].append(np.array([IMAGE_SIZE/2, IMAGE_SIZE/2]))
                 model_labels[status].append(f"Ensemble\n({status} - Error)")

        # Create 2x4 subplot
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        fig.suptitle(f'Model Comparison Grid for {selected_filename}', fontsize=16)
        for row, status in enumerate(['noaug', 'aug']):
            for col in range(4): # 3 models + 1 ensemble
                ax = axes[row, col]
                ax.imshow(img_display_np)
                ax.plot(gt_coords_resized_px[0], gt_coords_resized_px[1], 'go', ms=10, fillstyle='none', mew=2, label='GT')
                if col < len(preds_px[status]) and col < len(model_labels[status]):
                     ax.plot(preds_px[status][col][0], preds_px[status][col][1], 'rX', ms=10, mew=2, label='Pred')
                     ax.set_title(model_labels[status][col])
                else: ax.set_title(f"Plot Error {row},{col}")
                ax.axis('off')
                if row == 0 and col == 0: ax.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    print("--- Visualization Complete ---")

if __name__ == "__main__":
    main()