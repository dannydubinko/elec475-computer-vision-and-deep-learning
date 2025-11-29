import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import argparse # No longer needed for model paths

# --- Import models ---
try:
    from model import SnoutNet, SnoutNetAlex, SnoutNetVGG
except ImportError:
    print("\nError: Could not import models from model.py.")
    exit()

# --- Import dataset ---
try:
    from data_loader import SnoutDataset
except ImportError:
    print("\nError: Could not import SnoutDataset from data_loader.py.")
    exit()

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR_NAME = 'images-original'
TEST_ANNOTATION_FILE_NAME = 'test_noses.txt'
ROOT_DATA_DIR = os.path.join(SCRIPT_DIR, IMAGE_DIR_NAME)
IMAGE_DIR = os.path.join(ROOT_DATA_DIR, 'images')
TEST_ANNOTATION_FILE = os.path.join(SCRIPT_DIR, TEST_ANNOTATION_FILE_NAME)
IMAGE_SIZE = 227
BATCH_SIZE = 64

def load_model(model_class, model_path, device):
    """Helper function to load a single model."""
    if not os.path.exists(model_path):
        # Don't print error here, handled in the main loop
        return None
    try:
        model = model_class().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        # print(f"Successfully loaded model from {model_path}") # Keep output cleaner
        return model
    except Exception as e:
        print(f"Warning: Error loading model {model_path}: {e}")
        return None

def run_ensemble_evaluation(aug_suffix, device):
    """Runs the full ensemble evaluation for a given augmentation suffix."""

    ensemble_name = f"ensemble_{aug_suffix}"
    print(f"\n--- Starting Ensemble Evaluation for: {ensemble_name} ---")

    # --- Construct Model Paths ---
    model_filenames = {
        'custom': f'snoutnet_custom_{aug_suffix}_model.pth',
        'alexnet': f'snoutnet_alexnet_{aug_suffix}_model.pth',
        'vgg16': f'snoutnet_vgg16_{aug_suffix}_model.pth'
    }
    model_paths = {
        'custom': os.path.join(SCRIPT_DIR, model_filenames['custom']),
        'alexnet': os.path.join(SCRIPT_DIR, model_filenames['alexnet']),
        'vgg16': os.path.join(SCRIPT_DIR, model_filenames['vgg16'])
    }

    # --- Check if all models exist ---
    if not all(os.path.exists(p) for p in model_paths.values()):
        print(f"Skipping '{ensemble_name}': One or more required model files not found:")
        for name, path in model_paths.items():
            if not os.path.exists(path):
                print(f"  - Missing: {model_filenames[name]}")
        return

    # --- Create Results Directory ---
    results_base_dir = os.path.join(SCRIPT_DIR, 'results')
    output_dir = os.path.join(results_base_dir, ensemble_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # --- Load Models ---
    print("Loading models...")
    model_custom = load_model(SnoutNet, model_paths['custom'], device)
    model_alex = load_model(SnoutNetAlex, model_paths['alexnet'], device)
    model_vgg = load_model(SnoutNetVGG, model_paths['vgg16'], device)
    if not all([model_custom, model_alex, model_vgg]):
        print(f"Failed to load all models for '{ensemble_name}'. Skipping.")
        return
    models = [model_custom, model_alex, model_vgg]
    print("Models loaded successfully.")

    # --- Define Transformations ---
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    vis_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()
    ])

    # --- Load Test Dataset ---
    try:
        # Load dataset only once if possible, but safer to load inside function
        test_dataset = SnoutDataset(
            root_dir=ROOT_DATA_DIR, annotation_file=TEST_ANNOTATION_FILE,
            transform=data_transform, augmentations=None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        print(f"Loaded {len(test_dataset)} test samples.")
    except Exception as e: print(f"Error loading data: {e}"); return

    # --- Evaluation Prep & Loop ---
    all_labels, all_ensemble_preds, all_indices = [], [], []
    eval_loop = tqdm(test_loader, desc=f"Evaluating {ensemble_name}", leave=False)
    with torch.no_grad():
        for images, labels, batch_indices in eval_loop:
            images = images.to(device)
            batch_preds = [model(images).unsqueeze(0) for model in models]
            stacked_preds = torch.cat(batch_preds, dim=0)
            avg_preds = torch.mean(stacked_preds, dim=0)
            all_labels.append(labels.cpu().numpy())
            all_ensemble_preds.append(avg_preds.cpu().numpy())
            all_indices.append(batch_indices.cpu().numpy())

    # --- Calculate Statistics ---
    print("\nCalculating statistics...")
    try:
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_ensemble_preds, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)
        if len(all_labels) != len(test_dataset): print(f"Warning: Sample count mismatch.");
        all_labels_px = all_labels * IMAGE_SIZE
        all_preds_px = all_preds * IMAGE_SIZE
        distances = np.linalg.norm(all_preds_px - all_labels_px, axis=1)
        mean_dist, std_dist = np.mean(distances), np.std(distances)
        min_dist, max_dist = np.min(distances), np.max(distances)

        print(f"\n--- {ensemble_name} Evaluation Results ---")
        print(f"Models Ensembled: Custom, AlexNet, VGG16 ({aug_suffix})")
        print(f"Output Directory: {output_dir}")
        print(f"Total samples evaluated: {len(distances)}")
        print(f"Mean Euclidean Distance: {mean_dist:.4f} pixels")
        print(f"Std Dev of Distance:     {std_dist:.4f} pixels")
        print(f"Min Distance (Best):   {min_dist:.4f} pixels")
        print(f"Max Distance (Worst):  {max_dist:.4f} pixels")

        sorted_indices = np.argsort(distances)
        best_sample_indices = sorted_indices[:4]
        worst_sample_indices = sorted_indices[-4:][::-1]
        best_4_distances = distances[best_sample_indices]
        worst_4_distances = distances[worst_sample_indices]
        print("\n--- Stats for 4 BEST Ensemble Predictions ---")
        print(f"Distances: {best_4_distances}")
        print(f"Mean: {np.mean(best_4_distances):.4f}, Std: {np.std(best_4_distances):.4f}, Min: {np.min(best_4_distances):.4f}, Max: {np.max(best_4_distances):.4f} pixels")
        print("\n--- Stats for 4 WORST Ensemble Predictions ---")
        print(f"Distances: {worst_4_distances}")
        print(f"Mean: {np.mean(worst_4_distances):.4f}, Std: {np.std(worst_4_distances):.4f}, Min: {np.min(worst_4_distances):.4f}, Max: {np.max(worst_4_distances):.4f} pixels")

    except Exception as e: print(f"Error during stats: {e}"); return

    # --- Generate Plots ---
    plot_prefix = f"snoutnet_{ensemble_name}" # e.g., snoutnet_ensemble_flip_erase

    # Histogram Plot
    try:
        print("\nGenerating error histogram...")
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.title(f'Histogram of Ensemble Errors ({aug_suffix})')
        plt.xlabel('Error in Pixels'); plt.ylabel('Number of Samples')
        plt.axvline(mean_dist, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_dist:.2f}px')
        plt.legend(); plt.grid(True)
        hist_path = os.path.join(output_dir, f'{plot_prefix}_error_histogram.png')
        plt.savefig(hist_path); plt.close()
        print(f"Saved histogram to {hist_path}")
    except Exception as e: print(f"Error generating histogram: {e}")

    # Best/Worst Visuals
    try:
        print("\nGenerating best/worst visuals...")
        vis_dataset = SnoutDataset(
            root_dir=ROOT_DATA_DIR, annotation_file=TEST_ANNOTATION_FILE,
            transform=vis_transform, augmentations=None
        )
        indices_to_plot = np.concatenate([all_indices[best_sample_indices], all_indices[worst_sample_indices]])
        errors_to_plot = np.concatenate([best_4_distances, worst_4_distances])
        plot_types = ['Best'] * 4 + ['Worst'] * 4

        for i in range(len(indices_to_plot)):
            dataset_idx, error, plot_type = indices_to_plot[i], errors_to_plot[i], plot_types[i]
            vis_image, vis_label, _ = vis_dataset[dataset_idx]
            vis_image_np = vis_image.permute(1, 2, 0).numpy()
            vis_label_px = vis_label.numpy() * IMAGE_SIZE

            original_pred_index = np.where(all_indices == dataset_idx)[0]
            if not original_pred_index.size > 0: continue
            pred_px = all_preds_px[original_pred_index[0]]

            plt.figure(figsize=(8, 8))
            plt.imshow(vis_image_np)
            plt.plot(vis_label_px[0], vis_label_px[1], 'go', ms=10, fillstyle='none', mew=2.5, label=f'GT: ({vis_label_px[0]:.0f}, {vis_label_px[1]:.0f})')
            plt.plot(pred_px[0], pred_px[1], 'mX', ms=12, mew=2.5, label=f'Ensemble Pred: ({pred_px[0]:.0f}, {pred_px[1]:.0f})')
            plt.title(f"Ensemble ({aug_suffix}): {plot_type} #{i%4+1} (Idx: {dataset_idx})\nError: {error:.2f} pixels", fontsize=12)
            plt.legend(); plt.axis('off')
            img_filename = f"{plot_prefix}_result_{plot_type}_{i%4+1}_idx{dataset_idx}.png"
            save_path = os.path.join(output_dir, img_filename)
            plt.savefig(save_path); plt.close()
        print(f"Saved best/worst visuals to {output_dir}")
    except Exception as e:
        print(f"Error generating visual examples: {e}")

    print(f"--- Finished Ensemble Evaluation for: {ensemble_name} ---")


def main():
    print("--- ensemble_test.py script started ---")
    print("Will attempt to run evaluation for 'noaug' and 'flip_erase' ensembles.")

    # --- Setup Device ---
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Run evaluation for both suffixes ---
    run_ensemble_evaluation('noaug', device)
    run_ensemble_evaluation('flip_erase', device)

    print("\n--- All Ensemble Testing Complete ---")

if __name__ == "__main__":
    main()