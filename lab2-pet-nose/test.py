import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

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
TEST_ANNOTATION_FILE = os.path.join(SCRIPT_DIR, TEST_ANNOTATION_FILE_NAME)
IMAGE_SIZE = 227
BATCH_SIZE = 64
RESULTS_BASE_DIR = os.path.join(SCRIPT_DIR, 'results') # Base dir for all results

# --- Helper Function for a Single Test Run ---
def run_test(model_name, aug_suffix, device, data_transform, vis_transform):
    """Performs a single test run for a given model and augmentation status."""
    print(f"\n===== Starting Test for: {model_name} ({aug_suffix}) =====")

    output_dir = os.path.join(RESULTS_BASE_DIR, f'{model_name}_{aug_suffix}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # --- Load Model ---
    model_path = os.path.join(SCRIPT_DIR, f'snoutnet_{model_name}_{aug_suffix}_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Skipping this run.")
        return False # Indicate failure

    try:
        print(f"Initializing model: {model_name}")
        if model_name == 'custom': model = SnoutNet().to(device)
        elif model_name == 'alexnet': model = SnoutNetAlex().to(device)
        elif model_name == 'vgg16': model = SnoutNetVGG().to(device)
        else: raise ValueError(f"Unknown model name '{model_name}'")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return False

    # --- Load Test Dataset ---
    try:
        test_dataset = SnoutDataset(
            root_dir=ROOT_DATA_DIR, annotation_file=TEST_ANNOTATION_FILE,
            transform=data_transform, augmentations=None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        print(f"Loaded {len(test_dataset)} test samples.")
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return False

    # --- Evaluation ---
    all_labels, all_preds, all_indices = [], [], []
    eval_loop = tqdm(test_loader, desc=f"Evaluating {aug_suffix}", leave=False)
    with torch.no_grad():
        for images, labels, batch_indices in eval_loop:
            images = images.to(device)
            outputs = model(images)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            all_indices.append(batch_indices.cpu().numpy())

    # --- Statistics ---
    try:
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)
        if len(all_labels) != len(test_dataset):
             print(f"Warning: Sample count mismatch.")

        all_labels_px = all_labels * IMAGE_SIZE
        all_preds_px = all_preds * IMAGE_SIZE
        distances = np.linalg.norm(all_preds_px - all_labels_px, axis=1)
        mean_dist, std_dist = np.mean(distances), np.std(distances)
        min_dist, max_dist = np.min(distances), np.max(distances)

        print(f"\n--- Results for {model_name} ({aug_suffix}) ---")
        print(f"Mean Euclidean Distance: {mean_dist:.4f} pixels")
        print(f"Std Dev of Distance:     {std_dist:.4f} pixels")
        print(f"Min Distance (Best):   {min_dist:.4f} pixels")
        print(f"Max Distance (Worst):  {max_dist:.4f} pixels")

        sorted_indices = np.argsort(distances)
        best_indices = sorted_indices[:4]
        worst_indices = sorted_indices[-4:][::-1]
        best_distances = distances[best_indices]
        worst_distances = distances[worst_indices]
        print("\n  Stats for 4 BEST:")
        print(f"  Distances: {best_distances}")
        print(f"  Mean: {np.mean(best_distances):.4f}, Std: {np.std(best_distances):.4f}, Min: {np.min(best_distances):.4f}, Max: {np.max(best_distances):.4f} px")
        print("\n  Stats for 4 WORST:")
        print(f"  Distances: {worst_distances}")
        print(f"  Mean: {np.mean(worst_distances):.4f}, Std: {np.std(worst_distances):.4f}, Min: {np.min(worst_distances):.4f}, Max: {np.max(worst_distances):.4f} px")

    except Exception as e:
        print(f"Error during statistics: {e}")
        return False # Indicate failure if stats fail

    # --- Plotting ---
    plot_prefix = f"snoutnet_{model_name}_{aug_suffix}"

    # Histogram
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of Errors ({model_name.capitalize()}, {aug_suffix})')
        plt.xlabel('Error in Pixels'); plt.ylabel('Number of Samples')
        plt.axvline(mean_dist, color='red', ls='--', lw=2, label=f'Mean Error: {mean_dist:.2f}px')
        plt.legend(); plt.grid(True)
        hist_path = os.path.join(output_dir, f'{plot_prefix}_error_histogram.png')
        plt.savefig(hist_path); plt.close()
        print(f"\nSaved error histogram: {hist_path}")
    except Exception as e: print(f"Error saving histogram: {e}")

    # Best/Worst Visuals
    try:
        vis_dataset = SnoutDataset(
            root_dir=ROOT_DATA_DIR, annotation_file=TEST_ANNOTATION_FILE,
            transform=vis_transform, augmentations=None
        )
        indices_to_plot = np.concatenate([all_indices[best_indices], all_indices[worst_indices]])
        errors_to_plot = np.concatenate([best_distances, worst_distances])
        plot_types = ['Best'] * 4 + ['Worst'] * 4

        print("Saving best/worst prediction images...")
        for i in range(len(indices_to_plot)):
            dataset_idx, error, plot_type = indices_to_plot[i], errors_to_plot[i], plot_types[i]
            vis_image, vis_label, _ = vis_dataset[dataset_idx]
            vis_image_np = vis_image.permute(1, 2, 0).numpy()
            vis_label_px = vis_label.numpy() * IMAGE_SIZE

            # Find the corresponding prediction using the original index
            original_pred_index = np.where(all_indices == dataset_idx)[0]
            if not original_pred_index.size > 0: continue
            pred_px = all_preds_px[original_pred_index[0]]

            plt.figure(figsize=(8, 8))
            plt.imshow(vis_image_np)
            plt.plot(vis_label_px[0], vis_label_px[1], 'go', ms=10, fillstyle='none', mew=2, label=f'GT: ({vis_label_px[0]:.0f}, {vis_label_px[1]:.0f})')
            plt.plot(pred_px[0], pred_px[1], 'rX', ms=10, mew=2, label=f'Pred: ({pred_px[0]:.0f}, {pred_px[1]:.0f})')
            plt.title(f"{model_name.capitalize()} ({aug_suffix}): {plot_type} #{i%4+1} (Idx: {dataset_idx})\nError: {error:.2f} pixels", fontsize=12)
            plt.legend(); plt.axis('off')
            img_filename = f"{plot_prefix}_result_{plot_type}_{i%4+1}_idx{dataset_idx}.png"
            save_path = os.path.join(output_dir, img_filename)
            plt.savefig(save_path); plt.close()
        print("Saved visual examples.")
    except Exception as e: print(f"Error generating visual examples: {e}")

    print(f"===== Finished Test for: {model_name} ({aug_suffix}) =====")
    return True # Indicate success

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Test SnoutNet variants (noaug and flip_erase).")
    parser.add_argument('--model', type=str, required=True, choices=['custom', 'alexnet', 'vgg16'],
                        help='Which model architecture to test.')
    args = parser.parse_args()
    model_name = args.model

    print(f"--- test.py script started for model: {model_name} ---")

    # --- Setup Device ---
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Define Transformations ---
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    vis_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()
    ])

    # --- Run for both augmentation statuses ---
    augmentation_statuses = ['noaug', 'flip_erase']
    success_count = 0
    for aug_suffix in augmentation_statuses:
        if run_test(model_name, aug_suffix, device, data_transform, vis_transform):
            success_count += 1

    print(f"\n--- Testing Complete for {model_name} ---")
    print(f"Successfully completed {success_count} out of {len(augmentation_statuses)} test runs.")

if __name__ == "__main__":
    main()