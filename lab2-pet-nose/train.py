import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
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

def main():
    """ Main training loop """
    parser = argparse.ArgumentParser(description="Train SnoutNet variants.")
    parser.add_argument('--model', type=str, required=True, choices=['custom', 'alexnet', 'vgg16'],
                        help='Which model architecture to use.')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation (flip and erase). If not specified, both are enabled.')
    args = parser.parse_args()

    # --- Configuration ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR_NAME = 'images-original'
    TRAIN_ANNOTATION_FILE_NAME = 'train_noses.txt'
    TEST_ANNOTATION_FILE_NAME = 'test_noses.txt'
    ROOT_DATA_DIR = os.path.join(SCRIPT_DIR, IMAGE_DIR_NAME)
    TRAIN_ANNOTATION_FILE = os.path.join(SCRIPT_DIR, TRAIN_ANNOTATION_FILE_NAME)
    TEST_ANNOTATION_FILE = os.path.join(SCRIPT_DIR, TEST_ANNOTATION_FILE_NAME)

    # --- Hyperparameters ---
    IMAGE_SIZE = 227
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2

    # --- Augmentation Control ---
    if args.no_augmentation:
        geometric_augs_for_dataloader = []
        pixel_augs_for_transform = []
        aug_suffix = 'noaug'
        print("Data augmentation DISABLED.")
    else:
        geometric_augs_for_dataloader = ['flip']
        pixel_augs_for_transform = ['erase']
        aug_suffix = 'flip_erase'
        print("Data augmentation ENABLED: ['flip', 'erase']")
    print(f"Geometric augmentations (in DataLoader): {geometric_augs_for_dataloader}")
    print(f"Pixel augmentations (in transforms): {pixel_augs_for_transform}")

    # --- Setup Device ---
    if torch.backends.mps.is_available(): device = torch.device("mps"); pin_memory_device_str = "mps"
    elif torch.cuda.is_available(): device = torch.device("cuda"); pin_memory_device_str = "cuda"
    else: device = torch.device("cpu"); pin_memory_device_str = ""
    print(f"Using device: {device}")
    pin_memory = (device.type in ['cuda', 'mps'])
    if device.type == 'mps':
        try: torch.empty(0, device=device).pin_memory()
        except RuntimeError: print("MPS device does not support pin_memory, disabling."); pin_memory = False

    # --- Define Transformations ---
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), normalize])
    train_transform_list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    if 'erase' in pixel_augs_for_transform:
        train_transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0))
        print("Added RandomErasing to training transforms.")
    train_transform_list.append(normalize)
    train_transform = transforms.Compose(train_transform_list)

    # --- Create Datasets ---
    print(f"Loading datasets...")
    try:
        train_dataset = SnoutDataset(
            root_dir=ROOT_DATA_DIR, annotation_file=TRAIN_ANNOTATION_FILE,
            transform=train_transform, augmentations=geometric_augs_for_dataloader
        )
        val_dataset = SnoutDataset(
            root_dir=ROOT_DATA_DIR, annotation_file=TEST_ANNOTATION_FILE,
            transform=val_transform, augmentations=None
        )
        print(f"Loaded {len(train_dataset)} training samples.")
        print(f"Loaded {len(val_dataset)} validation samples.")
        print(f"Active augmentations suffix: {aug_suffix}")
    except FileNotFoundError: print("\nERROR: Data files not found..."); return
    except Exception as e: print(f"An error occurred while loading data: {e}"); return

    # --- Create DataLoaders ---
    loader_kwargs = { 'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'pin_memory': pin_memory }
    if pin_memory and pin_memory_device_str: loader_kwargs['pin_memory_device'] = pin_memory_device_str
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, **loader_kwargs)

    # --- Initialize Model ---
    model_name = args.model
    print(f"Initializing model: {model_name}")
    try:
        if model_name == 'custom': model = SnoutNet().to(device)
        elif model_name == 'alexnet': model = SnoutNetAlex().to(device)
        elif model_name == 'vgg16': model = SnoutNetVGG().to(device)
        else: raise ValueError("Unknown model")
    except Exception as e: print(f"Error initializing model: {e}"); return

    # --- Define Loss & Optimizer ---
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY )

    # --- Training Loop Setup ---
    best_val_loss = float('inf')
    model_save_path = os.path.join(SCRIPT_DIR, f'snoutnet_{model_name}_{aug_suffix}_model.pth')
    plot_save_path = os.path.join(SCRIPT_DIR, f'snoutnet_{model_name}_{aug_suffix}_loss_curve.png')
    epoch_history, train_loss_history, val_loss_history = [], [], []
    print("\n--- Starting Training ---")
    training_start_time = time.time()

    # --- Get total images per epoch for timing calculation ---
    total_epoch_images = len(train_dataset) + len(val_dataset)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time() # Start timing epoch

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", leave=False)
        for images, labels, _ in train_loop:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)
        epoch_train_loss = running_train_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Validate]", leave=False)
        with torch.no_grad():
            for images, labels, _ in val_loop:
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
        epoch_val_loss = running_val_loss / len(val_dataset)

        epoch_end_time = time.time() # End timing epoch
        epoch_duration = epoch_end_time - epoch_start_time

        # --- MODIFIED: Calculate average time per image ---
        avg_time_per_image_ms = (epoch_duration / total_epoch_images) * 1000 if total_epoch_images > 0 else 0
        # --- END MODIFIED ---

        # --- Epoch Summary ---
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} Complete")
        print(f"  Training Loss: {epoch_train_loss:.6f}")
        print(f"  Validation Loss: {epoch_val_loss:.6f}")
        # --- MODIFIED: Updated print statement ---
        print(f"  Epoch Time: {epoch_duration:.2f}s (Avg: {avg_time_per_image_ms:.2f} ms/image)")
        # --- END MODIFIED ---

        # --- Save Best Model ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Best model saved (Val Loss: {best_val_loss:.6f})")

        # --- Update & Save Plot ---
        epoch_history.append(epoch); train_loss_history.append(epoch_train_loss); val_loss_history.append(epoch_val_loss)
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(epoch_history, train_loss_history, 'o-', label='Training Loss')
            plt.plot(epoch_history, val_loss_history, 'o-', label='Validation Loss')
            plt.title(f'SnoutNet ({model_name.capitalize()}, {aug_suffix}): Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.legend(); plt.grid(True); plt.ylim(bottom=0); plt.tight_layout()
            plt.savefig(plot_save_path); plt.close()
        except Exception as e: print(f"  Warning: Could not save loss plot: {e}")

    # --- Training Complete Summary ---
    total_training_time = time.time() - training_start_time
    print("\n--- Training Complete ---")
    print(f"Total Time: {total_training_time:.2f}s | Best Val Loss: {best_val_loss:.6f}")
    print(f"Model saved to {model_save_path}")
    print(f"Plot saved to {plot_save_path}")

if __name__ == "__main__":
    main()