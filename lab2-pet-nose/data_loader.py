import os
import ast
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# --- ADDED ---
# F contains functional transforms (like hflip)
import torchvision.transforms.functional as F
import random # To randomly trigger augmentations
# ---
from PIL import Image
import matplotlib.pyplot as plt
            

class SnoutDataset(Dataset):
    """
    Custom PyTorch Dataset for the Oxford-IIIT Pets snout localization task.
    ...
    """
    
    def __init__(self, root_dir, annotation_file, transform=None, augmentations=None):
        """
        Args:
            root_dir (string): Path to the 'images-original' directory.
            annotation_file (string): Path to the 'train-noses.txt' or 'test-noses.txt' file.
            transform (callable, optional): Optional transform to be applied
                on an image sample.
            augmentations (list or set, optional): A list/set of strings specifying
                which augmentations to apply (e.g., ['flip', 'color']).
                Default is None (no augmentation).
        """
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_path = annotation_file
        self.transform = transform
        self.samples = []
        
        # --- MODIFIED ---
        # Convert augmentation list to a set for efficient lookup
        if augmentations is None:
            self.augmentations = set()
        else:
            self.augmentations = set(augmentations)

        # Define a ColorJitter transform to be applied manually
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        # ---
        
        print(f"Loading annotations from: {self.annotation_path}")
        if self.augmentations:
            print(f"Applying augmentations: {self.augmentations}")
        
        try:
            with open(self.annotation_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split on the *first* comma only
                    # Format: beagle_145.jpg,"(198, 304)"
                    parts = line.split(',', 1)
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed line: {line}")
                        continue

                    image_name = parts[0]
                    coord_str = parts[1].strip().strip('"') # Removes surrounding quotes
                    
                    try:
                        # ast.literal_eval safely evaluates the string "(x, y)" to a tuple
                        coords = ast.literal_eval(coord_str)
                        
                        # We will store the (x, y) coordinates
                        label = torch.tensor([coords[0], coords[1]], dtype=torch.float32)
                        
                        image_path = os.path.join(self.image_dir, image_name)
                        
                        if not os.path.exists(image_path):
                            print(f"Warning: Image file not found, skipping: {image_path}")
                            continue
                            
                        self.samples.append((image_path, label))

                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Skipping line with bad coord format: {line} -> {e}")
        
        except FileNotFoundError:
            print(f"ERROR: Annotation file not found at {self.annotation_path}")
            print("Please check your file paths.")
            raise
        
        print(f"Successfully loaded {len(self.samples)} samples.")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Returns:
            tuple: (image, label, index) where image is the transformed image
            tensor, label is the normalized [x, y] coordinate tensor,
            and index is the original index of the item.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, original_coords = self.samples[idx]
        
        try:
            # Open image and ensure it's RGB
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample if image is corrupt
            # In a real pipeline, you might want to filter these in __init__
            image = Image.new('RGB', (227, 227), (0, 0, 0))
            original_coords = torch.tensor([0.0, 0.0], dtype=torch.float32)

        original_w, original_h = image.size
        
        # --- Normalize coordinates ---
        # Convert absolute pixel coordinates (x, y) to
        # relative coordinates [0, 1]
        norm_x = original_coords[0] / original_w
        norm_y = original_coords[1] / original_h
        normalized_label = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        
        
        # --- START: DATA AUGMENTATION ---
        # Apply augmentations *before* the main transform (like ToTensor)
        
        # Augmentation 1: Random Horizontal Flip (Geometric)
        # Check if 'flip' is in our set and apply with 50% probability
        if 'flip' in self.augmentations and random.random() < 0.5:
            # Flip the image
            image = F.hflip(image)
            
            # Flip the x-coordinate of the label
            # 0.2 -> 0.8
            # 0.7 -> 0.3
            normalized_label[0] = 1.0 - normalized_label[0]

        # Augmentation 2: Color Jitter (Pixel-level)
        # Check if 'color' is in our set and apply with 50% probability
        if 'color' in self.augmentations and random.random() < 0.5:
            image = self.color_jitter(image)
                
        # --- END: DATA AUGMENTATION ---

        
        # Apply transforms (e.g., Resize, ToTensor, Normalize) to the image
        if self.transform:
            image = self.transform(image)
            
        return image, normalized_label, idx

