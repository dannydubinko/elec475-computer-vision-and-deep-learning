# dataset.py

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import InterpolationMode
from torch.utils.data import DataLoader

# --- Constants ---
NUM_CLASSES = 21
IGNORE_INDEX = 255
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CROP_SIZE = 480
RESIZE_SIZE = 520

# --- *** NEW TRANSFORM CLASS (Replaces v2.Lambda) *** ---
class ApplyIndividualTransforms(nn.Module):
    """
    A custom transform module that applies the correct transforms
    to the image and target tensors separately.
    """
    def __init__(self, mean, std):
        super().__init__()
        # Create the image-only transforms
        self.image_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])

    def forward(self, image, target):
        """
        This method is called by v2.Compose with two arguments: image and target.
        """
        
        # 1. Apply image-only transforms to the image
        image = self.image_transform(image)
        
        # 2. Apply target-only transform to the target
        # The raw mask is [C, H, W] (e.g., [3, 480, 480]) uint8.
        # We need [H, W] long for CrossEntropyLoss.
        target = target[0].to(torch.long)
        
        return image, target
# --- *** END NEW CLASS *** ---


# --- Main Function ---
def get_dataloaders(batch_size):
    """
    Creates and returns the PASCAL VOC 2012 train and val dataloaders
    with appropriate v2 transforms.
    """

    # --- Transforms ---
    
    # Combined v2 transform pipeline for TRAINING
    train_transforms = v2.Compose([
        # 1. Joint transforms (applied to PIL image/target)
        v2.Resize(RESIZE_SIZE, interpolation=InterpolationMode.NEAREST),
        v2.RandomCrop(CROP_SIZE),
        v2.RandomHorizontalFlip(p=0.5),
        
        # 2. Convert to Tensors
        v2.ToImage(), # -> (Image([C,H,W] uint8), Target([C,H,W] uint8))
        
        # 3. Apply separate transforms to image and target
        ApplyIndividualTransforms(mean=MEAN, std=STD)
    ])

    # Combined v2 transform pipeline for VALIDATION
    val_transforms = v2.Compose([
        # 1. Joint transforms
        v2.Resize(RESIZE_SIZE, interpolation=InterpolationMode.NEAREST),
        v2.CenterCrop(CROP_SIZE),
        
        # 2. Convert to Tensors
        v2.ToImage(),
        
        # 3. Apply separate transforms to image and target
        ApplyIndividualTransforms(mean=MEAN, std=STD)
    ])

    # --- Datasets ---
    
    # Create the training dataset
    train_dataset = datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='train',
        download=False, # Set to False as per our previous fix
        transforms=train_transforms
    )
    
    # Create the validation dataset
    val_dataset = datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=False, # Set to False as per our previous fix
        transforms=val_transforms
    )

    # --- DataLoaders ---
    
    # Create the training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create the validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader

# A simple check to see if the loader works
if __name__ == '__main__':
    print("Attempting to create DataLoaders...")
    train_loader, val_loader = get_dataloaders(batch_size=4)
    print("DataLoaders created successfully.")
    
    print("\nFetching one batch from train_loader...")
    try:
        images, targets = next(iter(train_loader))
        print(f"  Images batch shape: {images.shape}, dtype: {images.dtype}")
        print(f"  Targets batch shape: {targets.shape}, dtype: {targets.dtype}")
        print("  Batch fetched successfully!")
        
        if targets.dtype == torch.long and len(targets.shape) == 3:
            print("  [SUCCESS] Targets are 3D and torch.long as expected.")
        else:
            print(f"  [ERROR] Targets shape/dtype is wrong! Got {targets.shape}, {targets.dtype}")

    except Exception as e:
        print(f"\n[ERROR] Failed to fetch batch: {e}")
        import traceback
        traceback.print_exc()