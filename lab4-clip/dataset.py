# Save this file as: dataset.py
# Located at: /content/drive/MyDrive/Colab_Projects/elec475-computer-vision-and-deep-learning/lab4-clip/dataset.py

import os
import json
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from PIL import Image

# --- Constants based on Lab Manual and your paths ---

# The root directory of the COCO dataset
DATA_ROOT = "/root/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014"

# Paths to the required annotation files 
# CRITICAL: These files *must* exist, as discussed in the clarification.
ANNOTATION_PATHS = {
    "train": os.path.join(DATA_ROOT, "annotations/captions_train2014.json"),
    "val": os.path.join(DATA_ROOT, "annotations/captions_val2014.json")
}

# Paths to the image directories [cite: 46, 47]
IMAGE_PATHS = {
    "train": os.path.join(DATA_ROOT, "images/train2014"),
    "val": os.path.join(DATA_ROOT, "images/val2014")
}

# Pretrained model name for the processor [cite: 54]
MODEL_NAME = "openai/clip-vit-base-patch32"


class CocoDataset(Dataset):
    """
    PyTorch Dataset for COCO 2014 Captions.

    Loads image-caption pairs from the COCO 2014 dataset.
    This class loads the raw PIL Images and caption strings.
    The actual preprocessing (tokenization and image transforms)
    is handled by the CocoCollator class.
    """
    def __init__(self, mode: str = "train"):
        """
        Args:
            mode (str): 'train' or 'val' to specify which dataset split to load.
        """
        assert mode in ["train", "val"], "Mode must be 'train' or 'val'"
        self.mode = mode
        self.image_dir = IMAGE_PATHS[mode]
        self.caption_file = ANNOTATION_PATHS[mode]
        
        # This flat list will store (image_filename, caption_string) tuples
        self.pairs: List[Tuple[str, str]] = []

        self._load_annotations()

    def _load_annotations(self):
        """
        Loads annotations from the JSON file and creates a flat list
        of (image_filename, caption) pairs.
        """
        print(f"Loading {self.mode} annotations from: {self.caption_file}...")
        
        # Check if file exists before proceeding
        if not os.path.exists(self.caption_file):
            print(f"ERROR: Annotation file not found at {self.caption_file}")
            print("Please ensure 'captions_train2014.json' and 'captions_val2014.json' are present.")
            raise FileNotFoundError(f"Missing required file: {self.caption_file}")

        with open(self.caption_file, 'r') as f:
            data = json.load(f)

        # 1. Create a mapping from image_id to filename
        # e.g., {318556: "COCO_train2014_000000318556.jpg", ...}
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        
        # 2. Create the flat list of pairs
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            
            if image_id in id_to_filename:
                filename = id_to_filename[image_id]
                self.pairs.append((filename, caption))
            
        print(f"Loaded {len(self.pairs)} image-caption pairs for {self.mode} set.")

    def __len__(self) -> int:
        """Returns the total number of image-caption pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        """
        Returns a single raw (PIL Image, caption string) pair.
        """
        filename, caption = self.pairs[idx]
        
        # Construct the full image path
        image_path = os.path.join(self.image_dir, filename)
        
        # Load the image
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not load image {image_path}. Skipping.")
            # Return a placeholder or skip (here, we get the next item)
            # This is a simple way to handle corrupted/missing files
            return self.__getitem__((idx + 1) % len(self))
            
        return image, caption


class CocoCollator:
    """
    A collate_fn for the DataLoader that uses the CLIPProcessor
    to batch-process raw images and text.
    
    This applies the required resizing, normalization, and tokenization
    as specified in the lab manual[cite: 53, 54, 56].
    """
    def __init__(self):
        """
        Initializes the processor from Hugging Face.
        This processor contains the correct mean/std [cite: 56]
        and resizes images to 224x224[cite: 56].
        """
        print("Initializing CLIPProcessor...")
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("CLIPProcessor initialized.")

    def __call__(self, batch: List[Tuple[Image.Image, str]]) -> Dict[str, torch.Tensor]:
        """
        Processes a list of (image, caption) tuples.
        
        Args:
            batch: A list of (PIL.Image, str) tuples.
            
        Returns:
            A dictionary of batched tensors, ready for the model:
            - "pixel_values": (N, C, H, W) tensor for images
            - "input_ids": (N, L) tensor for text
            - "attention_mask": (N, L) tensor for text
        """
        # Unzip the batch
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        # Process the batch
        # The processor handles:
        # 1. Text: Tokenization, padding, truncation [cite: 65]
        # 2. Image: Resizing to 224x224 and normalization [cite: 53, 56, 64]
        inputs = self.processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return inputs