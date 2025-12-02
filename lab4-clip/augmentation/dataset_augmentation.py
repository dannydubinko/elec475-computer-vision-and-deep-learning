import os
import json
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from PIL import Image
from torchvision import transforms

# --- UPDATED PATHS FOR YOUR STRUCTURE ---
# Since this script runs inside 'project/augmentation/', 
# we step back one level (..) to get to 'project/coco2014/'
BASE_ROOT = "../coco2014"

IMAGE_PATHS = {
    "train": os.path.join(BASE_ROOT, "images/train2014"),
    "val": os.path.join(BASE_ROOT, "images/val2014")
}

ANNOTATION_PATHS = {
    "train": os.path.join(BASE_ROOT, "annotations/captions_train2014.json"),
    "val": os.path.join(BASE_ROOT, "annotations/captions_val2014.json")
}

MODEL_NAME = "openai/clip-vit-base-patch32"

class CocoDataset(Dataset):
    """
    PyTorch Dataset for COCO 2014 Captions with Augmentation.
    """
    def __init__(self, mode: str = "train"):
        assert mode in ["train", "val"], "Mode must be 'train' or 'val'"
        self.mode = mode
        self.image_dir = IMAGE_PATHS[mode]
        self.caption_file = ANNOTATION_PATHS[mode]
        self.pairs: List[Tuple[str, str]] = []
        
        self._load_annotations()

        # --- Data Augmentation ---
        if self.mode == "train":
            # Augmentation Pipeline for Training
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                # CLIP Normalization constants
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                     std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            # Validation: Deterministic Resize & CenterCrop
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                     std=(0.26862954, 0.26130258, 0.27577711))
            ])

    def _load_annotations(self):
        print(f"Loading {self.mode} annotations from: {self.caption_file}...")
        if not os.path.exists(self.caption_file):
            raise FileNotFoundError(f"Missing required file: {self.caption_file}")

        with open(self.caption_file, 'r') as f:
            data = json.load(f)

        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        
        for ann in data['annotations']:
            if ann['image_id'] in id_to_filename:
                self.pairs.append((id_to_filename[ann['image_id']], ann['caption']))
            
        print(f"Loaded {len(self.pairs)} image-caption pairs for {self.mode} set.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Image.Image]:
        # Return Tuple: (Transformed Tensor, Caption, Original PIL Image for Vis)
        filename, caption = self.pairs[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Skip corrupted/missing images
            return self.__getitem__((idx + 1) % len(self))
        
        # Apply transforms
        pixel_values = self.transform(image_pil)
            
        return pixel_values, caption, image_pil


class CocoCollator:
    """
    Handles tokenization of text batches. 
    Image transforms are already done in Dataset to support augmentation.
    """
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        # batch structure: [(pixel_values, caption, original_pil), ...]
        pixel_values = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        
        # Tokenize Text
        text_inputs = self.processor(
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "captions_raw": captions # Useful for debugging
        }