import torch
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import InterpolationMode
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import dataset as data_utils # Import our dataset file

def get_device():
    """Get the best available device (MPS, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device

def evaluate_pretrained_fcn():
    """
    Step 2.1: Load and evaluate the pretrained FCN-ResNet50 on PASCAL VOC 2012.
    """
    print("Starting Step 2.1: Evaluating Pretrained FCN-ResNet50...")

    # 1. Setup Device
    device = get_device()

    # 2. Load Model and Weights
    weights = segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = segmentation.fcn_resnet50(weights=weights).to(device)
    model.eval() # Set model to evaluation mode

    # 3. Load Data
    # We use the transforms recommended by the *model weights* for evaluation
    # This is a v1 pipeline, so it goes in the 'transform' (singular) argument
    image_transforms = weights.transforms()
    
    # Target transforms must be simple v2 transforms
    # This goes in the 'target_transform' (singular) argument
    target_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False)
    ])

    val_dataset = datasets.VOCSegmentation(
        root="./data",
        year="2012",
        image_set="val",
        download=False, # <-- FIXED as requested
        transform=image_transforms,
        target_transform=target_transforms
    )
    
    # We apply the joint transforms (resizing) from the transforms object
    # val_dataset = v2.WrappedDataset(val_dataset, joint_transform=image_transforms.get_joint_transforms())
    # ^-- THIS LINE IS DELETED. It was the original error and is not needed.

    val_loader = DataLoader(
        val_dataset,
        batch_size=1, # <-- FIXED: Use batch size 1 for evaluation
        shuffle=False,
        num_workers=2
    )
    print("Dataset loaded.")

    # 5. Setup Metric (mIoU)
    metric = JaccardIndex(
        task='multiclass',
        num_classes=data_utils.NUM_CLASSES,
        ignore_index=data_utils.IGNORE_INDEX
    ).to(device)

    # 6. Evaluation Loop
    print("Running evaluation...")
    with torch.no_grad(): # Disable gradient calculation
        for images, targets in tqdm(val_loader, desc="Evaluating Teacher"):
            images = images.to(device)
            targets = targets.to(device) # Shape: [1, 1, H, W]

            outputs = model(images)['out'] # Shape: [1, 21, H, W]
            
            # The model output and target may have slightly different sizes due to
            # model padding. We resize output to match target.
            outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)

            preds = torch.argmax(outputs, dim=1) # Shape: [1, H, W]
            targets = targets.squeeze(1) # Shape: [1, H, W]

            metric.update(preds, targets)

    # 7. Compute and Print Final mIoU
    miou = metric.compute()
    print("\n--- Evaluation Complete ---")
    print(f"Teacher FCN-ResNet50 mIoU (VOC 2012 val): {miou.item():.4f}")
    print("----------------------------")

if __name__ == "__main__":
    evaluate_pretrained_fcn()