import torch
import argparse
from student_model import CompactStudentModel
import torchvision.models.segmentation as segmentation
import dataset as data_utils

def count_params(model):
    """
    Counts the total and trainable parameters of a model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count Model Parameters")
    parser.add_argument(
        'model_type', 
        type=str, 
        choices=['teacher', 'student'], 
        help="Which model to count (teacher or student)"
    )
    args = parser.parse_args()

    if args.model_type == 'teacher':
        print("Loading Teacher Model (FCN-ResNet50)...")
        weights = segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = segmentation.fcn_resnet50(weights=weights)
        
        total, trainable = count_params(model)
        
        print("\n--- Teacher (FCN-ResNet50) ---")
        print(f"Total Parameters:     {total:,}")
        print(f"Trainable Parameters: {trainable:,}")
        print("--------------------------------")

    elif args.model_type == 'student':
        print("Loading Student Model (CompactStudentModel)...")
        # We need NUM_CLASSES to initialize the model
        model = CompactStudentModel(num_classes=data_utils.NUM_CLASSES)
        
        total, trainable = count_params(model)
        
        print("\n--- Student (CompactStudentModel) ---")
        print(f"Total Parameters:     {total:,}")
        print(f"Trainable Parameters: {trainable:,}")
        print("-------------------------------------")
        print("\n(Note: Total and Trainable are the same, which is correct\n as you are fine-tuning the entire student model.)")