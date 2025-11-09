import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt # <-- 1. IMPORT MATPLOTLIB

# Import models and data
from student_model import CompactStudentModel
import dataset as data_utils
from test_teacher import get_device # Reuse the MPS-aware device getter

# Import pre-trained teacher model
import torchvision.models.segmentation as segmentation
from torchvision.models.feature_extraction import create_feature_extractor

# Add a global flag for our one-time shape check
SHAPES_PRINTED = False

def train_distilled_model(args):
    """
    Main training and evaluation loop for Knowledge Distillation.
    """
    
    global SHAPES_PRINTED
    
    # --- 1. Setup ---
    device = get_device()
    
    # Create directory for models
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # --- MODIFIED: Auto-rename based on distillation arguments ---
    if args.alpha == 0.0 and args.beta == 0.0:
        print("Distillation disabled (alpha=0, beta=0). Saving to 'student_model_no_distillation.pth'")
        model_save_path = "models/student_model_no_distillation.pth"
    else:
        print(f"Distillation enabled (alpha={args.alpha}, beta={args.beta}). Saving to 'student_model_kd.pth'")
        model_save_path = "models/student_model_kd.pth"
    # ------------------------------------------------------------

    # --- 2. Data ---
    print("Loading datasets...")
    train_loader, val_loader = data_utils.get_dataloaders(args.batch_size)
    
    # --- 3. Models ---
    print("Loading models...")
    # Student Model (the one we're training)
    student_model = CompactStudentModel(num_classes=data_utils.NUM_CLASSES).to(device)
    
    # Teacher Model (FCN-ResNet50)
    teacher_weights = segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    base_teacher = segmentation.fcn_resnet50(weights=teacher_weights)
    
    # Create feature extractor for the teacher to get intermediate layers
    teacher_return_nodes = {
        'backbone.layer1': 'low',   # 256 channels
        'backbone.layer2': 'mid',   # 512 channels
        'backbone.layer3': 'high',  # 1024 channels
        'classifier.4': 'out'       # Final output (21 channels)
    }
    teacher_model = create_feature_extractor(base_teacher, teacher_return_nodes).to(device)
    
    # Freeze the teacher model
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    print(f"Student model parameters: {sum(p.numel() for p in student_model.parameters() if p.requires_grad):,}")
    print(f"Teacher model parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")

    # --- 4. Loss Functions & Optimizer ---
    
    # Optimizer for the STUDENT model
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 1. Standard Cross-Entropy (Hard Loss)
    criterion_hard = nn.CrossEntropyLoss(ignore_index=data_utils.IGNORE_INDEX)
    
    # 2. Response-Based Distillation (Soft Loss)
    # --- FIX: Use 'mean' reduction instead of 'batchmean' ---
    # 'batchmean' divides by B, 'mean' divides by (B*C*H*W)
    # This prevents the loss from being millions of times too large.
    criterion_soft = nn.KLDivLoss(reduction='mean', log_target=True)
    
    # 3. Feature-Based Distillation (Feature Loss)
    criterion_feat = nn.CosineEmbeddingLoss(margin=0.1)
    
    # Metric for validation (moved to CPU to prevent OOM)
    val_metric = JaccardIndex(
        task='multiclass', 
        num_classes=data_utils.NUM_CLASSES, 
        ignore_index=data_utils.IGNORE_INDEX
    )

    # --- 5. Training Loop ---
    best_miou = -1.0

    # --- CHANGED: Plotting Data Lists ---
    epochs_list = []
    train_loss_list = []
    val_miou_list = []
    # --- End Plot Setup ---


    print(f"Starting training for {args.epochs} epochs on {device}...")
    for epoch in range(args.epochs):
        student_model.train() # Set student to train mode
        train_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = images.to(device)
            targets = targets.to(device) # Shape: [B, H, W]

            # --- Forward Passes ---
            # 1. Student Forward Pass
            student_out_main, student_features = student_model(images)
            
            # 2. Teacher Forward Pass (with no_grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
                teacher_out_main = teacher_outputs['out']
                if teacher_out_main.shape[2:] != student_out_main.shape[2:]:
                    teacher_out_main = F.interpolate(teacher_out_main, size=student_out_main.shape[2:], mode='bilinear', align_corners=False)

            # --- START: Sanity Checks ---
            if not SHAPES_PRINTED:
                print("\n--- Running Shape Sanity Check (Epoch 0, Batch 0) ---")
                print(f"Images shape:           {images.shape}")
                print(f"Targets shape:          {targets.shape}  <-- Should be [B, H, W]")
                print(f"Student Output shape:   {student_out_main.shape}  <-- Should be [B, C, H, W]")
                print(f"Teacher Output shape:   {teacher_out_main.shape}  <-- Should be [B, C, H, W]")
                print("--- End Shape Check ---")
                SHAPES_PRINTED = True # Don't print this again
            # --- END: Sanity Checks ---

            # --- Calculate Losses ---
            
            # 1. L_CE: Hard Loss (Student vs. Ground Truth)
            loss_hard = criterion_hard(student_out_main, targets)
            
            # 2. L_KD_response: Soft Loss (Student vs. Teacher logits)
            soft_targets = F.log_softmax(teacher_out_main / args.temp, dim=1)
            soft_preds = F.log_softmax(student_out_main / args.temp, dim=1)
            loss_soft = (args.temp ** 2) * criterion_soft(soft_preds, soft_targets)

            # 3. L_KD_feature: Feature Loss (Student vs. Teacher features)
            loss_feat = 0.0
            
            # Only calculate feature loss if beta > 0
            if args.beta > 0:
                teacher_features = {k: teacher_outputs[k] for k in ['low', 'mid', 'high']}
                
                for key in student_features.keys():
                    s_feat = student_features[key]
                    t_feat = teacher_features[key]
                    
                    s_feat_resized = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
                    
                    batch_size, channels, h, w = s_feat_resized.shape
                    s_flat = s_feat_resized.permute(0, 2, 3, 1).reshape(-1, channels)
                    t_flat = t_feat.permute(0, 2, 3, 1).reshape(-1, channels)
                    
                    cos_target = torch.ones(s_flat.shape[0]).to(device)
                    loss_feat += criterion_feat(s_flat, t_flat, cos_target)
            
            # --- Joint Loss ---
            # Note: loss_soft will be 0 if alpha is 0, and loss_feat will be 0 if beta is 0
            loss = (1 - args.alpha) * loss_hard + \
                   args.alpha * loss_soft + \
                   args.beta * loss_feat

            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- 6. Validation Loop ---
        student_model.eval()
        val_metric.reset()
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = images.to(device)
                targets = targets.to(device)
                
                # During eval, model only returns main output
                outputs = student_model(images) 
                preds = torch.argmax(outputs, dim=1)
                
                # Move to CPU for metric calculation
                val_metric.update(preds.cpu(), targets.cpu())
        
        miou = val_metric.compute()
        scheduler.step(miou) # Update LR scheduler

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val mIoU: {miou.item():.4f}")
        
        if miou > best_miou:
            best_miou = miou
            torch.save(student_model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with mIoU: {best_miou:.4f}")

        # --- CHANGED: Update Plot Data & Save Figure ---
        epochs_list.append(epoch + 1)
        train_loss_list.append(avg_train_loss)
        val_miou_list.append(miou.item())
        
        # Create a new figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('Student Model Training')
        
        # Plot Training Loss
        ax1.plot(epochs_list, train_loss_list, 'b-', label='Train Loss')
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True)
        ax1.legend()
        
        # Plot Validation mIoU
        ax2.plot(epochs_list, val_miou_list, 'r-', label='Val mIoU')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mIoU")
        ax2.set_title("Validation mIoU")
        ax2.grid(True)
        ax2.legend()
        
        # Save and close the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("plots/training_plot.png")  # Overwrites the file each epoch
        plt.close(fig) # Close figure to free memory
        # --- End Save Plot ---

    print("--- Training Complete ---")
    print(f"Best Val mIoU: {best_miou:.4f}")
    print(f"Final model saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Student Model with Knowledge Distillation")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size (adjust for your VRAM)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--alpha', type=float, default=0.9, help="Weight for soft (KD response) loss")
    parser.add_argument('--beta', type=float, default=10.0, help="Weight for feature-based (KD feature) loss")
    parser.add_argument('--temp', type=float, default=4.0, help="Temperature for distillation")
    
    args = parser.parse_args()
    
    train_distilled_model(args)