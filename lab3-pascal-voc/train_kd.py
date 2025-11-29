import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
import time  # <-- 1. IMPORT TIME

# Import models and data
from student_model import CompactStudentModel
import dataset as data_utils
from test_teacher import get_device

# Import pre-trained teacher model
import torchvision.models.segmentation as segmentation
from torchvision.models.feature_extraction import create_feature_extractor

# Add a global flag for our one-time shape check
SHAPES_PRINTED = False

def train_distilled_model(args):
    """
    Main training and evaluation loop for Knowledge Distillation.
    """
    start_time = time.time()  # <-- 2. START TIMER
    
    global SHAPES_PRINTED
    
    # --- 1. Setup ---
    device = get_device()
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # --- 3. MODIFIED: Auto-rename based on 3 training modes ---
    model_name_suffix = ""
    if args.alpha == 0.0 and args.beta == 0.0:
        model_name_suffix = "no_distillation"
        print(f"Mode: Standard Training (No Distillation)")
    elif args.alpha > 0.0 and args.beta == 0.0:
        model_name_suffix = "response_kd"
        print(f"Mode: Response-Based Distillation (alpha={args.alpha})")
    elif args.alpha == 0.0 and args.beta > 0.0:
        model_name_suffix = "feature_kd"
        print(f"Mode: Feature-Based Distillation (beta={args.beta})")
    else:
        model_name_suffix = "combined_kd"
        print(f"Mode: Combined Distillation (alpha={args.alpha}, beta={args.beta})")

    model_save_path = f"models/student_model_{model_name_suffix}.pth"
    plot_save_path = f"plots/training_plot_{model_name_suffix}.png"
    print(f"Model will be saved to: {model_save_path}")
    # ------------------------------------------------------------

    # --- 2. Data ---
    print("Loading datasets...")
    train_loader, val_loader = data_utils.get_dataloaders(args.batch_size)
    
    # --- 3. Models ---
    print("Loading models...")
    student_model = CompactStudentModel(num_classes=data_utils.NUM_CLASSES).to(device)
    
    # Only load teacher if we're actually using it
    teacher_model = None
    if args.alpha > 0 or args.beta > 0:
        print("Loading teacher model for distillation...")
        teacher_weights = segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        base_teacher = segmentation.fcn_resnet50(weights=teacher_weights)
        
        teacher_return_nodes = {
            'backbone.layer1': 'low',
            'backbone.layer2': 'mid',
            'backbone.layer3': 'high',
            'classifier.4': 'out'
        }
        teacher_model = create_feature_extractor(base_teacher, teacher_return_nodes).to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print(f"Teacher model parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    
    print(f"Student model parameters: {sum(p.numel() for p in student_model.parameters() if p.requires_grad):,}")

    # --- 4. Loss Functions & Optimizer ---
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    criterion_hard = nn.CrossEntropyLoss(ignore_index=data_utils.IGNORE_INDEX)
    criterion_soft = nn.KLDivLoss(reduction='mean', log_target=True)
    criterion_feat = nn.CosineEmbeddingLoss(margin=0.1)
    
    val_metric = JaccardIndex(
        task='multiclass', 
        num_classes=data_utils.NUM_CLASSES, 
        ignore_index=data_utils.IGNORE_INDEX
    )

    # --- 5. Training Loop ---
    best_miou = -1.0
    epochs_list = []
    train_loss_list = []
    val_miou_list = []

    print(f"Starting training for {args.epochs} epochs on {device}...")
    for epoch in range(args.epochs):
        student_model.train()
        train_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = images.to(device)
            targets = targets.to(device)

            # --- Forward Pass ---
            student_out_main, student_features = student_model(images)
            
            # --- Calculate Losses ---
            loss = 0.0
            loss_hard = criterion_hard(student_out_main, targets)
            
            # 1. Hard Loss
            # We apply (1-alpha) *only* if alpha is > 0, otherwise it's just 1.0
            hard_loss_weight = 1.0
            if args.alpha > 0.0 and args.beta == 0.0: # Response-only
                hard_loss_weight = 1.0 - args.alpha
            elif args.alpha == 0.0 and args.beta > 0.0: # Feature-only
                hard_loss_weight = 1.0 # No alpha to subtract
            elif args.alpha > 0.0 and args.beta > 0.0: # Combined
                hard_loss_weight = 1.0 - args.alpha

            loss += hard_loss_weight * loss_hard

            # 2. Distillation Losses (if teacher is loaded)
            if teacher_model:
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)

                # Response-Based Loss (Soft)
                if args.alpha > 0.0:
                    teacher_out_main = teacher_outputs['out']
                    if teacher_out_main.shape[2:] != student_out_main.shape[2:]:
                        teacher_out_main = F.interpolate(teacher_out_main, size=student_out_main.shape[2:], mode='bilinear', align_corners=False)
                    
                    soft_targets = F.log_softmax(teacher_out_main / args.temp, dim=1)
                    soft_preds = F.log_softmax(student_out_main / args.temp, dim=1)
                    loss_soft = (args.temp ** 2) * criterion_soft(soft_preds, soft_targets)
                    loss += args.alpha * loss_soft

                # Feature-Based Loss
                if args.beta > 0.0:
                    loss_feat = 0.0
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
                    
                    loss += args.beta * loss_feat
            
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
                
                outputs = student_model(images) 
                preds = torch.argmax(outputs, dim=1)
                
                val_metric.update(preds.cpu(), targets.cpu())
        
        miou = val_metric.compute()
        scheduler.step(miou)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val mIoU: {miou.item():.4f}")
        
        if miou > best_miou:
            best_miou = miou
            torch.save(student_model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with mIoU: {best_miou:.4f}")

        # --- Update Plot Data & Save Figure ---
        epochs_list.append(epoch + 1)
        train_loss_list.append(avg_train_loss)
        val_miou_list.append(miou.item())
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f'Student Model Training ({model_name_suffix})')
        
        ax1.plot(epochs_list, train_loss_list, 'b-', label='Train Loss')
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(epochs_list, val_miou_list, 'r-', label='Val mIoU')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mIoU")
        ax2.set_title("Validation mIoU")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_save_path)
        plt.close(fig)

    # --- 4. CALCULATE AND PRINT TOTAL TIME ---
    end_time = time.time()
    total_seconds = end_time - start_time
    total_minutes = total_seconds / 60
    print("\n--- Training Complete ---")
    print(f"Best Val mIoU: {best_miou:.4f}")
    print(f"Final model saved to {model_save_path}")
    print(f"Total Training Time: {total_minutes:.2f} minutes ({total_seconds:.0f} seconds)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Student Model with Knowledge Distillation")
    parser.add_argument('--epochs', type=int, default=35, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size (adjust for your VRAM)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--alpha', type=float, default=0.0, help="Weight for soft (KD response) loss. If 0, this loss is not used.")
    parser.add_argument('--beta', type=float, default=0.0, help="Weight for feature-based (KD feature) loss. If 0, this loss is not used.")
    parser.add_argument('--temp', type=float, default=4.0, help="Temperature for distillation")
    
    args = parser.parse_args()
    
    train_distilled_model(args)