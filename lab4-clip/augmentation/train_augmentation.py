import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from augmentation files
from dataset_augmentation import CocoDataset, CocoCollator
from model_augmentation import CLIPModel

# --- Hyperparameters ---
BATCH_SIZE = 64  # M4 Pro usually handles 64 well. Reduce to 32 if you get memory errors.
LEARNING_RATE = 1e-4
EPOCHS = 5
SAVE_DIR = "checkpoints_augmentation"

# --- Device Selection (Mac M4 Optimization) ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple MPS (Metal Performance Shaders) acceleration.")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using CUDA.")
else:
    DEVICE = "cpu"
    print("Using CPU.")

os.makedirs(SAVE_DIR, exist_ok=True)

def info_nce_loss(image_embeddings, text_embeddings, temperature):
    # Logits: (N, N) matrix
    logits = (image_embeddings @ text_embeddings.T) * torch.exp(temperature)
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

def evaluate_loss(model, dataloader):
    """Computes average loss on validation set."""
    model.eval()
    total_loss = 0.0
    steps = 0
    print("Running validation loop...")
    with torch.no_grad():
        for batch in dataloader:
            # Limit validation steps to speed up training
            if steps > 200: break 
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pixel_values = batch["pixel_values"].to(DEVICE)
            
            img_emb, txt_emb = model(pixel_values, input_ids, attention_mask)
            loss = info_nce_loss(img_emb, txt_emb, model.temperature)
            total_loss += loss.item()
            steps += 1
    model.train()
    return total_loss / steps if steps > 0 else 0

def train():
    # 1. Data Loaders
    print("Preparing DataLoaders...")
    train_dataset = CocoDataset(mode="train")
    val_dataset = CocoDataset(mode="val") 
    collator = CocoCollator()
    
    # M4 Pro Optimization: num_workers=4 usually works best for Apple Silicon
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collator, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        collate_fn=collator, num_workers=2, pin_memory=True
    )
    
    # 2. Model, Optimizer, Scheduler
    model = CLIPModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        epoch_train_loss = 0.0
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pixel_values = batch["pixel_values"].to(DEVICE)
            
            optimizer.zero_grad()
            img_emb, txt_emb = model(pixel_values, input_ids, attention_mask)
            loss = info_nce_loss(img_emb, txt_emb, model.temperature)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Step Scheduler
        scheduler.step()
        
        # Calculate Averages
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation
        avg_val_loss = evaluate_loss(model, val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Results Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- SAVE CHECKPOINT EVERY EPOCH ---
        ckpt_path = os.path.join(SAVE_DIR, f"clip_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to: {ckpt_path}")
        
        # Update plot every epoch
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epochs')
        plt.ylabel('InfoNCE Loss')
        plt.title('Training vs Validation Loss (Augmented)')
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve_augmentation.png")
        plt.close() # Close to prevent memory buildup

    print(f"\nTraining Complete in {(time.time() - start_time)/60:.2f} minutes.")
    print("Final plot saved to 'loss_curve_augmentation.png'.")

if __name__ == "__main__":
    train()