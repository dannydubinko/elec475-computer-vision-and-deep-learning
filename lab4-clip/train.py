import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # Added for plotting

# Import custom modules
from dataset import CocoDataset, CocoCollator
from model import CLIPModel

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

def info_nce_loss(image_embeddings, text_embeddings, temperature):
    """
    InfoNCE Loss.
    """
    # Logits: (N, N) matrix
    logits = (image_embeddings @ text_embeddings.T) * torch.exp(temperature)
    
    # Targets: Diagonal indices [0, 1, 2, ...]
    labels = torch.arange(len(logits), device=logits.device)
    
    # Symmetric Loss
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2

def train():
    print(f"Starting training on {DEVICE}...")
    
    # Data
    train_dataset = CocoDataset(mode="train")
    collator = CocoCollator()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=2
    )
    
    # Model & Optimizer
    model = CLIPModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Track loss for plotting
    loss_history = []
    
    # Loop
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pixel_values = batch["pixel_values"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            img_emb, txt_emb = model(pixel_values, input_ids, attention_mask)
            
            # Loss
            loss = info_nce_loss(img_emb, txt_emb, model.temperature)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Logging
            current_loss = loss.item()
            epoch_loss += current_loss
            loss_history.append(current_loss) # Save for plot
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        # Save Checkpoint & Epoch Stats
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"clip_epoch_{epoch+1}.pt"))

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} minutes.")

    # --- PLOTTING ---
    print("Generating loss curve...")
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Iterations (Batches)')
    plt.ylabel('InfoNCE Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_curve.png")
    print("Saved plot to 'training_loss_curve.png'")
    
    # Save raw data
    with open("loss_history.txt", "w") as f:
        for l in loss_history:
            f.write(f"{l}\n")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()