import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import textwrap
from PIL import Image

from dataset import CocoDataset, CocoCollator
from model import CLIPModel

# --- Config ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
CHECKPOINT_PATH = "checkpoints/clip_epoch_5.pt" 

# Limit evaluation to 5,000 samples to prevent RAM crash (OOM)
MAX_EVAL_SAMPLES = 2000
BATCH_SIZE = 32

def get_embeddings(model, dataloader):
    """
    Computes embeddings for a subset of the dataset.
    """
    model.eval()
    img_embs = []
    txt_embs = []
    
    count = 0
    print(f"Computing embeddings for validation set (Limit: {MAX_EVAL_SAMPLES})...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pixel_values = batch["pixel_values"].to(DEVICE)
            
            # Get features
            img, txt = model(pixel_values, input_ids, attention_mask)
            
            img_embs.append(img.cpu())
            txt_embs.append(txt.cpu())
            
            # Stop if we reached the limit
            count += input_ids.size(0)
            if count >= MAX_EVAL_SAMPLES:
                break
            
    all_img = torch.cat(img_embs)[:MAX_EVAL_SAMPLES]
    all_txt = torch.cat(txt_embs)[:MAX_EVAL_SAMPLES]
    
    return all_img, all_txt

def calculate_recall(image_embs, text_embs, k_values=[1, 5, 10]):
    """
    Calculates Recall@K for Image->Text and Text->Image.
    """
    print("Computing similarity matrix...")
    sim_matrix = image_embs @ text_embs.T
    
    num_samples = sim_matrix.shape[0]
    targets = torch.arange(num_samples)
    
    print(f"\n--- Evaluation Metrics (N={num_samples}) ---")
    
    # 1. Image-to-Text Retrieval
    _, i2t_indices = sim_matrix.topk(max(k_values), dim=1)
    for k in k_values:
        correct = i2t_indices[:, :k].eq(targets.view(-1, 1).expand(-1, k))
        recall = correct.sum().float() / num_samples
        print(f"Image-to-Text Recall@{k}: {recall.item():.4f}")

    # 2. Text-to-Image Retrieval
    _, t2i_indices = sim_matrix.topk(max(k_values), dim=0)
    t2i_indices = t2i_indices.T 
    for k in k_values:
        correct = t2i_indices[:, :k].eq(targets.view(-1, 1).expand(-1, k))
        recall = correct.sum().float() / num_samples
        print(f"Text-to-Image Recall@{k}: {recall.item():.4f}")

def visualize_retrieval(model, dataset, text_query, k=5):
    """
    Qualitative Result: Given a text query, retrieve top K UNIQUE images.
    """
    print(f"\n--- Visualizing Retrieval for query: '{text_query}' ---")
    
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    
    input_ids = inputs["input_ids"].to(DEVICE)
    mask = inputs["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        text_emb = model.text_encoder(input_ids, mask) 
    
    # Helper to get embeddings for scanning
    candidates = []
    collator = CocoCollator() 
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    
    print(f"Scanning images for matches (Limit: {MAX_EVAL_SAMPLES})...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i * BATCH_SIZE >= MAX_EVAL_SAMPLES: break 
            pixel_values = batch["pixel_values"].to(DEVICE)
            img_emb = model.image_encoder(pixel_values)
            candidates.append(img_emb.cpu())
            
    all_img_embs = torch.cat(candidates).to(DEVICE) 
    # Compute Similarity
    sims = (text_emb @ all_img_embs.T).squeeze()
    
    # Get Top 50 candidates (to allow for filtering duplicates)
    values, indices = sims.topk(min(50, len(sims)))
    
    # Filter Duplicates
    seen_filenames = set()
    unique_indices = []
    unique_scores = []
    
    for i, idx in enumerate(indices):
        idx_item = idx.item()
        # Access filename directly from dataset pairs
        # pair structure: (filename, caption)
        filename = dataset.pairs[idx_item][0]
        
        if filename not in seen_filenames:
            seen_filenames.add(filename)
            unique_indices.append(idx_item)
            unique_scores.append(values[i].item())
            
        if len(unique_indices) == k:
            break
            
    # Plot
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Query: {text_query}")
    for i, idx in enumerate(unique_indices):
        image, _ = dataset[idx]
        
        ax = plt.subplot(1, k, i+1)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"Score: {unique_scores[i]:.2f}")
        
    plt.tight_layout()
    save_name = f"retrieval_{text_query.replace(' ', '_')}.png"
    plt.savefig(save_name)
    print(f"Saved visualization to {save_name}")

def main():
    # 1. Check for checkpoint
    ckpt_to_load = CHECKPOINT_PATH
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint {CHECKPOINT_PATH} not found.")
        import glob
        ckpts = glob.glob("checkpoints/clip_epoch_*.pt")
        if ckpts:
            ckpts.sort()
            ckpt_to_load = ckpts[-1]
            print(f"Falling back to latest found: {ckpt_to_load}")
        else:
            print("No checkpoints found. Please run train.py first.")
            return

    print(f"Loading model from {ckpt_to_load}...")
    model = CLIPModel()
    model.load_state_dict(torch.load(ckpt_to_load, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 2. Load Validation Data
    val_dataset = CocoDataset(mode="val")
    collator = CocoCollator()
    # Shuffle for metrics to get a random distribution
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=2)
    
    # 3. Calculate Quantitative Metrics
    img_embs, txt_embs = get_embeddings(model, val_loader)
    calculate_recall(img_embs, txt_embs)
    
    # 4. Qualitative Visualization
    # Use UNSHUFFLED dataset for visualization scanning to keep index alignment simple
    val_dataset_viz = CocoDataset(mode="val")
    visualize_retrieval(model, val_dataset_viz, "a person playing sports", k=5)
    visualize_retrieval(model, val_dataset_viz, "a cute animal", k=5)
    visualize_retrieval(model, val_dataset_viz, "food on a table", k=5)

if __name__ == "__main__":
    main()