print("--- Script Starting ---", flush=True)
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPProcessor

# Import from augmentation files
from dataset_augmentation import CocoDataset, CocoCollator
from model_augmentation import CLIPModel

print("--- Imports Complete ---", flush=True)

# --- Config ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Path to the specific epoch you want to evaluate
CHECKPOINT_PATH = "checkpoints_augmentation/clip_epoch_5.pt" 
MAX_EVAL_SAMPLES = 2000
BATCH_SIZE = 32

def get_embeddings(model, dataloader):
    model.eval()
    img_embs = []
    txt_embs = []
    count = 0
    
    print(f"Computing embeddings (Limit: {MAX_EVAL_SAMPLES})...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pixel_values = batch["pixel_values"].to(DEVICE)
            
            img, txt = model(pixel_values, input_ids, attention_mask)
            img_embs.append(img.cpu())
            txt_embs.append(txt.cpu())
            
            count += input_ids.size(0)
            if count >= MAX_EVAL_SAMPLES: break
            
    return torch.cat(img_embs)[:MAX_EVAL_SAMPLES], torch.cat(txt_embs)[:MAX_EVAL_SAMPLES]

def calculate_recall(image_embs, text_embs, k_values=[1, 5, 10]):
    print("\n--- Computing Similarity Matrix & Recall ---")
    sim_matrix = image_embs @ text_embs.T
    num_samples = sim_matrix.shape[0]
    targets = torch.arange(num_samples)
    
    print(f"Evaluated on {num_samples} pairs.")
    
    # Image-to-Text
    _, i2t_indices = sim_matrix.topk(max(k_values), dim=1)
    for k in k_values:
        correct = i2t_indices[:, :k].eq(targets.view(-1, 1).expand(-1, k))
        acc = correct.sum().float() / num_samples
        print(f"Image-to-Text Recall@{k}: {acc:.4f}")

    # Text-to-Image
    _, t2i_indices = sim_matrix.topk(max(k_values), dim=0)
    t2i_indices = t2i_indices.T 
    for k in k_values:
        correct = t2i_indices[:, :k].eq(targets.view(-1, 1).expand(-1, k))
        acc = correct.sum().float() / num_samples
        print(f"Text-to-Image Recall@{k}: {acc:.4f}")

def zero_shot_classification(model, dataset, image_idx, candidate_classes):
    print(f"\n--- Zero-Shot Classification (Image Index {image_idx}) ---")
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=candidate_classes, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        text_embs = model.text_encoder(inputs["input_ids"], inputs["attention_mask"]) 
        
        pixel_values, caption, orig_image = dataset[image_idx]
        pixel_values = pixel_values.unsqueeze(0).to(DEVICE) 
        
        img_emb = model.image_encoder(pixel_values)
        
        # Calculate probability via Softmax
        scores = (img_emb @ text_embs.T).squeeze()
        probs = scores.softmax(dim=0)
        
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_image)
    plt.title(f"True Caption: {caption[:30]}...")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    y_pos = range(len(candidate_classes))
    plt.barh(y_pos, probs.cpu().numpy())
    plt.yticks(y_pos, candidate_classes)
    plt.xlabel("Probability")
    plt.title("Predicted Class")
    
    plt.tight_layout()
    plt.savefig(f"zero_shot_idx_{image_idx}.png")
    print(f"Saved zero-shot result to zero_shot_idx_{image_idx}.png")

def visualize_text_to_image(model, dataset, query, k=5):
    print(f"\n--- Text-to-Image Retrieval: '{query}' ---")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=[query], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        text_emb = model.text_encoder(inputs["input_ids"], inputs["attention_mask"])
    
    # We scan more items to ensure we find decent matches
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=CocoCollator())
    all_img_embs = []
    
    print("Scanning dataset for matches...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 50: break # Scan ~1600 images
            pixel_values = batch["pixel_values"].to(DEVICE)
            all_img_embs.append(model.image_encoder(pixel_values).cpu())
            
    all_img_embs = torch.cat(all_img_embs)
    sims = (text_emb.cpu() @ all_img_embs.T).squeeze()
    
    # --- UPDATED LOGIC: Filter Duplicates ---
    # Get top 50 candidates to allow room for filtering
    values, indices = sims.topk(min(50, len(sims)))
    
    seen_filenames = set()
    unique_indices = []
    unique_scores = []
    
    for i, idx in enumerate(indices):
        idx_val = idx.item()
        # Access the filename directly from the dataset pairs
        # pair is (filename, caption)
        filename = dataset.pairs[idx_val][0]
        
        if filename not in seen_filenames:
            seen_filenames.add(filename)
            unique_indices.append(idx_val)
            unique_scores.append(values[i].item())
            
        if len(unique_indices) == k:
            break
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Query: {query}")
    for i, idx in enumerate(unique_indices):
        _, _, orig_image = dataset[idx]
        ax = plt.subplot(1, k, i+1)
        ax.imshow(orig_image)
        ax.axis('off')
        ax.set_title(f"Score: {unique_scores[i]:.2f}")
    plt.savefig(f"retrieval_{query.replace(' ', '_')}.png")
    print(f"Saved visualization for '{query}'.")

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Please run training first.", flush=True)
        return

    print(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}...", flush=True)
    model = CLIPModel()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Use validation set
    val_dataset = CocoDataset(mode="val")
    collator = CocoCollator()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=2)
    
    # 1. Metrics
    img_embs, txt_embs = get_embeddings(model, val_loader)
    calculate_recall(img_embs, txt_embs)
    
    # 2. Visualizations (Requested Queries)
    queries = [
        "food on the table",
        "a person playing sports",
        "a cute animal"
    ]
    
    for q in queries:
        visualize_text_to_image(model, val_dataset, q)
    
    # 3. Zero Shot
    # We use classes that match the queries above to see if it can classify them back
    classes = ["food", "sports", "animal", "vehicle", "furniture"]
    idx_to_test = 50 if len(val_dataset) > 50 else 0
    zero_shot_classification(model, val_dataset, image_idx=idx_to_test, candidate_classes=classes)

if __name__ == "__main__":
    main()