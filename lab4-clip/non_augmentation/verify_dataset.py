# Save this file as: verify_dataset.py
# Located at: /content/drive/MyDrive/Colab_Projects/elec475-computer-vision-and-deep-learning/lab4-clip/verify_dataset.py

import random
import textwrap

import matplotlib.pyplot as plt
from dataset import CocoDataset  # Import the class from dataset.py

def verify_dataset_integrity(num_samples: int = 5):
    """
    Loads the validation dataset and displays a number of
    random image-caption pairs to verify data integrity.
    
    """
    print("Initializing verification...")
    
    # 1. Load the validation dataset
    # We use 'val' set as it's smaller and good for checks
    try:
        val_dataset = CocoDataset(mode='val')
    except FileNotFoundError:
        print("\n--- VERIFICATION FAILED ---")
        print("Could not start verification. See error above.")
        print("Please ensure you have the *caption* files, not the *instance* files.")
        print("Required: 'captions_val2014.json'")
        return

    if len(val_dataset) == 0:
        print("\n--- VERIFICATION FAILED ---")
        print("Dataset loaded, but 0 pairs were found.")
        print("This almost certainly means the annotation file was wrong.")
        return

    print(f"\nSuccessfully loaded {len(val_dataset)} validation pairs.")
    print(f"Displaying {num_samples} random samples...\n")
    
    # 2. Get random indices
    random_indices = [random.randint(0, len(val_dataset) - 1) for _ in range(num_samples)]
    
    # 3. Display the samples
    plt.figure(figsize=(15, 3 * num_samples))
    for i, idx in enumerate(random_indices):
        
        # Get the raw image and caption 
        image, caption = val_dataset[idx]
        
        ax = plt.subplot(num_samples, 1, i + 1)
        ax.imshow(image)
        ax.axis('off')
        
        # Wrap the text for cleaner plotting
        wrapped_title = textwrap.fill(f"Caption: {caption}", width=80)
        ax.set_title(wrapped_title, ha='left', x=0)

    plt.tight_layout()
    
    # In a Colab/Jupyter environment, this will display the plot.
    # If running as a .py script, it will save it.
    save_path = "dataset_verification.png"
    plt.savefig(save_path)
    print(f"Saved verification plot to: {save_path}")
    # plt.show() # Uncomment this if running in an interactive notebook

if __name__ == "__main__":
    verify_dataset_integrity(num_samples=5)