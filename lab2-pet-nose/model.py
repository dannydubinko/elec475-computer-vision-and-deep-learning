import torch
import torch.nn as nn
# --- ADDED ---
import torchvision.models as models
# --- END ADDED ---

class SnoutNet(nn.Module):
    """
    Implementation of the SnoutNet model architecture as specified.
    (Original SnoutNet code remains unchanged)
    ... (rest of original SnoutNet class code) ...
    """
    def __init__(self):
        super(SnoutNet, self).__init__()

        # --- Convolutional Blocks ---
        # Each block = Conv(3x3) + ReLU + MaxPool

        # Block 1: Input (3, 227, 227) -> Output (64, 57, 57)
        # Conv(3, 64, k=3, s=1, p=1) -> (64, 227, 227)
        # MaxPool(k=3, s=4) -> floor((227 - 3) / 4) + 1 = 56 + 1 = 57
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=4)
        )

        # Block 2: Input (64, 57, 57) -> Output (128, 15, 15)
        # Conv(64, 128, k=3, s=1, p=1) -> (128, 57, 57)
        # MaxPool(k=1, s=4) -> floor((57 - 1) / 4) + 1 = 14 + 1 = 15
        # Note: A 1x1 MaxPool with stride 4 is unusual, but required to get 15x15
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=4)
        )

        # Block 3: Input (128, 15, 15) -> Output (256, 4, 4)
        # Conv(128, 256, k=3, s=1, p=1) -> (256, 15, 15)
        # MaxPool(k=3, s=4) -> floor((15 - 3) / 4) + 1 = 3 + 1 = 4
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=4)
        )

        # --- Fully Connected Layers ---
        # The flattened size from (256, 4, 4) is 256 * 4 * 4 = 4096

        # Per specification: "Fully connected layer but before it is a RELU"

        # FC Block 1
        self.fc1_relu = nn.ReLU()
        # NOTE: If input size was 224x224, this would be 2304 features
        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=1024)

        # FC Block 2
        self.fc2_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)

        # Output Layer
        # Maps the 1024 features to the final 2 output coordinates
        self.fc_out = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        """
        Defines the forward pass of the SnoutNet model.
        """
        # Pass through convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Flatten the output for the fully connected layers
        # x.shape is (batch_size, 256, 4, 4)
        # We flatten it to (batch_size, 4096)
        x = torch.flatten(x, 1) # Keep batch dimension (0)

        # Pass through fully connected layers (with ReLU *before* each)
        x = self.fc1_relu(x)
        x = self.fc1(x)

        x = self.fc2_relu(x)
        x = self.fc2(x)

        # Pass through the final output layer (no activation)
        x = self.fc_out(x)

        return x

# --- ADDED: SnoutNetAlex ---
class SnoutNetAlex(nn.Module):
    """
    SnoutNet using a pretrained AlexNet backbone.
    """
    def __init__(self):
        super(SnoutNetAlex, self).__init__()
        # Load pretrained AlexNet
        alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

        # Use the feature extractor part
        self.features = alexnet.features

        # AlexNet features output shape (batch_size, 256, 6, 6) for 227x227 input
        # Flattened size = 256 * 6 * 6 = 9216
        in_features_regressor = 9216

        # Replace the classifier with a regression head
        # Example: Linear -> ReLU -> Dropout -> Linear (Output 2)
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.5), # From original AlexNet classifier
            nn.Linear(in_features_regressor, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), # From original AlexNet classifier
            nn.Linear(4096, 1024), # Reduced intermediate size
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2) # Output layer for (x, y) coordinates
        )

    def forward(self, x):
        x = self.features(x)
        # AlexNet needs adaptive avg pooling before classifier typically,
        # but the original classifier handles the flatten based on expected 6x6.
        # Let's keep the flatten approach.
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x

# --- ADDED: SnoutNetVGG ---
class SnoutNetVGG(nn.Module):
    """
    SnoutNet using a pretrained VGG16 backbone.
    """
    def __init__(self):
        super(SnoutNetVGG, self).__init__()
        # Load pretrained VGG16
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Use the feature extractor part
        self.features = vgg16.features

        # VGG16 features output shape (batch_size, 512, 7, 7) for 227x227 input
        # Flattened size = 512 * 7 * 7 = 25088
        in_features_regressor = 25088

        # Replace the classifier with a regression head
        self.regressor = nn.Sequential(
            nn.Linear(in_features_regressor, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024), # Reduced intermediate size
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2) # Output layer for (x, y) coordinates
        )

    def forward(self, x):
        x = self.features(x)
        # VGG uses adaptive avg pooling typically, but let's flatten
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x
# --- END ADDED ---


# --- Test Script ---
if __name__ == "__main__":

    # --- Test Original SnoutNet ---
    print("\n--- Testing Original SnoutNet ---")
    model_custom = SnoutNet()
    # print(model_custom)
    try:
        dummy_input = torch.randn(2, 3, 227, 227) # Batch size 2
        print(f"Input shape: {dummy_input.shape}")
        model_custom.eval()
        with torch.no_grad():
            output = model_custom(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (2, 2), "Original SnoutNet shape mismatch!"
        print("SUCCESS: Original SnoutNet output shape OK.")
    except Exception as e:
        print(f"FAILURE: Error during Original SnoutNet test: {e}")

    # --- Test SnoutNetAlex ---
    print("\n--- Testing SnoutNetAlex ---")
    model_alex = SnoutNetAlex()
    # print(model_alex) # Can be very long
    try:
        dummy_input = torch.randn(2, 3, 227, 227) # Batch size 2
        print(f"Input shape: {dummy_input.shape}")
        model_alex.eval()
        with torch.no_grad():
            output = model_alex(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (2, 2), "SnoutNetAlex shape mismatch!"
        print("SUCCESS: SnoutNetAlex output shape OK.")
    except Exception as e:
        print(f"FAILURE: Error during SnoutNetAlex test: {e}")

    # --- Test SnoutNetVGG ---
    print("\n--- Testing SnoutNetVGG ---")
    model_vgg = SnoutNetVGG()
    # print(model_vgg) # Can be very long
    try:
        dummy_input = torch.randn(2, 3, 227, 227) # Batch size 2
        print(f"Input shape: {dummy_input.shape}")
        model_vgg.eval()
        with torch.no_grad():
            output = model_vgg(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (2, 2), "SnoutNetVGG shape mismatch!"
        print("SUCCESS: SnoutNetVGG output shape OK.")
    except Exception as e:
        print(f"FAILURE: Error during SnoutNetVGG test: {e}")

    print("\n--- Model Tests Complete ---")