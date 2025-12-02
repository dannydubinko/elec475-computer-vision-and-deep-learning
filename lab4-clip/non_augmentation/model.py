import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import CLIPTextModel
from torchvision.models import resnet50, ResNet50_Weights

# --- Constants ---
IMAGE_EMBED_DIM = 512  # 512-dim CLIP embedding space
TEXT_MODEL_NAME = "openai/clip-vit-base-patch32"

class ProjectionHead(nn.Module):
    """
    Implements the projection head for mapping image features.
    Architecture: Linear(2048) -> GELU -> Linear(512)
    """
    def __init__(self, input_dim: int = 2048, mid_dim: int = 1024, output_dim: int = IMAGE_EMBED_DIM):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ImageEncoder(nn.Module):
    """
    ResNet50 Backbone + Projection Head.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # 1. Load pretrained ResNet50
        if pretrained:
            print("Loading pretrained ResNet50...")
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
            
        # 2. Remove classification head (fc), use features from avgpool
        self.backbone.fc = nn.Identity()
        
        # 3. Projection head
        self.projection = ProjectionHead(input_dim=2048, output_dim=IMAGE_EMBED_DIM)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, 2048)
        features = self.backbone(x)  
        # (N, 512)
        embeddings = self.projection(features) 
        # L2 Normalize
        return F.normalize(embeddings, p=2, dim=1)


class TextEncoder(nn.Module):
    """
    Frozen CLIP Text Encoder.
    """
    def __init__(self):
        super().__init__()
        print(f"Loading frozen text encoder: {TEXT_MODEL_NAME}...")
        self.model = CLIPTextModel.from_pretrained(TEXT_MODEL_NAME)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output # (N, 512)
        return F.normalize(embeddings, p=2, dim=1)


class CLIPModel(nn.Module):
    """
    Main CLIP wrapper.
    """
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=True)
        self.text_encoder = TextEncoder()
        
        # Learnable temperature (initialized to 0.07)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor, text_mask: torch.Tensor):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(text_ids, text_mask)
        return image_embeddings, text_embeddings