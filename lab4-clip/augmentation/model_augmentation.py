import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import CLIPTextModel
from torchvision.models import resnet50, ResNet50_Weights

# --- Constants ---
IMAGE_EMBED_DIM = 512  
TEXT_MODEL_NAME = "openai/clip-vit-base-patch32"

class ProjectionHead(nn.Module):
    """
    Implements the projection head with Dropout (MODIFICATION 2).
    """
    def __init__(self, input_dim: int = 2048, mid_dim: int = 1024, output_dim: int = IMAGE_EMBED_DIM, dropout: float = 0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout), # Regularization
            nn.Linear(mid_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
            
        self.backbone.fc = nn.Identity()
        self.projection = ProjectionHead(input_dim=2048, output_dim=IMAGE_EMBED_DIM)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  
        embeddings = self.projection(features) 
        return F.normalize(embeddings, p=2, dim=1)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(TEXT_MODEL_NAME)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output 
        return F.normalize(embeddings, p=2, dim=1)


class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=True)
        self.text_encoder = TextEncoder()
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor, text_mask: torch.Tensor):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(text_ids, text_mask)
        return image_embeddings, text_embeddings