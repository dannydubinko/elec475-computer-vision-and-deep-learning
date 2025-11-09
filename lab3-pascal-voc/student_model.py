import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP)
    As described in the DeepLabV3 paper and lab hint.
    """
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        
        modules = []
        # 1. 1x1 Convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        # 2. Atrous Convolutions
        for rate in rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        # 3. Image-level Pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        self.convs = nn.ModuleList(modules)
        
        # 4. Final 1x1 conv to project all concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # Dropout from lab hint
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        
        # Resize image pooling output to match other feature map sizes
        img_pool_feat = _res[-1]
        _res[-1] = F.interpolate(img_pool_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat(_res, dim=1)
        return self.project(x)


class CompactStudentModel(nn.Module):
    """
    Student model based on the "Example Architecture"
    - Backbone: MobileNetV3-Small (pretrained)
    - Context: ASPP
    - Decoder: Bilinear upsampling + skip connection
    """
    def __init__(self, num_classes=21):
        super(CompactStudentModel, self).__init__()

        # --- 1. Backbone (Encoder) ---
        # Load pretrained MobileNetV3-Small
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # Define the nodes to extract features from, matching lab hints:
        # 'features.3': stride 4, 24 channels (low)
        # 'features.6': stride 8, 40 channels (mid)
        # 'features.12': stride 16, 576 channels (high)
        return_nodes = {
            'features.3': 'low',
            'features.6': 'mid',
            'features.12': 'high',
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

        # --- 2. Context Module (ASPP) ---
        # ASPP with rates {1, 6, 12, 18}
        # Input to ASPP is the 'high' feature map (576 channels)
        aspp_in_channels = 576
        aspp_out_channels = 256 # A common design choice
        self.aspp = ASPP(aspp_in_channels, aspp_out_channels, rates=[1, 6, 12, 18])

        # --- 3. Decoder ---
        # Skip connection from 'low-level' features (24 channels)
        low_level_channels = 24
        
        # 1x1 conv to reduce channels of low-level features before concatenation
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution block after concatenating ASPP + low-level features
        self.conv_final = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # --- 4. Classifier Head ---
        # Final 1x1 classifier to 21 classes
        self.classifier = nn.Conv2d(256, num_classes, 1)

        # --- 5. Adapters for Feature-based KD (for Step 2.4) ---
        # These 1x1 convs will match student feature channels to teacher's
        # Teacher (ResNet50): layer1 (256), layer2 (512), layer3 (1024)
        # Student (MobileNet): low (24), mid (40), high (576)
        self.adapt_low = nn.Conv2d(24, 256, 1)
        self.adapt_mid = nn.Conv2d(40, 512, 1)
        self.adapt_high = nn.Conv2d(576, 1024, 1)

    def forward(self, x):
        # We need the original input shape for final upsampling
        input_shape = x.shape[2:]

        # --- Encoder ---
        # Get features from the backbone
        features = self.backbone(x)
        low_feat = features['low']
        mid_feat = features['mid']
        high_feat = features['high']
        
        # --- Context ---
        # Pass high-level features through ASPP
        x = self.aspp(high_feat)
        
        # --- Decoder ---
        # 1. Upsample ASPP output to match low_feat spatial dimensions (stride 4)
        x_upsampled = F.interpolate(x, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 2. Process low-level features
        low_feat_processed = self.conv_low(low_feat)
        
        # 3. Concatenate upsampled ASPP features and processed low-level features
        x = torch.cat([x_upsampled, low_feat_processed], dim=1)
        
        # 4. Pass through final conv block
        x = self.conv_final(x)
        
        # 5. Classifier
        x = self.classifier(x)
        
        # 6. Final upsample to original image size
        x_out = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        # For Knowledge Distillation (Step 2.4), we also return intermediate
        # features *during training*.
        if self.training:
            # Adapt features for KD loss
            kd_low = self.adapt_low(low_feat)
            kd_mid = self.adapt_mid(mid_feat)
            kd_high = self.adapt_high(high_feat)
            
            # Return main output and a dict of adapted features
            return x_out, {'low': kd_low, 'mid': kd_mid, 'high': kd_high}
        
        # During inference, just return the main output
        return x_out