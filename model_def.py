import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


transform = transforms.Compose([
    transforms.Resize((224, 224)),   # change ONLY if you trained with another size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# FFT Transform
def fft_transform(img_tensor):
    fft = torch.fft.fft2(img_tensor)
    fft = torch.fft.fftshift(fft)
    magnitude = torch.log1p(torch.abs(fft))
    return magnitude


# EfficientNet-B3 Feature Extractor
class EfficientNetB3_Features(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # (B,1536)
  
    
    # Attention Fusion
class AttentionFusion(nn.Module):
    def __init__(self, dim=1536):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * 2),
            nn.Sigmoid()
        )

    def forward(self, rgb, fft):
        w = self.attn(torch.cat([rgb, fft], dim=1))
        w_rgb, w_fft = torch.split(w, rgb.size(1), dim=1)
        return torch.cat([rgb * w_rgb, fft * w_fft], dim=1)
    
    
class RGBFFT_Attention_EfficientNetB3(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.rgb_net = EfficientNetB3_Features()
        self.fft_net = EfficientNetB3_Features()
        self.fusion = AttentionFusion()

        self.classifier = nn.Sequential(
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb):
        fft = torch.stack([fft_transform(img) for img in rgb])
        rgb_f = self.rgb_net(rgb)
        fft_f = self.fft_net(fft)
        fused = self.fusion(rgb_f, fft_f)
        return self.classifier(fused)