import torch
import torch.nn as nn
from torchvision import models
from src.utils import config


class ChestXRayResNet(nn.Module):
    def __init__(self, num_classes=14):
        super(ChestXRayResNet, self).__init__()

        # ResNet50'yi önceden eğitilmiş ağırlıklarla yükle
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # --- A. Dondurma (Freezing) İşlemi ---
        # Layer4 ve FC hariç her şeyi dondur (Kapasite kısıtlama)
        for name, param in self.model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        num_ftrs = self.model.fc.in_features

        # --- B. Dropout Ekleme ---
        # %50 Dropout ekleyerek modelin ezber yapmasını (overfitting) engelliyoruz
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)