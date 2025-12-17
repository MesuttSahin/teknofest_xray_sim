import torch
import torch.nn as nn
from torchvision import models
from src.utils import config 


num_classes = config.NUM_CLASSES

class ChestXRayResNet(nn.Module):
    def __init__(self, num_classes=14): 
        super(ChestXRayResNet, self).__init__()
        
        # ResNet50'yi önceden eğitilmiş ağırlıklarla yükle
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        
        # Son katmanın (fc) giriş özellik sayısını al (ResNet-50 için 2048'dir)
        num_ftrs = self.model.fc.in_features 
        
        # Son katmanı (fc) istenen sınıf sayısına (14) uygun yeni bir nn.Linear katmanıyla değiştir
        # Bu, transfer öğrenimi (fine-tuning) için modelin uyarlanmasıdır.
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        # Standart ileri besleme (forward) akışı
        # Girdi (x), ResNet modelinden geçirilir.
        return self.model(x)

