import sys
import os
import torchvision.transforms as transforms

# ---------------------------------------------------------
# 1. CONFIG BAĞLANTISI
# ---------------------------------------------------------
# src/utils/config.py dosyasını görebilmek için yol ekliyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import config

# ImageNet standartları (Değişmez sabitler olduğu için burada kalabilir)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(mode: str) -> transforms.Compose:
    """
    Verilen moda (train/val) göre ön işleme borusunu hazırlar.
    """
    
    # ---------------------------------------------------------
    # 2. CONFIG'DEN BOYUT ÇEKME (DÜZELTME)
    # ---------------------------------------------------------
    # Elle 224 yazmak yerine config dosyasındaki ayarı kullanıyoruz.
    resize_transform = transforms.Resize(config.IMAGE_SIZE)
    
    # Normalizasyon
    normalize_transform = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

    if mode == 'train':
        # Train Modu: Veri Artırma (Data Augmentation) VAR
        return transforms.Compose([
            resize_transform,
            transforms.RandomHorizontalFlip(p=0.5), # %50 ihtimalle çevir
            transforms.RandomRotation(degrees=10),  # +/- 10 derece döndür
            transforms.ToTensor(),
            normalize_transform,
        ])
        
    elif mode == 'val':
        # Val Modu: Sadece Boyutlandırma ve Normalizasyon
        return transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            normalize_transform,
        ])
        
    else:
        raise ValueError(f"Hatalı mod: '{mode}'. Sadece 'train' veya 'val' olabilir.")

# ---------------------------------------------------------
# 3. SMOKE TEST (Duman Testi)
# ---------------------------------------------------------
# Bu dosya tek başına çalıştırılırsa transformları ekrana basar.
if __name__ == "__main__":
    print(f"🔧 Transform Testi Başladı...")
    try:
        train_t = get_transforms('train')
        print(f"✅ Train Transform Zinciri:\n{train_t}")
        print("-" * 30)
        val_t = get_transforms('val')
        print(f"✅ Val Transform Zinciri:\n{val_t}")
        print("\n🎉 BAŞARILI: Transformlar config ile uyumlu çalışıyor.")
    except Exception as e:
        print(f"❌ HATA: {e}")