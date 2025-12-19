import sys
import os
import torchvision.transforms as transforms

# ---------------------------------------------------------
# 1. CONFIG BAĞLANTISI
# ---------------------------------------------------------
# src/utils/config.py dosyasını görebilmek için yol ekliyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import config

# ImageNet standartları
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(mode: str) -> transforms.Compose:
    """
    Verilen moda (train/val) göre ön işleme borusunu hazırlar.
    """

    # Ortak Normalizasyon İşlemi
    normalize_transform = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

    if mode == 'train':
        # ---------------------------------------------------------
        # TRAIN MODU: Veri Artırma (Data Augmentation)
        # ---------------------------------------------------------
        return transforms.Compose([
            # 1. Kesip Büyütme: Modeli resmin tamamına değil, detaylara odaklanmaya zorlar.
            # config.IMAGE_SIZE (örn: 224) hedef boyuttur.
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),

            # 2. Döndürme: Hasta pozisyonundaki hafif sapmaları simüle eder (15 derece).
            transforms.RandomRotation(degrees=15),

            # 3. Işık/Kontrast: Farklı röntgen cihazlarının görüntü farklarını simüle eder.
            transforms.ColorJitter(brightness=0.2, contrast=0.2),

            # 4. Standart Çevirme: Yatay düzlemde aynalama.
            transforms.RandomHorizontalFlip(p=0.5),

            # 5. Tensor Dönüşümü ve Normalizasyon
            transforms.ToTensor(),
            normalize_transform,
        ])

    elif mode == 'val':
        # ---------------------------------------------------------
        # VAL MODU: Sadece Standartlaştırma
        # ---------------------------------------------------------
        # KRİTİK: Val verisi bozulmaz, sadece boyutu ayarlanır.
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize_transform,
        ])

    else:
        raise ValueError(f"Hatalı mod: '{mode}'. Sadece 'train' veya 'val' olabilir.")


# ---------------------------------------------------------
# 3. SMOKE TEST (Duman Testi)
# ---------------------------------------------------------
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