import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import argparse
from collections import OrderedDict

# --- PATH AYARLARI ---
# Scriptin çalıştığı yerden proje kök dizinini buluyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
project_root = os.path.dirname(current_dir)  # proje klasörü
sys.path.append(project_root)

try:
    from src.utils import config
except ImportError:
    print("❌ HATA: Config dosyası (src/utils/config.py) bulunamadı veya yüklenemedi!")
    exit(1)


# ---------------------

def get_validation_transforms():
    """
    Sadece resmi modele uygun hale getiren transformlar.
    DİKKAT: Burada veri çoğaltma (döndürme, flip vs.) ASLA yapılmaz.
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path):
    """Eğitilmiş modeli yükler ve ağırlıkları set eder."""
    from torchvision import models

    print(f"[INFO] Model mimarisi hazırlanıyor ve ağırlıklar yükleniyor...")

    # 1. Model Mimarisi (ResNet50)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)

    try:
        # 2. Ağırlıkları Yükle
        state_dict = torch.load(model_path, map_location=config.DEVICE)

        # --- FIX: Key İsimlerini Düzeltme ---
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k

            # 1. 'model.' önekini temizle (Varsa)
            if name.startswith('model.'):
                name = name[6:]

            # 2. 'fc.1.' sorununu düzelt (fc.1.weight -> fc.weight)
            # Eğitimde Sequential kullandığın için fc.1 olmuş, bunu fc yapıyoruz.
            if "fc.1." in name:
                name = name.replace("fc.1.", "fc.")

            new_state_dict[name] = v
        # ---------------------------------------

        # strict=False yapmıyoruz ki gerçekten doğru yüklediğinden emin olalım
        model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ Model başarıyla yüklendi: {model_path}")

    except FileNotFoundError:
        print(f"❌ HATA: Model dosyası bulunamadı: {model_path}")
        exit(1)
    except Exception as e:
        print(f"❌ Model yüklenirken beklenmedik hata: {e}")
        # Detay görmek için hatayı tekrar fırlatabiliriz veya exit
        exit(1)

    model.to(config.DEVICE)
    model.eval()
    return model


def predict_image(model, image_path):
    """Tek bir resim için tahmin üretir."""
    # Dosyanın gerçekten var olup olmadığını kontrol edelim
    if not os.path.exists(image_path):
        print(f"❌ HATA: Dosya bu konumda bulunamadı!")
        print(f"   Aranan yol: {image_path}")
        return None

    try:
        # X-Ray gri tonlamalı olsa bile ResNet 3 kanal (RGB) bekler
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        # BURAYI GÜNCELLEDİK: Hata detayını (e) yazdırıyoruz
        print(f"❌ Resim dosyası bozuk veya okunamıyor: {image_path}")
        print(f"   Hata Detayı: {e}")
        return None

    # Transform ve Boyut Ayarlama
    transform = get_validation_transforms()
    img_tensor = transform(image)

    # Batch boyutu ekle: [C, H, W] -> [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(config.DEVICE)

    # Tahmin
    with torch.no_grad():
        outputs = model(img_tensor)
        # Logitleri olasılığa (0-1 arası) çevir
        probs = torch.sigmoid(outputs)

    # Tensor'u listeye çevir
    probs = probs.cpu().squeeze().tolist()

    # Eğer tek sınıf varsa (Binary) probs bir sayı döner, listeye çevirelim
    if not isinstance(probs, list):
        probs = [probs]

    return probs


def generate_report(probabilities, threshold=0.5):
    """Olasılıkları insan tarafından okunabilir rapora çevirir."""
    findings = []

    for idx, prob in enumerate(probabilities):
        if prob > threshold:
            percentage = prob * 100
            disease_name = config.CLASS_NAMES[idx]
            findings.append(f"%{percentage:.0f} {disease_name}")

    print("\n" + "-" * 30)
    if findings:
        result_string = ", ".join(findings)
        print(f"Hasta: {result_string}")
    else:
        print("Hasta: No Finding (Sağlıklı)")
    print("-" * 30 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-Ray Canlı Test Aracı")

    # Argümanlar
    parser.add_argument('--image', type=str, required=True, help="Test edilecek resmin dosya yolu")
    parser.add_argument('--model', type=str, default=config.BEST_MODEL_PATH, help="Kullanılacak .pth model dosyası")

    args = parser.parse_args()

    # 1. Modeli Yükle
    model = load_model(args.model)

    # 2. Tahmin Yap
    probs = predict_image(model, args.image)

    # 3. Raporla
    if probs:
        generate_report(probs)