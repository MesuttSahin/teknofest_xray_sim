import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from models.model import ChestXRayResNet
from data.transforms import get_transforms
from src.data.dataset import ChestXRayDataset


def get_val_loader():
    # 1. Val transformlarını al
    val_ts = get_transforms('val')

    # 2. Dataset'i val_list üzerinden oluştur
    # (Buradaki parametre isimleri projerine göre 'list_file' veya 'csv_file' olabilir)
    val_dataset = ChestXRayDataset(
        csv_file='data/processed/val_list.csv',  # Parametre ismini 'csv_file' yaptık
        transform=get_transforms('val')  # img_dir parametresini sildik
    )

    # 3. Loader (Kritik: shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return val_loader


def evaluate():
    # Cihaz Ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Değerlendirme Başlıyor. Cihaz: {device}")

    # 1. Modeli Hazırla
    model = ChestXRayResNet().to(device)
    # Kritik: Validation hatasının en düşük olduğu epoch'u seç
    model_path = 'models/chest_xray_model_ep1.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Model yüklendi: {model_path}")
    model.eval()

    # 2. Veri Yükleyici (Kritik: shuffle=False)
    # train_loader, val_loader = get_loaders(...)

    all_probs = []
    all_labels = []

    print("📊 Tahminler ve gerçek etiketler toplanıyor...")
    with torch.no_grad():
        for images, labels in get_val_loader():
            images = images.to(device)

            outputs = model(images)
            # Analist için Sigmoid ile olasılığa (0-1) çeviriyoruz
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. Veri İhracı (Numpy Save)
    if not os.path.exists('logs'):
        os.makedirs('logs')

    np.save('logs/predictions.npy', np.array(all_probs))
    np.save('logs/true_labels.npy', np.array(all_labels))

    print("🎯 Analist için logs/predictions.npy ve logs/true_labels.npy dosyaları hazır!")


if __name__ == "__main__":
    evaluate()