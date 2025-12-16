import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Root dizini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Modülleri çağır
from src.utils import config
from src.data.dataset import ChestXRayDataset
from src.utils import plot_loss_curves  # Analistin fonksiyonu buradan geliyor
from src.models.model import ChestXRayResNet as ModelClass # Model Mimarı burayı düzenlemeli dikkat 

def train():    
    # 1. Transform İşlemleri
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Datasetlerin Yüklenmesi
    train_csv_path = os.path.join(config.PROCESSED_DATA_DIR, 'train_list.csv')
    val_csv_path = os.path.join(config.PROCESSED_DATA_DIR, 'val_list.csv')

    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        print(f"❌ HATA: CSV dosyaları bulunamadı:\n  {train_csv_path}\n  {val_csv_path}")
        return

    train_dataset = ChestXRayDataset(csv_file=train_csv_path, transform=train_transform)
    val_dataset = ChestXRayDataset(csv_file=val_csv_path, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS
    )

    print(f"🚀 Eğitim Başlıyor... Cihaz: {config.DEVICE}") # cuda config'de tanımlı.
    
    # 3. Model Kurulumu
    model = ModelClass(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    # 4. Loss ve Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Loglar için liste
    train_losses = []
    val_losses = []

    # 5. Eğitim Döngüsü
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for images, labels in loop:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Epoch Sonu Hesaplamaları
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # --- Validasyon Adımı ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for v_imgs, v_lbls in val_loader:
                v_imgs = v_imgs.to(config.DEVICE)
                v_lbls = v_lbls.to(config.DEVICE)
                v_out = model(v_imgs)
                v_loss = criterion(v_out, v_lbls)
                val_running_loss += v_loss.item()
        
        val_epoch_loss = val_running_loss / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(f"\t🏁 Epoch {epoch+1} Özeti -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

        # --- Model Kaydetme ---
        os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(config.MODEL_OUTPUT_DIR, f"chest_xray_model_ep{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        
        # --- Grafik Çizimi (Her Epoch Sonunda Güncellenir) ---
        # Analistin fonksiyonunu çağırdım
        plot_save_path = os.path.join("logs", "loss_curve.png")
        plot_loss_curves(train_losses, val_losses, save_path=plot_save_path)

    print("✅ Tüm eğitim süreci tamamlandı.")

if __name__ == "__main__":
    train()