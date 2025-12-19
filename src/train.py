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

from src.utils import config
from src.data.dataset import ChestXRayDataset
from src.utils.plots import plot_loss_curves
from src.models.model import ChestXRayResNet as ModelClass
from src.data.transforms import get_transforms


def train():
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')

    train_csv_path = os.path.join(config.PROCESSED_DATA_DIR, 'train_list.csv')
    val_csv_path = os.path.join(config.PROCESSED_DATA_DIR, 'val_list.csv')

    train_dataset = ChestXRayDataset(csv_file=train_csv_path, transform=train_transform)
    val_dataset = ChestXRayDataset(csv_file=val_csv_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # --- 2. Model ve Weighted Loss Kurulumu ---
    model = ModelClass(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Config'den gelen o hassas ağırlıkları Loss fonksiyonuna veriyoruz
    pos_weights = torch.tensor(config.POS_WEIGHTS).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # AdamW ile ağırlık aşınması (Weight Decay) ekleyerek ezberlemeyi (overfitting) zorlaştırıyoruz
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    # Scheduler: Val Loss 2 epoch boyunca düşmezse LR'yi 10'a böler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR
    )

    train_losses, val_losses = [], []

    # --- 3. Eğitim Döngüsü ---
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

        for images, labels in loop:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # --- Validasyon ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for v_imgs, v_lbls in val_loader:
                v_imgs, v_lbls = v_imgs.to(config.DEVICE), v_lbls.to(config.DEVICE)
                v_out = model(v_imgs)
                val_running_loss += criterion(v_out, v_lbls).item()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_losses.append(val_epoch_loss)

        # KRİTİK: Scheduler'a her epoch sonunda val_loss bilgisini veriyoruz
        scheduler.step(val_epoch_loss)

        print(f"\t🏁 Epoch {epoch + 1} Bitti -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

        # Kaydetme ve Grafik (Analistin fonksiyonu)
        os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.MODEL_OUTPUT_DIR, f"v2_model_ep{epoch + 1}.pth"))
        plot_loss_curves(train_losses, val_losses, save_path=os.path.join("logs", "v2_loss_curve.png"))

    print("✅ V2 Eğitimi (Daha Az Ezberleme, Daha Dengeli Öğrenme) Tamamlandı.")


if __name__ == "__main__":
    train()