import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
from tqdm import tqdm

# --- 1. AYARLAR ---

# ⚠️ BURAYA DİKKAT: Resimlerin olduğu klasör yolunu buraya yapıştır
# Örnek: r"D:\Downloads\Compressed\archive"
DATA_ROOT = r"D:\Downloads\Compressed\archive"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]


# --- 2. MODEL SINIFI ---
class ChestXRayResNet(nn.Module):
    def __init__(self, num_classes=14):
        super(ChestXRayResNet, self).__init__()
        try:
            self.model = models.resnet50(weights='IMAGENET1K_V1')
        except:
            self.model = models.resnet50(pretrained=True)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


# --- 3. AKILLI VERİ YÜKLEYİCİ (12 Klasör + Data_Entry Uyumu) ---
class NIHDatasetMultiFolder(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Hız için sadece 300 resim test et.
        # Gerçek sonuç için istersen bu satırı silip tamamını taratabilirsin.
        self.annotations = self.annotations.sample(n=300, random_state=42) if len(
            self.annotations) > 300 else self.annotations

        # Klasör listesi (images_001 ... images_012)
        self.subfolders = [f"images_{i:03d}" for i in range(1, 13)]

    def find_image(self, img_name):
        """ Resmi bulmak için 12 klasöre tek tek bakar """
        # 1. Root içinde mi?
        if os.path.exists(os.path.join(self.root_dir, img_name)):
            return os.path.join(self.root_dir, img_name)

        # 2. Alt klasörlerde mi?
        for sub in self.subfolders:
            path1 = os.path.join(self.root_dir, sub, "images", img_name)
            path2 = os.path.join(self.root_dir, sub, img_name)

            if os.path.exists(path1): return path1
            if os.path.exists(path2): return path2

        return None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        img_path = self.find_image(img_name)

        if img_path is None:
            image = Image.new('RGB', (224, 224))
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                image = Image.new('RGB', (224, 224))

        labels = torch.zeros(14)
        if 'Finding Labels' in self.annotations.columns:
            finding_labels = self.annotations.iloc[index]['Finding Labels']
            for i, cls in enumerate(CLASS_NAMES):
                if cls in finding_labels:
                    labels[i] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, labels


# --- 4. YARDIMCI FONKSİYONLAR ---
def get_loader():
    # Dosya yolunu proje yapına göre buluyoruz
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # GÜNCELLEME: Artık doğrudan 'Data_Entry_2017.csv' dosyasını arıyor!
    csv_path = os.path.join(project_root, "data", "raw", "Data_Entry_2017.csv")

    if not os.path.exists(csv_path):
        print(f"❌ HATA: Liste dosyası bulunamadı: {csv_path}")
        print("Lütfen 'data/raw/Data_Entry_2017.csv' dosyasının orada olduğundan emin ol.")
        return None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = NIHDatasetMultiFolder(csv_file=csv_path, root_dir=DATA_ROOT, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def run_test(model_path, loader):
    print(f"🔄 Test Ediliyor: {model_path}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    full_path = os.path.join(project_root, model_path)

    model = ChestXRayResNet(num_classes=14).to(DEVICE)

    if os.path.exists(full_path):
        try:
            state_dict = torch.load(full_path, map_location=DEVICE)
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            print("✅ Model ağırlıkları yüklendi.")
        except Exception as e:
            print(f"❌ Yükleme hatası: {e}")
            return [0.0] * 14
    else:
        print(f"❌ Dosya Yok: {full_path}")
        return [0.0] * 14

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    scores = []
    for i in range(14):
        scores.append(f1_score(all_labels[:, i], all_preds[:, i], zero_division=0))
    return scores


def main():
    print(f"⚙️ Cihaz: {DEVICE}")
    print(f"📂 Veri Yolu: {DATA_ROOT}")

    loader = get_loader()
    if loader is None: return

    # ESKİ MODEL (V1)
    scores_v1 = run_test("models/best_model.pth", loader)

    # YENİ MODEL (V2)
    scores_v2 = run_test("models_v2/v2_best_model.pth", loader)

    # GRAFİK
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    plt.figure(figsize=(14, 7))
    plt.bar(x - width / 2, scores_v1, width, label='V1 (Eski Model)', color='#e74c3c')
    plt.bar(x + width / 2, scores_v2, width, label='V2 (Yeni Model)', color='#2ecc71')

    plt.xlabel('Hastalıklar')
    plt.ylabel('F1 Skoru (Gerçek Veri)')
    plt.title('GERÇEK TEST SONUCU (NIH Veri Seti)')
    plt.xticks(x, CLASS_NAMES, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Log klasörüne kaydet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    output_path = os.path.join(log_dir, "comparison.png")
    plt.savefig(output_path)
    print(f"✅ Grafik şuraya kaydedildi: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()