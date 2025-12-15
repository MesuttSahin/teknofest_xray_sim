import os
import torch
import random
import numpy as np


#  AYARLAR

SEED = 42                # Şans faktörünü ortadan kaldırır
BATCH_SIZE = 16          # Her seferde GPU'ya girecek resim sayısı
IMAGE_SIZE = (224, 224)  # Resimlerin getirileceği boyut
NUM_CLASSES = 14         # Hastalık sayısı
LEARNING_RATE = 1e-4     # Öğrenme hızı
NUM_EPOCHS = 5           # Kaç tur eğitim yapılacağı
NUM_WORKERS = 2          # Veri yüklerken çalışacak işlemci çekirdeği (Kaggle'da 4 yap)


#DOSYA YOLLARI (PATHS)

# Eğer Drive bağlıysa orayı, değilse local klasörü gösterir
if os.path.exists('/content/drive'):
    BASE_PATH = '/content/drive/MyDrive/TEKNOFEST_XRAY_2025'
else:
    BASE_PATH = './data'  # Senin bilgisayarındaki yol

RAW_DATA_DIR = os.path.join(BASE_PATH, 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, 'processed')
MODEL_OUTPUT_DIR = os.path.join(BASE_PATH, 'models')
LOGS_DIR = os.path.join(BASE_PATH, 'logs')

# Hastalık İsimleri (Kaggle NIH Verisinden)
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


# Ekran kartı varsa onu seç, yoksa işlemciyi seç
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#  SEED SABİTLEME FONKSİYONU

def seed_everything(seed=42):
    """
    Tüm kütüphanelerin rastgele sayı üreteçlerini sabitler.
    Böylece deney tekrarlanabilir (Reproducible) olur.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random Seed sabitlendi: {seed}")

# Config dosyası import edildiğinde otomatik seed'i sabitle
seed_everything(SEED)