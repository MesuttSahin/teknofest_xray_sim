import os
import torch
import random
import numpy as np


SEED = 42
BATCH_SIZE = 16       # VRAM yetmezse 8 yapabilirsin
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 14

# Fine-Tuning (İnce Ayar) için LR düşürülür, Epoch artırılır
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10       # Model dondurulduğu için daha uzun süre eğitebiliriz (Eski: 5)
NUM_WORKERS = 2       # Windows'ta hata alırsan 0 yap


CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

BASE_PATH = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(BASE_PATH, 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, 'processed')
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')


BEST_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'v2_best_model.pth')


CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random Seed sabitlendi: {seed}")

seed_everything(SEED)



# A) SCHEDULER (Öğrenme Hızı Planlayıcı)
# Validation loss düşmezse Learning Rate'i küçült
SCHEDULER_PATIENCE = 2   # 2 epoch boyunca iyileşme olmazsa...
SCHEDULER_FACTOR = 0.1   # LR'yi 10'a böl (0.0001 -> 0.00001)

#  EARLY STOPPING (Erken Durdurma)
EARLY_STOPPING_PATIENCE = 5

# Hernia (494.88) gibi nadir sınıflar için ceza puanları
POS_WEIGHTS = [
    7.40, 38.28, 6.84, 4.01, 17.71, 15.33, 60.03,
    22.47, 20.80, 45.67, 39.07, 69.84, 31.52, 494.88
]