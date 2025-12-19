import os
import torch
import random
import numpy as np

# 1. AYARLAR
SEED = 42
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 14
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
NUM_WORKERS = 2  # Windows'ta hata verirse bunu 0 yap
RAW_DATA_DIR = 'data/raw'
# DOSYA YOLLARI 


CURRENT_FILE_PATH = os.path.abspath(__file__) # config.py'nin yeri
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

BASE_PATH = os.path.join(PROJECT_ROOT, 'data')

RAW_DATA_DIR = os.path.join(BASE_PATH, 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, 'processed')
MODEL_SRC_DIR = os.path.join(PROJECT_ROOT, 'src', 'models')
BEST_MODEL_PATH = os.path.join(MODEL_SRC_DIR, 'best_model.pth')
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models') # Modeller ana dizindeki models'e
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

#CİHAZ & SEED
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