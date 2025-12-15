from torch.utils.data import Dataset
import pandas as pd
class ChestXRayDataset(Dataset):import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from ..utils import config
from .transforms import get_transforms 


class ChestXRayDataset(Dataset):
    """
    Göğüs X-Ray görüntülerini yükleyen ve ön işleyen PyTorch Dataset sınıfı.
    """
    # config'ten gelen değişkenleri kullanıyoruz
    def __init__(self, csv_file, mode='train'):
        
        # 1. Veri Yollarını ve Dönüşümleri Başlatma
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = config.RAW_DATA_DIR # config.py'den çekildi
        self.transform = get_transforms(mode) # transforms.py'den çekildi
        
        # 2. Etiketleri Hazırlama
        self.labels = self.prepare_labels()
        
    def prepare_labels(self):
        """Çok Etiketli (Multi-label) sınıflandırma için etiketleri hazırlar."""
        
        all_labels = self.data_frame['Finding Labels'].str.split('|').fillna('')
        
        # config.py'den gelen CLASS_NAMES listesini kullanma
        label_matrix = torch.zeros((len(self.data_frame), len(config.CLASS_NAMES)), dtype=torch.float32)
        
        for i, labels_str in enumerate(all_labels):
            for j, class_name in enumerate(config.CLASS_NAMES):
                if class_name in labels_str:
                    label_matrix[i, j] = 1.0 
        
        return label_matrix

    def __len__(self):
        """Veri kümesindeki toplam örnek sayısını döndürür."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """Belirtilen dizine ait görüntüyü ve etiketi yükler."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Görüntü Yolu
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        
        # Görüntüyü Yükleme ve RGB'ye Çevirme
        image = Image.open(img_name).convert('RGB') 

        # Dönüşümleri Uygulama
        if self.transform:
            image = self.transform(image) 
        
        # Etiketler
        labels = self.labels[idx]

        return image, labels 


    if __name__ == "__main__":
         print("Dataset iskeleti oluşturuldu.")
