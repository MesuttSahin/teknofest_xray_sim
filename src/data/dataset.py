import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os

from src.utils import config


class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Generic olması için dosya yolu belirtmiyorum, dışarıdan gelecek.
        self.data = pd.read_csv(csv_file)

        # Kırpma veya Döndürme için koyduk EĞER yapılacaksa şart değil.
        self.transform = transform

        # Hasta listesini direkt config.py'den düzenleyebiliriz listeyi ordan çekiyoruz.
        self.class_names = config.CLASS_NAMES

    def __len__(self):
        # CSV dosyasındaki satır sayısı kadar uzunluk belirledik.
        return len(self.data)

    def __getitem__(self, indeks):
        # Sıradaki satırı seçiyor.
        row = self.data.iloc[indeks]
        img_name = row['Image Index']

        # Config'den gelen klasör yolu ile resim adını birleştiriyoruz sanırım böyle doğru oldu kontrol gerekebilir.
        img_path = os.path.join(config.RAW_DATA_DIR, 'images', img_name)

        try:
            # Resmi açıyoruz aynı zamanda RGB dönüşümü hata almamak için farz.
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Hata alırsan tam olarak nereye baktığını görmek için yolu yazdırıyoruz
            print(f"❌ HATA: Resim okunamadı!")
            print(f"   Aranan Yol: {img_path}")
            print(f"   Klasör yapısını kontrol et: 'data/raw/images' var mı?")
            raise e

        # Transform (Resize, Normalize vb.)
        if self.transform:
            # 224x224'e küçültmek genelde öyleymiş VEYA Tensöre çevirmek için kullandık.
            image = self.transform(image)

        label_string = row['Finding Labels']  # Örn: "Atelectasis|Effusion"

        # Config'den çektiğimiz listenin uzunluğu kadar 0 vektörü oluşturuyoruz.
        # Config'deki hastalık sayısı değişirse de bu sayede o kadar uzunluğa sahip olacağız.
        label_vector = np.zeros(len(self.class_names), dtype=np.float32)

        diseases = label_string.split(
            '|')  # Bizim veri setinde birden fazla hastalığı olanları gösterirken diğer hastalığı bu şekilde ayırmış

        for disease in diseases:
            # Config'deki listede bu hastalık varsa indeksini 1 yap
            if disease in self.class_names:
                # Hastalığın sırasını bul (Örn: Atelectasis 0. sırada)
                hastalik_idx = self.class_names.index(disease)
                # O sıradaki 0'ı 1 yap
                label_vector[hastalik_idx] = 1.0

        # Return: Görüntü ve Etiket vektörü
        return image, torch.tensor(label_vector)
