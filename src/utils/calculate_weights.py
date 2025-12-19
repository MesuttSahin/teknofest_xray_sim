import pandas as pd
import numpy as np
import os

# NIH Veri Seti Standart Sıralaması
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]


def calculate_pos_weights():
    # --- 1. DOSYA YOLLARINI BUL ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # DÜZELTME: Artık 'train_list.csv' dosyasını okuyoruz
    csv_path = os.path.join(project_root, "data", "raw", "train_list.csv")

    print(f"📊 Veri seti okunuyor: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"❌ HATA: '{csv_path}' bulunamadı!")
        print("Lütfen 'train_list.csv' dosyasını 'data/raw' klasörüne yapıştırdığından emin ol.")
        return

    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"✅ Toplam Görüntü Sayısı: {total_samples}")

    pos_weights = []

    print("\n⚖️  AĞIRLIKLAR HESAPLANIYOR (Negatif / Pozitif)...")
    print("-" * 50)
    print(f"{'HASTALIK':<20} | {'POZİTİF':<10} | {'NEGATİF':<10} | {'WEIGHT (Ceza Puanı)'}")
    print("-" * 50)

    # --- 2. HESAPLAMA DÖNGÜSÜ ---
    for label in CLASS_NAMES:
        # 'Finding Labels' sütunu yoksa hata vermesin diye kontrol
        if 'Finding Labels' not in df.columns:
            print("❌ HATA: CSV dosyasında 'Finding Labels' sütunu bulunamadı!")
            print(f"Mevcut Sütunlar: {df.columns}")
            return

        pos_count = df['Finding Labels'].str.contains(label).sum()
        neg_count = total_samples - pos_count

        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1e10

        pos_weights.append(round(weight, 2))

        print(f"{label:<20} | {pos_count:<10} | {neg_count:<10} | {weight:.2f}")

    print("-" * 50)
    print("\n🚀 KOPYALAMAN GEREKEN LİSTE (Config için):")
    print(f"POS_WEIGHTS = {pos_weights}")


if __name__ == "__main__":
    calculate_pos_weights()