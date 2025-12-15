import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import config

def main():
    print(f"️  Konfigürasyon Yüklendi. Root Dizini: {config.PROJECT_ROOT}")


    INPUT_FILE = os.path.join(config.RAW_DATA_DIR, "Data_Entry_2017.csv")
    OUTPUT_DIR = config.PROCESSED_DATA_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"\n KRİTİK HATA: CSV dosyası bulunamadı!")
        print(f"   Aranan Yol: {INPUT_FILE}")
        print(f"   Çözüm: 'Data_Entry_2017.csv' dosyasını 'data/raw/' klasörüne taşıyın.\n")
        return

    print(f" Veri okunuyor: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)


    # Her satır bir resimdir. Ama biz resimleri değil, HASTALARI bölmeliyiz.
    unique_patients = df['Patient ID'].unique()
    print(f" Toplam Benzersiz Hasta Sayısı: {len(unique_patients)}")

    # Config içindeki SEED (42) ile bölüyoruz. 
    # Böylece her çalıştırdığımızda AYNI hastalar Train grubuna düşer.
    train_ids, val_ids = train_test_split(
        unique_patients, 
        test_size=0.20, 
        random_state=config.SEED
    )

    # ID listelerine göre ana tabloyu filtrele
    train_df = df[df['Patient ID'].isin(train_ids)]
    val_df = df[df['Patient ID'].isin(val_ids)]

    print(f" Ayrıştırma Tamamlandı:")
    print(f"   - Train Seti: {len(train_df)} resim ({len(train_ids)} Hasta)")
    print(f"   - Val Seti  : {len(val_df)} resim ({len(val_ids)} Hasta)")


    # Train ve Val kümelerinde ORTAK hasta var mı? (Olmamalı!)
    t_set = set(train_df['Patient ID'].unique())
    v_set = set(val_df['Patient ID'].unique())
    intersect = t_set.intersection(v_set)

    # Eğer kesişim kümesi boş değilse programı durdur ve hata ver!
    assert len(intersect) == 0, f"❌ HATA: Data Leakage Var! {len(intersect)} hasta çakışıyor."
    print("  Güvenlik Kontrolü Geçildi: Veri sızıntısı yok (No Data Leakage).")


    # Dosyaları config.PROCESSED_DATA_DIR (data/processed) içine kaydet
    train_save_path = os.path.join(OUTPUT_DIR, 'train_list.csv')
    val_save_path = os.path.join(OUTPUT_DIR, 'val_list.csv')

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)

    print(f" Dosyalar başarıyla kaydedildi:\n   -> {train_save_path}\n   -> {val_save_path}")

if __name__ == "__main__":
    main()