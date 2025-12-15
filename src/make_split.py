import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Normalde bu bir config dosyasından gelir ama
# scriptin tek başına çalışması için buraya ekliyorum.
SEED = 42


def main():
    # 1. Veri Yolu Tanımları (Kendi klasör yapına göre kontrol et)
    INPUT_FILE = "../01_Sample_Data/Data_Entry_2017.csv"
    OUTPUT_DIR = "data/processed"

    # Klasör yoksa oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📖 Veri okunuyor: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # 2. HASTA BAZLI BÖLME (KRİTİK ADIM)
    # Görüntüleri değil, benzersiz hastaları alıyoruz
    unique_patients = df['Patient ID'].unique()
    print(f"🦠 Toplam Benzersiz Hasta Sayısı: {len(unique_patients)}")

    # Hastaları %80 Train / %20 Val olarak ayır
    # random_state=SEED (42) kullanarak sonucun her seferinde aynı olmasını sağlıyoruz
    train_ids, val_ids = train_test_split(unique_patients,
                                          test_size=0.20,
                                          random_state=SEED)

    # 3. LİSTELERİ OLUŞTURMA
    # Seçilen hasta ID'lerine sahip tüm satırları ana tablodan çekiyoruz
    train_df = df[df['Patient ID'].isin(train_ids)]
    val_df = df[df['Patient ID'].isin(val_ids)]

    print(f"✅ Ayrıştırma Tamamlandı:")
    print(f"   - Train Görüntü Sayısı: {len(train_df)} ({len(train_ids)} Hasta)")
    print(f"   - Val Görüntü Sayısı  : {len(val_df)} ({len(val_ids)} Hasta)")

    # 4. KANIT (ASSERTION) - Görev kartındaki en önemli madde!
    # Kesişim kümesi (intersection) BOŞ olmalı.
    train_patients = set(train_df['Patient ID'].unique())
    val_patients = set(val_df['Patient ID'].unique())

    intersect = train_patients.intersection(val_patients)

    assert len(intersect) == 0, f"❌ HATA! {len(intersect)} adet hasta her iki listede de var! (Data Leakage)"
    print("🛡️  GÜVENLİK KONTROLÜ BAŞARILI: Ortak hasta yok (No Data Leakage).")

    # 5. DOSYALARI KAYDETME
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_list.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val_list.csv'), index=False)
    print(f"💾 Dosyalar '{OUTPUT_DIR}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()