import pandas as pd
import os
import random



# --- AYARLAR (Kendi bilgisayarındaki yollara göre düzenle) ---
# Resimlerin olduğu klasör (İndirdiğin resimler nerede?)
RAW_IMAGE_DIR = "data/raw/images"
# Orijinal büyük CSV dosyasının yolu
ORIGINAL_CSV_PATH = "data/raw/Data_Entry_2017.csv"
# Çıktı dosyasının kaydedileceği yer (Mini CSV)
OUTPUT_CSV_PATH = "data/raw/Data_Entry_2017_Mini.csv"

# Hedeflenen Toplam Resim Sayısı (Örn: 5000)
TARGET_SIZE = 5000


def create_balanced_dataset():
    print(f"📂 Resimler taranıyor: {RAW_IMAGE_DIR}")

    # 1. Bilgisayardaki MEVCUT resimlerin listesini al
    # (Sadece klasörde olan resimlerle çalışmalıyız)
    try:
        available_images = [f for f in os.listdir(RAW_IMAGE_DIR) if f.endswith('.png')]
    except FileNotFoundError:
        print(f"❌ HATA: Klasör bulunamadı: {RAW_IMAGE_DIR}")
        print("Lütfen 'RAW_IMAGE_DIR' değişkenini resimlerin olduğu doğru klasöre yönlendir.")
        return

    print(f"✅ Klasörde {len(available_images)} adet resim bulundu.")

    # 2. Orijinal CSV'yi Oku
    print(f"📖 Büyük CSV okunuyor...")
    df = pd.read_csv(ORIGINAL_CSV_PATH)

    # 3. Sadece elimizde resmi olan satırları filtrele
    # (CSV'de 112k satır var ama bizde 5k resim var, eşleşmeyenleri at)
    df_existing = df[df['Image Index'].isin(available_images)]
    print(f"📉 Filtreleme sonucu elimizdeki veriler: {len(df_existing)} satır.")

    # 4. STRATEJİ: %50 Sağlıklı / %50 Hasta Ayrımı

    # Sağlıklı olanlar (No Finding)
    healthy_df = df_existing[df_existing['Finding Labels'] == "No Finding"]

    # Hasta olanlar (No Finding DIŞINDAKİ her şey)
    disease_df = df_existing[df_existing['Finding Labels'] != "No Finding"]

    print(f"   - Sağlıklı Aday Sayısı: {len(healthy_df)}")
    print(f"   - Hasta Aday Sayısı   : {len(disease_df)}")

    # 5. Örnekleme (Sampling)
    # Hedefimizin yarısı kadar sağlıklı, yarısı kadar hasta alacağız
    sample_count = TARGET_SIZE // 2

    # Eğer elimizde yeterince resim yoksa, olanın tamamını alalım (Hata vermesin)
    n_healthy = min(len(healthy_df), sample_count)
    n_disease = min(len(disease_df), sample_count)

    print(f"⚖️  Dengeleme yapılıyor: {n_healthy} Sağlıklı + {n_disease} Hasta seçilecek.")

    # Rastgele seçim yap
    sampled_healthy = healthy_df.sample(n=n_healthy, random_state=42)
    sampled_disease = disease_df.sample(n=n_disease, random_state=42)

    # İkisini birleştir
    mini_df = pd.concat([sampled_healthy, sampled_disease])

    # Karıştır (Shuffle) - Sıralı gelmesin
    mini_df = mini_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. Kaydet
    # Önce klasör var mı kontrol et
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    mini_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"💾 Mini CSV kaydedildi: {OUTPUT_CSV_PATH}")
    print(f"🎉 İşlem Tamam! Toplam {len(mini_df)} satırlık veri seti hazır.")


if __name__ == "__main__":
    create_balanced_dataset()