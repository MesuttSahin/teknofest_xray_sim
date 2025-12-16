import pandas as pd
import os
import shutil
import sys


def create_balanced_dataset():
    # --- AYARLAR ---

    # 1. KAYNAK KLASÖR (Bütün images_001, images_002... klasörlerinin olduğu yer)
    # Lütfen buraya kendi bilgisayarındaki yolu yaz (Ters slash \ yerine / veya r"..." kullan)
    # Örn: r"D:\Downloads\Compressed\archive"
    SOURCE_ROOT_DIR = r"D:\Downloads\Compressed\archive"

    # 2. PROJE YOLLARI
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Hedef Klasör (Resimlerin toplanacağı yer)
    dest_image_dir = os.path.join(project_root, "data", "raw", "images")

    # CSV Yolları
    # Büyük CSV dosyasının yeri (Genelde archive klasörünün içindedir, yoksa yolunu düzelt)
    source_csv_path = os.path.join(SOURCE_ROOT_DIR, "Data_Entry_2017.csv")
    output_csv_path = os.path.join(project_root, "data", "raw", "Data_Entry_2017_Mini.csv")

    TARGET_SIZE = 5000

    # --- İŞLEM BAŞLIYOR ---
    print("🚀 İşlem başladı...")
    print(f"📂 Kaynak taranıyor: {SOURCE_ROOT_DIR}")

    # 1. TÜM KLASÖRLERİ TARA VE RESİMLERİ BUL (Recursive)
    # Hangi resim nerede? (Dosya İsmi -> Tam Yol) sözlüğü oluşturuyoruz
    all_image_paths = {}

    for root, dirs, files in os.walk(SOURCE_ROOT_DIR):
        for file in files:
            if file.endswith(".png"):
                full_path = os.path.join(root, file)
                all_image_paths[file] = full_path

    total_found = len(all_image_paths)
    print(f"✅ Toplam {total_found} adet resim bulundu (Tüm klasörlerde).")

    if total_found == 0:
        print("❌ HATA: Hiç .png dosyası bulunamadı! SOURCE_ROOT_DIR yolunu kontrol et.")
        return

    # 2. CSV OKU
    print(f"📖 Büyük CSV okunuyor...")
    if not os.path.exists(source_csv_path):
        # Eğer CSV kaynak klasörde değilse proje içindekine bakalım
        source_csv_path = os.path.join(project_root, "01_Sample_Data", "Data_Entry_2017.csv")
        if not os.path.exists(source_csv_path):
            print(f"❌ HATA: CSV dosyası bulunamadı!")
            return

    df = pd.read_csv(source_csv_path)

    # 3. ELİMİZDEKİ RESİMLERE GÖRE FİLTRELE
    # CSV'de olup da bizde olmayanları at
    df_existing = df[df['Image Index'].isin(all_image_paths.keys())]
    print(f"📉 Eşleşen veri sayısı: {len(df_existing)}")

    # 4. DENGELEME (2500 / 2500)
    healthy_df = df_existing[df_existing['Finding Labels'] == "No Finding"]
    disease_df = df_existing[df_existing['Finding Labels'] != "No Finding"]

    sample_count = TARGET_SIZE // 2
    n_healthy = min(len(healthy_df), sample_count)
    n_disease = min(len(disease_df), sample_count)

    print(f"⚖️  Seçim yapılıyor: {n_healthy} Sağlıklı + {n_disease} Hasta...")

    sampled_healthy = healthy_df.sample(n=n_healthy, random_state=42)
    sampled_disease = disease_df.sample(n=n_disease, random_state=42)

    mini_df = pd.concat([sampled_healthy, sampled_disease])
    mini_df = mini_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. DOSYALARI KOPYALA
    print(f"📦 Seçilen {len(mini_df)} resim proje klasörüne kopyalanıyor...")

    # Hedef klasörü temizle/oluştur (Eski yanlış dosyalar gitsin)
    if os.path.exists(dest_image_dir):
        shutil.rmtree(dest_image_dir)  # Klasörü sil
    os.makedirs(dest_image_dir)  # Yeniden oluştur

    copy_count = 0
    for img_name in mini_df['Image Index']:
        src_path = all_image_paths[img_name]  # Resmin asıl yeri (örn: images_005 içinde)
        dst_path = os.path.join(dest_image_dir, img_name)
        shutil.copy2(src_path, dst_path)
        copy_count += 1
        if copy_count % 500 == 0:
            print(f"   ... {copy_count} resim kopyalandı.")

    # 6. CSV KAYDET
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    mini_df.to_csv(output_csv_path, index=False)

    print(f"🎉 İŞLEM TAMAM!")
    print(f"💾 Yeni CSV: {output_csv_path}")
    print(f"📂 Yeni Resimler: {dest_image_dir} (Toplam {copy_count} adet)")


if __name__ == "__main__":
    create_balanced_dataset()