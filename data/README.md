#  Veri Klasörü 

Bu klasör, proje için gerekli olan veri setlerini tutar.

##  ÖNEMLİ KURALLAR
1. **Asla GitHub'a Yükleme:** Büyük dosyalar (.zip, .tar.gz, resim klasörleri) GitHub'a atılmaz. `.gitignore` dosyası bunları engeller.
2. **Drive'dan Çek:** Verileri Google Drive'daki `01_Sample_Data` klasöründen indirip buraya koyun.

##  Klasör Yapısı
* **`raw/`**: Verinin hiç dokunulmamış, ham hali. (Örn: `Data_Entry_2017.csv` ve Kaggle'dan inen resimler).
* **`processed/`**: Kod tarafından işlenmiş, kırpılmış veya `.pt` (tensor) formatına çevrilmiş veriler.

#  Veri Klasörü
Büyük verileri GitHub'a yüklemiyoruz.
  **Google Drive:** `01_Sample_Data` klasörüne gidin.
  Oradan `Data_Entry_2017.csv` ve resimleri indirip buradaki `raw` klasörüne atın.