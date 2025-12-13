#  Veri İşleme Kodları (Data Scripts)

Bu klasör, verinin kendisini değil, **veriyi okuyan ve hazırlayan kodları** içerir.

##  Dosyalar Ne İşe Yarar?
* **`dataset.py`**: En kritik dosyadır.
    * `Data_Entry_2017.csv` dosyasını okur.
    * Resim yollarını (Path) bulur.
    * Görüntüyü diskten yükleyip PyTorch'a verir.
* **`transforms.py`**: Veri artırma işlemleri.
    * Resmi 224x224 boyutuna getirir (Resize).
    * Döndürme, kırpma gibi işlemlerle veriyi çoğaltır (Augmentation).

##  Uyarı
* Resim dosyalarını (`.png`, `.jpg`) buraya koymayın! Onlar ana dizindeki `data/raw/` klasöründe durmalı.