1. Bugün Ne Yaptım?

Kütüphane ve Ortam Kurulumu: Eksik olan torch, torchvision, numpy gibi kritik kütüphaneleri "System Interpreter" (Python 3.10) üzerine başarıyla kurdum ve GPU (CUDA) erişimini teyit ettim.

Evaluate.py Dosyasının Oluşturulması: Analist için gerekli olan model değerlendirme script’ini yazdım.

Modeli .eval() moduna alarak kararlı çalışmasını sağladım.

Kritik Ayar: Verilerin karışmaması için DataLoader üzerinde shuffle=False ayarını uyguladım.

Logit-Olasılık Dönüşümü: Modelden çıkan ham sayıları (logits), Analist'in ROC eğrisi çizebilmesi için torch.sigmoid() fonksiyonu ile 0-1 arası olasılık değerlerine dönüştürdüm.

Veri İhracı: Tahminleri (predictions.npy) ve gerçek etiketleri (true_labels.npy) logs/ klasörüne .npy formatında dışa aktardım.

Overfitting Analizi: src/data/transforms.py dosyasını inceleyerek modelin ezberlemesini önleyecek "Reçete"yi (RandomRotation artırımı ve ColorJitter eklenmesi) hazırladım.

2. Karşılaşılan Hatalar ve Çözümler

   
Hata: ModuleNotFoundError: No module named 'src'

Çözüm: Python'ın çalışma dizinini tanıması için $env:PYTHONPATH = "." komutu ile projenin kök dizinini sisteme tanıttık.

Hata: TypeError: 'type' object is not iterable (DataLoader Hatası)

Çözüm: DataLoader sınıfının kendisini değil, bu sınıftan üretilen val_loader nesnesini döngüye sokarak verilerin akışını sağladık.

Hata: TypeError: ChestXRayDataset.__init__() got an unexpected keyword argument 'list_file'

Çözüm: Dataset sınıfının __init__ metodunu inceleyerek beklenmeyen parametreleri temizledik ve sınıfın tam olarak beklediği csv_file parametresiyle (data/processed/val_list.csv) uyumlu hale getirdik.

Hata: Pip Version Warning

Çözüm: Pip aracını son sürüme güncelleyerek kurulum hatalarının önüne geçtik.

3. Sonuç

   
Analiz Hazır: Veri Analisti'nin ROC eğrisi ve performans metriklerini hesaplaması için gerekli olan tüm veriler logs/ altında başarıyla oluşturuldu.

Sistem Hazır: GPU desteği aktif edildi ve projenin klasör yapısı ile kod arasındaki "Import" sorunları tamamen çözüldü.

Yarın İçin Plan: Modelin eğitim sırasında ezber yapmasını engellemek amacıyla transforms.py dosyasına eklenecek veri artırma (augmentation) stratejileri belirlendi.






