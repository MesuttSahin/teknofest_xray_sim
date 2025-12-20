1. Bugün Ne Yaptım?


Açıklanabilir Yapa Zeka (XAI) Entegrasyonu: Modelin bir "kara kutu" gibi çalışmaması için Grad-CAM (Gradient-weighted Class Activation Mapping) tekniğini sisteme entegre ettim.

Kanca (Hook) Mekanizması Kurulumu: ResNet50 mimarisinin en derin ve anlamlı özelliklerinin bulunduğu layer4[2].conv3 katmanına erişim sağlayarak, ileri ve geri yayılım sırasında gradyanları yakalayan "Hook" yapılarını kurdum.

Görselleştirme Boru Hattı: Modelin kararlarını ısı haritasına (Heatmap) dönüştüren ve bu haritayı orijinal röntgen görüntüsü üzerine bindiren (superimposed) bir sistem geliştirdim.

Merkezi Yapı Uyumu: gradcam.py dosyasını config.BEST_MODEL_PATH ile tam uyumlu hale getirerek projenin genel mimarisiyle birleştirdim.


2. Karşılaşılan Hatalar ve Çözümler


Modül Bulunamama Hatası (ModuleNotFoundError): src klasörünün Python yolu (PYTHONPATH) olarak tanınmaması nedeniyle import hataları yaşandı.

Çözüm: Terminalde $env:PYTHONPATH = "." komutuyla çalışma dizinini tanımlayarak sistemin tüm klasörleri görmesini sağladım.

Versiyon Uyarıları (FutureWarning): PyTorch'un eski register_backward_hook metoduna dair uyarılar alındı.

Çözüm: Kodun çalıştığı doğrulandı ve ilerleyen aşamalarda register_full_backward_hook geçişi için planlama yapıldı.

İlk gradcam.py'yi çalıştırdığımızda renk yoğunluğu akciğerler değil de köprücük kemiği, omuz gibi akciğer dışı yerlerde çıkması : Bunun sebebi akciğerinden hasta olmayan bir bireyin akciğer görüntüsünü seçmemiz.

Çözüm: Hastalığı olan bir bireyin akciğer görüntüsü alındı.

  3. Sonuç

Model Onaylandı: Elimizdeki model sadece istatistiksel olarak değil (Loss değerleri), görsel olarak da (Grad-CAM) işe yaradığını kanıtladı.

