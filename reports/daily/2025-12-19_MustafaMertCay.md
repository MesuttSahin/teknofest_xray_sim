
    1. Bugün Ne Yaptım?

Model Kapasitesini Optimize Ettim: ResNet-50 mimarisinin ilk katmanlarını dondurarak (Freezing) ImageNet’ten gelen genel özellik bilgisini korudum; sadece layer4 ve fc katmanlarını akciğer röntgenlerine özel eğitim için serbest bıraktım.

Ezberlemeyi (Overfitting) Engelleyen Mekanizmalar Kurdum: Modelin sonuna %50 oranında bir Dropout katmanı ekleyerek nöronları rastgele devre dışı bıraktım ve modelin tek bir nörona aşırı güvenmesini önledim.

Veri Artırma (Data Augmentation) Entegrasyonu: transforms.py dosyasındaki RandomRotation, ColorJitter ve RandomResizedCrop gibi gelişmiş teknikleri train.py içine dahil ederek modelin her epoch’ta farklı varyasyonlar görmesini sağladım.

Dengeli Öğrenme Stratejisi Uyguladım: Analistten gelen ve nadir hastalıklar için (örneğin Hernia için 494.88) yüksek ceza puanları içeren pos_weight listesini BCEWithLogitsLoss fonksiyonuna tanımladım.

Eğitim Yönetimi: Öğrenme hızını (Learning Rate) takip eden ReduceLROnPlateau scheduler’ı ve ağırlık aşınması (Weight Decay) sağlayan AdamW optimizer’ı devreye aldım.

Hassas Eğitim (Fine-Tuning): Öğrenme hızını $1e-5$ yaparak modelin ağırlıklarını santim santim, en doğru noktaya yerleştirdim.
 
     2. Karşılaşılan Hatalar ve Çözümler

Hata (Overfitting): İlk denemelerde Train Loss düşerken Val Loss 2.0 seviyelerine fırladı ve model tamamen ezber yapmaya başladı.

Çözüm: Veri setinin çok kolay olduğunu tespit ettim. Data Augmentation (rastgele döndürme, ışık değiştirme) uygulayarak "sınav sorularını" zorlaştırdım ve modelin ezberlemesini imkansız hale getirdim.

Hata (Stabilite Kaybı): Eğitim sırasında Val Loss’ta ani sıçramalar (Epoch 3’te 1.62 gibi) gözlemlendi.

Çözüm: Scheduler ve Dropout mekanizmalarının yardımıyla modelin sakinleşmesini bekledim; 4. epoch’tan itibaren Val Loss tekrar 1.31 seviyelerine düşerek stabilize oldu.

Sorun: Önceki yüksek öğrenme hızında model hedefi ıskalıyor ve Val Loss dalgalanıyordu.

Çözüm: LR düşürerek modeli "sakinleştirdim". Sonuçta dalgalanma bitti ve stabil bir iniş sağlandı.


        3. Sonuç

Şampiyon Model: chest_xray_model_ep7.pth dosyası (Val Loss: 1.3008) projenin nihai en iyi modeli olarak tescillenmiştir.

Sistem Durumu: Eğitim 10 epoch boyunca hiçbir sapma göstermeden başarıyla tamamlandı.
