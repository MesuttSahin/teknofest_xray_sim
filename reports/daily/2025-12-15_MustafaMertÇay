1.Bugün Ne Yaptım?
Bu görevde, ön eğitimli (pre-trained) bir modelin beklediği formatta ve dengede veri sunulmasını sağlayacak olan PyTorch veri akışının tüm bileşenlerini tasarladım, kodladım ve entegre ettim.

Veri Dönüşümleri (Transforms) Tanımlandı: src/data/transforms.py dosyası oluşturuldu.

ImageNet istatistikleri (mean/std) kullanılarak normalizasyon adımı kesinleştirildi.

Eğitim verisi için genellemeyi artıracak Veri Artırma (Random Flip, Random Rotation) adımları eklendi. Doğrulama verisinden bu adımlar dışarıda bırakıldı.

Özel Veri Kümesi (Custom Dataset) Kuruldu: src/data/dataset.py dosyası, ChestXRayDataset sınıfını içerir.

Bu sınıf, CSV dosyasını okuyarak görüntü yollarını ve Çok Etiketli (Multi-Label) etiketleri ikili (binary) tensörlere dönüştürmekle görevlidir.

config.py ve transforms.py dosyalarından gelen değişkenler/fonksiyonlar başarıyla entegre edildi.

                                                            2.Karşılaşılan Hatalar ve Çözümler

Komut istemcisinin (Terminal/PowerShell) Yönetici Olarak Çalıştırılması ile izin sorunu giderildi ve PyTorch kurulumu tamamlandı.
Proje yapısı (src/data'dan src/utils'a) teyit edilerek doğru göreceli içe aktarma (from ..utils import config) yöntemi kullanıldı.

                                                            3.Sonuç:

Model mimarlığında Normalizasyon(Pre-train modelin beklentisi), batch size , main ve std gibi kavramları öğrendim.
Batch size'ı arttırırsakta genelleme riski oluşarak performansı düşebileceğini öğrendim.
