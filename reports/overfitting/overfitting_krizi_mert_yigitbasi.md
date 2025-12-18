Veri Analisti RAPOR: Model Neden Başarısız? (Ezberleme ve Dengesizlik Analizi)

1\. TEŞHİS: "Tembel Öğrenci" Sendromu Modelimiz şu an dersi öğrenmiyor, sistemin açığını bulmuş durumda. Veri setindeki hastaların çoğu "Sağlıklı" olduğu için, risk almayıp herkese "Sağlıklısın" diyor.

Kanıt: Confusion Matrix incelendiğinde durum vahimdir. Model; Mass (Kitle), Hernia (Fıtık), Pneumonia (Zatürre) gibi kritik hastalıkların hiçbirini bulamamış (0 Doğru Tespit), bu hastalara sahip kişilerin tamamına "Sağlıklı" demiştir.

2\. REÇETE: 3 Adımlı Çözüm Planı

Bu "sıfır çeken" tabloyu düzeltmek için şu üç stratejiyi aynı anda uygulamalıyız:

Veri Artırma (Data Augmentation - ÇOK KRİTİK): Elimizde sadece 4 tane 'Hernia' resmi var. Modelin bunu öğrenmesi imkansız. Bu az sayıdaki resmi sanal olarak çoğaltıp (döndür, bük, yakınlaştır) sayıyı artırmalıyız ki model mecburen hastalığı tanısın.

Modeli Zorlama (Dropout): Modelin "Herkese sağlıklı de, geç" ezberini bozmak için beynindeki nöronların yarısını rastgele kapatacağız. Model kolaya kaçamayacak.

Erken Müdahale (Early Stopping): Model ezber yapmaya başladığı (başarısı düşmeye başladığı) saniye eğitimi otomatik olarak keseceğiz.

