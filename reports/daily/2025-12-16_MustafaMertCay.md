1. Bugün Ne Yaptım?

Projemin kalbinde yer alan model.py dosyası, göğüs röntgeni görüntülerini sınıflandırmak için kullandığım derin öğrenme mimarisini barındırıyor. Bu sınıfın temel özellikleri şunlardır:

Temel Mimari: Prensip olarak ResNet-50 gibi önceden eğitilmiş (Pre-trained) bir Evrişimsel Sinir Ağı (CNN) kullanıldı. Bu, modelin görüntü tanıma yeteneğini sıfırdan öğrenmek yerine, halihazırda büyük veri kümelerinden (ImageNet) öğrendiği bilgileri kullanmasını sağlar (Transfer Learning).

Özelleştirme (Customization): Seçilen önceden eğitilmiş modelin (ResNet-50) son katmanı, projemin özel sınıflandırma görevine (örneğin X-Ray görüntülerinde 2 veya 3 farklı durumu ayırt etme) uygun hale getirilmesi için değiştirildi.

Orijinal sınıflandırma katmanı silindi.

Yerine, çıktı sayısı projenin sınıf sayısına eşit olan yeni bir Doğrusal Katman (Fully Connected Layer) eklendi.

Eğitilebilir Katmanlar: Performans ve hız dengesi için, modelin ilk katmanları genellikle dondurulur (freeze) ve sadece son katmanların ya da bazı blokların eğitilmesi (fine-tuning) tercih edilir.

Projemin eğitim hızını kat kat artırmak için NVIDIA GPU hızlandırmasını (CUDA) etkinleştirmek ve PyTorch'u nihayet doğru şekilde kurmaktı.

CUDA Sürümü Teyidi: Bilgisayarımdaki kurulu olan NVIDIA CUDA Toolkit sürümünün (V13.1) teyidini yaptım.

Sanal Ortam Oluşturma: Başta yaşadığım izin sorunları ve ortam karmaşasından kaçınmak için, GPU destekli kütüphaneleri izole etmek üzere gpu_env adında yepyeni ve temiz bir sanal ortam oluşturdum.

GPU PyTorch Kurulumu: CUDA 13.1 kurulu olmasına rağmen, uygun paket bulma hatası alınca, PyTorch'un en uyumlu GPU paketi olan CUDA 11.8 desteğiyle kurulumu Command Prompt (CMD) üzerinden başarıyla gerçekleştirdim.

2. Karşılaşılan Hatalar ve Çözümler

pip uninstall erişim engeli:	Terminali Yönetici (Administrator) olarak çalıştırmak.	
Sanal ortamın bulunamaması:	Eski yöntemler yerine, python -m venv gpu_env komutuyla yeni, temiz bir ortam oluşturmak.
PyTorch uyumlu sürüm bulunamadı:	Güncel CUDA 13.1 yerine, PyTorch'un CUDA 11.8 paket indeksini kullanmak.	
ModuleNotFoundError (tqdm):	Eksik kütüphaneleri pip install -r requirements.txt ile kurmak.

3. Sonuç

NVIDIA GPU hızlandırması projemde başarılı bir şekilde etkinleştirildi.

Performans: Model, CPU'da saatler sürecekken, GPU sayesinde dakikalar içinde eğitildi.

Eğitim Durumu: Model eğitimi (5 epoch) tamamlandı. Eğitim kaybım düşerken doğrulama kaybımın yükseldiğini gördüm, bu da modelimin Epoch 2 sonrası aşırı öğrenmeye (overfitting) başladığını gösteriyor.