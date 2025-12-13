#  Model Mimarisi (Model Source)

Yapay zeka modelinin "beyin yapısını" ve eğitim kurallarını içeren kodlar buradadır.

##  Dosyalar Ne İşe Yarar?
* **`model.py`**: Modelin iskeleti.
    * Örneğin `torchvision.models.resnet50` buradan çağrılır.
    * Son katman (Classifier) 14 hastalık için burada değiştirilir.
* **`train_mock.py`**: 
    * Sistemin çalışıp çalışmadığını test eden sahte eğitim kodu.
* **`train.py`** 
    * Gerçek eğitimi başlatan, epoch döngüsünü kuran ana dosya.

##  Uyarı
* Eğitilmiş model dosyaları (`.pth`) buraya kaydedilmez! Onlar ana dizindeki `models/` klasörüne veya Google Drive'a gider.





