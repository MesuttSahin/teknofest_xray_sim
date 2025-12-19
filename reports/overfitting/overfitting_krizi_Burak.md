# Günlük Rapor

**İsim:** [Burak Yıldırım]
**Tarih:** 2025-12-18


### 1.  Bugün Ne Yaptım?
Deep Learning Overfitting Solutions konusunu araştırdık. Bunun sebeplerine baktım ve kendi alanımla ilgili olan kısımları uzun uzadı okuyup bilgi edindim.
### 2.  Karşılaşılan Hatalar ve Çözümler
Hata ile karşılaşılmadı sadece öğrenme amaçlandı.

### 3.  Sonuç
Okuduğum bilgiler üzerine veri setimiz 5000 görselden oluştuğu için modelimize küçük geldi resnet50 23milyondan fazla parametreye sahipmiş. öğrenmek yerine ezberlemeyi seçti. bu şahsi fikrimdir. ancak farklı sebeplerde mevcuttur.
bu şahsi fikrim olan kısma çözüm önerim veri sayısını eğer var ise doğrudan kullanmak eğer veriye ulaşamıyorsak elimizdeki veriyi Augmentation ile çoğaltmamız lazım. 
benden kaynaklı oluşabilecek hatalar Data Leakage ancak bunun için koşullar koymuştum olması durumunda hata kodu çıktı olarak verecekti. Analistten kaynaklı olmuş olsa Yetersiz Karıştırma Türkçesi önce A labelının hepsi sonra B labelına sahip olanların hepsi geliyorsa model patern ezberliyormuş. 
En uç ve zayıf ihtimal olmadığından eminim ancak Duplicate Tekrarlayan veri demekmiş aynı görselden fazladan olması bu da mümkün değil.
Tüm araştırmalarım sonucunda modelin büyük olması ve bizim verimizin çok az olması bu ikisi bir araya gelince Overfitting farz oldu. 


