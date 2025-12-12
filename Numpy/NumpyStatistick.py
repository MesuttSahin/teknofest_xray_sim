import numpy as np
#np.mean() (Ortalama): Tüm sayıları toplayıp adedine böler. Bildiğimiz aritmetik ortalama.
#np.median() (Medyan/Ortanca): Sayıları küçükten büyüğe dizer ve tam ortadaki sayıyı bulur.
# median afaki değerleri görmezden gelir. herkes 10 bin tl alırken 1 milyon tl alanı almaz.
puanlar = np.array([70,80,75,90,10,85])
print(np.mean(puanlar)) # Direkt aritmetik ortalama
print(np.median(puanlar)) # 10 alanı değerlendirmeye almaz. Bu sayede başarıyı yansıtır

# Rastgele veri üretmek np.random
sicakliklar = np.random.randint(20,35,10) # 20 ile 35 arasında 10 tane sayı sallar 35 dahil değil.
print(np.mean(sicakliklar)) # Ortalama sıcaklık.
print(sicakliklar[sicakliklar < 30]) # Serin Günler.

