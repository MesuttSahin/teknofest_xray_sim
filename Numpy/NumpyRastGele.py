import numpy as np
rastgele = np.arange(10,50,5) # 10'dan başladı 50'ye kadar 50 dahil değil. 5'er 5'er artacak.
print(rastgele)
ontanesifir = np.zeros(10) 
print(ontanesifir)

# Örnek 
sayilar = np.arange(0,20,2)
print(sayilar[2]) # 3.sıradaki elemanı yazdır.
print(sayilar[0:5]) # 4'üncü indeksteki elemanı alır 5'inci indekstekini almaz.
print(sayilar[-1]) # Son elemanı yazdırdık.

# Örnek 2
hizlar = np.array([45,95,70,120,30,110])
ceza_yiyenler = hizlar[hizlar > 90]
print(ceza_yiyenler)

# Örnek 3 Matrisler ve Şekil Tespiti 
matris = np.arange(1,13)
matris = matris.reshape(4,3)
print(matris.shape)
print(matris)

# Örnek 4 sum() topla min() en küçük max() en büyük. sum(axis=0) sütun topla sum(axis=1) satır topla
satislar = np.array([[10,20,30],[40,50,60],[10,10,10]])
print(satislar.sum()) # tüm ciro 
print(satislar.sum(axis=0)) # günlük satışlar (sütun sütun toplattık.)
print(satislar.sum(axis=1)) # her şubenin satışını topladık (satırları topladık)