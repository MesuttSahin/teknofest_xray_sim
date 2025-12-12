import numpy as np
sayilar = [1,2,3] # Normal bir liste
numpy_sayilar = np.array(sayilar) # Listem artık bir numpy dizisi oldu.

# Basit bir örnek yapalım pratik amaçlı.

numbers = [1,2,3,4,5]
numpy_numbers = np.array(numbers)
islem = (numpy_numbers)*(10)-3
print(islem)

# Bir diger basit örneğimiz. Eğer boyutları aynı olmazsa hata veriyor.
vize = [40,60,75]
final = [70,80,45]
numpy_vize = np.array(vize)
numpy_final = np.array(final)

ortalama = (numpy_vize+numpy_final)/2 
print(ortalama)

