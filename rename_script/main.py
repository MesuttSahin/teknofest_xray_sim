import os

# -------- AYARLAR -------- #
FOLDER = r"C:\\Users\\messa\\Desktop\\dataset\\labels"   # Resim klasörünün yolu
PREFIX = "su_sisesi"                               # Başlık
# -------------------------- #

exts = [".jpg", ".jpeg", ".png",".txt"]

files = [f for f in os.listdir(FOLDER) if os.path.splitext(f)[1].lower() in exts]
files.sort()  # sıralı olsun

for i, filename in enumerate(files, 1):
    old_path = os.path.join(FOLDER, filename)
    ext = os.path.splitext(filename)[1]
    new_name = f"{PREFIX}_{i}{ext}"
    new_path = os.path.join(FOLDER, new_name)
    
    os.rename(old_path, new_path)

print("✔ Tüm resimler yeniden adlandırıldı!")
