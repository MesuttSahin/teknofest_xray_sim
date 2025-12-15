# Günlük Rapor

İsim: [Mustafa Mert Çay] Tarih: 2025-12-13

1. Bugün Ne Yaptım? 

* Ortam Kurulumu ve Smoke Test: Yeni bir Kaggle Notebook (GPU açık) ortamı oluşturdum ve GitHub repomuzu klonlayarak (teknofest_xray_sim) ortamın Git ile iletişim kurabildiğini kanıtladım.
* Kod Entegrasyonu: Yerel makinemde (VS Code) geliştirdiğim `src/models/train_mock.py` dosyasını, MustafaMertÇay/SmokeTest branch'ine başarıyla pushladım ve Kaggle ortamına çektim (`git pull`).
* Entegrasyon Testi: Klonlanan kodda bulunan `train_mock.py` dosyasını çalıştırarak `mock_result.txt` çıktı dosyasını ürettim.
* Veri Çıktısı Yönetimi: Üretilen çıktı dosyasını (`mock_result.txt`), Kaggle'da bağlı olan Google Drive'daki `03_Kaggle_Outputs` klasörüne kopyalayarak görev gerekliliğini yerine getirdim.

2. Karşılaşılan Hatalar ve Çözümler

Komutun Notebook hücresinde `!` öneki olmadan çalıştırılması. Çözüm: Terminal komutu olduğunu belirtmek için `!git clone` kullanıldı. 
Kaggle Notebook'un İnternet Ayarının kapalı olması. Çözüm: Notebook Ayarlarından (Settings) İnternet erişimi açıldı (On).
Kaggle ortamında `google.colab.drive` kütüphanesinin kullanılması. Çözüm: Kaggle arayüzündeki Bağlantılar (Connections) bölümünden Google Drive hesabı yetkilendirildi. 
`src/models/` klasörünün `.gitignore` tarafından göz ardı edilmesi. Çözüm: Dosyayı zorla eklemek için `git add -f src/models/train_mock.py` komutu kullanıldı. 

3. Sonuç

Sonuç olarak bugün kaggle ve github'ın birbiriyle kullanılabildiği ve yazdığımız herhangi eğitim kodunun kaggle'da çalışarak bir çıktı vermesini gördüm.

---
