# train_mock.py içeriği
import datetime

OUTPUT_FILE = 'mock_result.txt'

print("Mock Eğitim Başladı...")

report_content = (
    f"Mock Eğitim Başarılı - Yerel VS Code'da Oluşturuldu\n"
    f"Tarih: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
)

with open(OUTPUT_FILE, 'w') as f:
    f.write(report_content)

print(f"Sonuç dosyası ({OUTPUT_FILE}) başarıyla oluşturuldu.")