import numpy as np
import os
import sys
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Proje dizinini ayarla
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import config


def generate_metrics_report():
    print("📊 Model Karnesi Hazırlanıyor...\n")

    # 1. Kaydedilmiş tahminleri ve gerçek etiketleri yükle
    pred_path = os.path.join(config.LOGS_DIR, 'predictions.npy')
    true_path = os.path.join(config.LOGS_DIR, 'true_labels.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print("❌ HATA: .npy dosyaları bulunamadı! Önce 'evaluate.py' çalıştırılmalı.")
        return

    y_pred_probs = np.load(pred_path)  # Olasılıklar (0.0 - 1.0)
    y_true = np.load(true_path)  # Gerçekler (0 veya 1)

    # 2. Olasılıkları 0 veya 1'e çevir (Eşik Değeri: 0.5)
    threshold = 0.5
    y_pred_binary = (y_pred_probs > threshold).astype(int)

    class_names = config.CLASS_NAMES

    # -------------------------------------------------------
    # 📝 BÖLÜM 1: SINIF BAZLI DETAYLI RAPOR
    # -------------------------------------------------------
    print(f"{'HASTALIK':<20} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'DESTEK (Sayı)':<10}")
    print("-" * 75)

    # Her sınıf için tek tek hesapla
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred_binary, average=None,
                                                                     zero_division=0)

    for i, name in enumerate(class_names):
        print(f"{name:<20} | {precision[i]:.4f}     | {recall[i]:.4f}     | {f1[i]:.4f}     | {support[i]:<10}")

    print("-" * 75)

    # -------------------------------------------------------
    # 🏆 BÖLÜM 2: GENEL PERFORMANS (ORTALAMALAR)
    # -------------------------------------------------------
    # Micro Average: Toplam doğru/yanlış sayısına bakar (Dengesiz setlerde önemlidir)
    # Macro Average: Her sınıfı eşit sayar (Nadir hastalıkların başarısını gösterir)

    print("\n🌍 GENEL ÖZET:")

    # Subset Accuracy (Exact Match): Bir resimdeki tüm hastalıkları birebir doğru bilme oranı
    subset_acc = accuracy_score(y_true, y_pred_binary)
    print(f"🔹 Tam Eşleşme Doğruluğu (Exact Match Accuracy): %{subset_acc * 100:.2f}")
    print("   (Not: Multi-label'da bu düşük çıkar, çünkü 14 hastalıktan 1'ini bile kaçırsa yanlış sayılır.)")

    # Her sınıf için ortalama accuracy
    class_acc = np.mean(y_true == y_pred_binary)
    print(f"🔹 Sınıf Başına Ortalama Doğruluk (Hamming Accuracy): %{class_acc * 100:.2f}")

    print(f"🔹 Ortalama F1 Score (Macro - Önemli): {np.mean(f1):.4f}")


if __name__ == "__main__":
    generate_metrics_report()