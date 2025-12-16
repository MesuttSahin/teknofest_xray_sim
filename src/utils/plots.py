import matplotlib.pyplot as plt
import os


def plot_loss_curves(train_loss, val_loss, save_path="logs/loss_curve.png"):
    """
    Eğitim ve Doğrulama kayıp (loss) değerlerini çizer ve kaydeder.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', color='blue', linestyle='-')
    plt.plot(val_loss, label='Validation Loss', color='orange', linestyle='--')

    plt.title('Eğitim Süreci: Loss Grafiği')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Hata)')
    plt.legend()
    plt.grid(True)

    # Klasör yoksa oluştur
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    print(f"📈 Grafik kaydedildi: {save_path}")
    plt.close()