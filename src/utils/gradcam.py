import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from src.utils import config
from src.models.model import ChestXRayResNet

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Kanca (Hook) Mekanizması: İleri ve geri yayılımda veri yakalama
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx):
        # 1. Modeli değerlendirme moduna al ve tahmini yap
        self.model.eval()
        output = self.model(input_image)

        # Hedef sınıf için gradyanları sıfırla ve geri yayılım başlat
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # 2. Isı Haritası Hesaplama
        # Gradyanların kanal bazlı ortalamasını al (Global Average Pooling)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Özellik haritalarını bu ağırlıklarla çarp
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Kanalları birleştir ve ReLU'dan geçir (Pozitif etkileri al)
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)

        # Normalizasyon (0-1 arası)
        heatmap /= torch.max(heatmap)
        return heatmap.detach().cpu().numpy()


def apply_gradcam(image_path, model_path, target_class_idx):
    # Modeli Yükle
    model = ChestXRayResNet(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)

    # ResNet50'nin son konv. katmanına erişim: layer4[2].conv3
    target_layer = model.model.layer4[2].conv3
    cam = GradCAM(model, target_layer)

    # Resmi Hazırla
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, config.IMAGE_SIZE)

    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img_resized).unsqueeze(0).to(config.DEVICE)

    # Haritayı Üret
    heatmap = cam.generate_heatmap(input_tensor, target_class_idx)

    # 3. Görselleştirme ve Bindirme
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Orijinal resimle birleştir
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed_img, img_rgb


if __name__ == "__main__":
    import os
    from src.utils import config

    # 1. Ayarlar
    model_yolu = config.BEST_MODEL_PATH

    # Burayı klasöründeki gerçek bir resim ismiyle değiştir
    resim_yolu = 'data/raw/images/00022192_028.png'
    hastalik_id = 0  # İncelemek istediğin sınıfın indeksi

    if not os.path.exists(model_yolu):
        print(f"❌ HATA: Model dosyası bulunamadı: {model_yolu}")
    elif not os.path.exists(resim_yolu):
        print(f"❌ HATA: Resim bulunamadı: {resim_yolu}")
    else:
        print(f"🔍 Grad-CAM Analizi Başlıyor... (Model: {os.path.basename(model_yolu)})")

        # Grad-CAM uygula ve görselleştir
        result_img, original_rgb = apply_gradcam(resim_yolu, model_yolu, hastalik_id)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title("Orijinal Görüntü")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(result_img)
        plt.title(f"Grad-CAM Odak Noktası (Sınıf {hastalik_id})")
        plt.axis('off')

        plt.show()