import streamlit as st
import os
import sys
import tempfile
from PIL import Image
import torch
import numpy as np

# --- PATH AYARLARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from src import predict
    from src.utils import config
    from src import gradcam

    GRADCAM_AVAILABLE = True
except ImportError as e:
    # Gradcam yoksa hata vermesin, sadece false olsun
    if "gradcam" in str(e):
        GRADCAM_AVAILABLE = False
    else:
        st.error(f"⚠️ Modül hatası: Dosyaların yerini kontrol et! Hata: {e}")
        st.stop()

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Chest X-Ray AI Tanı", page_icon="🩻", layout="wide")

# Progress Bar Rengi (Görsellik)
st.markdown("""<style>.stProgress > div > div > div > div { background-color: #f63366; }</style>""",
            unsafe_allow_html=True)


# --- MODEL YÜKLEME (Normal Tahmin İçin) ---
@st.cache_resource
def get_model_cached():
    model_path = config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası yok: {model_path}")
        return None
    return predict.load_model(model_path)


model = get_model_cached()

# --- ARAYÜZ ---
st.title("🩻 Chest X-Ray AI Diagnosis")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("1. Görüntü Seçimi")
    uploaded_file = st.file_uploader("X-Ray Yükle", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hasta Görüntüsü", use_container_width=True)

with col2:
    st.subheader("2. Yapay Zeka Analizi")

    if uploaded_file and model:
        # Geçici dosya oluştur (Mimarın kodu path istiyor)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.spinner('Röntgen taranıyor...'):
            try:
                # 1. TAHMİN YAP (Bizim predict.py)
                probs = predict.predict_image(model, tmp_file_path)

                # En yüksek olasılıklı hastalığın indexini bul (GradCAM için lazım)
                top_class_idx = np.argmax(probs)

                # Sonuçları Sırala
                results = sorted(zip(config.CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)

                for disease, probability in results:
                    percent = int(probability * 100)
                    if percent > 5:
                        color = ":red" if percent > 50 else ":green"
                        st.markdown(f"**{disease}** {color}[%{percent}]")
                        st.progress(probability)

            except Exception as e:
                st.error(f"Tahmin Hatası: {e}")

        # --- GRAD-CAM BÖLÜMÜ ---
        st.divider()
        st.info(f"Yapay zeka en çok **{config.CLASS_NAMES[top_class_idx]}** şüphesi taşıyor.")

        if st.button("🔍 Neden Böyle Düşündün? (Odak Haritası)"):
            if GRADCAM_AVAILABLE:
                with st.spinner("Mimarın kodu çalışıyor: Isı haritası oluşturuluyor..."):
                    try:
                        # --- KRİTİK NOKTA ---
                        # Mimarın fonksiyonu şöyleydi: apply_gradcam(image_path, model_path, target_class_idx)
                        # Biz de aynen istediklerini veriyoruz:

                        heatmap_img, original_rgb = gradcam.apply_gradcam(
                            image_path=tmp_file_path,  # Resmin dosya yolu
                            model_path=config.BEST_MODEL_PATH,  # Modelin dosya yolu
                            target_class_idx=top_class_idx  # Hastalık ID'si
                        )

                        st.image(heatmap_img,
                                 caption=f"Yapay Zekanın {config.CLASS_NAMES[top_class_idx]} için Odaklandığı Bölge",
                                 use_container_width=True)

                    except Exception as e:
                        st.error(f"GradCAM Hatası: {e}")
                        st.warning("İpucu: 'src/models/model.py' dosyan var mı? Mimarın kodu bunu arıyor olabilir.")
            else:
                st.warning("⚠️ 'gradcam.py' bulunamadı veya hatalı.")

        # Temizlik (İş bitince geçici dosyayı sil)
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    elif not uploaded_file:
        st.info("👈 Lütfen soldan resim yükleyin.")