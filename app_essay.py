import streamlit as st
import torch
import json
import numpy as np
from PIL import Image
from pathlib import Path

# Impor arsitektur model Anda dari file proyek Anda
from app_essay.dataset import EssayScoringModel, get_transforms_and_tokenizer

# ── 1. CONFIG & LOAD MODEL (DI-CACHE) ──────────────────────────────────
@st.cache_resource
def load_multimodal_model():
    # Load konfigurasi rentang nilai skor
    with open("outputs/meta_config.json", "r") as f:
        meta = json.load(f)

    # Inisialisasi arsitektur model 
    model = EssayScoringModel()

    # Load checkpoint ke CPU (Penting agar tidak error di Streamlit Cloud)
    checkpoint_path = "outputs/checkpoints/baseline_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval() # Set ke mode evaluasi

    # Ambil tokenizer dan transformator gambar bawaan dari proyek Anda
    # Catatan: Sesuaikan argumen jika fungsi get_transforms_and_tokenizer membutuhkan model name
    tokenizer, transform_fn = get_transforms_and_tokenizer()

    return model, tokenizer, transform_fn, meta

# Muat komponen model sekali saja ke dalam memori server
try:
    model, tokenizer, transform_fn, meta_config = load_multimodal_model()
    score_min = meta_config["score_min"]
    score_max = meta_config["score_max"]
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file arsitektur dan checkpoint sudah benar. Error: {e}")
    st.stop()


# ── 2. ANTARMUKA APLIKASI (STREAMLIT UI) ───────────────────────────────
st.title("📝 Sistem Multimodal Penilaian Essay")
st.write(f"Aplikasi penilai esai otomatis berbasis teks dan gambar. (Skala Nilai: {int(score_min)} - {int(score_max)})")

# Input 1: Teks Esai
essay_text = st.text_area("Masukkan Teks Esai di Sini:", height=200, placeholder="Tuliskan jawaban esai...")

# Input 2: Unggah Gambar Pendukung (misal: lembar jawaban fisik/diagram)
uploaded_image = st.file_uploader("Unggah Gambar Pendukung (Opsional):", type=["jpg", "jpeg", "png"])


# ── 3. PROSES PREDIKSI MODEL ───────────────────────────────────────────
if st.button("Hitung Skor Esai", type="primary"):
    if not essay_text.strip():
        st.warning("Mohon masukkan teks esai terlebih dahulu!")
    else:
        with st.spinner("Model sedang menganalisis esai Anda..."):
            try:
                # A. Preprocessing Teks
                inputs = tokenizer(
                    essay_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512 # sesuaikan dengan konfigurasi training Anda
                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # B. Preprocessing Gambar
                if uploaded_image is not None:
                    image = Image.open(uploaded_image).convert("RGB")
                    # Tampilkan gambar preview
                    st.image(image, caption="Gambar yang diproses", width=300)
                    image_tensor = transform_fn(image).unsqueeze(0) # Tambah dimensi batch [1, C, H, W]
                else:
                    # Jika opsional dan kosong, buat tensor nol (sesuaikan dengan arsitektur model Anda)
                    image_tensor = torch.zeros(1, 3, 224, 224)

                # C. Jalankan Prediksi Model (Forward Pass Tanpa Gradien)
                with torch.no_grad():
                    raw_prediction = model(image_tensor, input_ids, attention_mask)
                    # Konversi ke numpy float
                    pred_normalized = raw_prediction.squeeze().item()

                # D. Denormalisasi Skor 
                scale = score_max - score_min
                final_score = np.round(pred_normalized * scale + score_min)
                final_score = np.clip(final_score, int(score_min), int(score_max))

                # E. Tampilkan Hasil Ke Pengguna
                st.success("🎉 Analisis Selesai!")

                # Visualisasi metrik nilai dengan kolom
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Skor Prediksi Akhir", value=f"{int(final_score)} / {int(score_max)}")
                with col2:
                    st.progress(float((final_score - score_min) / scale))
                    st.caption(f"Posisi nilai dalam rentang {int(score_min)}-{int(score_max)}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")
