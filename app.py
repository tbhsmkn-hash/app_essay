import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Identifikasi Nama Buah")
st.write("Unggah foto buah, dan AI akan menebak namanya!")

# Load model yang sudah dilatih
model = tf.keras.models.load_model('model_klasifikasibuah.keras')
labels = ['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon'] # Sesuaikan dengan dataset Anda

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    # Pre-processing gambar agar sesuai input model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    result = labels[np.argmax(predictions)]

    st.success(f"Hasil Identifikasi: **{result}**")
