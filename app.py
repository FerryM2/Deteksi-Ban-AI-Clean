import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Deteksi Keausan Ban", layout="centered")

st.title("🚗 Deteksi Kelayakan Ban Menggunakan AI")
st.write("Upload foto ban untuk mengetahui apakah ban masih layak dipakai.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Load labels
labels = open("labels.txt", "r").read().splitlines()

uploaded_file = st.file_uploader("Upload foto ban", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Foto ban yang diupload", use_column_width=True)

    # PREPROCESSING
    img_resized = img.resize((224, 224))  # default Teachable Machine
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # PREDIKSI
    preds = model.predict(img_array)
    index = np.argmax(preds)
    label = labels[index]
    confidence = preds[0][index] * 100

    st.subheader("Hasil Prediksi")
    st.write(f"**{label}** ({confidence:.2f}% yakin)")

    if "tidak" in label.lower() or "aus" in label.lower():
        st.error("❌ Ban sudah tidak layak, sebaiknya segera diganti.")
    else:
        st.success("✅ Ban masih dalam kondisi layak dipakai.")
