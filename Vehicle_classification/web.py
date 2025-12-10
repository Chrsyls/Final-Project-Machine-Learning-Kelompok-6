import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ============================================
# PAGE CONFIG & STYLING
# ============================================
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling modern
st.markdown("""
    <style>
    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-container h1 {
        color: white;
        margin: 0;
        font-size: 2.5em;
    }
    
    .header-container p {
        color: rgba(255, 255, 255, 0.9);
        margin: 10px 0 0 0;
        font-size: 1.1em;
    }
    
    /* Result card styling */
    .result-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
    }
    
    .metric-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .prediction-title {
        color: #667eea;
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .confidence-score {
        font-size: 2.5em;
        font-weight: bold;
        color: #667eea;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown("""
    <div class="header-container">
        <h1>Vehicle Classifier</h1>
        <p>Klasifikasi kendaraan menggunakan Deep Learning</p>
    </div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('vehicle_classifier_model.h5')
    return model

model = load_model()
class_names = ['Bus', 'Car', 'Motorcycle']

# ============================================
# MAIN CONTENT
# ============================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih file gambar kendaraan",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Informasi")
    st.info("""
    **Panduan:**
    - Upload gambar kendaraan (Bus, Car, atau Motorcycle)
    - Gambar akan dianalisis secara otomatis
    - Hasil menunjukkan prediksi dan confidence score
    """)

# ============================================
# PROCESS & DISPLAY RESULTS
# ============================================
if uploaded_file is not None:
    col_img, col_result = st.columns([1.2, 1], gap="large")
    
    with col_img:
        st.subheader("📸 Gambar Input")
        image = Image.open(uploaded_file)
        # Use `use_container_width` (replacement for deprecated `use_column_width`)
        st.image(image, use_container_width=True, caption="Gambar yang diupload")
    
    with col_result:
        st.subheader("🎯 Hasil Analisis")
        
        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Prediksi
        with st.spinner("Menganalisis gambar..."):
            predictions = model.predict(img_array, verbose=0)
        
        # Ambil hasil
        predicted_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_idx]
        confidence = np.max(predictions[0]) * 100
        
        # Display prediction
        st.markdown(f"""
            <div class="result-card">
                <div class="prediction-title">Prediksi:</div>
                <div style="font-size: 2em; font-weight: bold; color: #667eea; margin: 15px 0;">
                    {predicted_class}
                </div>
                <div class="prediction-title" style="margin-top: 20px;">Confidence:</div>
                <div class="confidence-score">{confidence:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # PROBABILITY BREAKDOWN
    # ============================================
    st.subheader("📊 Probabilitas Setiap Kelas")
    
    # Create dataframe for better visualization
    prob_data = pd.DataFrame({
        'Kelas': class_names,
        'Probabilitas': predictions[0],
        'Persentase': [f"{p*100:.1f}%" for p in predictions[0]]
    })
    
    col_chart, col_table = st.columns([1.5, 1])
    
    with col_chart:
        # Bar chart dengan warna gradien
        chart_data = pd.DataFrame({
            'Kelas': class_names,
            'Nilai': predictions[0]
        }).set_index('Kelas')
        st.bar_chart(chart_data, color=['#667eea'])
    
    with col_table:
        st.dataframe(
            prob_data[['Kelas', 'Persentase']],
            use_container_width=True,
            hide_index=True
        )
    
    # ============================================
    # CONFIDENCE INDICATOR
    # ============================================
    st.subheader("📈 Tingkat Kepercayaan")
    
    # Color-coded confidence indicator
    if confidence >= 80:
        color = "🟢"
        level = "Sangat Tinggi"
    elif confidence >= 60:
        color = "🟡"
        level = "Tinggi"
    else:
        color = "🔴"
        level = "Rendah"
    
    col_conf1, col_conf2 = st.columns([2, 1])
    with col_conf1:
        # Compute a native Python float in range [0.0, 1.0] for st.progress
        progress_value = float(np.clip(confidence / 100.0, 0.0, 1.0))
        st.progress(progress_value)
    with col_conf2:
        st.metric("Status", level, f"{color}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em; margin-top: 20px;">
        <p>🤖 Powered by TensorFlow & MobileNetV2</p>
        <p>Vehicle Classification Model v1.0</p>
    </div>
""", unsafe_allow_html=True)