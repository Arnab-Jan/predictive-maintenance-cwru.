import streamlit as st
import numpy as np
import pandas as pd
import joblib
import scipy.io
import matplotlib.pyplot as plt

from src.feature_extraction import extract_time_features, extract_freq_features

# Dashboard Page Configuration
st.set_page_config(
    page_title="Bearing Fault Diagnostics System",
    page_icon="⚙️",
    layout="wide"
)

CLASS_NAMES = {0: 'Normal Baseline', 1: 'Ball Fault', 2: 'Inner Race Fault', 3: 'Outer Race Fault'}

@st.cache_resource
def load_model():
    return joblib.load('bearing_fault_rf_model.pkl')

model = load_model()

st.title("⚙️ Predictive Maintenance: CWRU Bearing Diagnostics")
st.markdown("Upload raw 48 kHz drive end vibration data (`.mat` file) to extract time/frequency features and run fault classification.")

# Sidebar Controls
st.sidebar.header("Data Selection")
uploaded_file = st.sidebar.file_uploader("Upload raw CWRU MAT file", type=['mat'])

if uploaded_file is not None:
    try:
        mat = scipy.io.loadmat(uploaded_file)
        de_keys = [k for k in mat.keys() if 'DE_time' in k]
        
        if not de_keys:
            st.error("No Drive End (DE_time) signal found in the uploaded file.")
            st.stop()
            
        raw_signal = mat[de_keys[0]].flatten()
        st.sidebar.success(f"Loaded {len(raw_signal):,} raw signal samples.")
        
        # Window Selection Slider
        max_start = max(0, len(raw_signal) - 2048)
        start_sample = st.sidebar.slider("Select Window Start Sample", 0, max_start, 0, step=2048)
        
        window = raw_signal[start_sample : start_sample + 2048]
        
        # Calculate features
        t_feats = extract_time_features(window)
        f_feats = extract_freq_features(window, fs=48000)
        feature_dict = {**t_feats, **f_feats}
        df_features = pd.DataFrame([feature_dict])
        
        # Perform Model Inference
        prediction = model.predict(df_features)[0]
        probabilities = model.predict_proba(df_features)[0]
        
        # Layout: Prediction Banner & Gauges
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Diagnostic Result")
            pred_name = CLASS_NAMES[prediction]
            
            if prediction == 0:
                st.success(f"**Status:** {pred_name}")
            else:
                st.error(f"**Fault Detected:** {pred_name}")
                
            st.write("### Prediction Confidence")
            for idx, prob in enumerate(probabilities):
                st.progress(float(prob), text=f"{CLASS_NAMES[idx]}: {prob*100:.1f}%")

        with col2:
            st.subheader("Raw Signal Window (2048 Samples)")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(window, color='#1f77b4', linewidth=0.8)
            ax.set_title("Time-Domain Waveform")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Acceleration (g)")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        st.markdown("---")
        
        # Feature Breakdown Section
        st.subheader("Extracted Diagnostic Features")
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            st.markdown("**Time-Domain Metrics**")
            st.dataframe(pd.DataFrame(t_feats.items(), columns=['Metric', 'Value']), use_container_width=True)
            
        with f_col2:
            st.markdown("**Frequency-Domain Metrics**")
            st.dataframe(pd.DataFrame(f_feats.items(), columns=['Metric', 'Value']), use_container_width=True)
            
    except Exception as e:
        st.error(f"Error processing .mat file: {e}")

else:
    st.info("👈 Upload a `.mat` file from `data/raw/` in the sidebar to begin analysis.")
