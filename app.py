import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from fpdf import FPDF
from auth_oauth import google_oauth_login
import os

# 1️⃣ Authenticate via Google OAuth
st.set_page_config(page_title="CyberShield ML", layout="wide")
st.title("🛡️ CyberShield ML – Cyber Threat Detection")

# Add session reset option in sidebar
if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# Authentication
user_email = google_oauth_login()
if not user_email:
    st.info("Please log in with Google to continue.")
    st.stop()

st.sidebar.success(f"Logged in as {user_email}")

# 2️⃣ Load pretrained models (Decision Tree, XGBoost, Neural Network)
@st.cache_resource
def load_models():
    try:
        # Check if model files exist
        model_files = {
            "dt": "models/dt_model.pkl",
            "xgb": "models/xgb_model.pkl", 
            "nn": "models/nn_model.h5"
        }
        
        for name, path in model_files.items():
            if not os.path.exists(path):
                st.error(f"Model file not found: {path}")
                st.info("Please run train_models.py first to create the model files.")
                st.stop()
        
        dt = pickle.load(open("models/dt_model.pkl", "rb"))
        xgb = pickle.load(open("models/xgb_model.pkl", "rb"))
        nn = tf.keras.models.load_model("models/nn_model.h5")
        return dt, xgb, nn
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure all model files are properly trained and saved.")
        st.stop()

dt_model, xgb_model, nn_model = load_models()

# 3️⃣ Upload traffic log CSV
uploaded_file = st.file_uploader("📁 Upload Network Traffic CSV (from CICIDS2017 or similar)", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to begin threat analysis.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data")
    st.dataframe(df.head())
    
    # Validate required columns
    required_columns = ["Source IP", "Destination IP", "Timestamp"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"Some expected columns are missing: {missing_cols}")
        st.info("The analysis will continue with available data.")
    
except Exception as e:
    st.error(f"Error reading CSV file: {str(e)}")
    st.stop()

# 4️⃣ Extract numerical features for prediction
try:
    X = df.select_dtypes(include=[np.number])
    if X.empty:
        st.error("No numerical features found in the dataset.")
        st.info("Please ensure your CSV contains numerical columns for analysis.")
        st.stop()
    
    st.info(f"Found {X.shape[1]} numerical features for analysis.")
except Exception as e:
    st.error(f"Error processing features: {str(e)}")
    st.stop()

# 5️⃣ Predict threats using pretrained models
st.subheader("🧠 Running Threat Classification...")

try:
    predictions = {}
    
    # Decision Tree predictions
    with st.spinner("Running Decision Tree..."):
        predictions["Decision Tree"] = dt_model.predict(X)
    
    # XGBoost predictions  
    with st.spinner("Running XGBoost..."):
        predictions["XGBoost"] = xgb_model.predict(X)
    
    # Neural Network predictions
    with st.spinner("Running Neural Network..."):
        nn_pred = nn_model.predict(X)
        predictions["Neural Network"] = np.argmax(nn_pred, axis=1)
    
except Exception as e:
    st.error(f"Error during prediction: {str(e)}")
    st.info("This might be due to feature mismatch between training and test data.")
    st.stop()

# 6️⃣ Add predictions to DataFrame
df["Threat_Prediction"] = predictions["XGBoost"]
st.success("✅ Classification complete using XGBoost")

# Show model comparison
st.subheader("🔍 Model Comparison")
comparison_df = pd.DataFrame(predictions)
st.dataframe(comparison_df.head())

# 7️⃣ Show result table
st.subheader("🔍 Detected Threats")
display_columns = []
for col in ["Source IP", "Destination IP", "Timestamp"]:
    if col in df.columns:
        display_columns.append(col)
display_columns.append("Threat_Prediction")

st.dataframe(df[display_columns].head(15))

# 8️⃣ Pie Chart of threats
st.subheader("📊 Threat Distribution")
try:
    pie_data = df["Threat_Prediction"].value_counts()
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.pie(pie_data.values, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Distribution of Detected Threats")
    st.pyplot(fig1)
    plt.close()
except Exception as e:
    st.error(f"Error creating pie chart: {str(e)}")

# 9️⃣ Line Chart – Threats Over Time
if "Timestamp" in df.columns:
    st.subheader("⏱️ Threats Over Time")
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        df_valid_time = df.dropna(subset=['Timestamp'])
        
        if not df_valid_time.empty:
            hourly_counts = df_valid_time.groupby(df_valid_time["Timestamp"].dt.hour)["Threat_Prediction"].count()
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(hourly_counts.index, hourly_counts.values, marker='o', linewidth=2)
            ax2.set_xlabel("Hour of Day")
            ax2.set_ylabel("Number of Threats")
            ax2.set_title("Threat Activity by Hour")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()
        else:
            st.warning("No valid timestamps found for time-based analysis.")
    except Exception as e:
        st.warning(f"Could not parse timestamps: {e}")

# 🔟 Generate PDF report
st.subheader("📄 Export Threat Summary Report")
if st.button("📥 Generate Threat Report"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(0, 10, "CyberShield ML Threat Report", ln=True, align="C")
        pdf.ln(5)
        
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Generated by: {user_email}", ln=True)
        pdf.cell(0, 10, f"Total records analyzed: {len(df)}", ln=True)
        pdf.cell(0, 10, f"Threats detected: {sum(df['Threat_Prediction'] > 0)}", ln=True)
        pdf.ln(10)
        
        pdf.cell(0, 10, "Top 10 Threat Records:", ln=True)
        pdf.ln(5)
        
        # Add table headers
        pdf.set_font("Arial", size=10)
        for i, row in enumerate(df[display_columns].head(10).values):
            if i == 0:
                # Add headers
                pdf.cell(0, 8, " | ".join(display_columns), ln=True)
                pdf.ln(2)
            
            # Add data rows
            row_str = " | ".join([str(x)[:20] + "..." if len(str(x)) > 20 else str(x) for x in row])
            pdf.cell(0, 8, row_str, ln=True)
        
        # Create download button
        pdf_output = pdf.output(dest="S").encode("latin-1")
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_output,
            file_name=f"cybershield_threat_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
        st.success("✅ PDF report generated successfully!")
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")

# Additional Statistics
st.subheader("📈 Analysis Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", len(df))

with col2:
    threats_detected = sum(df["Threat_Prediction"] > 0)
    st.metric("Threats Detected", threats_detected)

with col3:
    threat_rate = (threats_detected / len(df)) * 100 if len(df) > 0 else 0
    st.metric("Threat Rate", f"{threat_rate:.1f}%")

with col4:
    unique_threats = len(df["Threat_Prediction"].unique())
    st.metric("Threat Types", unique_threats)
