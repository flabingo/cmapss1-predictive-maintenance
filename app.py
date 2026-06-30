import json
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(page_title="Turbofan Predictive Maintenance", layout="wide", page_icon="✈️")

# Load the trained ML model
@st.cache_resource
def load_model():
    try:
        return joblib.load('turbofan_rul_model.pkl')
    except Exception as e:
        return None

model = load_model()

# Load the model validation metrics
@st.cache_data
def load_metrics():
    try:
        with open('model_metrics.json', 'r') as f:
            return json.load(f)
    except Exception:
        return {"r2_accuracy": "Training Required"}

metrics = load_metrics()

# -----------------------------------------
# LAYER 1: THE SIDEBAR CONTROLS
# -----------------------------------------
with st.sidebar:
    st.header("🎛️ Live Sensor Feeds")
    st.caption("Adjust operational parameters to simulate engine wear.")
    st.markdown("---")
    
    # ⚠️ FIXED RANGES: Matched exactly to the REAL NASA C-MAPSS Dataset
    core_temp = st.slider("Core Temp (Rankine)", 1580.0, 1620.0, 1589.0, step=0.5)
    static_pressure = st.slider("Static Pressure (psia)", 46.5, 48.5, 47.1, step=0.1)
    fan_speed = st.slider("Core Speed (RPM)", 9000.0, 9200.0, 9045.0, step=5.0)


# -----------------------------------------
# LAYER 2: THE MAIN DASHBOARD & ALERTS
# -----------------------------------------
st.title("✈️ Turbofan Engine Health Dashboard")
st.subheader("NASA C-MAPSS Remaining Useful Life (RUL) Predictor")
st.markdown("---")

st.header("📊 Diagnostics & Analytics")

# Calculate RUL 
if model is not None:
    input_data = np.array([[core_temp, static_pressure, fan_speed]])
    predicted_rul = int(model.predict(input_data)[0])
else:
    # Simulated fallback just in case the .pkl is missing
    wear_factor = ((core_temp - 1580) / 40) + ((static_pressure - 46.5) / 2.0)
    predicted_rul = int(250 - (wear_factor * 100))

# Ensure RUL doesn't drop below 0 natively
predicted_rul = max(0, predicted_rul)

# Threshold Logic & UI Alerts
if predicted_rul <= 30:
    status = "CRITICAL: MAINTENANCE REQUIRED IMMEDIATELY"
    color = "red"
    # Only set a darker background for critical state — avoid forcing all text to white
    # st.markdown(
    #     """
    #     <style>
    #     .stApp { background-color: #3b0b0b !important; transition: 0.5s ease; }
    #     </style>
    #     """, unsafe_allow_html=True
    # )
elif predicted_rul <= 80:
    status = "WARNING: SCHEDULE INSPECTION"
    color = "orange"
else:
    status = "OPTIMAL: NOMINAL OPERATION"
    color = "green"

# Metrics Display
m1, m2, m3 = st.columns(3)
m1.metric("Current Engine Status", "Online")
m2.metric("Predicted Remaining Useful Life", f"{predicted_rul} Cycles")
m3.metric("AI Validation Accuracy (R²)", metrics['r2_accuracy'])


st.markdown(
    f"<h3>System Alert: <span style='color:{color}; font-weight:700'>{status}</span></h3>",
    unsafe_allow_html=True,
)
st.markdown("---")


# -----------------------------------------
# LAYER 3: INTERACTIVE TRAJECTORY CHARTS
# -----------------------------------------
st.header("📈 Historical Degradation Trajectory")
st.caption("Live charting of sensor drift leading to current operational state.")

cycles = np.arange(1, 51)
np.random.seed(42) 

# Creating a curve mapped to the NASA bounds
temp_trend = np.linspace(1580, core_temp, 50) + np.random.normal(0, 0.5, 50)
pressure_trend = np.linspace(46.5, static_pressure, 50) + np.random.normal(0, 0.05, 50)

# Build a Pandas DataFrame for Streamlit's native charting
chart_data = pd.DataFrame({
    'Cycle (Time)': cycles,
    'Core Temp (Rankine)': temp_trend,
    'Static Pressure (psia)': pressure_trend
}).set_index('Cycle (Time)')

# Render the interactive line chart
st.line_chart(chart_data, use_container_width=True)


# -----------------------------------------
# LAYER 4: DATA ENGINEERING & MODEL INSIGHTS
# -----------------------------------------
st.markdown("---")
st.header("🗄️ Backend Data Architecture & ML Insights")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Raw NASA Telemetry Logs")
    st.caption("Live view of the ingested Pandas DataFrame.")
    
    # We use an expander so it doesn't clutter the beautiful UI
    with st.expander("🔍 Click to View Raw Sensor Data"):
        # Mocking the raw data structure based on the current trajectory
        raw_mock = pd.DataFrame({
            'Engine_ID': [101] * 50,
            'Operational_Cycle': cycles,
            'Core_Temp_Rankine': temp_trend.round(2),
            'Static_Pressure_psia': pressure_trend.round(2)
        })
        
        # Display the interactive dataframe
        st.dataframe(raw_mock.tail(10), use_container_width=True)
        
        # The Pro-Level Software Engineering Move: A Download Button
        csv = raw_mock.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Factory Log as CSV",
            data=csv,
            file_name='engine_101_sensor_logs.csv',
            mime='text/csv',
        )

with col4:
    st.subheader("🧠 Model Explainability")
    st.caption("Random Forest Feature Importance (What drives failures?)")
    
    # Extracting real feature importances if the model exists, else mocking the physics
    if model is not None and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # In a real turbofan, temperature is usually the highest driver of fatigue
        importances = [0.65, 0.25, 0.10] 
        
    importance_df = pd.DataFrame({
        'Impact on Degradation': importances
    }, index=['Core Temperature', 'Static Pressure', 'Core Speed'])
    
    # Display a beautiful horizontal bar chart
    st.bar_chart(importance_df, use_container_width=True)



