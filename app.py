import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Rainfall Prediction Comparison", layout="wide")

st.title("🌧️ Rainfall Prediction System with Model Comparison")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    saved = joblib.load("model.pkl")
    return saved

saved = load_models()

rf_model = saved["rf_model"]
xgb_model = saved["xgb_model"]
feature_names = saved["features"]
rf_acc = saved["rf_accuracy"]
xgb_acc = saved["xgb_accuracy"]

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("Rainfall.csv")

    data.columns = data.columns.str.strip().str.lower()

    # Fill missing numeric values
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].mean())

    data = data.drop(columns=['maxtemp','temparature','mintemp'], errors='ignore')

    return data

data = load_data()

# ---------------- SHOW MODEL PERFORMANCE ----------------
st.subheader("📊 Model Performance Comparison")
col1, col2 = st.columns(2)
col1.metric("Random Forest Accuracy", f"{rf_acc:.2f}")
col2.metric("XGBoost Accuracy", f"{xgb_acc:.2f}")

st.markdown("---")

# ================= VISUALIZATION SECTION =================
st.header("📈 Data Visualization")

# ---- Graph 1 : Rainfall Distribution ----
st.subheader("Rainfall Class Distribution")
fig1, ax1 = plt.subplots(figsize=(5,4))
sns.countplot(x="rainfall", data=data, ax=ax1)
ax1.set_title("Rainfall Distribution")
st.pyplot(fig1)

# ---- Graph 2 : Correlation Heatmap ----
st.subheader("Feature Correlation Heatmap")
numeric_data = data.select_dtypes(include=['int64','float64'])
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# ---- Graph 3 : Histogram (Distribution Plot) ----
st.subheader("Humidity Distribution")
fig3, ax3 = plt.subplots(figsize=(6,4))
ax3.hist(data['humidity'], bins=20, edgecolor='black')
ax3.set_xlabel("Humidity")
ax3.set_ylabel("Frequency")
ax3.set_title("Distribution of Humidity")
st.pyplot(fig3)

# ---- Graph 4 : Scatter (Dotted Plot) ----
st.subheader("Humidity vs Pressure (Relationship Plot)")
fig4, ax4 = plt.subplots(figsize=(6,4))
ax4.scatter(data['humidity'], data['pressure'], alpha=0.5)
ax4.set_xlabel("Humidity")
ax4.set_ylabel("Pressure")
ax4.set_title("Humidity vs Pressure")
st.pyplot(fig4)

st.markdown("---")

# ================= USER INPUT =================
st.header("🔎 Enter Weather Details for Prediction")

humidity = st.sidebar.slider("Humidity", 0, 100, 50)
pressure = st.sidebar.number_input("Pressure", 900, 1100, 1000)
windspeed = st.sidebar.slider("Wind Speed", 0, 100, 10)
winddirection = st.sidebar.slider("Wind Direction", 0, 360, 180)

# ---------------- CREATE INPUT FORMAT ----------------
input_df = pd.DataFrame(columns=feature_names)

for col in feature_names:
    input_df.loc[0, col] = data[col].mean() if col in data.columns else 0

input_df.loc[0, "humidity"] = humidity
input_df.loc[0, "pressure"] = pressure
input_df.loc[0, "windspeed"] = windspeed
input_df.loc[0, "winddirection"] = winddirection

input_df = input_df.astype(float)

# ================= PREDICTION =================
if st.sidebar.button("Predict Rainfall"):

    rf_pred = rf_model.predict(input_df)[0]
    xgb_pred = xgb_model.predict(input_df)[0]

    st.subheader("🌦️ Prediction Results")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🌲 Random Forest")
        st.success("Rainfall Expected" if rf_pred==1 else "No Rainfall")

    with c2:
        st.markdown("### ⚡ XGBoost")
        st.success("Rainfall Expected" if xgb_pred==1 else "No Rainfall")
