import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Rainfall Prediction Comparison", layout="wide")

st.title("🌧️ Rainfall Prediction System (Model Comparison)")

# ---------------- LOAD MODEL PACKAGE ----------------
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

    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].mean())

    data = data.drop(columns=['maxtemp','temparature','mintemp'], errors='ignore')
    return data

data = load_data()

# ---------------- SHOW MODEL ACCURACY ----------------
st.subheader("📊 Model Performance")
st.write(f"✅ Random Forest Accuracy: **{rf_acc:.2f}**")
st.write(f"✅ XGBoost Accuracy: **{xgb_acc:.2f}**")

# ---------------- VISUALIZATIONS ----------------
st.subheader("Rainfall Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="rainfall", data=data, ax=ax1)
st.pyplot(fig1)

st.subheader("Correlation Heatmap")
numeric_data = data.select_dtypes(include=['int64','float64'])
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# ---------------- USER INPUT ----------------
st.sidebar.header("Enter Weather Details")

humidity = st.sidebar.slider("Humidity", 0, 100, 50)
pressure = st.sidebar.number_input("Pressure", 900, 1100, 1000)
windspeed = st.sidebar.slider("Wind Speed", 0, 100, 10)
winddirection = st.sidebar.slider("Wind Direction", 0, 360, 180)

# ---------------- PREPARE INPUT ----------------
input_df = pd.DataFrame(columns=feature_names)

for col in feature_names:
    input_df.loc[0, col] = data[col].mean() if col in data.columns else 0

input_df.loc[0, "humidity"] = humidity
input_df.loc[0, "pressure"] = pressure
input_df.loc[0, "windspeed"] = windspeed
input_df.loc[0, "winddirection"] = winddirection

input_df = input_df.astype(float)

# ---------------- PREDICTIONS ----------------
if st.sidebar.button("Predict Rainfall"):

    rf_pred = rf_model.predict(input_df)[0]
    xgb_pred = xgb_model.predict(input_df)[0]

    st.subheader("🔎 Prediction Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌲 Random Forest Result")
        st.success("Rainfall Expected" if rf_pred==1 else "No Rainfall")

    with col2:
        st.markdown("### ⚡ XGBoost Result")
        st.success("Rainfall Expected" if xgb_pred==1 else "No Rainfall")
