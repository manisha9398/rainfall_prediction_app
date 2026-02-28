import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Rainfall Prediction App", layout="wide")

st.title("🌧️ Rainfall Prediction System")
st.write("Machine Learning Based Rainfall Prediction using Weather Parameters")

# ---------------- LOAD DATA (ONLY ONCE) ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("Rainfall.csv")

    # Standardize column names (fix humidity error)
    data.columns = data.columns.str.strip().str.lower()

    # Handle Missing Values (important for deployment)
    if 'humidity' in data.columns:
        data['humidity'] = data['humidity'].fillna(data['humidity'].mean())

    if 'pressure' in data.columns:
        data['pressure'] = data['pressure'].fillna(data['pressure'].mean())

    if 'windspeed' in data.columns:
        data['windspeed'] = data['windspeed'].fillna(data['windspeed'].mean())

    if 'winddirection' in data.columns:
        data['winddirection'] = data['winddirection'].fillna(data['winddirection'].mean())

    # Drop unwanted columns (same as training)
    data = data.drop(columns=['maxtemp','temparature','mintemp'], errors='ignore')

    return data

data = load_data()

# ---------------- DATA PREVIEW ----------------
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# ---------------- VISUALIZATIONS ----------------
st.subheader("📈 Rainfall Class Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="rainfall", data=data, ax=ax1)
ax1.set_title("Rainfall Distribution")
st.pyplot(fig1)

st.subheader("📊 Correlation Heatmap")
numeric_data = data.select_dtypes(include=['int64', 'float64'])
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax2)
ax2.set_title("Feature Correlation")
st.pyplot(fig2)

st.subheader("🌡️ Humidity Distribution")
fig3, ax3 = plt.subplots()
ax3.hist(data['humidity'], bins=20, edgecolor='black')
ax3.set_xlabel("Humidity")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

st.subheader("🌦️ Humidity vs Pressure Relationship")
fig4, ax4 = plt.subplots()
ax4.scatter(data['humidity'], data['pressure'], alpha=0.5)
ax4.set_xlabel("Humidity")
ax4.set_ylabel("Pressure")
st.pyplot(fig4)

# ---------------- LOAD TRAINED MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()

# ---------------- USER INPUT ----------------
st.sidebar.header("🔎 Enter Weather Details")

humidity = st.sidebar.slider("Humidity", 0, 100, 50)
pressure = st.sidebar.number_input("Pressure", 900, 1100, 1000)
windspeed = st.sidebar.slider("Wind Speed", 0, 100, 10)
winddirection = st.sidebar.slider("Wind Direction", 0, 360, 180)

# Create Feature Array (same order as training)
features = np.array([[humidity, pressure, windspeed, winddirection]])

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Rainfall"):

    prediction = model.predict(features)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("🌧️ Rainfall Expected")
    else:
        st.success("☀️ No Rainfall Expected")

st.markdown("---")
st.write("Developed using Machine Learning, Streamlit & Deployed via GitHub")
