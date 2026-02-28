import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Rainfall Prediction", layout="wide")

st.title("🌧️ Rainfall Prediction System")

# ---------------- LOAD MODEL + FEATURES ----------------
@st.cache_resource
def load_model():
    saved = joblib.load("model.pkl")
    return saved["model"], saved["features"]

model, feature_names = load_model()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("Rainfall.csv")

    data.columns = data.columns.str.strip().str.lower()

    # Fill missing values
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].mean())

    # Drop unused columns (same as training)
    data = data.drop(columns=['maxtemp','temparature','mintemp'], errors='ignore')

    return data

data = load_data()

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

st.subheader("Humidity Distribution")
fig3, ax3 = plt.subplots()
ax3.hist(data['humidity'], bins=20, edgecolor='black')
st.pyplot(fig3)

st.subheader("Humidity vs Pressure")
fig4, ax4 = plt.subplots()
ax4.scatter(data['humidity'], data['pressure'], alpha=0.5)
st.pyplot(fig4)

# ---------------- USER INPUT ----------------
st.sidebar.header("Enter Weather Details")

humidity = st.sidebar.slider("Humidity", 0, 100, 50)
pressure = st.sidebar.number_input("Pressure", 900, 1100, 1000)
windspeed = st.sidebar.slider("Wind Speed", 0, 100, 10)
winddirection = st.sidebar.slider("Wind Direction", 0, 360, 180)

# ---------------- CREATE FULL FEATURE VECTOR ----------------
# Create empty dataframe with SAME columns used in training
input_df = pd.DataFrame(columns=feature_names)

# Fill with dataset mean values first
for col in feature_names:
    if col in data.columns:
        input_df.loc[0, col] = data[col].mean()
    else:
        input_df.loc[0, col] = 0

# Now overwrite with user inputs (only the ones user provides)
if "humidity" in input_df.columns:
    input_df.loc[0, "humidity"] = humidity

if "pressure" in input_df.columns:
    input_df.loc[0, "pressure"] = pressure

if "windspeed" in input_df.columns:
    input_df.loc[0, "windspeed"] = windspeed

if "winddirection" in input_df.columns:
    input_df.loc[0, "winddirection"] = winddirection

# Ensure numeric
input_df = input_df.astype(float)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Rainfall"):

    prediction = model.predict(input_df)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("🌧️ Rainfall Expected")
    else:
        st.success("☀️ No Rainfall Expected")
