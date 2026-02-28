import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Rainfall Prediction", layout="wide")

st.title("🌧️ Rainfall Prediction System")
st.write("Predict whether it will rain based on weather conditions.")

# Upload Dataset for Visualization
uploaded_file = st.file_uploader("Upload Rainfall Dataset", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # ------------------ GRAPH 1 ------------------
    st.subheader("Rainfall Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="rainfall", data=data, ax=ax1)
    st.pyplot(fig1)

    # ------------------ GRAPH 2 ------------------
    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['int64','float64'])
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # ------------------ GRAPH 3 ------------------
    st.subheader("Humidity Distribution")
    fig3, ax3 = plt.subplots()
    ax3.hist(data['humidity'], bins=20, edgecolor='black')
    st.pyplot(fig3)

    # ------------------ GRAPH 4 ------------------
    st.subheader("Humidity vs Pressure")
    fig4, ax4 = plt.subplots()
    ax4.scatter(data['humidity'], data['pressure'], alpha=0.5)
    ax4.set_xlabel("Humidity")
    ax4.set_ylabel("Pressure")
    st.pyplot(fig4)

st.sidebar.header("Enter Weather Details")

humidity = st.sidebar.slider("Humidity", 0, 100, 50)
pressure = st.sidebar.number_input("Pressure", 900, 1100, 1000)
windspeed = st.sidebar.slider("Wind Speed", 0, 100, 10)
winddirection = st.sidebar.slider("Wind Direction", 0, 360, 180)

features = np.array([[humidity, pressure, windspeed, winddirection]])

if st.sidebar.button("Predict Rainfall"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("🌧️ Rainfall Expected")
    else:
        st.success("☀️ No Rainfall Expected")