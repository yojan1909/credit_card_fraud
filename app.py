import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title(" Credit Card Fraud Detection")
st.write("Upload transaction details to check for fraud.")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0)
features = [st.number_input(f"V{i}", format="%.5f") for i in range(1, 29)]

if st.button("Predict"):
    data = np.array(features + [amount]).reshape(1, -1)
    data[:, -1] = scaler.transform(data[:, -1].reshape(-1, 1)).flatten()  # scale Amount
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("️ Fraudulent Transaction Detected!")
    else:
        st.success(" Normal Transaction")
