import streamlit as st
import gdown
import os
import pickle
FILE_ID = "1DiFxT6UP34qCTJQ4n68mKFZsnJkOMnG_"
MODEL_FILE = "model.pkl"
if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
st.title("AirBNB Price Pridiction App")
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)

if st.button("Predict"):
    prediction = model.predict([[feature1, feature2]])
    st.success(f"Prediction: {prediction[0]}")
pip install streamlit gdown
streamlit run app.py
