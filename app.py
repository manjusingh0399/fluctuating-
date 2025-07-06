import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

# Page config
st.set_page_config(page_title="Airbnb NYC 2019", layout="wide")

# Subtle UI styling
st.markdown("""
    <style>
        h1, h3, h4 {
            color: #343a40;
        }
        .stMetric {
            font-size: 18px !important;
        }
        .main > div {
            padding-top: 20px;
        }
        .block-container {
            padding-top: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .css-ffhzg2 {
            background-color: #f1f3f5 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏙️ Airbnb NYC 2019 Dashboard + Price Predictor")

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("AB_NYC_2019.csv")
    except FileNotFoundError:
        st.error("❌ 'AB_NYC_2019.csv' not found in the same folder as this file.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Sidebar - Booking filters
    st.sidebar.header("🔍 Filter Listings")

    st.sidebar.subheader("📅 Booking Dates")
    checkin = st.sidebar.date_input("Check-in", datetime.today())
    checkout = st.sidebar.date_input("Check-out", datetime.today())
    nights_stayed = (checkout - checkin).days if checkout > checkin else 0

    if nights_stayed <= 0:
        st.sidebar.warning("Please select valid check-in and check-out dates.")
        st.stop()

    st.sidebar.info(f"⏳ Stay Duration: {nights_stayed} nights")

    # Sidebar - Filters
    selected_group = st.sidebar.multiselect(
        "Neighbourhood Group",
        options=df['neighbourhood_group'].unique(),
        default=df['neighbourhood_group'].unique()
    )

    selected_room = st.sidebar.multiselect(
        "Room Type",
        options=df['room_type'].unique(),
        default=df['room_type'].unique()
    )

    # Filter for availability and selections
    available_df = df[df['availability_365'] >= nights_stayed]
    filtered_df = available_df[
        (available_df['neighbourhood_group'].isin(selected_group)) &
        (available_df['room_type'].isin(selected_room))
    ]

    # Summary Metrics
    st.markdown(f"### ✅ {len(filtered_df)} Listings Available for {nights_stayed} Nights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", len(filtered_df))
    col2.metric("Avg. Price", f"${filtered_df['price'].mean():.2f}")
    col3.metric("Unique Hosts", filtered_df['host_id'].nunique())

    # Side-by-side charts
    st.markdown("### 📈 Visualizations")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("##### 💵 Price Distribution (Under $500)")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.set_theme(style="whitegrid")
        sns.histplot(filtered_df[filtered_df['price'] < 500]['price'], bins=40, kde=True, ax=ax1, color="#adb5bd")
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with colB:
        st.markdown("##### 🏘️ Listings by Neighbourhood Group")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.countplot(
            data=filtered_df,
            x='neighbourhood_group',
            order=filtered_df['neighbourhood_group'].value_counts().index,
            palette="Greys", ax=ax2
        )
        ax2.set_ylabel("Number of Listings")
        ax2.set_xlabel("Neighbourhood Group")
        st.pyplot(fig2)

    # Map view
    st.markdown("### 🗺️ Available Listings Map")
    st.map(filtered_df[['latitude', 'longitude']].dropna())

    # ML Section
    st.markdown("### 🤖 Predict Airbnb Price")

    df_model = df[df['price'] < 500]
    features = ['neighbourhood_group', 'room_type', 'minimum_nights', 'number_of_reviews', 'availability_365']
    target = 'price'

    X = df_model[features]
    y = df_model[target]
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction input
    st.sidebar.header("💡 Predict Price")
    input_group = st.sidebar.selectbox("Neighbourhood Group", df['neighbourhood_group'].unique())
    input_room = st.sidebar.selectbox("Room Type", df['room_type'].unique())
    input_nights = nights_stayed
    input_reviews = st.sidebar.slider("Number of Reviews", 0, 300, 20)
    input_availability = st.sidebar.slider("Availability (days/year)", 0, 365, 180)

    input_df = pd.DataFrame({
        'neighbourhood_group': [input_group],
        'room_type': [input_room],
        'minimum_nights': [input_nights],
        'number_of_reviews': [input_reviews],
        'availability_365': [input_availability]
    })

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    predicted_price = model.predict(input_encoded)[0]
    st.sidebar.markdown("### 💰 Predicted Price")
    st.sidebar.success(f"${predicted_price:.2f}")
