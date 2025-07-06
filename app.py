import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os

st.set_page_config(page_title="Airbnb NYC Dashboard", layout="wide")
st.title("🏙️ Airbnb NYC 2019 Dashboard + Price Predictor")

# Load data
@st.cache_data
def load_data():
    if os.path.exists("AB_NYC_2019.csv"):
        df = pd.read_csv("AB_NYC_2019.csv")
        return df
    else:
        st.error("⚠️ 'AB_NYC_2019.csv' not found.")
        return pd.DataFrame()

df = load_data()

if not df.empty:

    ## ------------------------ Sidebar Filters ------------------------
    st.sidebar.header("🔍 Filter Listings")
    selected_group = st.sidebar.multiselect("Neighbourhood Group", df['neighbourhood_group'].unique(), default=df['neighbourhood_group'].unique())
    selected_room = st.sidebar.multiselect("Room Type", df['room_type'].unique(), default=df['room_type'].unique())

    filtered_df = df[
        (df['neighbourhood_group'].isin(selected_group)) &
        (df['room_type'].isin(selected_room))
    ]

    ## ------------------------ Metrics ------------------------
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", len(filtered_df))
    col2.metric("Average Price", f"${filtered_df['price'].mean():.2f}")
    col3.metric("Active Hosts", filtered_df['host_id'].nunique())

    ## ------------------------ Visuals ------------------------
    st.markdown("### 💵 Price Distribution (Under $500)")
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df[filtered_df['price'] < 500]['price'], bins=40, kde=True, ax=ax1, color="#FF5A5F")
    st.pyplot(fig1)

    st.markdown("### 🏘️ Listings by Neighbourhood Group")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=filtered_df, x='neighbourhood_group', order=filtered_df['neighbourhood_group'].value_counts().index, palette="pastel", ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 🗺️ Map of Listings")
    st.map(filtered_df[['latitude', 'longitude']].dropna())

    ## ------------------------ Machine Learning Model ------------------------

    st.markdown("### 🤖 Predict Airbnb Price")

    # Drop outliers
    df_model = df[df['price'] < 500]

    # Select features
    features = ['neighbourhood_group', 'room_type', 'minimum_nights', 'number_of_reviews', 'availability_365']
    target = 'price'

    X = df_model[features]
    y = df_model[target]

    # Encode categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    ## ------------------------ Live Prediction ------------------------

    st.sidebar.header("💡 Predict Price")
    input_group = st.sidebar.selectbox("Neighbourhood Group", df['neighbourhood_group'].unique())
    input_room = st.sidebar.selectbox("Room Type", df['room_type'].unique())
    input_nights = st.sidebar.slider("Minimum Nights", 1, 30, 3)
    input_reviews = st.sidebar.slider("Number of Reviews", 0, 300, 10)
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
    st.sidebar.success(f"💰 Predicted Price: ${predicted_price:.2f}")
