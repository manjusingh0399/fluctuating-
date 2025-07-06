import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

# Page config
st.set_page_config(page_title="Book Your NYC BNB", layout="wide")

# Theme & style
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container { padding-top: 1rem; }
        h1, h2, h3 { color: #ff4b6e; }
        .stButton>button {
            background-color: #ffce56 !important;
            color: black;
            border-radius: 10px;
            font-weight: 600;
        }
        .stMetric { background-color: #fff3cd; border-radius: 10px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🌸 Book Your NYC BNB")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("AB_NYC_2019.csv")

df = load_data()

if df.empty:
    st.error("Data not found.")
    st.stop()

# 🌟 Main Filters
with st.form("main_filter_form"):
    st.subheader("🛎️ Customize Your Stay")

    col1, col2, col3 = st.columns([1.3, 1.3, 1])
    with col1:
        checkin = st.date_input("📅 Check-in", value=None)
    with col2:
        checkout = st.date_input("📅 Check-out", value=None)
    with col3:
        guests = st.number_input("👥 Guests", min_value=1, max_value=16, step=1)

    submitted = st.form_submit_button("🔍 Search Listings")

if not submitted or checkin is None or checkout is None:
    st.info("Please select check-in and check-out dates above.")
    st.stop()

nights_stayed = (checkout - checkin).days
if nights_stayed <= 0:
    st.warning("Check-out must be after check-in.")
    st.stop()

filtered_df = df[df['availability_365'] >= nights_stayed]

# Sidebar filters
st.sidebar.header("✨ More Filters")

selected_group = st.sidebar.multiselect("🏙️ Neighbourhood Group", df['neighbourhood_group'].unique())
selected_room = st.sidebar.multiselect("🛏️ Room Type", df['room_type'].unique())
selected_hood = st.sidebar.multiselect("📍 Neighbourhood", sorted(df['neighbourhood'].unique()))
min_price, max_price = st.sidebar.slider("💰 Price Range", 10, 1000, (50, 300))
min_nights = st.sidebar.slider("📆 Minimum Nights", 1, 30, 1)
max_reviews = st.sidebar.slider("💬 Max Reviews", 0, 500, 300)
min_avail = st.sidebar.slider("✅ Min Availability", 0, 365, 30)

# Apply sidebar filters
if selected_group:
    filtered_df = filtered_df[filtered_df['neighbourhood_group'].isin(selected_group)]
if selected_room:
    filtered_df = filtered_df[filtered_df['room_type'].isin(selected_room)]
if selected_hood:
    filtered_df = filtered_df[filtered_df['neighbourhood'].isin(selected_hood)]

filtered_df = filtered_df[
    (filtered_df['price'].between(min_price, max_price)) &
    (filtered_df['minimum_nights'] >= min_nights) &
    (filtered_df['number_of_reviews'] <= max_reviews) &
    (filtered_df['availability_365'] >= min_avail)
]

# 🧾 Overview
st.markdown(f"### ✨ {len(filtered_df)} Listings Matching Your Criteria")
col1, col2, col3 = st.columns(3)
col1.metric("🌟 Listings", len(filtered_df))
col2.metric("💸 Avg. Price", f"${filtered_df['price'].mean():.2f}")
col3.metric("👩‍💼 Hosts", filtered_df['host_id'].nunique())

# 🎨 Charts
st.markdown("### 📊 Explore Your Options")

sns.set_theme(style="whitegrid", palette="pastel")
colA, colB = st.columns(2)

with colA:
    st.markdown("##### 💵 Price Spread (Under $500)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered_df[filtered_df['price'] < 500]['price'], bins=40, kde=True, color="#ffb6b9", ax=ax1)
    ax1.set_title("Prices You’ll Love 💕")
    st.pyplot(fig1)

with colB:
    st.markdown("##### 🗺️ Area Popularity")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    order = filtered_df['neighbourhood_group'].value_counts().index
    sns.countplot(data=filtered_df, x='neighbourhood_group', order=order, palette="YlOrRd", ax=ax2)
    ax2.set_title("Neighbourhood Hotspots 🔥")
    st.pyplot(fig2)

# 📍 Map
st.markdown("### 📍 Where You’ll Be Staying")
st.map(filtered_df[['latitude', 'longitude']].dropna())

# 🔮 Price Prediction
st.markdown("### 🔮 Smart Price Prediction")

model_df = df[df['price'] < 500]
features = ['neighbourhood_group', 'room_type', 'minimum_nights', 'number_of_reviews', 'availability_365']
X = pd.get_dummies(model_df[features], drop_first=True)
y = model_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Input for prediction
input_group = st.sidebar.selectbox("🔮 Predict: Neighbourhood Group", df['neighbourhood_group'].unique())
input_room = st.sidebar.selectbox("🔮 Predict: Room Type", df['room_type'].unique())
input_reviews = st.sidebar.slider("🔮 Predict: Reviews", 0, 300, 20)
input_avail = st.sidebar.slider("🔮 Predict: Availability", 0, 365, 180)

input_df = pd.DataFrame({
    'neighbourhood_group': [input_group],
    'room_type': [input_room],
    'minimum_nights': [nights_stayed],
    'number_of_reviews': [input_reviews],
    'availability_365': [input_avail]
})

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
predicted_price = model.predict(input_encoded)[0]

# 🎯 Show prediction
st.markdown(f"""
    <div style='
        background-color:#fff0ba;
        padding: 20px 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 30px;
        font-size: 20px;
        color: #212529;
        font-weight: 500;'>
    🎯 <b>Estimated Price for Your Stay:</b><br>
    <span style='font-size:32px; color:#ff4b6e;'>${predicted_price:.2f}</span> / night
    </div>
""", unsafe_allow_html=True)
