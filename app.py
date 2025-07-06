import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(page_title="Book Your NYC BNB", layout="wide")

# Display avatar (Statue of Liberty waving)
with st.container():
    st.markdown("""
    <div style='text-align:center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Statue_of_Liberty_7.jpg/200px-Statue_of_Liberty_7.jpg' width='100' style='border-radius: 10px;'>
        <h1>🗽 Book Your NYC BNB</h1>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        return pd.read_csv("AB_NYC_2019.csv")
    except FileNotFoundError:
        st.error("❌ 'AB_NYC_2019.csv' not found. Please upload it to the app directory.")
        return pd.DataFrame()

# Load data
df = load_data()
if df.empty:
    st.stop()

# Select page
page = st.query_params.get("page", "Search")

if page == "Search":
    with st.container():
        st.subheader("Welcome! We hope your stay is convenient and memorable 💞")
        with st.form("search_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                checkin = st.date_input("📅 Check-in")
            with col2:
                checkout = st.date_input("📅 Check-out")
            with col3:
                guests = st.number_input("👥 Guests", min_value=1, max_value=16, step=1)

            area = st.multiselect("🏙️ Choose Your Neighbourhood Group", df['neighbourhood_group'].unique())
            submit = st.form_submit_button("🔍 Search Listings")

        if submit:
            if checkin is None or checkout is None:
                st.warning("Please select both check-in and check-out dates.")
                st.stop()

            nights = (checkout - checkin).days
            if nights <= 0:
                st.warning("Check-out must be after check-in.")
                st.stop()

            st.session_state['checkin'] = checkin
            st.session_state['checkout'] = checkout
            st.session_state['guests'] = guests
            st.session_state['nights'] = nights
            st.session_state['area'] = area

            st.query_params["page"] = "Results"
            st.rerun()

elif page == "Results":
    if 'checkin' not in st.session_state:
        st.warning("Please start from the Search page.")
        st.stop()

    with st.sidebar:
        st.header("✨ Refine Search")
        selected_group = st.multiselect("🏘️ Neighbourhood Group", df['neighbourhood_group'].unique(), default=st.session_state.get('area', []))
        selected_room = st.multiselect("🛏️ Room Type", df['room_type'].unique())
        selected_hood = st.multiselect("📍 Neighbourhood", sorted(df['neighbourhood'].unique()))
        min_price, max_price = st.slider("💰 Price Range", 10, 1000, (50, 300))
        min_nights = st.slider("🌙 Minimum Nights", 1, 30, 1)
        max_reviews = st.slider("💬 Max Reviews", 0, 500, 300)
        min_avail = st.slider("📆 Min Availability", 0, 365, 30)

    df_filtered = df.copy()
    if selected_group:
        df_filtered = df_filtered[df_filtered['neighbourhood_group'].isin(selected_group)]
    if selected_room:
        df_filtered = df_filtered[df_filtered['room_type'].isin(selected_room)]
    if selected_hood:
        df_filtered = df_filtered[df_filtered['neighbourhood'].isin(selected_hood)]

    df_filtered = df_filtered[
        (df_filtered['price'].between(min_price, max_price)) &
        (df_filtered['minimum_nights'] >= min_nights) &
        (df_filtered['number_of_reviews'] <= max_reviews) &
        (df_filtered['availability_365'] >= min_avail)
    ]

    st.subheader("🎉 Matching Listings")
    st.write(f"✅ {len(df_filtered)} listings found")

    col1, col2, col3 = st.columns(3)
    col1.metric("Listings", len(df_filtered))
    col2.metric("Avg. Price", f"${df_filtered['price'].mean():.2f}")
    col3.metric("Hosts", df_filtered['host_id'].nunique())

    # Predict price using RandomForest
    model_df = df[df['price'] < 500]
    features = ['neighbourhood_group', 'room_type', 'minimum_nights', 'number_of_reviews', 'availability_365']
    X = pd.get_dummies(model_df[features], drop_first=True)
    y = model_df['price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    input_df = pd.DataFrame({
        'neighbourhood_group': [selected_group[0] if selected_group else 'Manhattan'],
        'room_type': [selected_room[0] if selected_room else 'Private room'],
        'minimum_nights': [st.session_state['nights']],
        'number_of_reviews': [max_reviews],
        'availability_365': [min_avail]
    })
    input_encoded = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(input_encoded)[0]

    st.markdown(f"""
        <div style='background-color:#fff0ba; padding: 15px; border-radius: 12px; text-align: center;'>
        💵 <b>Estimated Price:</b> <span style='font-size:24px; color:#ff4b6e;'>${prediction:.2f}</span> / night
        </div>
    """, unsafe_allow_html=True)

    # Charts
    st.markdown("### 📊 Visual Insights")
    sns.set_theme(style="whitegrid")
    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.histplot(df_filtered[df_filtered['price'] < 500]['price'], bins=30, kde=True, color="#FFC0CB", ax=ax1)
        ax1.set_title("Price Distribution")
        st.pyplot(fig1)

    with colB:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df_filtered, x='neighbourhood_group', palette="YlOrRd", ax=ax2)
        ax2.set_title("Neighbourhood Popularity")
        st.pyplot(fig2)

    st.markdown("### 🗺️ Listings on Map")
    st.map(df_filtered[['latitude', 'longitude']].dropna())

    st.markdown("### 🗽 Hotspots Near You")
    if selected_group:
        if 'Manhattan' in selected_group:
            st.markdown("- Central Park\n- Times Square\n- The High Line")
        if 'Brooklyn' in selected_group:
            st.markdown("- Brooklyn Bridge\n- DUMBO\n- Prospect Park")
        if 'Queens' in selected_group:
            st.markdown("- Flushing Meadows\n- Gantry Plaza")
        if 'Bronx' in selected_group:
            st.markdown("- Bronx Zoo\n- Botanical Garden")
        if 'Staten Island' in selected_group:
            st.markdown("- Staten Island Ferry\n- Staten Island Greenbelt")

    st.markdown("""
        ---
        <div style='text-align: center;'>
            <h4>📞 Contact Us</h4>
            <p>Instagram: <a href='https://instagram.com'>@nyc_bnb</a><br>
            Email: <a href='mailto:contact@nycbnb.com'>contact@nycbnb.com</a><br>
            Phone: +1 234 567 8900</p>
        </div>
    """, unsafe_allow_html=True)
