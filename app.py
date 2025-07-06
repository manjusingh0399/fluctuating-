import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(page_title="Book Your NYC BNB", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("AB_NYC_2019.csv")

df = load_data()
if df.empty:
    st.error("Data not found.")
    st.stop()

# Define current page
page = st.query_params.get("page", "Search")

if page == "Search":
    with st.container():
        st.title("\U0001F4C5 Book Your NYC BNB")
        st.subheader("Welcome! We hope your stay is convenient and memorable \U0001F49E")
        with st.form("initial_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                checkin = st.date_input("\U0001F4C5 Check-in")
            with col2:
                checkout = st.date_input("\U0001F4C5 Check-out")
            with col3:
                guests = st.number_input("\U0001F465 Guests", min_value=1, max_value=16, step=1)

            area = st.multiselect("\U0001F3D9\uFE0F Choose Your Neighbourhood Group", df['neighbourhood_group'].unique())
            submit = st.form_submit_button("\U0001F50D Search Listings")

        if submit:
            if checkin is None or checkout is None:
                st.warning("Please select both check-in and check-out dates.")
                st.stop()

            nights_stayed = (checkout - checkin).days
            if nights_stayed <= 0:
                st.warning("Check-out must be after check-in.")
                st.stop()

            st.session_state['checkin'] = checkin
            st.session_state['checkout'] = checkout
            st.session_state['guests'] = guests
            st.session_state['nights_stayed'] = nights_stayed
            st.session_state['area'] = area

            st.query_params["page"] = "Results"
            st.rerun()

elif page == "Results":
    # Sidebar Filters (only for Results page)
    with st.sidebar:
        st.header("\u2728 Refine Your Search")
        selected_group = st.multiselect("\U0001F3E9 Neighbourhood Group", df['neighbourhood_group'].unique(), default=st.session_state.get('area', []))
        selected_room = st.multiselect("\U0001F6CC Room Type", df['room_type'].unique())
        selected_hood = st.multiselect("\U0001F4CD Neighbourhood", sorted(df['neighbourhood'].unique()))
        min_price, max_price = st.slider("\U0001F4B0 Price Range", 10, 1000, (50, 300))
        min_nights = st.slider("\U0001F4C5 Minimum Nights", 1, 30, 1)
        max_reviews = st.slider("\U0001F4AC Max Reviews", 0, 500, 300)
        min_avail = st.slider("✅ Min Availability", 0, 365, 30)

    if 'checkin' not in st.session_state:
        st.warning("Please start from the Search page.")
        st.stop()

    checkin = st.session_state['checkin']
    checkout = st.session_state['checkout']
    guests = st.session_state['guests']
    nights_stayed = st.session_state['nights_stayed']
    area = st.session_state['area']

    st.title("\U0001F389 Your NYC BNB Listings")
    st.markdown("_We hope your stay is convenient and full of memories!_")

    # Apply Filters
    filtered_df = df[df['availability_365'] >= nights_stayed]
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

    st.markdown(f"### ✨ {len(filtered_df)} Listings Matching Your Criteria")
    col1, col2, col3 = st.columns(3)
    col1.metric("\U0001F31F Listings", len(filtered_df))
    col2.metric("\U0001F4B8 Avg. Price", f"${filtered_df['price'].mean():.2f}")
    col3.metric("\U0001F469‍\U0001F4BC Hosts", filtered_df['host_id'].nunique())

    # ML Prediction
    df_model = df[df['price'] < 500]
    features = ['neighbourhood_group', 'room_type', 'minimum_nights', 'number_of_reviews', 'availability_365']
    X = pd.get_dummies(df_model[features], drop_first=True)
    y = df_model['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    input_df = pd.DataFrame({
        'neighbourhood_group': [selected_group[0] if selected_group else 'Manhattan'],
        'room_type': [selected_room[0] if selected_room else 'Private room'],
        'minimum_nights': [nights_stayed],
        'number_of_reviews': [max_reviews],
        'availability_365': [min_avail]
    })
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    predicted_price = model.predict(input_encoded)[0]

    st.markdown(f"""
        <div style='background-color:#fff0ba; padding: 20px; border-radius: 12px; text-align: center; margin: 20px 0; font-size: 20px;'>
        🌟 <b>Estimated Price for Your Stay:</b><br>
        <span style='font-size:32px; color:#ff4b6e;'>${predicted_price:.2f}</span> / night
        </div>
    """, unsafe_allow_html=True)

    # Charts
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
        st.markdown("##### 🗌️ Area Popularity")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        order = filtered_df['neighbourhood_group'].value_counts().index
        sns.countplot(data=filtered_df, x='neighbourhood_group', order=order, palette="YlOrRd", ax=ax2)
        ax2.set_title("Neighbourhood Hotspots 🔥")
        st.pyplot(fig2)

    st.markdown("### 📍 Map of Listings")
    st.map(filtered_df[['latitude', 'longitude']].dropna())

    # Recommended NYC Hotspots
    st.markdown("""
        ### 🗽 Must-Visit NYC Hotspots Nearby:
        - **Central Park** – Perfect for morning walks and relaxing afternoons.
        - **Times Square** – For vibrant nightlife, lights, and Broadway shows.
        - **Brooklyn Bridge** – A scenic walk with skyline views.
        - **Statue of Liberty** – NYC's most iconic attraction.
        - **The High Line** – An elevated park with great food, art, and views.
        - **Chelsea Market** – Delicious eats and unique local shops.
        - **SoHo** – Trendy shopping and cozy cafés.
    """)

    # Center-aligned Contact Section
    st.markdown("""
        ---
        <div style='text-align: center;'>
            <h3>📢 Contact Us</h3>
            <p>Instagram: <a href='https://instagram.com'>@nyc_bnb</a><br>
            Email: <a href='mailto:contact@nycbnb.com'>contact@nycbnb.com</a><br>
            Phone: +1 234 567 8900</p>
        </div>
    """, unsafe_allow_html=True)
