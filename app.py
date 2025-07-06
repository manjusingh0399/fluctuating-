import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("🏠 Airbnb NYC 2019 Analysis")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/AB_NYC_2019.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Listings")
neighbourhood_group = st.sidebar.multiselect("Neighbourhood Group", options=df['neighbourhood_group'].unique(), default=df['neighbourhood_group'].unique())
room_type = st.sidebar.multiselect("Room Type", options=df['room_type'].unique(), default=df['room_type'].unique())

# Filtered DataFrame
filtered_df = df[(df['neighbourhood_group'].isin(neighbourhood_group)) & (df['room_type'].isin(room_type))]

st.markdown("### 📊 Dataset Preview")
st.dataframe(filtered_df.head(100))

# Basic KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Listings", len(filtered_df))
col2.metric("Average Price", f"${filtered_df['price'].mean():.2f}")
col3.metric("Total Hosts", filtered_df['host_id'].nunique())

# Price Distribution
st.markdown("### 💰 Price Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(filtered_df[filtered_df['price'] < 500]['price'], bins=50, kde=True, ax=ax1)
st.pyplot(fig1)

# Listings by Neighborhood Group
st.markdown("### 📍 Listings by Neighbourhood Group")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.countplot(data=filtered_df, x='neighbourhood_group', order=filtered_df['neighbourhood_group'].value_counts().index, ax=ax2)
st.pyplot(fig2)

# Map
st.markdown("### 🗺️ Map of Listings")
st.map(filtered_df[['latitude', 'longitude']].dropna())


