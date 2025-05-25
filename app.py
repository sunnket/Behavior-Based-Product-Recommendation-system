import streamlit as st
from preprocess import load_data, preprocess_sessions
from apriori_model import run_apriori
from clustering_model import extract_features, run_kmeans
from visualize import show_rules

st.title("🛒 Behavior-Based Product Recommendation Engine")

data_path = "data/ecommerce_data.csv"
df = load_data(data_path)

st.subheader("📊 Raw Interaction Data")
st.dataframe(df.head())

# Preprocess for Apriori
sessions = preprocess_sessions(df)

# Apriori
st.subheader("🔗 Association Rule Mining (Apriori)")
rules = run_apriori(sessions)
st.dataframe(show_rules(rules))

# K-Means
st.subheader("🎯 User Clustering (K-Means)")
features = extract_features(df)
clustered, _ = run_kmeans(features)
st.dataframe(clustered)

st.success("✅ Model Executed Successfully!")
