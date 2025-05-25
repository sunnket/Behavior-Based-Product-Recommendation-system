from sklearn.cluster import KMeans
import pandas as pd

def extract_features(df):
    # Count interactions per user
    features = df.groupby('user_id')['event_type'].value_counts().unstack(fill_value=0)
    return features

def run_kmeans(features, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    features['cluster'] = model.fit_predict(features)
    return features, model
